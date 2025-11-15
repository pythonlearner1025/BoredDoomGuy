import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from utils import (
    encode_images_to_latents, 
    decode_latents_to_images, 
    create_rollout_comparison_grid
)
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers.optimization import Adafactor


@dataclass
class DoomSimCfg:
    n_hist: int = 64
    alpha_max: float = 0.7
    k_buckets: int = 10
    cfg_weight: float = 1.5
    num_inference_steps: int = 4
    text_hidden_size: int = 768
    noise_embed_dim: int = 256
    action_vocab_size: int = 6
    grad_clip: float = 1.0
    mixed_precision: bool = False
    latent_channels: int = 4
    vae_scaling_factor: float = 0.18215
    height: int = 120
    width: int = 160
    prediction_type: str = "sample"
    context_dropout_prob: float = 0.1  # Probability of dropping context frames during training for CFG
    gradient_checkpointing: bool = True  # Enable gradient checkpointing to save VRAM

VALID_ACTIONS = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
    "USE",
]

def buttons_to_action_one_hot(buttons: list[str]) -> torch.Tensor:
    one_hot = torch.zeros(len(VALID_ACTIONS), dtype=torch.float32)
    for button in buttons:
        if button in VALID_ACTIONS:
            one_hot[VALID_ACTIONS.index(button)] = 1.0
    return one_hot

class DoomDataset(Dataset):
    def __init__(self, data_dir: str, n_hist: int, height: int = 240, width: int = 320, preload_to_ram: bool = True):
        """
        Args:
            data_dir: Directory containing episode subdirectories
            n_hist: Number of history frames to load
            height: Target image height
            width: Target image width
            preload_to_ram: If True, preload all images and actions into RAM (default: True)
        """
        self.data_dir = data_dir
        self.n_hist = n_hist
        self.height = height
        self.width = width
        
        # Find all episodes (exclude hidden directories like .cache)
        self.episode_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                             if os.path.isdir(os.path.join(data_dir, f)) and not f.startswith('.')]
        self.episode_paths.sort()
        
        print(f"Loading {len(self.episode_paths)} episodes from {data_dir}...")
        
        # Preload all data into RAM for maximum efficiency
        self.episodes_data = []  # List of (frames_tensor, actions_tensor) tuples
        
        # Always compute episode lengths as fallback (needed if preload fails)
        self.episode_lengths = [len([f for f in os.listdir(ep) if f.endswith(".png")]) 
                               for ep in self.episode_paths]
        
    def __len__(self):
        return len(self.episodes_data) if self.episodes_data else len(self.episode_paths)
    
    def load_episode_frames(self, idx: int, n_frames: int, start_idx: int = 0):
        """Load specific frames from an episode (used for test/validation)."""
        ep_path = self.episode_paths[idx]
        actual_n_frames = min(n_frames, self.episode_lengths[idx] - start_idx)
        
        frames, actions = [], []
        frame_files = sorted([f for f in os.listdir(ep_path) if f.endswith(".png")])
        json_files = sorted([f for f in os.listdir(ep_path) if f.startswith("frame_") and f.endswith(".json")])
        
        for i in range(start_idx, start_idx + actual_n_frames):
            img = Image.open(os.path.join(ep_path, frame_files[i])).convert("RGB")
            img = img.resize((self.width, self.height), Image.BILINEAR)
            frames.append(torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0)
            
            with open(os.path.join(ep_path, json_files[i]), "r") as f:
                buttons = json.load(f)["action"]["buttons"]
                actions.append(buttons_to_action_one_hot(buttons))
        
        while len(frames) < n_frames:
            frames.append(torch.zeros(3, self.height, self.width))
            actions.append(torch.zeros(len(VALID_ACTIONS)))
        
        return torch.stack(frames), torch.stack(actions)
    
    def __getitem__(self, idx):
        """Fast random access from preloaded RAM data."""
        ep_path = self.episode_paths[idx]
        ep_len = self.episode_lengths[idx]
        start_idx = 0 if ep_len <= self.n_hist else random.randint(0, ep_len - self.n_hist)
        actual_n_hist = min(ep_len, self.n_hist)
        
        frames, actions = [], []
        frame_files = sorted([f for f in os.listdir(ep_path) if f.endswith(".png")])
        json_files = sorted([f for f in os.listdir(ep_path) if f.startswith("frame_") and f.endswith(".json")])
        
        for i in range(start_idx, start_idx + actual_n_hist):
            img = Image.open(os.path.join(ep_path, frame_files[i])).convert("RGB")
            img = img.resize((self.width, self.height), Image.BILINEAR)
            frames.append(torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0)
            
            with open(os.path.join(ep_path, json_files[i]), "r") as f:
                buttons = json.load(f)["action"]["buttons"]
                actions.append(buttons_to_action_one_hot(buttons))
        
        while len(frames) < self.n_hist:
            frames.append(torch.zeros(3, self.height, self.width))
            actions.append(torch.zeros(len(VALID_ACTIONS)))
        
        return torch.stack(frames), torch.stack(actions)
    

class Conditioner(nn.Module):
    def __init__(
        self,
        action_vocab_size: int,
        out_hidden_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_proj = nn.Linear(action_vocab_size, out_hidden_size)
    
    def forward(self, actions: torch.FloatTensor, caption_embeds: Optional[torch.FloatTensor]) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        '''Output is pure action embeddings[B, N_hist, H] if caption is None'''
        # actions: [B, N_hist, action_vocab_size] (one-hot vectors)
        act_tokens = self.action_proj(actions)  # [B, N_hist, H]
        obs_mask_parts = [torch.zeros(actions.shape[:2], dtype=torch.bool, device=actions.device)]
        tokens = [act_tokens]
        if caption_embeds is not None:
            tokens.append(caption_embeds)
            obs_mask_parts.append(torch.ones(caption_embeds.shape[:2], dtype=torch.bool, device=caption_embeds.device))
        enc = torch.cat(tokens, dim=1)  # [B, N_hist(+T_cap), H]
        obs_mask = torch.cat(obs_mask_parts, dim=1)  # True where observation tokens
        return enc, obs_mask

class NoiseBucketer(nn.Module):
    def __init__(self, k_buckets: int, alpha_max: float, noise_embed_dim: int):
        super().__init__()
        self.k = k_buckets
        self.alpha_max = alpha_max
        self.embed = nn.Embedding(k_buckets, noise_embed_dim)
    
    def forward(self, ids: torch.LongTensor) -> torch.FloatTensor:
        return self.embed(ids)
    
    @torch.no_grad()
    def sample_alphas(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch, device=device) * self.alpha_max

    def bucketize(self, alpha: torch.Tensor) -> torch.LongTensor:
        # alpha in [0, alpha_max] -> [0, k-1]
        scaled = (alpha / max(self.alpha_max, 1e-8)) * self.k
        ids = torch.clamp(scaled.long(), 0, self.k - 1)
        return ids

def inflate_conv_in_for_history(unet: UNet2DConditionModel, old_in: int, new_in: int) -> None:
    '''
        Updates conv_in weight and operator to input new_in instead of old_in. Output channels remain the same.

        OLD
        - x_t: [B, 4, W/8, H/8] (VAE encoder compresses W, H and outputs 4 channels)
        - conv_in(x_t): [B, C_out, W/8, H/8]
        NEW
        - x_t: [B, 4 + (N_hist - 1)*4, W/8, H/8] (concatenate channels from N_hist-1 frames and x_t)
        - conv_in(x_t): [B, C_out, W/8, H/8]
    '''
    with torch.no_grad():
        w = unet.conv_in.weight  # [C_out, old_in, k, k]
        b = unet.conv_in.bias
        C_out, _, k1, k2 = w.shape
        if unet.conv_in.in_channels != old_in:
            raise ValueError(f"Expected old_in={old_in}, found layer.in_channels={unet.conv_in.in_channels}")
        new_w = torch.zeros(C_out, new_in, k1, k2, device=w.device, dtype=w.dtype) # expand so new_in = 4 + (N_hist - 1)*4
        new_w[:, :old_in] = w  # copy the first 4 channels for x_t
        new_conv = nn.Conv2d(new_in, C_out, kernel_size=w.shape[-1], padding=unet.conv_in.padding)
        new_conv = new_conv.to(w.device, w.dtype)
        new_conv.weight.copy_(new_w)
        new_conv.bias.copy_(b) # bias stays the same
        unet.conv_in = new_conv

def build_models(cfg: DoomSimCfg, device: torch.device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    print(f"Using dtype: {dtype}")
    
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=dtype
    ).to(device)
    
    # Freeze entire VAE and set to eval mode to prevent memory leaks
    vae.requires_grad_(False)
    vae.eval()

    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet", torch_dtype=dtype
    ).to(device)
    
    # Enable gradient checkpointing to save VRAM (trades compute for memory)
    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    in_channels = cfg.latent_channels * (cfg.n_hist - 1)
    total_in_channels = cfg.latent_channels + in_channels
    inflate_conv_in_for_history(unet, old_in=4, new_in=total_in_channels)
    
    if unet.time_embedding.cond_proj is None:
        time_embed_input_dim = unet.time_embedding.linear_1.in_features
        unet.time_embedding.cond_proj = nn.Linear(cfg.noise_embed_dim, time_embed_input_dim).to(device, dtype=dtype)
    
    scheduler = DDIMScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="scheduler", prediction_type=cfg.prediction_type
    )
    
    conditioner = Conditioner(
        action_vocab_size=cfg.action_vocab_size,
        out_hidden_size=cfg.text_hidden_size,
    ).to(device, dtype=dtype)
    
    noise_bucketer = NoiseBucketer(cfg.k_buckets, cfg.alpha_max, cfg.noise_embed_dim).to(device, dtype=dtype)
    
    caption_encoder = None
    unet.requires_grad_(True)
    total_params = sum(p.numel() for p in unet.parameters())
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"UNet: {trainable_params:,} / {total_params:,} parameters trainable ({100 * trainable_params / total_params:.2f}%)")

    return vae, unet, scheduler, conditioner, noise_bucketer, caption_encoder

def cfg_with_context_dropout(unet, x_noisy, context_latents, t, enc_states, timestep_cond, cfg_weight):
    """
    Apply Classifier-Free Guidance by running the model twice:
    1. With context latents (conditional)
    2. Without context latents (unconditional - zeros)
    
    This matches the paper's approach of dropping context frames with probability 0.1 during training
    and using CFG at inference time.
    """
    model_dtype = next(unet.parameters()).dtype
    # Conditional: use actual context
    x_cond = torch.cat([x_noisy, context_latents.flatten(1, 2)], dim=1).to(dtype=model_dtype)
    
    # Unconditional: use zero context (simulating context dropout)
    context_uncond = torch.zeros_like(context_latents)
    x_uncond = torch.cat([x_noisy, context_uncond.flatten(1, 2)], dim=1).to(dtype=model_dtype)
    
    # Run both through the model in a single batch for efficiency
    x_cat = torch.cat([x_uncond, x_cond], dim=0)
    t_cat = torch.cat([t, t], dim=0)
    enc_cat = torch.cat([enc_states, enc_states], dim=0).to(dtype=model_dtype)
    cond_cat = torch.cat([timestep_cond, timestep_cond], dim=0).to(dtype=model_dtype)
    
    out = unet(sample=x_cat, timestep=t_cat, encoder_hidden_states=enc_cat, timestep_cond=cond_cat).sample
    pred_uncond, pred_cond = out.chunk(2, dim=0)
    
    # Blend predictions using CFG weight
    return pred_uncond + cfg_weight * (pred_cond - pred_uncond)


def train_step_teacher_forced(vae, unet, scheduler, conditioner, noise_bucketer, caption_encoder, cfg, 
                              past_images, actions, current_images, captions=None): 
    device, B, N_hist = current_images.device, *past_images.shape[:2]
    if captions is not None:
        assert len(captions) == B

    x0 = encode_images_to_latents(vae, current_images, cfg.vae_scaling_factor)
    ctxt = encode_images_to_latents(vae, past_images.flatten(0, 1), cfg.vae_scaling_factor).reshape(B, N_hist, *x0.shape[1:])

    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long)
    # Noise must be detached - it's a target, not part of the computation graph
    noise = torch.randn_like(x0, dtype=x0.dtype).detach()
    x_t = scheduler.add_noise(x0, noise, timesteps)

    alpha = torch.rand(B, device=device) * cfg.alpha_max
    ctxt_noisy = ctxt + alpha.view(B, 1, 1, 1, 1) * torch.randn_like(ctxt, dtype=ctxt.dtype)
    bucket_ids = noise_bucketer.bucketize(alpha)
    
    # Context dropout: randomly zero out context LATENTS for CFG training
    # This teaches the model to work both with and without context
    if random.random() < cfg.context_dropout_prob:
        ctxt_noisy = torch.zeros_like(ctxt_noisy)

    caption_embeds = caption_encoder(captions) if caption_encoder and captions else None
    action_text_embeds, obs_mask = conditioner(actions, caption_embeds)
    timestep_cond = noise_bucketer(bucket_ids)

    # During training: simple forward pass without CFG (saves memory)
    # Context dropout already trains the model for CFG at inference time
    model_dtype = next(unet.parameters()).dtype
    x_in = torch.cat([x_t, ctxt_noisy.flatten(1, 2)], dim=1).to(dtype=model_dtype)
    model_output = unet(
        sample=x_in, 
        timestep=timesteps, 
        encoder_hidden_states=action_text_embeds.to(dtype=model_dtype), 
        timestep_cond=timestep_cond.to(dtype=model_dtype)
    ).sample
    
    return F.mse_loss(model_output, x0)

def train(data_dir: str = "debug/rnd", batch_size: int = 4, num_steps: int = 10000, lr: float = 5e-5, prediction_type: str = "v_prediction", gradient_checkpointing: bool = True, preload_to_ram: bool = False, rollout_interval: int = 100, use_adafactor: bool = False, use_cosine_decay: bool = False, warmup_steps: int = 0, hold_steps: int = 0, full_ft: bool = False):
    cfg = DoomSimCfg()
    cfg.prediction_type = prediction_type
    cfg.gradient_checkpointing = gradient_checkpointing
    cfg.use_lora = not full_ft  # Use LoRA unless full fine-tuning is requested
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vae, unet, scheduler, conditioner, noise_bucketer, caption_encoder = build_models(cfg, device)
    train_dataset = DoomDataset(data_dir=data_dir, n_hist=cfg.n_hist, height=cfg.height, width=cfg.width, preload_to_ram=preload_to_ram)
    print(f"Dataset mode: {'Preloaded to RAM' if preload_to_ram else 'Load on-demand from disk'}")
    
    # DataLoader with multiple workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,  # 16 parallel workers for data loading
        pin_memory=True,  # Faster CPU-to-GPU transfers
    )
    print(f"DataLoader configured with 16 parallel workers")
    
    model_dtype = next(unet.parameters()).dtype
    
    # Use Adafactor optimizer (memory-efficient alternative to AdamW)
    if use_adafactor:
        optimizer = Adafactor(
            list(unet.parameters()) + list(conditioner.parameters()) + list(noise_bucketer.parameters()),
            lr=lr,
            scale_parameter=True,  # Scale learning rate by RMS of parameters
            relative_step=False,   # Use fixed learning rate (not adaptive)
            warmup_init=False,      # No warmup phase
            weight_decay=0.0,       # No weight decay
            decay_mode="cosine",   # Cosine decay
            max_grad_norm=1.0,     # No gradient clipping
            lr_scheduler="constant",  # No learning rate scheduler
        )
    else:
        optimizer = torch.optim.AdamW(
            list(unet.parameters()) + list(conditioner.parameters()) + list(noise_bucketer.parameters()), lr=lr
        )

    lr_scheduler = None
    if use_cosine_decay or warmup_steps > 0 or hold_steps > 0:
        def lr_lambda(step):
            """
            LR schedule with three phases:
            1. Warmup: [0, warmup_steps] - linear from 0 to target lr (lr parameter)
            2. Hold: [warmup_steps, warmup_steps + hold_steps] - constant at target lr
            3. Decay: [warmup_steps + hold_steps, num_steps] - cosine decay from target lr to 0
            """
            if step < warmup_steps:
                # Warmup phase: linear warmup from 0 to 1.0 (target lr)
                return float(step) / float(max(1, warmup_steps))
            elif step < warmup_steps + hold_steps:
                # Hold phase: constant at target lr
                return 1.0
            else:
                # Cosine decay phase: from target lr to 0
                if not use_cosine_decay:
                    # If no cosine decay requested, stay at target lr
                    return 1.0
                decay_steps = num_steps - (warmup_steps + hold_steps)
                progress = (step - warmup_steps - hold_steps) / float(max(1, decay_steps))
                # Cosine annealing from lr to 0
                cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
                return cosine_decay
        
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    rollout_steps = 8
    test_frames, test_actions = train_dataset.load_episode_frames(idx=0, n_frames=cfg.n_hist + rollout_steps, start_idx=0)
    # save the first test_frames to check if it is all black or not
    test_context = test_frames[:cfg.n_hist].unsqueeze(0).to(device=device, dtype=model_dtype)
    test_action_seq = test_actions[cfg.n_hist:].unsqueeze(0).to(device=device, dtype=model_dtype)
    test_gt_frames = test_frames[cfg.n_hist:].unsqueeze(0).to(device=device, dtype=model_dtype)
    # Initialize action history buffer with the first n_hist actions that produced the initial context
    test_action_history = test_actions[:cfg.n_hist].unsqueeze(0).to(device=device, dtype=model_dtype)

    rollout_dir = f"debug/rollouts/{timestamp}"
    os.makedirs(rollout_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Training mode: {'Full Fine-Tuning' if full_ft else f'LoRA (r={cfg.lora_r}, alpha={cfg.lora_alpha})'}")
    print(f"Dataset: {len(train_dataset)} episodes")
    print(f"Batch size: {batch_size}")
    print(f"Total training steps: {num_steps}")
    if lr_scheduler is not None:
        schedule_desc = []
        if warmup_steps > 0:
            schedule_desc.append(f"warmup to {lr:.2e} over {warmup_steps} steps")
        else:
            schedule_desc.append(f"start at {lr:.2e}")
        if hold_steps > 0:
            schedule_desc.append(f"hold for {hold_steps} steps")
        if use_cosine_decay:
            schedule_desc.append(f"cosine decay to 0")
        print(f"Learning rate schedule: {' → '.join(schedule_desc)}")
    else:
        print(f"Learning rate: {lr:.2e} (constant)")
    print(f"Context length (n_hist): {cfg.n_hist}")
    print(f"Context dropout probability: {cfg.context_dropout_prob}")
    print(f"Noise augmentation: alpha_max={cfg.alpha_max}, buckets={cfg.k_buckets}")
    print(f"CFG weight: {cfg.cfg_weight}")
    print(f"Prediction type: {cfg.prediction_type}")
    print(f"Gradient checkpointing: {'ENABLED' if cfg.gradient_checkpointing else 'DISABLED'}")
    print(f"Rollout interval: every {rollout_interval} step(s)")
    print(f"Checkpoint save interval: every 1000 steps")
    print(f"Rollouts directory: {rollout_dir}")
    print("="*60 + "\n")
    
    # Create infinite data iterator without caching batches
    def infinite_dataloader(dataloader):
        """Infinite iterator that doesn't cache batches (unlike itertools.cycle)."""
        while True:
            for batch in dataloader:
                yield batch
    
    data_iter = infinite_dataloader(train_loader)
    
    global_step = 0
    running_loss = 0.0
    log_interval = 10
    
    print(f"\nStarting training for {num_steps} steps...")
    
    while global_step < num_steps:
        frames, actions = next(data_iter)
        frames, actions = frames.to(device=device, dtype=model_dtype), actions.to(device=device, dtype=model_dtype)
        past_images, current_images, past_actions = frames[:, :-1], frames[:, -1], actions[:, :-1]
        
        loss = train_step_teacher_forced(
            vae, unet, scheduler, conditioner, noise_bucketer, caption_encoder,
            cfg, past_images, past_actions, current_images, captions=None
        )
        
        optimizer.zero_grad(set_to_none=True)  # More memory efficient than setting to zeros
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(unet.parameters()) + list(conditioner.parameters()) + list(noise_bucketer.parameters()),
            cfg.grad_clip
        )
        optimizer.step()
        
        # Step LR scheduler after each optimizer step (important for warmup/rampup)
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        global_step += 1
        running_loss += loss.item()
        
        # Log every N steps
        if global_step % log_interval == 0:
            avg_loss = running_loss / log_interval
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Step {global_step}/{num_steps}] Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
            running_loss = 0.0
        
        # Run rollout at specified intervals
        if global_step % rollout_interval == 0:
            print(f"\nRunning rollout at step {global_step}...")
            unet.eval()
            conditioner.eval()
            noise_bucketer.eval()
            
            # Temporarily disable gradient checkpointing during rollout for speed
            # (no need to save memory during inference since no gradients)
            if cfg.gradient_checkpointing:
                unet.disable_gradient_checkpointing()
            
            with torch.no_grad():
                predicted_frames = []
                current_context = test_context.clone()
                action_history = test_action_history.clone()

                for step in range(rollout_steps):
                    # Use the last n_hist-1 actions (the actual action history for the context frames)
                    # Decode the action history to get the actual actions
                    action_context = action_history[:, -(cfg.n_hist-1):]
                    context_frames = current_context[:, -(cfg.n_hist-1):]

                    ctx = encode_images_to_latents(
                        vae, context_frames.flatten(0, 1), cfg.vae_scaling_factor
                    ).reshape(1, cfg.n_hist - 1, cfg.latent_channels, cfg.height//8, cfg.width//8)

                    action_text_embeds, obs_mask = conditioner(action_context, None)
                    timestep_cond = noise_bucketer(torch.zeros(1, dtype=torch.long, device=device))

                    lat = torch.randn(1, cfg.latent_channels, cfg.height // 8, cfg.width // 8,
                                     device=device, dtype=model_dtype)
                    scheduler.set_timesteps(cfg.num_inference_steps, device=device)

                    for t in scheduler.timesteps:
                        x0_pred = cfg_with_context_dropout(
                            unet, lat, ctx, t.expand(1), action_text_embeds, timestep_cond, cfg.cfg_weight
                        )
                        lat = scheduler.step(x0_pred, t, lat).prev_sample

                    next_frame = decode_latents_to_images(vae, lat, cfg.vae_scaling_factor)
                    predicted_frames.append(next_frame)

                    # Roll the buffers: remove oldest, add newest
                    current_context = torch.cat([current_context[:, 1:], next_frame.unsqueeze(1)], dim=1)
                    next_action = test_action_seq[:, step:step+1]
                    action_history = torch.cat([action_history[:, 1:], next_action], dim=1)
                
                step_dir = os.path.join(rollout_dir, f"step_{global_step:06d}")
                os.makedirs(step_dir, exist_ok=True)
                
                for i in range(cfg.n_hist):
                    frame = (test_context[0, i].cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(frame).save(os.path.join(step_dir, f"context_{i:02d}.png"))
                
                for i, frame in enumerate(predicted_frames):
                    frame_np = (frame[0].cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(frame_np).save(os.path.join(step_dir, f"pred_{i:02d}.png"))
                
                for i in range(rollout_steps):
                    frame = (test_gt_frames[0, i].cpu().float().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(frame).save(os.path.join(step_dir, f"gt_{i:02d}.png"))
                
                create_rollout_comparison_grid(step_dir, cfg.n_hist, rollout_steps)
                print(f"Rollout saved to: {step_dir}")
            
            # Re-enable gradient checkpointing for training
            if cfg.gradient_checkpointing:
                unet.enable_gradient_checkpointing()

            unet.train()
            conditioner.train()
            noise_bucketer.train()

        # Save checkpoint every 1000 steps
        if global_step % 1000 == 0:
            ckpt_dir = f"checkpoints/doom_full/{timestamp}/step_{global_step:06d}"
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                'unet': unet.state_dict(),
                'conditioner': conditioner.state_dict(),
                'noise_bucketer': noise_bucketer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
            }, f"{ckpt_dir}/checkpoint.pt")
            print(f"✓ Full FT checkpoint saved at step {global_step}: {ckpt_dir}/")
    
    save_dir = f"checkpoints/doom_full/{timestamp}/final"
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'unet': unet.state_dict(),
        'conditioner': conditioner.state_dict(),
        'noise_bucketer': noise_bucketer.state_dict(),
        'optimizer': optimizer.state_dict(),
        'global_step': global_step,
    }, f"{save_dir}/checkpoint.pt")
    print(f"✓ Final full FT checkpoint saved to {save_dir}/")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Doom world model with diffusion + LoRA")
    parser.add_argument("--data_dir", type=str, default="debug/rnd/20251023_150242", help="Path to episode data directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (use 2 for MPS, 4+ for CUDA)")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps (paper uses 700,000)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Target learning rate (default: 5e-5, paper uses 2e-5)")
    parser.add_argument("--prediction_type", type=str, default="v_prediction", choices=["v_prediction", "epsilon", "sample"], help="Prediction type: v_prediction (GameNGen paper default), epsilon, or sample")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing to save VRAM (default: True)")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false", help="Disable gradient checkpointing")
    parser.add_argument("--preload", action="store_true", help="Preload all data to RAM (default: False, load on-demand from disk)")
    parser.add_argument("--rollout_interval", type=int, default=100, help="Run rollout every N steps (default: 100)")
    parser.add_argument("--use_cosine_decay", action="store_true", default=False, help="Use cosine decay for learning rate after warmup/hold (default: False)")
    parser.add_argument("--use_adafactor", action="store_true", default=False, help="Use Adafactor optimizer (default: False, use AdamW)")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps (linear warmup from 0 to target LR)")
    parser.add_argument("--hold_steps", type=int, default=0, help="Number of steps to hold at target LR before decay")
    parser.add_argument("--full_ft", action="store_true", default=False, help="Use full fine-tuning instead of LoRA (default: False, use LoRA)")
    args = parser.parse_args()
    
    train(data_dir=args.data_dir, batch_size=args.batch_size, num_steps=args.steps, lr=args.lr, prediction_type=args.prediction_type, gradient_checkpointing=args.gradient_checkpointing, preload_to_ram=args.preload, rollout_interval=args.rollout_interval, use_adafactor=args.use_adafactor, use_cosine_decay=args.use_cosine_decay, warmup_steps=args.warmup_steps, hold_steps=args.hold_steps, full_ft=args.full_ft)