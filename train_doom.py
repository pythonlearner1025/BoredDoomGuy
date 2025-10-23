import json
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model, TaskType

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

@dataclass
class DoomSimCfg:
    # Context window and noise augmentation
    n_hist: int = 2  # reduced from 64 for memory efficiency
    alpha_max: float = 0.25  # tune for AR stability
    k_buckets: int = 8
    # Guidance and sampling
    cfg_weight: float = 1.5  # apply only on observation tokens
    num_inference_steps: int = 4  # DDIM steps
    # Embedding sizes
    text_hidden_size: int = 768  # SD1.5 cross-attn dim
    noise_embed_dim: int = 256  # fed through UNet `time_cond_proj_dim`
    action_vocab_size: int = 6  # number of valid actions
    action_embed_dim: int = 12
    caption_vocab_size: int = 0  # 0 to disable caption path
    caption_embed_dim: int = 256
    # CLIP 
    use_clip_captions: bool = False  # disabled to save memory on MPS
    clip_model_name_or_path: str = "openai/clip-vit-large-patch14"  # SD 1.x text encoder
    caption_max_length: int = 77
    # Optimization
    grad_clip: float = 1.0
    mixed_precision: bool = False
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 1
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # Model defaults (keep SD1.5-like)
    latent_channels: int = 4
    vae_scaling_factor: float = 0.18215
    height: int = 240  # input image H
    width: int = 320   # input image W

VALID_ACTIONS = [
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
    "USE",
]

def buttons_to_action_one_hot(buttons: list[str]) -> torch.Tensor:
    """Convert button list to one-hot encoded action vector."""
    one_hot = torch.zeros(len(VALID_ACTIONS), dtype=torch.float32)
    for button in buttons:
        if button in VALID_ACTIONS:
            idx = VALID_ACTIONS.index(button)
            one_hot[idx] = 1.0
    return one_hot

class DoomDataset(Dataset):
    def __init__(self, data_dir: str, n_hist: int, height: int = 240, width: int = 320):
        self.data_dir = data_dir
        self.n_hist = n_hist
        self.height = height
        self.width = width
        self.episode_paths = []
        
        # Collect all episode directories
        for file in os.listdir(data_dir):
            ep_path = os.path.join(data_dir, file)
            if os.path.isdir(ep_path):
                self.episode_paths.append(ep_path)
        self.episode_paths.sort()
        
        # Cache episode lengths for efficient sampling
        self.episode_lengths = []
        for ep_path in self.episode_paths:
            n_frames = len([f for f in os.listdir(ep_path) if f.endswith(".png")])
            self.episode_lengths.append(n_frames)
    
    def __len__(self):
        return len(self.episode_paths)
    
    def __getitem__(self, idx):
        """Sample n_hist sequential frames and actions from episode idx."""
        ep_path = self.episode_paths[idx]
        ep_len = self.episode_lengths[idx]
        
        # Sample random start index ensuring we have n_hist frames
        if ep_len <= self.n_hist:
            start_idx = 0
            actual_n_hist = ep_len
        else:
            start_idx = random.randint(0, ep_len - self.n_hist)
            actual_n_hist = self.n_hist
        
        # Load frames and actions
        frames = []
        actions = []
        
        frame_files = sorted([f for f in os.listdir(ep_path) if f.endswith(".png")])
        json_files = sorted([f for f in os.listdir(ep_path) if f.endswith(".json")])
        
        for i in range(start_idx, start_idx + actual_n_hist):
            # Load image
            img_path = os.path.join(ep_path, frame_files[i])
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.width, self.height), Image.BILINEAR)
            img_array = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            frames.append(img_array)
            
            # Load action
            json_path = os.path.join(ep_path, json_files[i])
            with open(json_path, "r") as f:
                json_data = json.load(f)
                buttons = json_data["action"]["buttons"]
                action_one_hot = buttons_to_action_one_hot(buttons)
                actions.append(action_one_hot)
        
        # Pad if needed
        while len(frames) < self.n_hist:
            frames.append(torch.zeros(3, self.height, self.width))
            actions.append(torch.zeros(len(VALID_ACTIONS)))
        
        frames = torch.stack(frames)  # [n_hist, 3, H, W]
        actions = torch.stack(actions)  # [n_hist, act_dim]
        
        return frames, actions
    

class Conditioner(nn.Module):
    def __init__(
        self,
        action_vocab_size: int,
        action_embed_dim: int,
        out_hidden_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_proj = nn.Linear(action_vocab_size, out_hidden_size)
        self.out_hidden_size = out_hidden_size
    
    def forward(self, actions: torch.FloatTensor, caption_embeds: Optional[torch.FloatTensor]) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
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

class CaptionEncoderCLIP(nn.Module):
    def __init__(
        self,
        name_or_path: str,
        max_length: int = 77,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(name_or_path)
        self.text_encoder = CLIPTextModel.from_pretrained(name_or_path)
        if device is not None and dtype is not None:
            self.text_encoder = self.text_encoder.to(device=device, dtype=dtype)
        elif device is not None:
            self.text_encoder = self.text_encoder.to(device)
        elif dtype is not None:
            self.text_encoder = self.text_encoder.to(dtype)
        self.max_length = max_length
    
    def forward(self, texts: list[str]) -> torch.FloatTensor:
        tok = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(next(self.text_encoder.parameters()).device) for k, v in tok.items()}
        out = self.text_encoder(**tok)
        return out.last_hidden_state  # [B, T, hidden]

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
    with torch.no_grad():
        w = unet.conv_in.weight  # [C_out, old_in, k, k]
        b = unet.conv_in.bias
        C_out, _, k1, k2 = w.shape
        if unet.conv_in.in_channels != old_in:
            raise ValueError(f"Expected old_in={old_in}, found layer.in_channels={unet.conv_in.in_channels}")
        new_w = torch.zeros(C_out, new_in, k1, k2, device=w.device, dtype=w.dtype)
        new_w[:, :old_in] = w  # copy the first 4 channels for x_t
        new_conv = nn.Conv2d(new_in, C_out, kernel_size=w.shape[-1], padding=unet.conv_in.padding)
        new_conv = new_conv.to(w.device, w.dtype)
        new_conv.weight.copy_(new_w)
        new_conv.bias.copy_(b)
        unet.conv_in = new_conv

def build_models(cfg: DoomSimCfg, device: torch.device):
    # Determine optimal dtype
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("Using dtype: bfloat16")
        else:
            dtype = torch.float16
            print("Using dtype: float16")
    elif device.type == "mps":
        # MPS has mixed precision bugs with float16 - use float32 for stability
        dtype = torch.float32
        print("Using dtype: float32 (MPS - float16 causes dtype mismatch errors)")
    else:
        dtype = torch.float32
        print("Using dtype: float32 (CPU)")
    
    vae = AutoencoderKL(
        sample_size=cfg.height // 8,
        latent_channels=cfg.latent_channels,
        scaling_factor=cfg.vae_scaling_factor,
    ).to(device, dtype=dtype)
    for p in vae.encoder.parameters():
        p.requires_grad = False

    in_channels = cfg.latent_channels * cfg.n_hist  # current + (n_hist-1) context
    unet = UNet2DConditionModel(
        in_channels=in_channels,
        out_channels=cfg.latent_channels,
        cross_attention_dim=cfg.text_hidden_size,
        time_cond_proj_dim=cfg.noise_embed_dim,
        # SD1.5-like defaults
        block_out_channels=(320, 640, 1280, 1280),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        layers_per_block=2,
        attention_head_dim=8,
    ).to(device, dtype=dtype)
    
    scheduler = DDIMScheduler()
    
    conditioner = Conditioner(
        action_vocab_size=cfg.action_vocab_size,
        action_embed_dim=cfg.action_embed_dim,
        out_hidden_size=cfg.text_hidden_size,
    ).to(device, dtype=dtype)
    
    noise_bucketer = NoiseBucketer(cfg.k_buckets, cfg.alpha_max, cfg.noise_embed_dim).to(device, dtype=dtype)
    
    caption_encoder = None
    if cfg.use_clip_captions:
        try:
            caption_encoder = CaptionEncoderCLIP(
                cfg.clip_model_name_or_path, max_length=cfg.caption_max_length, device=device, dtype=dtype
            )
        except Exception as e:
            print(f"[warn] CLIP caption encoder not available: {e}. Falling back to learned caption embeddings if any.")

    # Apply LoRA to trainable modules
    if cfg.use_lora:
        print(f"\nApplying LoRA (r={cfg.lora_r}, alpha={cfg.lora_alpha})...")
        
        # LoRA for UNet (target attention and conv layers)
        unet_lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
            lora_dropout=cfg.lora_dropout,
        )
        unet = get_peft_model(unet, unet_lora_config)
        unet.print_trainable_parameters()
        
        # LoRA for Conditioner
        conditioner_lora_config = LoraConfig(
            r=cfg.lora_r,  # smaller rank for smaller model
            lora_alpha=cfg.lora_alpha,
            target_modules=["action_proj", "action_out"],
            lora_dropout=cfg.lora_dropout,
        )
        conditioner = get_peft_model(conditioner, conditioner_lora_config)
        print("Conditioner LoRA applied")
        
        # LoRA for NoiseBucketer
        bucketer_lora_config = LoraConfig(
            r=cfg.lora_r,  # even smaller rank
            lora_alpha=cfg.lora_alpha,
            target_modules=["embed"],
            lora_dropout=cfg.lora_dropout,
        )
        noise_bucketer = get_peft_model(noise_bucketer, bucketer_lora_config)
        print("NoiseBucketer LoRA applied\n")

    return vae, unet, scheduler, conditioner, noise_bucketer, caption_encoder

def encode_images_to_latents(vae: AutoencoderKL, images: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    # images: [B, 3, H, W] -> latents [B, C_vae, H/8, W/8]
    # Ensure dtype matches VAE to avoid MPS dtype mismatches
    images = images.to(device=vae.device, dtype=next(vae.parameters()).dtype)
    latents = vae.encode(images).latent_dist.sample()
    return latents * scaling_factor

def cfg_only_on_observation(
    unet: UNet2DConditionModel,
    x_in: torch.Tensor,
    t: torch.IntTensor,
    enc_states: torch.Tensor,  # [B, T, H]
    obs_mask: torch.BoolTensor,  # [B, T]
    timestep_cond: torch.Tensor,  # [B, D]
    cfg_weight: float,
):
    enc_uncond = enc_states.clone()
    # mask text conditioning 
    enc_uncond[obs_mask] = 0.0
    # input both masked, unmasked for text cfg
    enc_cat = torch.cat([enc_uncond, enc_states], dim=0)  # [2B, T, H]
    x_cat = torch.cat([x_in, x_in], dim=0)
    t_cat = torch.cat([t, t], dim=0)
    cond_cat = torch.cat([timestep_cond, timestep_cond], dim=0)
    out = unet(sample=x_cat, timestep=t_cat, encoder_hidden_states=enc_cat, timestep_cond=cond_cat).sample
    eps_uncond, eps_cond = out.chunk(2, dim=0)
    eps = eps_uncond + cfg_weight * (eps_cond - eps_uncond)
    return eps

def train_step_teacher_forced(
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    conditioner: Conditioner,
    noise_bucketer: NoiseBucketer,
    caption_encoder: Optional[CaptionEncoderCLIP],
    cfg: DoomSimCfg,
    past_images: torch.Tensor,   # [B, N_hist, 3, H, W]
    actions: torch.FloatTensor,  # [B, N_hist, action_vocab_size] (one-hot)
    current_images: torch.Tensor,  # [B, 3, H, W]
    captions: Optional[list] = None,  # current frame caps
): 
    device = current_images.device
    B, N_hist = past_images.shape[:2]
    
    if captions is not None:
        assert len(captions) == B

    # Encode target and context 
    x0 = encode_images_to_latents(vae, current_images, cfg.vae_scaling_factor)  # [B, C, h, w]
    ctxt = encode_images_to_latents(vae, past_images.flatten(0, 1), cfg.vae_scaling_factor)
    ctxt = ctxt.reshape(B, N_hist, *x0.shape[1:])  # [B, N_hist, C, h, w]

    # Diffusion target: sample t, add noise to current frame only
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = scheduler.add_noise(x0, noise, timesteps)

    # Context 
    alpha = torch.rand(B, device=device) * cfg.alpha_max  # [B]
    ctx_noise = torch.randn_like(ctxt)
    ctxt_noisy = ctxt + alpha.view(B, 1, 1, 1, 1) * ctx_noise

    # Bucketize alpha and get noise-bucket embeddings (to be passed as timestep_cond)
    bucket_ids = noise_bucketer.bucketize(alpha)  # [B]

    # Channel-concat: [x_t || noisy_context_latents]
    ctxt_cat = ctxt_noisy.flatten(1, 2)  # [B, N_hist*C, h, w]
    x_in = torch.cat([x_t, ctxt_cat], dim=1)  # [B, C + N_hist*C, h, w]

    # x_in, noise, x0, timesteps, bucket_ids computed above

    caption_embeds = None
    if caption_encoder is not None and captions is not None:
        caption_embeds = caption_encoder(captions)

    action_text_embeds, obs_mask = conditioner(actions, caption_embeds)
    timestep_cond = noise_bucketer(bucket_ids)  # [B, D]

    # Standard epsilon-pred loss on current frame only
    eps = cfg_only_on_observation(
        unet, x_in, timesteps, action_text_embeds, obs_mask, timestep_cond, cfg.cfg_weight
    )
    loss = F.mse_loss(eps, noise)
    return loss

def train(data_dir: str = "debug/rnd", batch_size: int = 4, num_epochs: int = 10, lr: float = 1e-4):
    cfg = DoomSimCfg()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")
    
    vae, unet, scheduler, conditioner, noise_bucketer, caption_encoder = build_models(cfg, device)
    
    # Initialize dataset and dataloader
    train_dataset = DoomDataset(data_dir=data_dir, n_hist=cfg.n_hist, height=cfg.height, width=cfg.width)
    print(f"Dataset loaded with {len(train_dataset)} episodes")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Get model dtype
    model_dtype = next(unet.parameters()).dtype
    print(f"Model dtype: {model_dtype}")
    
    # Optimizer for trainable parameters only (vae encoder is frozen)
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(conditioner.parameters()) + list(noise_bucketer.parameters()),
        lr=lr
    )
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (frames, actions) in enumerate(train_loader):
            # frames: [B, n_hist, 3, H, W]
            # actions: [B, n_hist, act_dim]
            frames = frames.to(device=device, dtype=model_dtype)
            actions = actions.to(device=device, dtype=model_dtype)
            
            # Split into context (past) and target (current)
            past_images = frames[:, :-1]  # [B, n_hist-1, 3, H, W]
            current_images = frames[:, -1]  # [B, 3, H, W]
            past_actions = actions[:, :-1]  # [B, n_hist-1, act_dim]
            
            if batch_idx == 0:
                print(f"Debug: frames shape: {frames.shape}, past_images shape: {past_images.shape}, current_images shape: {current_images.shape}")
            
            loss = train_step_teacher_forced(
                vae, unet, scheduler, conditioner, noise_bucketer, caption_encoder,
                cfg, past_images, past_actions, current_images, captions=None
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(unet.parameters()) + list(conditioner.parameters()) + list(noise_bucketer.parameters()),
                cfg.grad_clip
            )
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
    
    # Save LoRA adapters
    if cfg.use_lora:
        save_dir = "checkpoints/doom_lora"
        os.makedirs(save_dir, exist_ok=True)
        unet.save_pretrained(f"{save_dir}/unet")
        conditioner.save_pretrained(f"{save_dir}/conditioner")
        noise_bucketer.save_pretrained(f"{save_dir}/noise_bucketer")
        print(f"\nLoRA adapters saved to {save_dir}/")

@torch.no_grad()
def rollout(
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    conditioner: Conditioner,
    noise_bucketer: NoiseBucketer,
    caption_encoder: Optional[CaptionEncoderCLIP],
    cfg: DoomSimCfg,
    init_context_images: torch.Tensor,  # [B, N_hist, 3, H, W]
    actions_seq: torch.FloatTensor,     # [B, T_total, action_vocab_size] action one_hot vecs for rollout
    captions_seq: Optional[torch.LongTensor] = None,  # [B, T_cap] per step or None (legacy)
    caption_texts: Optional[list[str]] = None,        # list[str] for CLIP (kept constant here for simplicity)
    steps: int = 8,
):
    device = init_context_images.device
    B, N_hist, _, H, W = init_context_images.shape

    # Maintain a rolling context of decoded frames; re-encode each predicted frame
    context_images = init_context_images.clone()

    # Prepare fixed timestep_cond representing zero context noise (bucket id 0)
    bucket_ids = torch.zeros(B, dtype=torch.long, device=device)
    timestep_cond = noise_bucketer(bucket_ids)  # [B, D]

    for t_idx in range(steps):
        actions_t = actions_seq[:, t_idx].unsqueeze(1).repeat(1, N_hist, 1)  # [B, N_hist, act_dim]
        caption_embeds = None
        if caption_encoder is not None and caption_texts is not None:
            caption_embeds = caption_encoder(caption_texts)
        action_text_embeds, obs_mask = conditioner(actions_t, caption_embeds)
        # Build context latents (no train-time noise corruption here)
        ctx = encode_images_to_latents(
            vae, context_images.flatten(0, 1), cfg.vae_scaling_factor
        ).reshape(B, N_hist, cfg.latent_channels, H//8, W//8)

        lat = torch.randn(B, cfg.latent_channels, H // 8, W // 8, device=device)
        scheduler.set_timesteps(cfg.num_inference_steps, device=device)

        for t in scheduler.timesteps:
            # Channel concat
            ctx_cat = ctx.flatten(1,2)
            x_in = torch.cat([lat, ctx_cat], dim=1)
            
            eps = cfg_only_on_observation(
                unet, 
                x_in, 
                t.expand(B), 
                action_text_embeds, 
                obs_mask, 
                timestep_cond, 
                cfg.cfg_weight
            )
            pred = scheduler.step(eps, t, lat)
            lat = pred.prev_sample

        # Decode to image and roll context
        image = vae.decode(lat / cfg.vae_scaling_factor).sample  # [B, 3, H, W]
        context_images = torch.cat([context_images[:, 1:], image.unsqueeze(1)], dim=1)

    return context_images[:, -1]  # last predicted frame

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Doom world model with diffusion + LoRA")
    parser.add_argument("--data_dir", type=str, default="debug/rnd/20251022_224904", help="Path to episode data directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (use 2 for MPS, 4+ for CUDA)")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    
    train(data_dir=args.data_dir, batch_size=args.batch_size, num_epochs=args.num_epochs, lr=args.lr)

