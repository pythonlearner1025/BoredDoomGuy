#!/usr/bin/env python3
"""
Reference wiring for your SD1.5-based, action‑conditioned AR UNet variant

What this script demonstrates
- UNet I/O surgery: in_channels = C_x_t + N_hist * C_vae, with channel-concat of context latents
- Conditioning via cross-attn: action tokens (+ optional caption tokens) projected to text hidden size
- Bucketed context-noise: Uniform alpha -> K buckets -> learned embedding -> fed via UNet time_cond
- Train step (teacher-forced): standard diffusion MSE on current frame latent only
- Inference rollout: DDIM, low CFG applied only to observation tokens (not action tokens)
- VAE usage: encode frames with the encoder (frozen), decode with decoder (LoRA optional)

Notes
- This is a minimal reference. It runs with random tensors; replace the data stubs with your dataset.
- No network calls; if you want pretrained weights, install diffusers and peft, then load them in build_models().
- LoRA injection is sketched (PEFT). Uncomment the blocks and tailor target_modules to your needs.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _append_diffusers_src_to_path():
    """Ensure `import diffusers` resolves from a local clone without install.

    Looks for either:
      - this_file/../diffusers/src (when sim_doom.py is one folder above repo)
      - this_file/src (when placed at repo root)
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(here, "diffusers", "src"),  # ../diffusers/src
        os.path.join(here, "src"),  # ./src
    ]
    for p in candidate_paths:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            break


_append_diffusers_src_to_path()

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


# ----------------------------
# Config knobs you should set
# ----------------------------
@dataclass
class DoomSimCfg:
    # Context window and noise augmentation
    n_hist: int = 64
    alpha_max: float = 0.25  # tune for AR stability
    k_buckets: int = 8
    # Guidance and sampling
    cfg_weight: float = 1.5  # apply only on observation tokens
    num_inference_steps: int = 4  # DDIM steps
    # Embedding sizes
    text_hidden_size: int = 768  # SD1.5 cross-attn dim
    noise_embed_dim: int = 256  # fed through UNet `time_cond_proj_dim`
    action_vocab_size: int = 64  # placeholder; set to your action space size
    action_embed_dim: int = 64
    caption_vocab_size: int = 0  # 0 to disable caption path
    caption_embed_dim: int = 256
    # CLIP caption encoder
    use_clip_captions: bool = True
    clip_model_name_or_path: str = "openai/clip-vit-large-patch14"  # SD 1.x text encoder
    caption_max_length: int = 77
    # Optimization (illustrative; actual training loop not implemented here)
    grad_clip: float = 1.0
    mixed_precision: bool = True
    # Model defaults (keep SD1.5-like)
    latent_channels: int = 4
    vae_scaling_factor: float = 0.18215
    height: int = 240  # input image H
    width: int = 320   # input image W


class ActionTextConditioner(nn.Module):
    """Projects action tokens and optionally concatenates caption tokens (either learned or CLIP).

    - Actions: learned embedding -> project to `out_hidden_size`.
    - Captions: either
        a) learned embedding from token ids (legacy path), or
        b) precomputed caption embeddings (e.g., CLIP last_hidden_state); project only if dims mismatch.

    Returns encoder_hidden_states and a boolean mask marking which tokens are observation tokens (captions only).
    """

    def __init__(
        self,
        action_vocab_size: int,
        action_embed_dim: int,
        caption_vocab_size: int,
        caption_embed_dim: int,
        out_hidden_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.has_captions = caption_vocab_size > 0

        self.action_emb = nn.Embedding(action_vocab_size, action_embed_dim)
        self.action_proj = nn.Linear(action_embed_dim, out_hidden_size)

        # Legacy learned caption path (used only if token ids are provided and CLIP is disabled)
        if self.has_captions:
            self.caption_emb = nn.Embedding(caption_vocab_size, caption_embed_dim)
            self.caption_proj = nn.Linear(caption_embed_dim, out_hidden_size)
        else:
            self.caption_emb = None
            self.caption_proj = None

        # Optional projection for externally provided caption embeddings (e.g., CLIP) if dim != out_hidden_size
        self.caption_in_proj: Optional[nn.Linear] = None
        self.out_hidden_size = out_hidden_size
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        actions: torch.LongTensor,  # [B, N_hist]
        captions: Optional[torch.LongTensor] = None,  # [B, T_cap] legacy token ids
        caption_embeds: Optional[torch.FloatTensor] = None,  # [B, T_cap, D_embed] e.g., CLIP outputs
    ) -> Tuple[torch.FloatTensor, torch.BoolTensor]:
        bsz, n_hist = actions.shape
        act_tokens = self.action_proj(self.action_emb(actions))  # [B, N_hist, H]
        tokens = [act_tokens]
        obs_mask_parts = [torch.zeros(bsz, n_hist, dtype=torch.bool, device=actions.device)]

        if caption_embeds is not None:
            cap = caption_embeds
            # Project to out_hidden_size if needed
            if cap.shape[-1] != self.out_hidden_size:
                if self.caption_in_proj is None:
                    self.caption_in_proj = nn.Linear(cap.shape[-1], self.out_hidden_size).to(cap.device, cap.dtype)
                cap = self.caption_in_proj(cap)
            tokens.append(cap)
            obs_mask_parts.append(torch.ones(bsz, cap.shape[1], dtype=torch.bool, device=cap.device))
        elif self.has_captions and captions is not None:
            cap_tokens = self.caption_proj(self.caption_emb(captions))  # [B, T_cap, H]
            tokens.append(cap_tokens)
            obs_mask_parts.append(torch.ones(bsz, cap_tokens.shape[1], dtype=torch.bool, device=cap_tokens.device))

        enc = torch.cat(tokens, dim=1)  # [B, N_hist(+T_cap), H]
        enc = self.dropout(enc)
        obs_mask = torch.cat(obs_mask_parts, dim=1)  # True where observation tokens
        return enc, obs_mask


class CaptionEncoderCLIP(nn.Module):
    """Wrapper around CLIP text encoder for SD‑1.x style captions.

    - Loads tokenizer and text encoder from a local directory or model id.
    - Returns `last_hidden_state` [B, T, hidden] suitable for UNet cross‑attn.

    PEFT LoRA example (commented):
        from peft import LoraConfig, get_peft_model
        lora_rank = 128
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            bias="none",
            task_type="CAUSAL_LM",  # arbitrary label; required by PEFT API
        )
        self.text_encoder = get_peft_model(self.text_encoder, lora_cfg)
        # Then optimize only LoRA parameters during training.
    """

    def __init__(
        self,
        name_or_path: str,
        max_length: int = 77,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        try:
            from transformers import CLIPTokenizer, CLIPTextModel
        except Exception as e:
            raise ImportError(
                "Transformers is required for CLIP captions. Install `transformers` and provide a local checkpoint."
            ) from e

        self.tokenizer = CLIPTokenizer.from_pretrained(name_or_path)
        self.text_encoder = CLIPTextModel.from_pretrained(name_or_path)
        if device is not None:
            self.text_encoder = self.text_encoder.to(device)
        if dtype is not None:
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
    """Uniform alpha in [0, alpha_max] -> K buckets -> embedding to `noise_embed_dim`.

    You will pass the embedding to UNet via `timestep_cond`.
    """

    def __init__(self, k_buckets: int, alpha_max: float, noise_embed_dim: int):
        super().__init__()
        self.k = k_buckets
        self.alpha_max = alpha_max
        self.embed = nn.Embedding(k_buckets, noise_embed_dim)

    @torch.no_grad()
    def sample_alphas(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch, device=device) * self.alpha_max

    def bucketize(self, alpha: torch.Tensor) -> torch.LongTensor:
        # alpha in [0, alpha_max] -> [0, k-1]
        scaled = (alpha / max(self.alpha_max, 1e-8)) * self.k
        ids = torch.clamp(scaled.long(), 0, self.k - 1)
        return ids

    def forward(self, ids: torch.LongTensor) -> torch.FloatTensor:
        return self.embed(ids)


def inflate_conv_in_for_history(unet: UNet2DConditionModel, old_in: int, new_in: int) -> None:
    """UNet I/O surgery: expand `conv_in` from old_in to new_in channels by copying weights for the
    target-latent channels and zero-initializing the extra history channels.

    Use this if you load a 4-channel pretrained UNet and need 4*(1+N_hist).
    """
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
    # VAE (encoder frozen, decoder can receive LoRA)
    vae = AutoencoderKL(
        sample_size=cfg.height // 8,
        latent_channels=cfg.latent_channels,
        scaling_factor=cfg.vae_scaling_factor,
    ).to(device)
    for p in vae.encoder.parameters():
        p.requires_grad = False

    # UNet with widened input channels and a time_cond for noise buckets
    in_channels = cfg.latent_channels * (1 + cfg.n_hist)
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
    ).to(device)

    # Scheduler (DDIM)
    scheduler = DDIMScheduler()

    # Action/caption conditioner and noise bucketer
    conditioner = ActionTextConditioner(
        action_vocab_size=cfg.action_vocab_size,
        action_embed_dim=cfg.action_embed_dim,
        caption_vocab_size=cfg.caption_vocab_size,
        caption_embed_dim=cfg.caption_embed_dim,
        out_hidden_size=cfg.text_hidden_size,
    ).to(device)
    noise_bucketer = NoiseBucketer(cfg.k_buckets, cfg.alpha_max, cfg.noise_embed_dim).to(device)

    # Optional CLIP caption encoder (recommended for SD1.5)
    caption_encoder = None
    if cfg.use_clip_captions:
        try:
            caption_encoder = CaptionEncoderCLIP(
                cfg.clip_model_name_or_path, max_length=cfg.caption_max_length, device=device
            )
        except Exception as e:
            print(f"[warn] CLIP caption encoder not available: {e}. Falling back to learned caption embeddings if any.")

    # Optional: attach LoRA with PEFT to UNet cross-attn (Q/K/V/out) and VAE decoder
    # try:
    #     from peft import LoraConfig, get_peft_model
    #     # High-rank LoRA suggested: 128–256 for decoder, 64–256 for UNet attn
    #     lora_rank_unet = 128
    #     lora_cfg_unet = LoraConfig(
    #         r=lora_rank_unet,
    #         lora_alpha=lora_rank_unet,
    #         target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # cross-attn paths
    #         bias="none",
    #         task_type="CAUSAL_LM",  # PEFT requires a task; doesn't affect injected modules here
    #     )
    #     unet = get_peft_model(unet, lora_cfg_unet)
    #
    #     lora_rank_vae = 256
    #     lora_cfg_vae = LoraConfig(
    #         r=lora_rank_vae,
    #         lora_alpha=lora_rank_vae,
    #         target_modules=["conv_out", "conv1", "conv2", "conv_shortcut"],  # late decoder blocks
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #     )
    #     vae.decoder = get_peft_model(vae.decoder, lora_cfg_vae)
    # except Exception as e:
    #     print(f"[info] PEFT not configured ({e}). Proceeding without LoRA injection.")

    return vae, unet, scheduler, conditioner, noise_bucketer, caption_encoder


def encode_images_to_latents(vae: AutoencoderKL, images: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    # images: [B, 3, H, W] -> latents [B, C_vae, H/8, W/8]
    posterior = vae.encode(images).latent_dist
    latents = posterior.sample()
    latents = latents * scaling_factor
    return latents


def prepare_concat_latents_and_targets(
    vae: AutoencoderKL,
    scheduler: DDIMScheduler,
    current_images: torch.Tensor,  # [B, 3, H, W]
    past_images: torch.Tensor,     # [B, N_hist, 3, H, W]
    alpha_max: float,
    k_buckets: int,
    noise_bucketer: NoiseBucketer,
    scaling_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.IntTensor, torch.LongTensor]:
    device = current_images.device
    B, N_hist = past_images.shape[:2]

    # Encode target and context
    x0 = encode_images_to_latents(vae, current_images, scaling_factor)  # [B, C, h, w]
    context = encode_images_to_latents(vae, past_images.flatten(0, 1), scaling_factor)
    context = context.reshape(B, N_hist, *x0.shape[1:])  # [B, N_hist, C, h, w]

    # Diffusion target: sample t, add noise to current frame only
    timesteps = torch.randint(0, scheduler.num_train_timesteps, (B,), device=device, dtype=torch.long)
    noise = torch.randn_like(x0)
    x_t = scheduler.add_noise(x0, noise, timesteps)

    # Context noise augmentation: alpha ~ U(0, alpha_max), add scalar noise to context latents
    alpha = torch.rand(B, device=device) * alpha_max  # [B]
    ctx_noise = torch.randn_like(context)
    context_noisy = context + alpha.view(B, 1, 1, 1, 1) * ctx_noise

    # Bucketize alpha and get noise-bucket embeddings (to be passed as timestep_cond)
    bucket_ids = noise_bucketer.bucketize(alpha)  # [B]

    # Channel-concat: [x_t || noisy_context_latents]
    context_cat = context_noisy.flatten(1, 2)  # [B, N_hist*C, h, w]
    x_in = torch.cat([x_t, context_cat], dim=1)  # [B, C + N_hist*C, h, w]

    return x_in, noise, x0, timesteps, bucket_ids


def cfg_only_on_observation(
    unet: UNet2DConditionModel,
    x_in: torch.Tensor,
    t: torch.IntTensor,
    enc_states: torch.Tensor,  # [B, T, H]
    obs_mask: torch.BoolTensor,  # [B, T]
    timestep_cond: torch.Tensor,  # [B, D]
    cfg_weight: float,
):
    """Classifier-free guidance applied only to the observation tokens.

    We create two encoder_hidden_states streams per sample:
      - cond: actions + observations
      - uncond: actions + (null/zero) observations

    Then stack batch-wise and run a single UNet forward.
    """
    bsz = enc_states.shape[0]

    # Build unconditional tokens by zeroing the observation positions
    enc_uncond = enc_states.clone()
    enc_uncond[obs_mask] = 0.0

    # Stack for a single pass
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
    conditioner: ActionTextConditioner,
    noise_bucketer: NoiseBucketer,
    caption_encoder: Optional[CaptionEncoderCLIP],
    cfg: DoomSimCfg,
    past_images: torch.Tensor,   # [B, N_hist, 3, H, W]
    actions: torch.LongTensor,   # [B, N_hist]
    current_images: torch.Tensor,  # [B, 3, H, W]
    captions: Optional[torch.LongTensor] = None,  # [B, T_cap] legacy ids
    caption_texts: Optional[list[str]] = None,  # list of strings for CLIP
):
    x_in, noise, x0, timesteps, bucket_ids = prepare_concat_latents_and_targets(
        vae, scheduler, current_images, past_images, cfg.alpha_max, cfg.k_buckets, noise_bucketer, cfg.vae_scaling_factor
    )

    caption_embeds = None
    if caption_encoder is not None and caption_texts is not None:
        caption_embeds = caption_encoder(caption_texts)

    enc_states, obs_mask = conditioner(actions, captions=captions, caption_embeds=caption_embeds)
    timestep_cond = noise_bucketer(bucket_ids)  # [B, D]

    # Standard epsilon-pred loss on current frame only
    eps = cfg_only_on_observation(
        unet, x_in, timesteps, enc_states, obs_mask, timestep_cond, cfg.cfg_weight
    )
    loss = F.mse_loss(eps, noise)
    return loss

@torch.no_grad()
def rollout(
    vae: AutoencoderKL,
    unet: UNet2DConditionModel,
    scheduler: DDIMScheduler,
    conditioner: ActionTextConditioner,
    noise_bucketer: NoiseBucketer,
    caption_encoder: Optional[CaptionEncoderCLIP],
    cfg: DoomSimCfg,
    init_context_images: torch.Tensor,  # [B, N_hist, 3, H, W]
    actions_seq: torch.LongTensor,      # [B, T_total] actions for rollout
    captions_seq: Optional[torch.LongTensor] = None,  # [B, T_cap] per step or None (legacy)
    caption_texts: Optional[list[str]] = None,        # list[str] for CLIP (kept constant here for simplicity)
    steps: int = 8,
):
    device = init_context_images.device
    B, N_hist, _, H, W = init_context_images.shape

    # Maintain a rolling context of decoded frames; re-encode each predicted frame
    context_images = init_context_images.clone()
    obs_mask_template = None

    # Prepare fixed timestep_cond representing zero context noise (bucket id 0)
    bucket_ids = torch.zeros(B, dtype=torch.long, device=device)
    timestep_cond = noise_bucketer(bucket_ids)  # [B, D]

    # Actions are consumed T one-by-one; captions optional and constant in this stub
    for t_idx in range(steps):
        actions_t = actions_seq[:, t_idx].unsqueeze(1).repeat(1, N_hist)  # re-use last N_hist actions if needed
        caption_embeds = None
        if caption_encoder is not None and caption_texts is not None:
            caption_embeds = caption_encoder(caption_texts)
        enc_states, obs_mask = conditioner(actions_t, captions=captions_seq, caption_embeds=caption_embeds)
        if obs_mask_template is None:
            obs_mask_template = obs_mask  # save shape/positions

        # Build context latents (no train-time noise corruption here)
        ctx = encode_images_to_latents(
            vae, context_images.flatten(0, 1), cfg.vae_scaling_factor
        ).reshape(B, N_hist, cfg.latent_channels, H // 8, W // 8)

        # Initialize DDIM latent
        lat = torch.randn(B, cfg.latent_channels, H // 8, W // 8, device=device)
        scheduler.set_timesteps(cfg.num_inference_steps, device=device)

        for t in scheduler.timesteps:
            # Channel concat
            ctx_cat = ctx.flatten(1, 2)
            x_in = torch.cat([lat, ctx_cat], dim=1)

            # CFG only on obs tokens (captions). Actions are kept identical across cond/uncond.
            eps = cfg_only_on_observation(
                unet, x_in, t.expand(B), enc_states, obs_mask_template, timestep_cond, cfg.cfg_weight
            )

            pred = scheduler.step(eps, t, lat)
            lat = pred.prev_sample

        # Decode to image and roll context
        image = vae.decode(lat / cfg.vae_scaling_factor).sample  # [B, 3, H, W]
        context_images = torch.cat([context_images[:, 1:], image.unsqueeze(1)], dim=1)

    return context_images[:, -1]  # last predicted frame


def _toy_batch(cfg: DoomSimCfg, device: torch.device, batch_size: int = 2):
    H, W = cfg.height, cfg.width
    past = torch.randn(batch_size, cfg.n_hist, 3, H, W, device=device)
    cur = torch.randn(batch_size, 3, H, W, device=device)
    actions = torch.randint(0, cfg.action_vocab_size, (batch_size, cfg.n_hist), device=device)
    captions = None
    if cfg.caption_vocab_size > 0:
        captions = torch.randint(0, cfg.caption_vocab_size, (batch_size, 1), device=device)
    return past, actions, cur, captions


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = DoomSimCfg()
    vae, unet, scheduler, conditioner, bucketer, capenc = build_models(cfg, device)

    # One teacher-forced loss with random tensors
    past, actions, cur, captions = _toy_batch(cfg, device)
    loss = train_step_teacher_forced(
        vae, unet, scheduler, conditioner, bucketer, capenc, cfg, past, actions, cur, captions, caption_texts=None
    )
    print(f"toy loss: {loss.item():.4f}")

    # One short rollout
    with torch.no_grad():
        T = 4
        actions_seq = torch.randint(0, cfg.action_vocab_size, (past.shape[0], T), device=device)
        pred = rollout(
            vae, unet, scheduler, conditioner, bucketer, capenc, cfg,
            init_context_images=past, actions_seq=actions_seq, steps=2
        )
        print("pred frame shape:", tuple(pred.shape))


if __name__ == "__main__":
    main()
