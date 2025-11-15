import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
import os
import numpy as np
from PIL import Image

def encode_images_to_latents(vae: AutoencoderKL, images: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    images = (images * 2 - 1).to(device=vae.device, dtype=next(vae.parameters()).dtype)
    return vae.encode(images).latent_dist.sample() * scaling_factor

def decode_latents_to_images(vae: AutoencoderKL, latents: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    imgs = vae.decode(latents / scaling_factor).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    return imgs

def epsilon_to_x0(epsilon_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor, 
                  alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    Convert epsilon (noise) prediction to x0 (clean sample) prediction.
    
    This is used when fine-tuning a pretrained diffusion model (like Stable Diffusion)
    that was trained to predict epsilon, but we want to supervise with x0 losses.
    
    Formula: x0 = (x_t - sqrt(1 - alpha_t) * epsilon) / sqrt(alpha_t)
    
    Args:
        epsilon_pred: Model's epsilon prediction [B, C, H, W]
        x_t: Noisy latent at timestep t [B, C, H, W]
        t: Timestep(s) [B] or scalar
        alphas_cumprod: Cumulative product of alphas from scheduler [num_timesteps]
    
    Returns:
        x0_pred: Predicted clean sample [B, C, H, W]
    """
    # Get alpha_t for the current timestep(s)
    if t.dim() == 0:
        # Scalar timestep
        alpha_t = alphas_cumprod[t].view(1, 1, 1, 1)
    else:
        # Batch of timesteps
        alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
    
    x0_pred = (x_t - sqrt_one_minus_alpha_t * epsilon_pred) / sqrt_alpha_t
    return x0_pred

def get_v_target(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor,
                 alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    Compute v-prediction target from clean sample and noise.
    
    V-prediction is the velocity parameterization used in the GameNGen paper and SD 2.1.
    Formula: v = sqrt(alpha_t) * noise - sqrt(1 - alpha_t) * x0
    
    Args:
        x0: Clean sample [B, C, H, W]
        noise: Noise tensor [B, C, H, W]
        t: Timestep(s) [B] or scalar
        alphas_cumprod: Cumulative product of alphas from scheduler [num_timesteps]
    
    Returns:
        v_target: V-prediction target [B, C, H, W]
    """
    # Get alpha_t for the current timestep(s)
    if t.dim() == 0:
        # Scalar timestep
        alpha_t = alphas_cumprod[t].view(1, 1, 1, 1)
    else:
        # Batch of timesteps
        alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
    
    # v = sqrt(alpha_t) * epsilon - sqrt(1 - alpha_t) * x0
    v_target = sqrt_alpha_t * noise - sqrt_one_minus_alpha_t * x0
    return v_target

def v_to_x0(v_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor,
            alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    Convert v-prediction to x0 (clean sample) prediction.
    
    V-prediction is defined as: v = sqrt(alpha_t) * epsilon - sqrt(1 - alpha_t) * x0
    Rearranging to solve for x0:
    x0 = sqrt(alpha_t) * x_t - sqrt(1 - alpha_t) * v
    
    Args:
        v_pred: Model's v prediction [B, C, H, W]
        x_t: Noisy latent at timestep t [B, C, H, W]
        t: Timestep(s) [B] or scalar
        alphas_cumprod: Cumulative product of alphas from scheduler [num_timesteps]
    
    Returns:
        x0_pred: Predicted clean sample [B, C, H, W]
    """
    # Get alpha_t for the current timestep(s)
    if t.dim() == 0:
        # Scalar timestep
        alpha_t = alphas_cumprod[t].view(1, 1, 1, 1)
    else:
        # Batch of timesteps
        alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1.0 - alpha_t)
    
    # x0 = sqrt(alpha_t) * x_t - sqrt(1 - alpha_t) * v
    x0_pred = sqrt_alpha_t * x_t - sqrt_one_minus_alpha_t * v_pred
    return x0_pred

def create_rollout_comparison_grid(epoch_dir: str, n_context: int, n_rollout: int):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        # Load context frames (show last 4)
        n_show_context = min(4, n_context)
        context_imgs = []
        for i in range(n_context - n_show_context, n_context):
            img_path = os.path.join(epoch_dir, f"context_{i:02d}.png")
            if os.path.exists(img_path):
                context_imgs.append(np.array(Image.open(img_path)))
        
        # Load predicted and ground truth frames
        pred_imgs = []
        gt_imgs = []
        for i in range(n_rollout):
            pred_path = os.path.join(epoch_dir, f"pred_{i:02d}.png")
            gt_path = os.path.join(epoch_dir, f"gt_{i:02d}.png")
            if os.path.exists(pred_path):
                pred_imgs.append(np.array(Image.open(pred_path)))
            if os.path.exists(gt_path):
                gt_imgs.append(np.array(Image.open(gt_path)))
        
        if not pred_imgs and not gt_imgs:
            return
        
        # Create grid: 3 rows (context, predicted, ground truth)
        n_cols = max(len(context_imgs), len(pred_imgs), len(gt_imgs))
        fig, axes = plt.subplots(3, n_cols, figsize=(2.5*n_cols, 7.5))
        
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Row 0: Context frames
        for col in range(n_cols):
            if col < len(context_imgs):
                axes[0, col].imshow(context_imgs[col])
                axes[0, col].set_title(f"Context {n_context - n_show_context + col}", fontsize=8)
            axes[0, col].axis('off')
        
        # Row 1: Predicted frames
        for col in range(n_cols):
            if col < len(pred_imgs):
                axes[1, col].imshow(pred_imgs[col])
                axes[1, col].set_title(f"Predicted {col}", fontsize=8, color='blue')
            axes[1, col].axis('off')
        
        # Row 2: Ground truth frames
        for col in range(n_cols):
            if col < len(gt_imgs):
                axes[2, col].imshow(gt_imgs[col])
                axes[2, col].set_title(f"Ground Truth {col}", fontsize=8, color='green')
            axes[2, col].axis('off')
        
        # Add row labels
        axes[0, 0].text(-0.3, 0.5, 'Context', transform=axes[0, 0].transAxes, 
                       fontsize=12, va='center', ha='right', rotation=90, weight='bold')
        axes[1, 0].text(-0.3, 0.5, 'Predicted', transform=axes[1, 0].transAxes, 
                       fontsize=12, va='center', ha='right', rotation=90, weight='bold', color='blue')
        axes[2, 0].text(-0.3, 0.5, 'Ground Truth', transform=axes[2, 0].transAxes, 
                       fontsize=12, va='center', ha='right', rotation=90, weight='bold', color='green')
        
        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, "rollout_comparison.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not create rollout comparison grid: {e}")