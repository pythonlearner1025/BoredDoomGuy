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