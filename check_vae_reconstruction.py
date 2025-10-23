"""
Simple script to check VAE reconstruction quality on Doom frames.
Loads frames from RND data, encodes and decodes them, then creates comparison plots.
"""

import os
import glob
import random
from datetime import datetime
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


def load_random_frames(data_dir, n_samples=100, target_width=160, target_height=120):
    """Load random frames from RND episode data."""
    print(f"Loading {n_samples} random frames from {data_dir}...")
    
    # Find all PNG files
    all_frames = []
    episode_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    for episode_dir in episode_dirs:
        episode_path = os.path.join(data_dir, episode_dir)
        frame_files = glob.glob(os.path.join(episode_path, "*.png"))
        all_frames.extend(frame_files)
    
    print(f"Found {len(all_frames)} total frames across {len(episode_dirs)} episodes")
    
    # Sample random frames
    if len(all_frames) > n_samples:
        sampled_frames = random.sample(all_frames, n_samples)
    else:
        sampled_frames = all_frames
        print(f"Warning: Only {len(all_frames)} frames available, using all of them")
    
    # Load and preprocess frames
    frames = []
    frame_paths = []
    for frame_path in sampled_frames:
        img = Image.open(frame_path).convert("RGB")
        img = img.resize((target_width, target_height), Image.BILINEAR)
        frame_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        frames.append(frame_tensor)
        frame_paths.append(frame_path)
    
    frames = torch.stack(frames)  # [N, 3, H, W]
    print(f"Loaded {len(frames)} frames with shape {frames.shape}")
    
    return frames, frame_paths


def reconstruct_with_vae(vae, frames, scaling_factor=0.18215, batch_size=16):
    """Encode and decode frames through VAE."""
    print(f"Reconstructing frames through VAE (batch_size={batch_size})...")
    
    device = next(vae.parameters()).device
    dtype = next(vae.parameters()).dtype
    
    reconstructions = []
    
    # Process in batches to avoid OOM
    n_batches = (len(frames) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(frames))
            batch = frames[start_idx:end_idx].to(device=device, dtype=dtype)
            
            # Encode to latents
            latents = vae.encode(batch).latent_dist.sample() * scaling_factor
            
            # Decode back to images
            reconstructed = vae.decode(latents / scaling_factor).sample
            
            reconstructions.append(reconstructed.cpu())
            
            if (i + 1) % 5 == 0:
                print(f"  Processed batch {i+1}/{n_batches}")
    
    reconstructions = torch.cat(reconstructions, dim=0)
    print(f"Reconstruction complete! Shape: {reconstructions.shape}")
    
    return reconstructions


def tensor_to_image(tensor):
    """Convert tensor [3, H, W] in range [0, 1] to numpy image [H, W, 3] in range [0, 255]."""
    img = tensor.float().permute(1, 2, 0).numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def create_comparison_plot(original, reconstructed, save_path, n_show=10):
    """Create a grid comparing original and reconstructed frames."""
    n_show = min(n_show, len(original))
    
    fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5))
    
    if n_show == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_show):
        # Original
        orig_img = tensor_to_image(original[i])
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"Original {i}", fontsize=8)
        axes[0, i].axis('off')
        
        # Reconstructed
        recon_img = tensor_to_image(reconstructed[i])
        axes[1, i].imshow(recon_img)
        axes[1, i].set_title(f"Reconstructed {i}", fontsize=8)
        axes[1, i].axis('off')
    
    # Add row labels
    axes[0, 0].text(-0.3, 0.5, 'Ground Truth', transform=axes[0, 0].transAxes,
                   fontsize=12, va='center', ha='right', rotation=90, weight='bold', color='green')
    axes[1, 0].text(-0.3, 0.5, 'Reconstructed', transform=axes[1, 0].transAxes,
                   fontsize=12, va='center', ha='right', rotation=90, weight='bold', color='blue')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to: {save_path}")


def save_individual_comparisons(original, reconstructed, output_dir):
    """Save individual frame comparisons."""
    individual_dir = os.path.join(output_dir, "individual_comparisons")
    os.makedirs(individual_dir, exist_ok=True)
    
    print(f"Saving {len(original)} individual comparisons...")
    
    for i in range(len(original)):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # Original
        orig_img = tensor_to_image(original[i])
        axes[0].imshow(orig_img)
        axes[0].set_title("Ground Truth", fontsize=12, weight='bold', color='green')
        axes[0].axis('off')
        
        # Reconstructed
        recon_img = tensor_to_image(reconstructed[i])
        axes[1].imshow(recon_img)
        axes[1].set_title("Reconstructed", fontsize=12, weight='bold', color='blue')
        axes[1].axis('off')
        
        # Compute MSE
        mse = np.mean((orig_img.astype(float) - recon_img.astype(float)) ** 2)
        fig.suptitle(f"Frame {i} - MSE: {mse:.2f}", fontsize=14, weight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(individual_dir, f"comparison_{i:03d}.png"), dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"Individual comparisons saved to: {individual_dir}")


def compute_metrics(original, reconstructed):
    """Compute reconstruction metrics."""
    print("\nComputing reconstruction metrics...")
    
    # Convert to numpy for easier calculation (convert to float32 first for numpy compatibility)
    orig_np = original.float().permute(0, 2, 3, 1).numpy()  # [N, H, W, 3]
    recon_np = reconstructed.float().permute(0, 2, 3, 1).numpy()  # [N, H, W, 3]
    
    # MSE per frame
    mse_per_frame = np.mean((orig_np - recon_np) ** 2, axis=(1, 2, 3))
    
    # PSNR per frame
    psnr_per_frame = 20 * np.log10(1.0 / (np.sqrt(mse_per_frame) + 1e-8))
    
    print(f"  MSE  - Mean: {mse_per_frame.mean():.6f}, Std: {mse_per_frame.std():.6f}, "
          f"Min: {mse_per_frame.min():.6f}, Max: {mse_per_frame.max():.6f}")
    print(f"  PSNR - Mean: {psnr_per_frame.mean():.2f} dB, Std: {psnr_per_frame.std():.2f} dB, "
          f"Min: {psnr_per_frame.min():.2f} dB, Max: {psnr_per_frame.max():.2f} dB")
    
    # Create metrics plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(mse_per_frame, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('MSE', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('MSE Distribution', fontsize=14, weight='bold')
    axes[0].axvline(mse_per_frame.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {mse_per_frame.mean():.6f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(psnr_per_frame, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_xlabel('PSNR (dB)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('PSNR Distribution', fontsize=14, weight='bold')
    axes[1].axvline(psnr_per_frame.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {psnr_per_frame.mean():.2f} dB')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return mse_per_frame, psnr_per_frame, fig


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Check VAE reconstruction quality on Doom frames")
    parser.add_argument("--data_dir", type=str, default="debug/rnd/20251023_014026", 
                        help="Path to RND data directory")
    parser.add_argument("--n_samples", type=int, default=100, 
                        help="Number of frames to sample")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for VAE processing")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: vae_reconstruction_TIMESTAMP)")
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"vae_reconstruction_{timestamp}"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load VAE
    print("\nLoading VAE from Stable Diffusion 1.5...")
    vae = AutoencoderKL.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        subfolder="vae", 
        torch_dtype=dtype
    ).to(device)
    vae.eval()
    print("VAE loaded successfully!")
    
    # Load frames
    frames, frame_paths = load_random_frames(args.data_dir, n_samples=args.n_samples)
    
    # Reconstruct
    reconstructed = reconstruct_with_vae(vae, frames, batch_size=args.batch_size)
    
    # Compute metrics
    mse_per_frame, psnr_per_frame, metrics_fig = compute_metrics(frames, reconstructed)
    metrics_fig.savefig(os.path.join(output_dir, "metrics.png"), dpi=150, bbox_inches='tight')
    plt.close(metrics_fig)
    print(f"Metrics plot saved to: {os.path.join(output_dir, 'metrics.png')}")
    
    # Create comparison plots (show first 10)
    create_comparison_plot(frames, reconstructed, 
                          os.path.join(output_dir, "comparison_grid.png"), 
                          n_show=10)
    
    # Save individual comparisons
    save_individual_comparisons(frames, reconstructed, output_dir)
    
    # Save all reconstructed images
    recon_dir = os.path.join(output_dir, "reconstructed")
    os.makedirs(recon_dir, exist_ok=True)
    print(f"\nSaving all {len(reconstructed)} reconstructed frames...")
    for i in range(len(reconstructed)):
        recon_img = tensor_to_image(reconstructed[i])
        Image.fromarray(recon_img).save(os.path.join(recon_dir, f"recon_{i:03d}.png"))
    print(f"Reconstructed frames saved to: {recon_dir}")
    
    # Save all original images
    orig_dir = os.path.join(output_dir, "original")
    os.makedirs(orig_dir, exist_ok=True)
    print(f"Saving all {len(frames)} original frames...")
    for i in range(len(frames)):
        orig_img = tensor_to_image(frames[i])
        Image.fromarray(orig_img).save(os.path.join(orig_dir, f"orig_{i:03d}.png"))
    print(f"Original frames saved to: {orig_dir}")
    
    # Save metadata
    metadata = {
        "data_dir": args.data_dir,
        "n_samples": len(frames),
        "device": str(device),
        "dtype": str(dtype),
        "vae_model": "runwayml/stable-diffusion-v1-5",
        "frame_paths": frame_paths,
        "metrics": {
            "mse_mean": float(mse_per_frame.mean()),
            "mse_std": float(mse_per_frame.std()),
            "mse_min": float(mse_per_frame.min()),
            "mse_max": float(mse_per_frame.max()),
            "psnr_mean": float(psnr_per_frame.mean()),
            "psnr_std": float(psnr_per_frame.std()),
            "psnr_min": float(psnr_per_frame.min()),
            "psnr_max": float(psnr_per_frame.max()),
        }
    }
    
    import json
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to: {os.path.join(output_dir, 'metadata.json')}")
    
    print("\n" + "="*70)
    print("VAE RECONSTRUCTION CHECK COMPLETE!")
    print("="*70)
    print(f"Results saved to: {output_dir}")
    print(f"  - comparison_grid.png: Grid of first 10 frames")
    print(f"  - individual_comparisons/: All {len(frames)} frame-by-frame comparisons")
    print(f"  - reconstructed/: All reconstructed frames")
    print(f"  - original/: All original frames")
    print(f"  - metrics.png: MSE and PSNR distributions")
    print(f"  - metadata.json: Configuration and metrics")
    print("="*70)


if __name__ == "__main__":
    main()

