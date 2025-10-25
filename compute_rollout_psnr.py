#!/usr/bin/env python3
"""
Compute PSNR from rollout data to compare with GameNGen paper Figure 13.

According to the GameNGen paper, Figure 13 shows PSNR degradation over rollout steps,
with initial PSNR around 26-29 dB that degrades over autoregressive steps.

PSNR formula (standard):
    PSNR = 10 * log10(MAX^2 / MSE)
    
For 8-bit images (pixel values 0-255):
    PSNR = 10 * log10(255^2 / MSE) = 20 * log10(255 / sqrt(MSE))
    
For normalized images (0-1 range):
    PSNR = 10 * log10(1.0 / MSE) = 20 * log10(1.0 / sqrt(MSE))
"""

import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json


def compute_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """
    Compute PSNR between two images.
    
    Args:
        img1: First image (H, W, C) with values in [0, max_val]
        img2: Second image (H, W, C) with values in [0, max_val]
        max_val: Maximum pixel value (255 for 8-bit images)
    
    Returns:
        PSNR in dB
    """
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10(max_val ** 2 / mse)
    return psnr


def load_image(path: str) -> np.ndarray:
    """Load image as numpy array."""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def analyze_single_rollout(rollout_dir: str, verbose: bool = False) -> Dict:
    """
    Analyze a single rollout directory containing context_*.png, gt_*.png, pred_*.png.
    
    Args:
        rollout_dir: Path to rollout directory
        verbose: Print detailed information
    
    Returns:
        Dictionary with PSNR statistics
    """
    # Find all gt and pred frames
    gt_files = sorted([f for f in os.listdir(rollout_dir) if f.startswith('gt_') and f.endswith('.png')])
    pred_files = sorted([f for f in os.listdir(rollout_dir) if f.startswith('pred_') and f.endswith('.png')])
    
    if len(gt_files) == 0 or len(pred_files) == 0:
        return None
    
    if len(gt_files) != len(pred_files):
        print(f"Warning: Mismatch in {rollout_dir}: {len(gt_files)} GT vs {len(pred_files)} pred frames")
        # Use minimum
        n_frames = min(len(gt_files), len(pred_files))
        gt_files = gt_files[:n_frames]
        pred_files = pred_files[:n_frames]
    
    psnr_per_frame = []
    
    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_path = os.path.join(rollout_dir, gt_file)
        pred_path = os.path.join(rollout_dir, pred_file)
        
        gt_img = load_image(gt_path)
        pred_img = load_image(pred_path)
        
        # Compute PSNR using 8-bit image range (0-255)
        psnr = compute_psnr(gt_img, pred_img, max_val=255.0)
        psnr_per_frame.append(psnr)
        
        if verbose:
            print(f"  {gt_file} vs {pred_file}: PSNR = {psnr:.2f} dB")
    
    psnr_array = np.array(psnr_per_frame)
    
    results = {
        'n_frames': len(psnr_per_frame),
        'psnr_mean': float(np.mean(psnr_array)),
        'psnr_std': float(np.std(psnr_array)),
        'psnr_min': float(np.min(psnr_array)),
        'psnr_max': float(np.max(psnr_array)),
        'psnr_per_frame': psnr_per_frame,
    }
    
    return results


def analyze_epoch_directory(epoch_dir: str, epoch_name: str, verbose: bool = False) -> Dict:
    """
    Analyze an epoch directory (e.g., epoch_100 or timestamp directories).
    
    Args:
        epoch_dir: Path to epoch directory
        epoch_name: Name of the epoch (for display)
        verbose: Print detailed information
    
    Returns:
        Dictionary with analysis results
    """
    if not os.path.exists(epoch_dir):
        return None
    
    # Check if this is a single rollout or contains multiple step directories
    subdirs = [d for d in os.listdir(epoch_dir) if os.path.isdir(os.path.join(epoch_dir, d))]
    
    if len(subdirs) > 0 and all(d.startswith('step_') for d in subdirs):
        # This directory contains multiple steps
        step_results = {}
        for step_dir in sorted(subdirs):
            step_path = os.path.join(epoch_dir, step_dir)
            step_num = int(step_dir.split('_')[1])
            result = analyze_single_rollout(step_path, verbose=verbose)
            if result:
                step_results[step_num] = result
        
        if len(step_results) == 0:
            return None
        
        # Compute aggregate statistics
        all_psnr_means = [r['psnr_mean'] for r in step_results.values()]
        
        return {
            'type': 'multi_step',
            'epoch_name': epoch_name,
            'n_steps': len(step_results),
            'step_results': step_results,
            'overall_psnr_mean': float(np.mean(all_psnr_means)),
            'overall_psnr_std': float(np.std(all_psnr_means)),
        }
    else:
        # This is a single rollout directory
        result = analyze_single_rollout(epoch_dir, verbose=verbose)
        if result:
            result['type'] = 'single_rollout'
            result['epoch_name'] = epoch_name
        return result


def plot_psnr_progression(results_dict: Dict[str, Dict], output_path: str):
    """
    Plot PSNR progression across training steps or epochs.
    
    Args:
        results_dict: Dictionary mapping epoch names to results
        output_path: Where to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data for plotting
    epoch_names = []
    mean_psnr = []
    std_psnr = []
    
    # Sort by epoch number if available
    sorted_items = sorted(results_dict.items(), key=lambda x: (
        int(x[0].split('_')[1]) if x[0].startswith('epoch_') or x[0].startswith('step_') else 
        int(x[0]) if x[0].isdigit() else 0
    ))
    
    for epoch_name, result in sorted_items:
        if result is None:
            continue
        
        if result['type'] == 'single_rollout':
            epoch_names.append(epoch_name)
            mean_psnr.append(result['psnr_mean'])
            std_psnr.append(result['psnr_std'])
        elif result['type'] == 'multi_step':
            epoch_names.append(epoch_name)
            mean_psnr.append(result['overall_psnr_mean'])
            std_psnr.append(result['overall_psnr_std'])
    
    if len(mean_psnr) == 0:
        print("No PSNR data to plot")
        return
    
    # Plot 1: Mean PSNR across epochs
    ax1 = axes[0, 0]
    x_pos = np.arange(len(epoch_names))
    ax1.plot(x_pos, mean_psnr, 'o-', linewidth=2, markersize=6, color='#2E86DE', label='Mean PSNR')
    ax1.fill_between(x_pos, 
                      np.array(mean_psnr) - np.array(std_psnr), 
                      np.array(mean_psnr) + np.array(std_psnr),
                      alpha=0.3, color='#2E86DE', label='±1 Std Dev')
    ax1.axhline(y=26, color='red', linestyle='--', linewidth=2, label='GameNGen Target (~26 dB)')
    ax1.set_xlabel('Training Progress', fontsize=12, weight='bold')
    ax1.set_ylabel('PSNR (dB)', fontsize=12, weight='bold')
    ax1.set_title('Mean PSNR Across Training', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_xticks(x_pos[::max(1, len(x_pos)//10)])
    ax1.set_xticklabels([epoch_names[i] for i in range(0, len(epoch_names), max(1, len(epoch_names)//10))], rotation=45, ha='right')
    
    # Plot 2: PSNR distribution
    ax2 = axes[0, 1]
    ax2.hist(mean_psnr, bins=20, edgecolor='black', alpha=0.7, color='#10AC84')
    ax2.axvline(x=26, color='red', linestyle='--', linewidth=2, label='GameNGen Target')
    ax2.set_xlabel('PSNR (dB)', fontsize=12, weight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, weight='bold')
    ax2.set_title('PSNR Distribution', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Per-frame PSNR for selected checkpoints (if available)
    ax3 = axes[1, 0]
    plotted_any = False
    
    # Plot first, middle, and last checkpoints
    indices_to_plot = [0, len(sorted_items)//2, -1] if len(sorted_items) > 2 else list(range(len(sorted_items)))
    colors = ['#EE5A6F', '#FFA502', '#26DE81']
    
    for idx, color in zip(indices_to_plot[:3], colors):
        if idx >= len(sorted_items):
            continue
        epoch_name, result = sorted_items[idx]
        if result and 'psnr_per_frame' in result:
            frames = np.arange(len(result['psnr_per_frame']))
            ax3.plot(frames, result['psnr_per_frame'], 'o-', 
                    label=f"{epoch_name} (mean: {result['psnr_mean']:.2f} dB)",
                    linewidth=2, markersize=4, color=color)
            plotted_any = True
    
    if plotted_any:
        ax3.axhline(y=26, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='GameNGen Target')
        ax3.set_xlabel('Frame in Rollout', fontsize=12, weight='bold')
        ax3.set_ylabel('PSNR (dB)', fontsize=12, weight='bold')
        ax3.set_title('Per-Frame PSNR (Selected Checkpoints)', fontsize=14, weight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No per-frame data available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    
    # Plot 4: Statistics summary table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = "PSNR Statistics Summary\n" + "="*40 + "\n\n"
    stats_text += f"Total checkpoints analyzed: {len(mean_psnr)}\n\n"
    stats_text += f"Overall PSNR:\n"
    stats_text += f"  Mean:   {np.mean(mean_psnr):.2f} dB\n"
    stats_text += f"  Std:    {np.std(mean_psnr):.2f} dB\n"
    stats_text += f"  Min:    {np.min(mean_psnr):.2f} dB\n"
    stats_text += f"  Max:    {np.max(mean_psnr):.2f} dB\n\n"
    
    stats_text += f"GameNGen Target: ~26 dB\n"
    above_target = sum(1 for p in mean_psnr if p >= 26)
    stats_text += f"Checkpoints >= 26 dB: {above_target}/{len(mean_psnr)} ({100*above_target/len(mean_psnr):.1f}%)\n\n"
    
    if len(mean_psnr) > 0:
        stats_text += f"Latest checkpoint ({epoch_names[-1]}):\n"
        stats_text += f"  PSNR: {mean_psnr[-1]:.2f} ± {std_psnr[-1]:.2f} dB\n"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ PSNR progression plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compute PSNR from rollout data to compare with GameNGen paper Figure 13'
    )
    parser.add_argument('--rollout_dir', type=str, default=None,
                       help='Specific rollout directory to analyze (e.g., debug/rollouts/20251024_103057)')
    parser.add_argument('--rollouts_dir', type=str, default='debug/rollouts',
                       help='Directory containing rollout subdirectories (used if --rollout_dir not specified)')
    parser.add_argument('--specific_step', type=str, default=None,
                       help='Analyze a specific step/timestamp directory')
    parser.add_argument('--output', type=str, default='psnr_analysis.png',
                       help='Output plot filename')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed per-frame PSNR')
    parser.add_argument('--save_json', type=str, default=None,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PSNR Analysis for GameNGen-style Model Training")
    print("="*70)
    print(f"\nTarget PSNR from GameNGen paper Figure 13: ~26 dB")
    print(f"(This is the autoregressive prediction quality baseline)\n")
    
    results = {}
    
    # Determine which directory to analyze
    if args.rollout_dir:
        # Analyze a specific rollout directory (contains step_* subdirectories)
        target_dir = args.rollout_dir
        print(f"\nAnalyzing rollout directory: {target_dir}\n")
        
        if not os.path.exists(target_dir):
            print(f"Error: Rollout directory not found: {target_dir}")
            return
        
        # Find all step subdirectories
        subdirs = [d for d in os.listdir(target_dir) 
                  if os.path.isdir(os.path.join(target_dir, d)) and d.startswith('step_')]
        
        if len(subdirs) == 0:
            print(f"Error: No step_* subdirectories found in {target_dir}")
            return
        
        # Sort by step number
        subdirs.sort(key=lambda x: int(x.split('_')[1]))
        
        print(f"Found {len(subdirs)} training steps\n")
        
        for subdir in subdirs:
            step_path = os.path.join(target_dir, subdir)
            result = analyze_epoch_directory(step_path, subdir, verbose=args.verbose)
            
            if result:
                results[subdir] = result
                if result['type'] == 'single_rollout':
                    print(f"✓ {subdir:20s}: PSNR = {result['psnr_mean']:6.2f} ± {result['psnr_std']:5.2f} dB "
                          f"({result['n_frames']} frames)")
                else:
                    print(f"✓ {subdir:20s}: PSNR = {result['overall_psnr_mean']:6.2f} ± {result['overall_psnr_std']:5.2f} dB "
                          f"({result['n_steps']} substeps)")
    
    elif args.specific_step:
        # Analyze a specific step
        step_path = os.path.join(args.rollouts_dir, args.specific_step)
        if not os.path.exists(step_path):
            step_path = args.specific_step  # Try as absolute path
        
        print(f"\nAnalyzing specific step: {args.specific_step}")
        print(f"Path: {step_path}")
        
        result = analyze_epoch_directory(step_path, args.specific_step, verbose=args.verbose)
        if result:
            results[args.specific_step] = result
            print(f"\n✓ {args.specific_step}:")
            if result['type'] == 'single_rollout':
                print(f"    PSNR: {result['psnr_mean']:.2f} ± {result['psnr_std']:.2f} dB")
                print(f"    Range: [{result['psnr_min']:.2f}, {result['psnr_max']:.2f}] dB")
                print(f"    Frames: {result['n_frames']}")
            else:
                print(f"    Overall PSNR: {result['overall_psnr_mean']:.2f} ± {result['overall_psnr_std']:.2f} dB")
                print(f"    Steps analyzed: {result['n_steps']}")
        else:
            print(f"✗ Could not analyze {args.specific_step}")
    else:
        # Analyze all subdirectories in the rollouts directory
        print(f"\nScanning rollouts directory: {args.rollouts_dir}\n")
        
        if not os.path.exists(args.rollouts_dir):
            print(f"Error: Rollouts directory not found: {args.rollouts_dir}")
            return
        
        # Find all subdirectories (steps or timestamps)
        subdirs = [d for d in os.listdir(args.rollouts_dir) 
                  if os.path.isdir(os.path.join(args.rollouts_dir, d))]
        
        if len(subdirs) == 0:
            print(f"Error: No subdirectories found in {args.rollouts_dir}")
            return
        
        # Sort subdirectories (handle both step_XXX and timestamp formats)
        def sort_key(name):
            if name.startswith('step_'):
                return (0, int(name.split('_')[1]))
            elif name.startswith('epoch_'):
                return (1, int(name.split('_')[1]))
            else:
                return (2, name)
        
        subdirs.sort(key=sort_key)
        
        print(f"Found {len(subdirs)} rollout directories\n")
        
        for subdir in subdirs:
            step_path = os.path.join(args.rollouts_dir, subdir)
            result = analyze_epoch_directory(step_path, subdir, verbose=args.verbose)
            
            if result:
                results[subdir] = result
                if result['type'] == 'single_rollout':
                    print(f"✓ {subdir:20s}: PSNR = {result['psnr_mean']:6.2f} ± {result['psnr_std']:5.2f} dB "
                          f"({result['n_frames']} frames)")
                else:
                    print(f"✓ {subdir:20s}: PSNR = {result['overall_psnr_mean']:6.2f} ± {result['overall_psnr_std']:5.2f} dB "
                          f"({result['n_steps']} steps)")
    
    if len(results) == 0:
        print("\nNo valid rollout data found to analyze.")
        return
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_mean_psnr = []
    for result in results.values():
        if result['type'] == 'single_rollout':
            all_mean_psnr.append(result['psnr_mean'])
        else:
            all_mean_psnr.append(result['overall_psnr_mean'])
    
    print(f"\nTotal checkpoints analyzed: {len(all_mean_psnr)}")
    print(f"Overall PSNR across all checkpoints:")
    print(f"  Mean: {np.mean(all_mean_psnr):.2f} dB")
    print(f"  Std:  {np.std(all_mean_psnr):.2f} dB")
    print(f"  Min:  {np.min(all_mean_psnr):.2f} dB")
    print(f"  Max:  {np.max(all_mean_psnr):.2f} dB")
    
    above_target = sum(1 for p in all_mean_psnr if p >= 26)
    print(f"\nCheckpoints with PSNR >= 26 dB: {above_target}/{len(all_mean_psnr)} ({100*above_target/len(all_mean_psnr):.1f}%)")
    
    # Compare with GameNGen target
    avg_psnr = np.mean(all_mean_psnr)
    diff = avg_psnr - 26
    print(f"\nComparison with GameNGen target (~26 dB):")
    if diff >= 0:
        print(f"  Your model: {avg_psnr:.2f} dB (✓ {diff:+.2f} dB vs target)")
    else:
        print(f"  Your model: {avg_psnr:.2f} dB ({diff:.2f} dB vs target)")
    
    # Create visualization
    if len(results) > 1:
        plot_psnr_progression(results, args.output)
    else:
        print("\n(Skipping plot generation - need multiple checkpoints for progression plot)")
    
    # Save JSON if requested
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.save_json}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

