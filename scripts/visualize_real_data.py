#!/usr/bin/env python3
"""
Visualize processed real microscopy data.
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
import argparse


def visualize_wf_2p_pairs(data_dir: str, num_samples: int = 4):
    """Visualize WF/2P pairs."""
    data_path = Path(data_dir)
    wf_dir = data_path / "splits" / "train" / "wf"
    tp_dir = data_path / "splits" / "train" / "2p"
    
    # Get sample files
    wf_files = sorted(list(wf_dir.glob("*.png")))[:num_samples]
    tp_files = sorted(list(tp_dir.glob("*.png")))[:num_samples]
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (wf_file, tp_file) in enumerate(zip(wf_files, tp_files)):
        # Load images
        wf_img = np.array(Image.open(wf_file))
        tp_img = np.array(Image.open(tp_file))
        
        # Plot WF (noisy)
        axes[0, i].imshow(wf_img, cmap='gray')
        axes[0, i].set_title(f'WF (noisy)\n{wf_file.name}')
        axes[0, i].axis('off')
        
        # Plot 2P (clean)
        axes[1, i].imshow(tp_img, cmap='gray')
        axes[1, i].set_title(f'2P (clean)\n{tp_file.name}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(data_path / "wf_2p_pairs_sample.png", dpi=150, bbox_inches='tight')
    print(f"Saved WF/2P pairs visualization to {data_path / 'wf_2p_pairs_sample.png'}")
    plt.close()


def visualize_beads(data_dir: str):
    """Visualize bead data."""
    data_path = Path(data_dir)
    beads_dir = data_path / "beads"
    
    if not beads_dir.exists():
        print("No bead data found")
        return
    
    # Load bead projections
    bead_no_ao = None
    bead_with_ao = None
    
    if (beads_dir / "bead_no_AO_projection.png").exists():
        bead_no_ao = np.array(Image.open(beads_dir / "bead_no_AO_projection.png"))
    
    if (beads_dir / "bead_with_AO_projection.png").exists():
        bead_with_ao = np.array(Image.open(beads_dir / "bead_with_AO_projection.png"))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    if bead_no_ao is not None:
        axes[0].imshow(bead_no_ao, cmap='hot')
        axes[0].set_title('Bead without Adaptive Optics\n(Max Projection)')
        axes[0].axis('off')
    
    if bead_with_ao is not None:
        axes[1].imshow(bead_with_ao, cmap='hot')
        axes[1].set_title('Bead with Adaptive Optics\n(Max Projection)')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(data_path / "beads_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved bead comparison to {data_path / 'beads_comparison.png'}")
    plt.close()


def plot_data_statistics(data_dir: str):
    """Plot statistics of the processed data."""
    data_path = Path(data_dir)
    
    # Count files in each split
    splits = ['train', 'val', 'test']
    counts = {}
    
    for split in splits:
        wf_dir = data_path / "splits" / split / "wf"
        if wf_dir.exists():
            counts[split] = len(list(wf_dir.glob("*.png")))
        else:
            counts[split] = 0
    
    # Plot split distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Split distribution
    ax1.bar(counts.keys(), counts.values(), color=['skyblue', 'lightgreen', 'coral'])
    ax1.set_title('Data Split Distribution')
    ax1.set_ylabel('Number of Image Pairs')
    for i, (split, count) in enumerate(counts.items()):
        ax1.text(i, count + 50, str(count), ha='center', va='bottom', fontweight='bold')
    
    # Sample intensity distribution
    wf_intensities = []
    tp_intensities = []
    
    wf_files = list((data_path / "splits" / "train" / "wf").glob("*.png"))[:100]  # Sample 100 images
    tp_files = list((data_path / "splits" / "train" / "2p").glob("*.png"))[:100]
    
    for wf_file, tp_file in zip(wf_files, tp_files):
        wf_img = np.array(Image.open(wf_file))
        tp_img = np.array(Image.open(tp_file))
        wf_intensities.append(wf_img.mean())
        tp_intensities.append(tp_img.mean())
    
    ax2.hist(wf_intensities, bins=20, alpha=0.7, label='WF (noisy)', color='red')
    ax2.hist(tp_intensities, bins=20, alpha=0.7, label='2P (clean)', color='blue')
    ax2.set_title('Mean Intensity Distribution (Sample)')
    ax2.set_xlabel('Mean Intensity')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(data_path / "data_statistics.png", dpi=150, bbox_inches='tight')
    print(f"Saved data statistics to {data_path / 'data_statistics.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize processed real microscopy data")
    parser.add_argument("--data-dir", default="data/real_microscopy", 
                       help="Directory with processed data")
    parser.add_argument("--num-samples", type=int, default=4,
                       help="Number of sample pairs to visualize")
    
    args = parser.parse_args()
    
    print(f"Visualizing data from: {args.data_dir}")
    
    # Create visualizations
    visualize_wf_2p_pairs(args.data_dir, args.num_samples)
    visualize_beads(args.data_dir)
    plot_data_statistics(args.data_dir)
    
    print("Visualization complete!")


if __name__ == "__main__":
    main()
