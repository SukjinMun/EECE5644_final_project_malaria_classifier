# Exploratory data analysis: file counts, sample image grid, image size histograms.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from src.config import (
    PARASITIZED_DIR,
    UNINFECTED_DIR,
    OUTPUTS_DIR,
    RANDOM_STATE,
)

def list_images(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])

def main():
    parasitized_files = list_images(PARASITIZED_DIR)
    uninfected_files = list_images(UNINFECTED_DIR)

    print(f'parasitized PNG files: {len(parasitized_files)}')
    print(f'uninfected PNG files:  {len(uninfected_files)}')
    total = len(parasitized_files) + len(uninfected_files)
    print(f'total: {total}')
    print(f'class balance ratio: {len(parasitized_files) / total:.4f}')

    # Sample image grid: 8 from each class
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    rng = np.random.default_rng(RANDOM_STATE)
    para_sample = rng.choice(parasitized_files, 8, replace=False)
    unin_sample = rng.choice(uninfected_files, 8, replace=False)
    for i, fname in enumerate(para_sample):
        img = imread(os.path.join(PARASITIZED_DIR, fname))
        axes[0, i].imshow(img)
        axes[0, i].set_axis_off()
        if i == 0:
            axes[0, i].set_title('parasitized', loc='left', fontsize=12)
    for i, fname in enumerate(unin_sample):
        img = imread(os.path.join(UNINFECTED_DIR, fname))
        axes[1, i].imshow(img)
        axes[1, i].set_axis_off()
        if i == 0:
            axes[1, i].set_title('uninfected', loc='left', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'eda_samples.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # Image size distribution from a 1000-sample-per-class subsample
    def sample_sizes(folder, files, n=1000):
        chosen = rng.choice(files, min(n, len(files)), replace=False)
        sizes = []
        for f in chosen:
            img = imread(os.path.join(folder, f))
            sizes.append(img.shape[:2])
        return np.array(sizes)

    t0 = time.time()
    para_sizes = sample_sizes(PARASITIZED_DIR, parasitized_files)
    unin_sizes = sample_sizes(UNINFECTED_DIR, uninfected_files)
    print(f'sampled sizes in {time.time() - t0:.1f}s')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(para_sizes[:, 0], bins=30, alpha=0.6, label='parasitized')
    axes[0].hist(unin_sizes[:, 0], bins=30, alpha=0.6, label='uninfected')
    axes[0].set_xlabel('image height (pixels)')
    axes[0].set_ylabel('count')
    axes[0].set_title('image height distribution')
    axes[0].legend()
    axes[1].hist(para_sizes[:, 1], bins=30, alpha=0.6, label='parasitized')
    axes[1].hist(unin_sizes[:, 1], bins=30, alpha=0.6, label='uninfected')
    axes[1].set_xlabel('image width (pixels)')
    axes[1].set_ylabel('count')
    axes[1].set_title('image width distribution')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'eda_sizes.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    print('saved eda_samples.png and eda_sizes.png')

if __name__ == '__main__':
    main()
