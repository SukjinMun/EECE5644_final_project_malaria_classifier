"""Handcrafted color and texture feature extraction.

Each cell image is reduced to a fixed-length feature vector consisting of
18 color statistics (RGB and HSV channels, mean / std / skew) and 16
GLCM-based texture statistics (contrast / correlation / energy / homogeneity
at four angles). Total: 34 features per image.
"""
import os
import time
import numpy as np
from skimage.io import imread
from skimage.transform import resize as sk_resize
from skimage.color import rgb2hsv, rgb2gray
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew

from .config import (
    IMG_SIZE,
    GLCM_LEVELS,
    GLCM_ANGLES,
    GLCM_PROPS,
    ALL_FEATURE_NAMES,
)


def color_features(img_rgb_uint8):
    """Color statistics in RGB and HSV color spaces, 18 dimensions."""
    img_rgb = img_rgb_uint8.astype(np.float32) / 255.0
    img_hsv = rgb2hsv(img_rgb)
    feats = []
    for img in (img_rgb, img_hsv):
        for ch in range(3):
            c = img[:, :, ch].ravel()
            feats.append(float(c.mean()))
            feats.append(float(c.std()))
            s = skew(c)
            if not np.isfinite(s):
                s = 0.0
            feats.append(float(s))
    return np.array(feats, dtype=np.float32)


def texture_features(img_rgb_uint8):
    """GLCM-based texture statistics from a quantized grayscale image, 16 dimensions."""
    gray = rgb2gray(img_rgb_uint8.astype(np.float32) / 255.0)
    gray_q = np.clip(
        (gray * (GLCM_LEVELS - 1)).round().astype(np.uint8),
        0, GLCM_LEVELS - 1,
    )
    glcm = graycomatrix(
        gray_q,
        distances=[1],
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    feats = []
    for prop in GLCM_PROPS:
        v = graycoprops(glcm, prop)[0]  # shape (len(angles),)
        feats.extend(v.tolist())
    return np.array(feats, dtype=np.float32)


def extract_one(path):
    """Extract a 34-dimensional feature vector from a single image file."""
    img = imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img_resized = sk_resize(
        img, (IMG_SIZE, IMG_SIZE),
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)
    cf = color_features(img_resized)
    tf = texture_features(img_resized)
    return np.concatenate([cf, tf])


def extract_features_for_paths(paths, label='split', cache_path=None):
    """Extract features for every path with disk caching keyed by cache_path."""
    if cache_path is not None and os.path.exists(cache_path):
        print(f'  loading cached features from {cache_path}')
        data = np.load(cache_path)
        return data['X']
    n = len(paths)
    X = np.zeros((n, len(ALL_FEATURE_NAMES)), dtype=np.float32)
    t0 = time.time()
    report_every = max(1, n // 20)
    for i, p in enumerate(paths):
        X[i] = extract_one(p)
        if (i + 1) % report_every == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (n - i - 1) / max(rate, 1e-9)
            print(
                f'  [{label}] {i+1:>6}/{n}  '
                f'{rate:.1f} img/s  elapsed {elapsed:.0f}s  eta {eta:.0f}s'
            )
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, X=X)
        print(f'  saved cache to {cache_path}')
    return X
