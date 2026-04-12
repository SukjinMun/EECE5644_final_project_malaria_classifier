# Build the file index, perform the stratified train/test split, extract features.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import (
    PARASITIZED_DIR,
    UNINFECTED_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    IMG_SIZE,
    RANDOM_STATE,
    TEST_SIZE,
    SUBSAMPLE_PER_CLASS,
)
from src.feature_extraction import extract_features_for_paths

def list_images(folder):
    return sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])

def build_paths():
    parasitized_files = list_images(PARASITIZED_DIR)
    uninfected_files = list_images(UNINFECTED_DIR)
    paths = []
    labels = []
    for f in parasitized_files:
        paths.append(os.path.join(PARASITIZED_DIR, f))
        labels.append(1)
    for f in uninfected_files:
        paths.append(os.path.join(UNINFECTED_DIR, f))
        labels.append(0)
    return np.array(paths), np.array(labels)

def main():
    all_paths, all_labels = build_paths()

    if SUBSAMPLE_PER_CLASS is not None:
        rng = np.random.default_rng(RANDOM_STATE)
        para_idx = np.where(all_labels == 1)[0]
        unin_idx = np.where(all_labels == 0)[0]
        para_keep = rng.choice(para_idx, SUBSAMPLE_PER_CLASS, replace=False)
        unin_keep = rng.choice(unin_idx, SUBSAMPLE_PER_CLASS, replace=False)
        keep = np.concatenate([para_keep, unin_keep])
        all_paths = all_paths[keep]
        all_labels = all_labels[keep]
        print(f'subsampled to {len(all_paths)} images')

    paths_train, paths_test, y_train, y_test = train_test_split(
        all_paths, all_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=all_labels,
    )
    print(f'training set: {len(paths_train)} images, train parasitized ratio = {y_train.mean():.4f}')
    print(f'test set:     {len(paths_test)} images,  test parasitized ratio  = {y_test.mean():.4f}')

    # Persist the path/label split so downstream scripts use the same partition
    np.savez(
        FEATURES_DIR / 'split.npz',
        paths_train=paths_train,
        paths_test=paths_test,
        y_train=y_train,
        y_test=y_test,
    )
    print(f'saved split to {FEATURES_DIR / "split.npz"}')

    # Extract features for both splits, with disk caching
    cache_train = FEATURES_DIR / f'features_train_{IMG_SIZE}_sub{SUBSAMPLE_PER_CLASS}.npz'
    cache_test = FEATURES_DIR / f'features_test_{IMG_SIZE}_sub{SUBSAMPLE_PER_CLASS}.npz'

    print('extracting training features ...')
    X_train_raw = extract_features_for_paths(paths_train, 'train', str(cache_train))
    print('extracting test features ...')
    X_test_raw = extract_features_for_paths(paths_test, 'test', str(cache_test))
    print(f'X_train_raw shape: {X_train_raw.shape}')
    print(f'X_test_raw shape:  {X_test_raw.shape}')

    # Fit a StandardScaler on the training matrix only and persist it
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    joblib.dump(scaler, MODELS_DIR / 'standard_scaler.joblib')
    print(f'saved scaler to {MODELS_DIR / "standard_scaler.joblib"}')

    # Persist the scaled feature matrices for downstream scripts
    np.savez(
        FEATURES_DIR / 'features_scaled.npz',
        X_train=X_train, X_test=X_test,
        X_train_raw=X_train_raw, X_test_raw=X_test_raw,
        y_train=y_train, y_test=y_test,
    )
    print(f'saved scaled features to {FEATURES_DIR / "features_scaled.npz"}')

if __name__ == '__main__':
    main()
