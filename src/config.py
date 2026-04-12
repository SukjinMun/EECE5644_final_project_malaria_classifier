"""Central configuration: paths, hyperparameters, random state.

All paths are relative to this file so the project is portable. Override the
data location by setting the MALARIA_DATA_DIR environment variable to the
folder that contains the Parasitized/ and Uninfected/ subdirectories.
"""
import os
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DATA_DIR = Path(os.environ.get(
    'MALARIA_DATA_DIR',
    PROJECT_ROOT / 'data' / 'cell_images',
))
PARASITIZED_DIR = DATA_DIR / 'Parasitized'
UNINFECTED_DIR = DATA_DIR / 'Uninfected'

OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
MODELS_DIR = PROJECT_ROOT / 'models'
FEATURES_DIR = PROJECT_ROOT / 'features'

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Pipeline hyperparameters
IMG_SIZE = 128
RANDOM_STATE = 42
TEST_SIZE = 0.2
SUBSAMPLE_PER_CLASS = None  # set to an int for fast development runs

# Feature extraction constants
GLCM_LEVELS = 32
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ['contrast', 'correlation', 'energy', 'homogeneity']

COLOR_FEATURE_NAMES = [
    f'{space}_ch{ch}_{stat}'
    for space in ('rgb', 'hsv')
    for ch in (0, 1, 2)
    for stat in ('mean', 'std', 'skew')
]
TEXTURE_FEATURE_NAMES = [
    f'glcm_{prop}_a{k}'
    for prop in GLCM_PROPS
    for k in range(len(GLCM_ANGLES))
]
ALL_FEATURE_NAMES = COLOR_FEATURE_NAMES + TEXTURE_FEATURE_NAMES

# CNN training hyperparameters
CNN_EPOCHS = 8
CNN_LR = 1e-3
CNN_BATCH_SIZE = 64
CNN_DROPOUT = 0.3
