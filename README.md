# Malaria Cell Image Classification — Source Code

Python script port of the Jupyter notebook from this project. Each numbered
script corresponds to one logical section of the notebook and is intended to
be run in order. Together they reproduce every figure and metric the project
reports.

## What this code does

Given the NIH Malaria Cell Images Dataset, the pipeline:

1. Visualizes the dataset (sample image grid, image size distribution).
2. Builds a stratified 80/20 train/test split and extracts a 34-dimensional
   handcrafted feature vector per image (18 color statistics in RGB and HSV,
   16 GLCM-based texture statistics).
3. Trains seven classical classifiers plus a Dummy majority-class baseline.
4. Hyperparameter-tunes the strongest classical model with `GridSearchCV`
   inside a sklearn `Pipeline` so each cross-validation fold refits its own
   scaler.
5. Performs a feature category ablation (color only vs texture only vs
   combined), per-feature importance from the tuned tree model, and a
   logistic regression coefficient interpretation.
6. Projects the test set into 2D PCA space and plots the decision regions
   of every classical classifier.
7. Trains a small from-scratch CNN as a deep-learning baseline.
8. Assembles a final comparison table of every model and benchmarks the
   results against Molina et al. (2020).

## File layout

```
github/
├── README.md                       this file
├── requirements.txt                pip dependencies
├── src/                            reusable importable modules
│   ├── README.md                   per-file description of the src/ folder
│   ├── __init__.py                 marks src/ as a Python package
│   ├── config.py                   central paths, hyperparameters, random state
│   ├── feature_extraction.py       handcrafted color and texture functions
│   └── cnn_model.py                CNN architecture and Dataset class
└── scripts/                        numbered entry-point scripts (run in order)
    ├── README.md                   per-file description of the scripts/ folder
    ├── 01_eda.py                   sample cells + image size distribution
    ├── 02_extract_features.py      train/test split + feature extraction + scaling
    ├── 03_train_classical.py       7 classical models + Dummy baseline
    ├── 04_tune_xgboost.py          GridSearchCV on the best default classifier
    ├── 05_feature_analysis.py      feature ablation + importance + LogReg coefs
    ├── 06_pca_boundaries.py        2D PCA scatter + decision boundary grid
    ├── 07_train_cnn.py             small CNN deep-learning baseline
    └── 08_final_comparison.py      final ranking + literature comparison
```

Each file in `scripts/` is a thin entry point that imports its building
blocks from `src/`. The numbered prefixes signal the intended run order.

The pipeline produces three folders of artifacts on first run:

```
features/                     cached feature matrices (.npz)
models/                       trained model files (.joblib + .pt)
outputs/                      figures (.png) and metric tables (.csv)
```

These three folders are created automatically by `config.py` on import and
are intended to be added to `.gitignore` so generated artifacts do not get
committed.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The pinned versions in `requirements.txt` are minimum versions that have
been tested. Newer versions usually work too.

### 2. Get the dataset

Download the NIH Malaria Cell Images Dataset from Kaggle:
<https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria>

Extract the archive. After extraction the layout should look like:

```
some_folder/
└── cell_images/
    ├── Parasitized/         (~13,779 PNG files)
    └── Uninfected/          (~13,779 PNG files)
```

### 3. Tell the scripts where the data is

Either place the extracted `cell_images/` folder at `github/data/cell_images/`
(the default), or set the environment variable `MALARIA_DATA_DIR` to the
folder that contains the `Parasitized/` and `Uninfected/` subdirectories:

```bash
export MALARIA_DATA_DIR=/absolute/path/to/cell_images
```

On Windows PowerShell:

```powershell
$env:MALARIA_DATA_DIR = "C:\path\to\cell_images"
```

The scripts read this variable on import in `config.py` so the rest of the
code stays portable.

## How to run

Run the eight numbered scripts in order **from the `github/` folder**, not
from inside `scripts/`:

```bash
python scripts/01_eda.py
python scripts/02_extract_features.py
python scripts/03_train_classical.py
python scripts/04_tune_xgboost.py
python scripts/05_feature_analysis.py
python scripts/06_pca_boundaries.py
python scripts/07_train_cnn.py
python scripts/08_final_comparison.py
```

Each script puts its parent directory on `sys.path` so that
`from src.config import ...` resolves regardless of the working directory,
but it is still cleanest to run them from `github/` so the relative
output paths land correctly.

Each script reads the artifacts produced by earlier scripts (cached
features in `features/`, trained models in `models/`, intermediate outputs
in `outputs/`), so they must be run in order at least once. After the
first full run, individual scripts can be re-run independently as long as
their upstream artifacts still exist.

## Reproducibility

All randomness is controlled by `RANDOM_STATE = 42` in `config.py`, including:

- the stratified train/test split
- the `random_state` argument of every classifier and `GridSearchCV`
- the NumPy generators used for sampling test points and image inspection

Classical model results are deterministic across re-runs. The CNN may drift
by a few hundredths of a percentage point between runs because of PyTorch
GPU nondeterminism.

## Hardware

- The classical pipeline (scripts 1-6 and 8) runs comfortably on a modern
  CPU laptop. Total wall time is roughly 10-15 minutes for the first run
  (the feature extraction loop is the slowest single step) and under a
  minute for re-runs that hit the cache.
- The CNN script (7) is much faster on a GPU. On a free Colab T4 it trains
  in 2-4 minutes for 8 epochs. On a CPU it takes roughly 30 minutes.

## Configuration knobs

All tunable settings live in `config.py`:

- `IMG_SIZE`: image resize target (default 128)
- `RANDOM_STATE`: reproducibility seed (default 42)
- `TEST_SIZE`: held-out test fraction (default 0.2)
- `SUBSAMPLE_PER_CLASS`: set to a small int for fast development runs;
  leave as `None` to use the full dataset
- `CNN_EPOCHS`, `CNN_LR`, `CNN_BATCH_SIZE`, `CNN_DROPOUT`: CNN training
  hyperparameters

If `IMG_SIZE` or `SUBSAMPLE_PER_CLASS` change, the cached feature files in
`features/` are keyed on those values, so a different config will trigger
a fresh extraction without colliding with old caches.

## Citation

If you use this code, please cite the dataset providers and the published
baseline used for benchmarking:

- NIH Malaria Cell Images Dataset (mirrored on Kaggle by `iarunava`):
  <https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria>
- Molina, A., Alferez, S., Boldu, L., Acevedo, A., Rodellar, J., and
  Merino, A. (2020). Sequential classification system for recognition of
  malaria infection using peripheral blood cell images. *Journal of
  Clinical Pathology*, 73(10):665-670.
