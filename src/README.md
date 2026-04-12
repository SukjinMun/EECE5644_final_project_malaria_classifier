# src/

Reusable importable modules for the project. The numbered scripts in
`../scripts/` import everything they need from this folder.

## Files

| File | Purpose | Source |
|---|---|---|
| `__init__.py` | Marks `src/` as a Python package so the scripts can do `from src.config import ...`. Empty apart from a one-line docstring. | Hand-written. |
| `config.py` | Central project configuration. Defines `PROJECT_ROOT`, the data and output paths (with `MALARIA_DATA_DIR` env-var override), the random seed, the image resize target, the GLCM constants, the full feature name list (color + texture), and the CNN training hyperparameters. **Every other module imports its paths and constants from here.** | Hand-written. |
| `feature_extraction.py` | Handcrafted feature extraction functions: `color_features` (18 RGB+HSV stats), `texture_features` (16 GLCM stats), `extract_one` (single image to 34-D vector), and `extract_features_for_paths` (loop with disk caching and progress reporting). | Hand-written. |
| `cnn_model.py` | Small CNN architecture and PyTorch Dataset class: `CellImageDataset` loads cell images on demand and converts them to CHW float tensors in `[0, 1]`, and `SmallCNN` defines a 3-block conv network with about 1M parameters. | Hand-written. |

## Why these are separate from `scripts/`

The `src/` modules contain logic that is reused by more than one script
(for example `feature_extraction.py` is used by both `02_extract_features.py`
during the initial extraction and could be reused by an inference script
later). Keeping them in a dedicated package makes the import boundary
clear, lets a future contributor unit-test these functions in isolation,
and avoids duplicating code across the eight scripts.

## How the scripts find these modules

Each script in `../scripts/` starts with three lines that put the
`github/` folder onto `sys.path`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

After that line, `from src.config import ...` resolves naturally because
the script's parent directory contains the `src/` package.

## Modifying these files

If you change something in `config.py`, every downstream script picks up
the change on its next run. The most common edits are:

- Changing `IMG_SIZE` (will trigger a fresh feature extraction because
  the cache filenames are keyed on `IMG_SIZE`)
- Changing `SUBSAMPLE_PER_CLASS` for fast development runs
- Changing `RANDOM_STATE` to test sensitivity to the train/test split
- Changing the `CNN_*` hyperparameters
