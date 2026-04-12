# scripts/

The eight numbered entry-point scripts that reproduce every result in the
project. Each script is a thin runner that imports its building blocks
from `../src/` and writes its outputs to the `outputs/`, `models/`, and
`features/` folders at the `github/` root.

## Run order

The scripts are numbered in the order they should be executed on a
fresh checkout. Each script reads artifacts produced by earlier scripts,
so they must be run in order at least once. After a full first run,
individual scripts can be re-run independently as long as their upstream
artifacts still exist.

## Files

| File | What it does | Inputs | Outputs |
|---|---|---|---|
| `01_eda.py` | Exploratory data analysis: counts files per class, draws a 16-image sample grid (8 parasitized + 8 uninfected), and plots image height and width histograms from a 1000-cell sample per class. | The dataset folder configured in `src/config.py`. | `outputs/eda_samples.png`, `outputs/eda_sizes.png` |
| `02_extract_features.py` | Builds the file index, performs the stratified 80/20 train/test split, runs the feature extraction loop with disk caching, fits a `StandardScaler` on the training matrix, and persists everything for downstream scripts. | The dataset folder. | `features/split.npz`, `features/features_train_*.npz`, `features/features_test_*.npz`, `features/features_scaled.npz`, `models/standard_scaler.joblib` |
| `03_train_classical.py` | Trains seven classical classifiers (LogReg, KNN, SVM-RBF, DecisionTree, RandomForest, XGBoost, GaussianNB) plus a `DummyClassifier` majority-class baseline. Computes ROC and PR curves and a confusion matrix for the best model on both splits. | `features/features_scaled.npz` | `models/model_*.joblib` (8 files), `outputs/classical_metrics_default.csv`, `outputs/classical_roc_pr.png`, `outputs/classical_best_confusion.png`, `outputs/best_classical_name.npz` |
| `04_tune_xgboost.py` | Runs `GridSearchCV` with stratified 5-fold cross-validation on the best default classifier, wrapped in a sklearn `Pipeline` so each fold refits its own scaler. Compares tuned vs default test metrics. | `features/features_scaled.npz`, `outputs/best_classical_name.npz`, `outputs/classical_metrics_default.csv` | `models/model_*_tuned.joblib`, `outputs/tuning_comparison.csv` |
| `05_feature_analysis.py` | Three interpretability analyses: feature category ablation (color only vs texture only vs combined), per-feature importance from the tuned tree model, and logistic regression coefficient inspection with a top-15 bar chart. | `features/features_scaled.npz`, `models/model_*_tuned.joblib`, `models/model_LogReg.joblib` | `outputs/feature_subset_comparison.csv`, `outputs/feature_subset_f1.png`, `outputs/feature_importance.csv`, `outputs/feature_importance.png`, `outputs/logreg_coefficients.csv`, `outputs/logreg_coefficients.png` |
| `06_pca_boundaries.py` | Projects the standardized features into 2D PCA space and plots a 7-panel decision boundary grid for every classical classifier trained on the 2D projection. | `features/features_scaled.npz` | `outputs/pca_2d.png`, `outputs/decision_boundaries.png` |
| `07_train_cnn.py` | Trains the small from-scratch CNN deep-learning baseline using the same train/test split as the classical models, prints the architecture and parameter count, saves the final state dict, computes full test metrics, and saves the training curve. | `features/split.npz`, raw image files | `models/cnn_state_dict.pt`, `outputs/cnn_metrics.csv`, `outputs/cnn_training_curve.png` |
| `08_final_comparison.py` | Assembles every model into a single ranking table including the dummy baseline, the seven default classical models, the tuned classical model, and the CNN. Produces the final F1 bar chart and the literature comparison vs Molina et al. (2020) at 97.7% cell-level accuracy. | `outputs/classical_metrics_default.csv`, `outputs/tuning_comparison.csv`, `outputs/cnn_metrics.csv` | `outputs/final_comparison.csv`, `outputs/final_f1_bars.png`, `outputs/literature_comparison.csv` |

## How a script knows where `src/` is

Every script begins with three lines that put the parent of `scripts/`
(that is, the `github/` folder) on `sys.path`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

After that line runs, `from src.config import ...` resolves correctly
no matter what the current working directory is.

## Recommended invocation

Run from the `github/` folder, not from inside `scripts/`:

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
