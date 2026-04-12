"""Hyperparameter tuning for the best classical model via GridSearchCV.

Uses a sklearn Pipeline so each cross-validation fold refits its own scaler,
avoiding the train-validation leakage that occurs if a globally fitted
scaler is reused inside the search loop.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

from src.config import FEATURES_DIR, MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE


GRID_BY_TARGET = {
    'XGBoost': (
        XGBClassifier(eval_metric='logloss', n_jobs=-1, random_state=RANDOM_STATE),
        {
            'n_estimators': [200, 400],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
        },
    ),
    'RandomForest': (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {
            'n_estimators': [100, 200, 400],
            'max_depth': [None, 10, 20],
            'min_samples_leaf': [1, 2, 5],
        },
    ),
    'SVM-RBF': (
        SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        {'C': [0.5, 1, 5, 10], 'gamma': ['scale', 0.01, 0.1]},
    ),
    'KNN': (
        KNeighborsClassifier(),
        {'n_neighbors': [3, 5, 7, 11, 15], 'weights': ['uniform', 'distance']},
    ),
    'LogReg': (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {'C': [0.1, 1, 10], 'penalty': ['l2']},
    ),
    'DecisionTree': (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {'max_depth': [None, 5, 10, 20], 'min_samples_leaf': [1, 5, 10]},
    ),
}


def main():
    data = np.load(FEATURES_DIR / 'features_scaled.npz')
    X_train_raw = data['X_train_raw']
    X_test_raw = data['X_test_raw']
    y_train = data['y_train']
    y_test = data['y_test']

    # Pick the best default model from the previous step
    best_blob = np.load(OUTPUTS_DIR / 'best_classical_name.npz', allow_pickle=True)
    best_name = str(best_blob['name'][0])
    if best_name not in GRID_BY_TARGET:
        # Fall back to XGBoost if the winner is one we did not define a grid for
        best_name = 'XGBoost'
    print(f'tuning target: {best_name}')

    base_clf, raw_grid = GRID_BY_TARGET[best_name]
    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', base_clf),
    ])
    param_grid = {f'clf__{k}': v for k, v in raw_grid.items()}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(base_pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
    t0 = time.time()
    gs.fit(X_train_raw, y_train)
    print(f'grid search done in {time.time() - t0:.1f}s')
    print(f'best params: {gs.best_params_}')
    print(f'best CV F1:  {gs.best_score_:.4f}')

    tuned_model = gs.best_estimator_
    y_pred_tuned = tuned_model.predict(X_test_raw)
    if hasattr(tuned_model, 'predict_proba'):
        y_score_tuned = tuned_model.predict_proba(X_test_raw)[:, 1]
    elif hasattr(tuned_model, 'decision_function'):
        y_score_tuned = tuned_model.decision_function(X_test_raw)
    else:
        y_score_tuned = y_pred_tuned.astype(float)

    tuned_metrics = {
        'model': f'{best_name} (tuned)',
        'accuracy': accuracy_score(y_test, y_pred_tuned),
        'precision': precision_score(y_test, y_pred_tuned),
        'recall': recall_score(y_test, y_pred_tuned),
        'f1': f1_score(y_test, y_pred_tuned),
        'auc': roc_auc_score(y_test, y_score_tuned),
    }

    default_metrics_df = pd.read_csv(OUTPUTS_DIR / 'classical_metrics_default.csv').set_index('model')
    default_row = default_metrics_df.loc[best_name].to_dict()
    default_row['model'] = f'{best_name} (default)'
    compare_df = pd.DataFrame([default_row, tuned_metrics]).set_index('model')[
        ['accuracy', 'precision', 'recall', 'f1', 'auc']
    ]
    print(compare_df.round(4).to_string())
    compare_df.to_csv(OUTPUTS_DIR / 'tuning_comparison.csv')

    safe_name = best_name.replace(' ', '_')
    joblib.dump(tuned_model, MODELS_DIR / f'model_{safe_name}_tuned.joblib')
    print(f'saved tuned model to {MODELS_DIR / f"model_{safe_name}_tuned.joblib"}')


if __name__ == '__main__':
    main()
