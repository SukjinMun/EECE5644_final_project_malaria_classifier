"""Train seven classical classifiers + a Dummy baseline. Save metrics + ROC/PR + confusion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
)

from src.config import FEATURES_DIR, MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE


def evaluate_model(model, name, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_time = time.time() - t0
    y_pred = model.predict(X_te)
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_te)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_score = model.decision_function(X_te)
    else:
        y_score = y_pred.astype(float)
    metrics = {
        'model': name,
        'accuracy': accuracy_score(y_te, y_pred),
        'precision': precision_score(y_te, y_pred),
        'recall': recall_score(y_te, y_pred),
        'f1': f1_score(y_te, y_pred),
        'auc': roc_auc_score(y_te, y_score),
        'fit_time_s': fit_time,
    }
    return model, metrics, y_pred, y_score


def main():
    data = np.load(FEATURES_DIR / 'features_scaled.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    classifiers = {
        'Dummy (majority)': DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE),
        'LogReg': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM-RBF': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
        'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(n_estimators=200, eval_metric='logloss',
                                 n_jobs=-1, random_state=RANDOM_STATE),
        'GaussianNB': GaussianNB(),
    }

    results = {}
    for name, clf in classifiers.items():
        print(f'training {name} ...')
        model, metrics, y_pred, y_score = evaluate_model(
            clf, name, X_train, y_train, X_test, y_test
        )
        results[name] = {
            'model': model, 'metrics': metrics,
            'y_pred': y_pred, 'y_score': y_score,
        }
        print(
            f'  acc={metrics["accuracy"]:.4f}  f1={metrics["f1"]:.4f}  '
            f'auc={metrics["auc"]:.4f}  fit_time={metrics["fit_time_s"]:.1f}s'
        )

    # Persist every model
    safe_names = {n: n.replace(' ', '_').replace('(', '').replace(')', '') for n in classifiers.keys()}
    for name, r in results.items():
        joblib.dump(r['model'], MODELS_DIR / f'model_{safe_names[name]}.joblib')
    print(f'saved {len(results)} models to {MODELS_DIR}')

    metrics_df = pd.DataFrame([results[n]['metrics'] for n in classifiers.keys()]).set_index('model')
    print(metrics_df.round(4).to_string())
    metrics_df.to_csv(OUTPUTS_DIR / 'classical_metrics_default.csv')

    # ROC + PR curves side by side
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r['y_score'])
        ax.plot(fpr, tpr, label=f'{name} (AUC = {r["metrics"]["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')
    ax.set_title('ROC curves (test set)')
    ax.legend(loc='lower right', fontsize=8)

    ax = axes[1]
    for name, r in results.items():
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, r['y_score'])
        ap = average_precision_score(y_test, r['y_score'])
        ax.plot(rec_arr, prec_arr, label=f'{name} (AP = {ap:.3f})')
    pos_rate = float(y_test.mean())
    ax.axhline(pos_rate, ls='--', color='gray', linewidth=1, label=f'chance (AP = {pos_rate:.3f})')
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.set_title('precision-recall curves (test set)')
    ax.legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'classical_roc_pr.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # Confusion matrix and classification report for the best model by F1
    best_name = max(results.keys(), key=lambda n: results[n]['metrics']['f1'])
    best_model = results[best_name]['model']
    print(f'best model by test F1: {best_name}')

    y_train_pred = best_model.predict(X_train)
    y_test_pred = results[best_name]['y_pred']

    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['uninfected', 'parasitized'],
                yticklabels=['uninfected', 'parasitized'])
    axes[0].set_title(f'{best_name} confusion matrix (train)')
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['uninfected', 'parasitized'],
                yticklabels=['uninfected', 'parasitized'])
    axes[1].set_title(f'{best_name} confusion matrix (test)')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'classical_best_confusion.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(classification_report(y_test, y_test_pred, target_names=['uninfected', 'parasitized']))

    # Persist best model name for downstream scripts
    np.savez(OUTPUTS_DIR / 'best_classical_name.npz', name=np.array([best_name]))


if __name__ == '__main__':
    main()
