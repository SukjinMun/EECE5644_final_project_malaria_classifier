# Feature category ablation, per-feature importance, logistic regression coefficients.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from src.config import (
    FEATURES_DIR, MODELS_DIR, OUTPUTS_DIR,
    COLOR_FEATURE_NAMES, TEXTURE_FEATURE_NAMES, ALL_FEATURE_NAMES,
)

def main():
    data = np.load(FEATURES_DIR / 'features_scaled.npz')
    X_train_raw = data['X_train_raw']
    X_test_raw = data['X_test_raw']
    y_train = data['y_train']
    y_test = data['y_test']

    # ----- Feature category comparison: color vs texture vs combined -----
    best_blob = np.load(OUTPUTS_DIR / 'best_classical_name.npz', allow_pickle=True)
    best_name = str(best_blob['name'][0])
    safe_name = best_name.replace(' ', '_')
    tuned_pipeline = joblib.load(MODELS_DIR / f'model_{safe_name}_tuned.joblib')
    tuned_clf = tuned_pipeline.named_steps['clf']

    color_idx = list(range(len(COLOR_FEATURE_NAMES)))
    texture_idx = list(range(
        len(COLOR_FEATURE_NAMES),
        len(COLOR_FEATURE_NAMES) + len(TEXTURE_FEATURE_NAMES),
    ))
    subsets = {
        'color only': color_idx,
        'texture only': texture_idx,
        'color+texture': color_idx + texture_idx,
    }

    def fresh_pipeline():
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clone(tuned_clf)),
        ])

    rows = []
    for sub_name, idx in subsets.items():
        Xtr_sub = X_train_raw[:, idx]
        Xte_sub = X_test_raw[:, idx]
        m = fresh_pipeline()
        m.fit(Xtr_sub, y_train)
        yp = m.predict(Xte_sub)
        if hasattr(m, 'predict_proba'):
            ys = m.predict_proba(Xte_sub)[:, 1]
        elif hasattr(m, 'decision_function'):
            ys = m.decision_function(Xte_sub)
        else:
            ys = yp.astype(float)
        rows.append({
            'feature subset': sub_name,
            'n features': len(idx),
            'accuracy': accuracy_score(y_test, yp),
            'precision': precision_score(y_test, yp),
            'recall': recall_score(y_test, yp),
            'f1': f1_score(y_test, yp),
            'auc': roc_auc_score(y_test, ys),
        })
    subset_df = pd.DataFrame(rows).set_index('feature subset')
    print(subset_df.round(4).to_string())
    subset_df.to_csv(OUTPUTS_DIR / 'feature_subset_comparison.csv')

    fig, ax = plt.subplots(figsize=(7, 4))
    subset_df['f1'].plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('test F1 score')
    ax.set_title(f'feature subset comparison ({best_name}, tuned)')
    ax.set_ylim([min(subset_df['f1'].min() - 0.05, 0.5), 1.0])
    for i, v in enumerate(subset_df['f1']):
        ax.text(i, v + 0.005, f'{v:.4f}', ha='center')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'feature_subset_f1.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ----- Per-feature importance from the tuned tree-based model -----
    if hasattr(tuned_clf, 'feature_importances_'):
        importances = np.asarray(tuned_clf.feature_importances_, dtype=float)
        categories = (
            ['color'] * len(COLOR_FEATURE_NAMES)
            + ['texture'] * len(TEXTURE_FEATURE_NAMES)
        )
        fi_df = pd.DataFrame({
            'feature': ALL_FEATURE_NAMES,
            'importance': importances,
            'category': categories,
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        print('top 10 features by importance:')
        print(fi_df.head(10).to_string(index=False))

        top_n = 15
        top = fi_df.head(top_n)
        colors_bar = ['#1f77b4' if c == 'color' else '#ff7f0e' for c in top['category']]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(range(top_n), top['importance'], color=colors_bar)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('feature importance')
        ax.set_title(f'top {top_n} features ({best_name} tuned)')
        ax.legend(handles=[
            mpatches.Patch(color='#1f77b4', label='color feature'),
            mpatches.Patch(color='#ff7f0e', label='texture feature'),
        ], loc='lower right')
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / 'feature_importance.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        fi_df.to_csv(OUTPUTS_DIR / 'feature_importance.csv', index=False)

    # ----- Logistic regression coefficient interpretation -----
    safe_logreg = 'LogReg'
    logreg = joblib.load(MODELS_DIR / f'model_{safe_logreg}.joblib')
    if hasattr(logreg, 'coef_'):
        coefs = logreg.coef_[0]
        categories = (
            ['color'] * len(COLOR_FEATURE_NAMES)
            + ['texture'] * len(TEXTURE_FEATURE_NAMES)
        )
        coef_df = pd.DataFrame({
            'feature': ALL_FEATURE_NAMES,
            'coefficient': coefs,
            'abs_coef': np.abs(coefs),
            'category': categories,
        }).sort_values('abs_coef', ascending=False).reset_index(drop=True)
        print('top 10 logistic regression coefficients:')
        print(coef_df.head(10)[['feature', 'coefficient', 'category']].to_string(index=False))

        top_n = 15
        top = coef_df.head(top_n)
        colors_bar = ['#1f77b4' if c == 'color' else '#ff7f0e' for c in top['category']]
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(range(top_n), top['coefficient'], color=colors_bar)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top['feature'])
        ax.invert_yaxis()
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlabel('coefficient (positive values push toward parasitized)')
        ax.set_title('top 15 logistic regression coefficients by |coef|')
        ax.legend(handles=[
            mpatches.Patch(color='#1f77b4', label='color feature'),
            mpatches.Patch(color='#ff7f0e', label='texture feature'),
        ], loc='lower right')
        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / 'logreg_coefficients.png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        coef_df.to_csv(OUTPUTS_DIR / 'logreg_coefficients.csv', index=False)

if __name__ == '__main__':
    main()
