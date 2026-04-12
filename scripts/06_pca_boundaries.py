# PCA visualization in 2D + decision boundary plots for every classical model.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from src.config import FEATURES_DIR, OUTPUTS_DIR, RANDOM_STATE

def main():
    data = np.load(FEATURES_DIR / 'features_scaled.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f'explained variance ratio: {pca.explained_variance_ratio_}')

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, color, name in [(0, '#1f77b4', 'uninfected'), (1, '#d62728', 'parasitized')]:
        mask = y_test == label
        ax.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                   s=8, alpha=0.4, color=color, label=name)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)')
    ax.set_title('test set projected onto first two principal components')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'pca_2d.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # Decision boundary grid in 2D PCA space
    boundary_models = {
        'LogReg': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
        'SVM-RBF': SVC(kernel='rbf', random_state=RANDOM_STATE),
        'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss',
                                 n_jobs=-1, random_state=RANDOM_STATE),
        'GaussianNB': GaussianNB(),
    }

    x_min, x_max = X_train_pca[:, 0].min() - 0.5, X_train_pca[:, 0].max() + 0.5
    y_min, y_max = X_train_pca[:, 1].min() - 0.5, X_train_pca[:, 1].max() + 0.5
    grid_res = 200
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    rng = np.random.default_rng(RANDOM_STATE)
    n_show = 300
    idx_0 = np.where(y_test == 0)[0]
    idx_1 = np.where(y_test == 1)[0]
    pick_0 = rng.choice(idx_0, min(n_show, len(idx_0)), replace=False)
    pick_1 = rng.choice(idx_1, min(n_show, len(idx_1)), replace=False)

    cmap_bg = ListedColormap(['#c7dbff', '#ffc7c7'])
    fig, axes = plt.subplots(2, 4, figsize=(17, 9))
    axes_flat = axes.ravel()
    for ax, (name, bm) in zip(axes_flat, boundary_models.items()):
        bm.fit(X_train_pca, y_train)
        Z = bm.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.45, cmap=cmap_bg)
        ax.scatter(X_test_pca[pick_0, 0], X_test_pca[pick_0, 1],
                   s=10, c='#1f77b4', alpha=0.6, edgecolors='none', label='uninfected')
        ax.scatter(X_test_pca[pick_1, 0], X_test_pca[pick_1, 1],
                   s=10, c='#d62728', alpha=0.6, edgecolors='none', label='parasitized')
        train_acc2d = bm.score(X_train_pca, y_train)
        test_acc2d = bm.score(X_test_pca, y_test)
        ax.set_title(f'{name}\n(2D: train {train_acc2d:.3f} / test {test_acc2d:.3f})',
                     fontsize=10)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    for leftover in axes_flat[len(boundary_models):]:
        leftover.set_axis_off()
    axes[0, 0].legend(loc='lower right', fontsize=8, framealpha=0.9)
    plt.suptitle('decision boundaries in 2D PCA space (models trained on 2 PCs only)',
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUTS_DIR / 'decision_boundaries.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()
