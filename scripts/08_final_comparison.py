# Assemble the final comparison table and the literature comparison vs Molina et al. (2020).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import matplotlib.pyplot as plt

from src.config import OUTPUTS_DIR, ALL_FEATURE_NAMES

MOLINA_ACCURACY = 0.9770  # Molina et al. (2020) cell-level accuracy

def main():
    # Load all the per-source metric tables
    classical_df = pd.read_csv(OUTPUTS_DIR / 'classical_metrics_default.csv')
    tuning_df = pd.read_csv(OUTPUTS_DIR / 'tuning_comparison.csv')
    cnn_df = pd.read_csv(OUTPUTS_DIR / 'cnn_metrics.csv')

    tuned_row = tuning_df[tuning_df['model'].str.contains('tuned')].iloc[0].to_dict()
    cnn_row = cnn_df.iloc[0].to_dict()

    final_rows = []
    cols = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'fit_time_s']
    for _, r in classical_df.iterrows():
        final_rows.append({c: r[c] for c in ['model'] + cols})
    final_rows.append({c: tuned_row[c] for c in ['model'] + cols})
    final_rows.append({c: cnn_row[c] for c in ['model'] + cols})

    final_df = pd.DataFrame(final_rows).set_index('model')[cols]
    print(final_df.round(4).to_string())
    final_df.to_csv(OUTPUTS_DIR / 'final_comparison.csv')

    # F1 bar chart
    order = final_df['f1'].sort_values(ascending=False)
    colors = [
        '#2ca02c' if 'CNN' in n else ('#ff7f0e' if 'tuned' in n else '#1f77b4')
        for n in order.index
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    order.plot(kind='bar', ax=ax, color=colors)
    ax.set_ylabel('test F1 score')
    ax.set_title('test F1 across all models')
    ax.set_ylim([min(order.min() - 0.05, 0.5), 1.0])
    for i, v in enumerate(order):
        ax.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'final_f1_bars.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # Literature comparison vs Molina et al. (2020)
    literature_rows = [
        {
            'source': 'Molina et al. (2020), cell-level',
            'accuracy': MOLINA_ACCURACY,
            'f1': float('nan'),
            'n_features': 2852,
            'dataset': '15,660 erythrocytes (6 classes)',
        },
        {
            'source': f'Ours: {tuned_row["model"]}',
            'accuracy': float(tuned_row['accuracy']),
            'f1': float(tuned_row['f1']),
            'n_features': len(ALL_FEATURE_NAMES),
            'dataset': 'NIH (27,558 cells, binary)',
        },
        {
            'source': 'Ours: small CNN',
            'accuracy': float(cnn_row['accuracy']),
            'f1': float(cnn_row['f1']),
            'n_features': 'end-to-end (128x128 RGB pixels)',
            'dataset': 'NIH (27,558 cells, binary)',
        },
    ]
    lit_df = pd.DataFrame(literature_rows).set_index('source')
    lit_df['accuracy'] = lit_df['accuracy'].map(
        lambda v: f'{v:.4f}' if pd.notna(v) else 'not reported'
    )
    lit_df['f1'] = lit_df['f1'].map(
        lambda v: f'{v:.4f}' if pd.notna(v) else 'not reported'
    )
    print(lit_df.to_string())
    lit_df.to_csv(OUTPUTS_DIR / 'literature_comparison.csv')

    gap_classical = MOLINA_ACCURACY - float(tuned_row['accuracy'])
    gap_cnn = MOLINA_ACCURACY - float(cnn_row['accuracy'])
    print(f'gap vs Molina (tuned classical): {gap_classical * 100:+.2f} percentage points')
    print(f'gap vs Molina (small CNN):       {gap_cnn * 100:+.2f} percentage points')

if __name__ == '__main__':
    main()
