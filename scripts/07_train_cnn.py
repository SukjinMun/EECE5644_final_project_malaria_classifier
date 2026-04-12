"""Train the small CNN deep-learning baseline on raw cell images."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

from src.config import (
    FEATURES_DIR, MODELS_DIR, OUTPUTS_DIR,
    CNN_EPOCHS, CNN_LR, CNN_BATCH_SIZE,
)
from src.cnn_model import CellImageDataset, SmallCNN


def main():
    split = np.load(FEATURES_DIR / 'split.npz', allow_pickle=True)
    paths_train = split['paths_train']
    paths_test = split['paths_test']
    y_train = split['y_train']
    y_test = split['y_test']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'torch device: {device}')
    print(f'torch version: {torch.__version__}')

    train_ds = CellImageDataset(paths_train, y_train)
    test_ds = CellImageDataset(paths_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=CNN_BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)
    print(f'train batches: {len(train_loader)}, test batches: {len(test_loader)}')

    model = SmallCNN().to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'total parameters: {n_params:,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LR)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_accs = []
    for epoch in range(1, CNN_EPOCHS + 1):
        model.train()
        running = 0.0
        n_seen = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = running / n_seen
        train_losses.append(train_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        test_acc = correct / total
        test_accs.append(test_acc)
        print(f'epoch {epoch}/{CNN_EPOCHS}  train loss {train_loss:.4f}  '
              f'test acc {test_acc:.4f}  time {time.time() - t0:.1f}s')

    torch.save(model.state_dict(), MODELS_DIR / 'cnn_state_dict.pt')
    print(f'saved CNN state_dict to {MODELS_DIR / "cnn_state_dict.pt"}')

    # Final evaluation with full metric set
    model.eval()
    all_pred = []
    all_score = []
    all_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_pred.append(preds)
            all_score.append(probs)
            all_true.append(yb.numpy())
    all_pred = np.concatenate(all_pred)
    all_score = np.concatenate(all_score)
    all_true = np.concatenate(all_true)

    cnn_metrics = {
        'model': 'CNN (DL baseline)',
        'accuracy': accuracy_score(all_true, all_pred),
        'precision': precision_score(all_true, all_pred),
        'recall': recall_score(all_true, all_pred),
        'f1': f1_score(all_true, all_pred),
        'auc': roc_auc_score(all_true, all_score),
    }
    pd.DataFrame([cnn_metrics]).to_csv(OUTPUTS_DIR / 'cnn_metrics.csv', index=False)
    print(cnn_metrics)

    # Training curve
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ep = list(range(1, CNN_EPOCHS + 1))
    ax1.plot(ep, train_losses, marker='o', color='#1f77b4', label='train loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('train loss', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax2 = ax1.twinx()
    ax2.plot(ep, test_accs, marker='s', color='#d62728', label='test accuracy')
    ax2.set_ylabel('test accuracy', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    plt.title('CNN training curve')
    fig.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'cnn_training_curve.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
