"""
train.py
--------
Full training loop for the Degradation Estimator.

Two-phase training strategy to prevent overfitting:
    Phase A (epochs 1-10)  : backbone frozen, only heads train
    Phase B (epochs 11-30) : full fine-tuning at lower lr

Usage:
    python train.py --data_dir data/degraded --epochs 30 --batch_size 32
"""

import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_dataloaders
from model import DegradationEstimator


def train_one_epoch(model, loader, optimizer, device,
                    loss_type_fn, loss_severity_fn):
    """Single training epoch. Returns avg loss, type acc, severity acc."""
    model.train()
    total_loss = 0.0
    type_correct = 0
    severity_correct = 0
    total = 0

    for images, type_labels, severity_labels in tqdm(loader, leave=False):
        images         = images.to(device)
        type_labels    = type_labels.to(device)       # (batch, 4) multi-hot
        severity_labels = severity_labels.to(device)  # (batch,)  integer

        # Forward pass
        type_logits, severity_logits = model(images)

        # Compute losses
        loss_t = loss_type_fn(type_logits, type_labels)
        loss_s = loss_severity_fn(severity_logits, severity_labels)
        loss   = loss_t + loss_s

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy metrics
        type_preds = (torch.sigmoid(type_logits) > 0.5).float()
        type_correct += (type_preds == type_labels).all(dim=1).sum().item()

        sev_preds = severity_logits.argmax(dim=1)
        severity_correct += (sev_preds == severity_labels).sum().item()

        total += images.size(0)

    return (
        total_loss / len(loader),
        type_correct / total * 100,
        severity_correct / total * 100
    )


@torch.no_grad()
def evaluate(model, loader, device, loss_type_fn, loss_severity_fn):
    """Evaluation on val or test set. Returns avg loss, type acc, severity acc."""
    model.eval()
    total_loss = 0.0
    type_correct = 0
    severity_correct = 0
    total = 0

    for images, type_labels, severity_labels in loader:
        images          = images.to(device)
        type_labels     = type_labels.to(device)
        severity_labels = severity_labels.to(device)

        type_logits, severity_logits = model(images)

        loss_t = loss_type_fn(type_logits, type_labels)
        loss_s = loss_severity_fn(severity_logits, severity_labels)
        total_loss += (loss_t + loss_s).item()

        type_preds = (torch.sigmoid(type_logits) > 0.5).float()
        type_correct += (type_preds == type_labels).all(dim=1).sum().item()

        sev_preds = severity_logits.argmax(dim=1)
        severity_correct += (sev_preds == severity_labels).sum().item()

        total += images.size(0)

    return (
        total_loss / len(loader),
        type_correct / total * 100,
        severity_correct / total * 100
    )


def plot_curves(history: dict, save_path: str):
    """Saves training curves to a PNG file."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'],   label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    axes[1].plot(history['train_type_acc'], label='Train')
    axes[1].plot(history['val_type_acc'],   label='Val')
    axes[1].set_title('Type accuracy (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()

    axes[2].plot(history['train_sev_acc'], label='Train')
    axes[2].plot(history['val_sev_acc'],   label='Val')
    axes[2].set_title('Severity accuracy (%)')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Training curves saved to {save_path}")


def main(args):
    # ── Device ───────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Mac M3 GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU — training will be slow")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # Phase A: backbone frozen → only heads train
    model = DegradationEstimator(
        dropout_rate=0.4,
        freeze_backbone=True   # backbone frozen for first phase
    ).to(device)

    # ── Loss functions ────────────────────────────────────────────────────────
    # BCEWithLogitsLoss for type  (multi-label, sigmoid inside)
    # CrossEntropyLoss  for severity (single-label, softmax inside)
    loss_type_fn     = nn.BCEWithLogitsLoss()
    loss_severity_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Phase A optimizer: only head parameters ───────────────────────────────
    head_params = list(model.type_head.parameters()) + \
                  list(model.severity_head.parameters())
    optimizer = torch.optim.Adam(head_params, lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # ── Training history ──────────────────────────────────────────────────────
    history = {
        'train_loss': [], 'val_loss': [],
        'train_type_acc': [], 'val_type_acc': [],
        'train_sev_acc': [],  'val_sev_acc': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    PHASE_B_START = 10   # unfreeze backbone after epoch 10

    print(f"\nStarting training for {args.epochs} epochs")
    print(f"Phase A: epochs 1-{PHASE_B_START} (heads only)")
    print(f"Phase B: epochs {PHASE_B_START+1}-{args.epochs} (full fine-tune)")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):

        # ── Switch to Phase B ──────────────────────────────────────────────
        if epoch == PHASE_B_START + 1:
            model.unfreeze_backbone()
            # Lower learning rate for full model fine-tuning
            # Too high = backbone forgets ImageNet knowledge
            optimizer = torch.optim.Adam(
                model.parameters(), lr=1e-5, weight_decay=1e-5
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=5, gamma=0.5
            )
            print(f"\nPhase B started — full fine-tuning at lr=1e-5")

        # ── Train ──────────────────────────────────────────────────────────
        train_loss, train_tacc, train_sacc = train_one_epoch(
            model, train_loader, optimizer, device,
            loss_type_fn, loss_severity_fn
        )

        # ── Validate ────────────────────────────────────────────────────────
        val_loss, val_tacc, val_sacc = evaluate(
            model, val_loader, device,
            loss_type_fn, loss_severity_fn
        )

        scheduler.step()

        # ── Log ─────────────────────────────────────────────────────────────
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_type_acc'].append(train_tacc)
        history['val_type_acc'].append(val_tacc)
        history['train_sev_acc'].append(train_sacc)
        history['val_sev_acc'].append(val_sacc)

        gap = val_loss - train_loss
        flag = "  ** OVERFIT WARNING **" if gap > 0.3 else ""

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} (gap:{gap:+.3f}){flag} | "
            f"Type: {train_tacc:.1f}%/{val_tacc:.1f}% | "
            f"Severity: {train_sacc:.1f}%/{val_sacc:.1f}%"
        )

        # ── Save best model ──────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), args.save_path)
            print(f"  → Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ── Final plots ───────────────────────────────────────────────────────────
    plot_curves(history, 'training_curves.png')
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Degradation Estimator')
    parser.add_argument('--data_dir',    type=str,   required=True,           help='Path to degraded dataset')
    parser.add_argument('--epochs',      type=int,   default=30,              help='Total training epochs')
    parser.add_argument('--batch_size',  type=int,   default=32,              help='Batch size')
    parser.add_argument('--num_workers', type=int,   default=4,               help='DataLoader workers')
    parser.add_argument('--save_path',   type=str,   default='best_model.pth',help='Where to save best model')
    parser.add_argument('--patience',    type=int,   default=7,               help='Early stopping patience')
    args = parser.parse_args()

    main(args)
