"""
train_specialist.py
-------------------
Trains all 4 severity specialist models — one per degradation type.

WHY THIS EXISTS:
    The main model (train.py) uses a shared severity head for all 4 types.
    This reaches ~62% severity accuracy because the head must learn
    subtle differences across all types simultaneously.

    Each specialist is trained on ONE type only — so it focuses entirely
    on learning XS vs S vs M vs L vs XL for that specific degradation.
    Expected improvement: ~62% → ~85%+ severity accuracy.

HOW IT WORKS:
    - Reuses the SAME 50,000 images you already generated
    - Filters to 12,500 images per type using SpecialistDataset
    - Trains 4 separate SeveritySpecialist models
    - Saves each to specialists/ folder

Usage:
    # Train all 4 specialists (recommended — takes ~2 hours total on M3)
    python train_specialist.py --data_dir data_main/degraded --train_all

    # Train just one specialist (e.g. denoise only)
    python train_specialist.py --data_dir data_main/degraded --type denoise

After training, run evaluation:
    python evaluate_specialist.py --data_dir data_main/degraded
"""

import os
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Imports from your existing files — nothing changed in those files
from dataset import get_specialist_dataloaders, TYPES, LEVELS
from model import SeveritySpecialist


def train_one_epoch(model, loader, optimizer, device, loss_fn):
    """One training epoch for a specialist. Returns avg loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, severity_labels in tqdm(loader, leave=False):
        images          = images.to(device)
        severity_labels = severity_labels.to(device)

        # Forward — specialist only returns severity logits
        severity_logits = model(images)

        loss = loss_fn(severity_logits, severity_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy
        preds   = severity_logits.argmax(dim=1)
        correct += (preds == severity_labels).sum().item()
        total   += images.size(0)

    return total_loss / len(loader), correct / total * 100


@torch.no_grad()
def evaluate_epoch(model, loader, device, loss_fn):
    """Evaluation for one epoch. Returns avg loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, severity_labels in loader:
        images          = images.to(device)
        severity_labels = severity_labels.to(device)

        severity_logits = model(images)
        loss            = loss_fn(severity_logits, severity_labels)

        total_loss += loss.item()
        preds       = severity_logits.argmax(dim=1)
        correct    += (preds == severity_labels).sum().item()
        total      += images.size(0)

    return total_loss / len(loader), correct / total * 100


def train_one_specialist(deg_type: str, data_dir: str, save_dir: str,
                          epochs: int = 30, batch_size: int = 32,
                          patience: int = 7, device: torch.device = None):
    """
    Trains one severity specialist for the given degradation type.

    Args:
        deg_type  : one of 'upsample', 'denoise', 'deartifact', 'inpaint'
        data_dir  : path to your degraded dataset (data_main/degraded)
        save_dir  : where to save the trained model (specialists/)
        epochs    : max training epochs
        patience  : early stopping patience
    """
    type_idx  = TYPES.index(deg_type)
    save_path = os.path.join(save_dir, f"{deg_type}_specialist.pth")

    print(f"\n{'='*60}")
    print(f"Training [{deg_type}] severity specialist")
    print(f"Save path: {save_path}")
    print(f"{'='*60}")

    # ── Data ──────────────────────────────────────────────────────────────
    # Uses SpecialistDataset which filters to only this type's images
    train_loader, val_loader, _ = get_specialist_dataloaders(
        data_dir, type_idx, batch_size=batch_size
    )

    # ── Model ─────────────────────────────────────────────────────────────
    # SeveritySpecialist — same backbone, only severity head
    model = SeveritySpecialist(
        deg_type=deg_type,
        dropout_rate=0.5,
        freeze_backbone=True    # freeze backbone for phase A
    ).to(device)

    # Label smoothing helps severity generalize better
    # Prevents overconfident predictions for adjacent levels
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)

    # Phase A — only severity head trains, backbone frozen
    head_params = list(model.severity_head.parameters())
    optimizer   = torch.optim.Adam(head_params, lr=1e-3, weight_decay=1e-5)
    scheduler   = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    best_val_loss    = float('inf')
    patience_counter = 0
    history          = {'train_loss': [], 'val_loss': [],
                        'train_acc': [],  'val_acc': []}

    PHASE_B_START = 10

    for epoch in range(1, epochs + 1):

        # Switch to Phase B — unfreeze backbone at epoch 11
        if epoch == PHASE_B_START + 1:
            model.unfreeze_backbone()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=1e-5, weight_decay=1e-5
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=5, gamma=0.5
            )
            print(f"Phase B started — full fine-tuning")

        # Train and validate
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, device, loss_fn
        )
        scheduler.step()

        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"  Epoch {epoch:02d}/{epochs} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Severity: {train_acc:.1f}%/{val_acc:.1f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model (val_acc={val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Save training curve plot
    plot_path = os.path.join(save_dir, f"{deg_type}_training_curve.png")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'],   label='Val')
    plt.title(f'{deg_type} — Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'],   label='Val')
    plt.title(f'{deg_type} — Severity Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()

    print(f"\n[{deg_type}] Best val loss: {best_val_loss:.4f}")
    print(f"[{deg_type}] Model saved to: {save_path}")
    print(f"[{deg_type}] Training curve: {plot_path}")

    return best_val_loss


def main(args):
    # ── Device ────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Mac M3 GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Create specialists folder
    os.makedirs(args.save_dir, exist_ok=True)

    if args.train_all:
        # Train all 4 specialists one after another
        print("\nTraining all 4 specialists...")
        print("Expected time: ~30 mins each → ~2 hours total on M3\n")
        results = {}
        for deg_type in TYPES:
            best_loss = train_one_specialist(
                deg_type   = deg_type,
                data_dir   = args.data_dir,
                save_dir   = args.save_dir,
                epochs     = args.epochs,
                batch_size = args.batch_size,
                patience   = args.patience,
                device     = device,
            )
            results[deg_type] = best_loss

        print("\n" + "="*60)
        print("ALL SPECIALISTS TRAINED")
        print("="*60)
        for t, loss in results.items():
            print(f"  {t:12s} : best val loss = {loss:.4f}")
        print(f"\nAll models saved to: {args.save_dir}/")

    else:
        # Train one specific specialist
        if args.type not in TYPES:
            print(f"Error: --type must be one of {TYPES}")
            return
        train_one_specialist(
            deg_type   = args.type,
            data_dir   = args.data_dir,
            save_dir   = args.save_dir,
            epochs     = args.epochs,
            batch_size = args.batch_size,
            patience   = args.patience,
            device     = device,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train severity specialist models'
    )
    parser.add_argument('--data_dir',   type=str,  required=True,
                        help='Path to degraded dataset (data_main/degraded)')
    parser.add_argument('--save_dir',   type=str,  default='specialists',
                        help='Where to save specialist models')
    parser.add_argument('--train_all',  action='store_true',
                        help='Train all 4 specialists')
    parser.add_argument('--type',       type=str,  default='denoise',
                        help='Which type to train (if not --train_all)')
    parser.add_argument('--epochs',     type=int,  default=30)
    parser.add_argument('--batch_size', type=int,  default=32)
    parser.add_argument('--patience',   type=int,  default=7)
    args = parser.parse_args()
    main(args)
