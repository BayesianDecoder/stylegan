"""
train_specialist.py
-------------------
Trains severity specialist models — best backbone per degradation type:

  upsample   → MobileNetV3-Small   (already >90%, keep it)
  denoise    → EfficientNet-B0 + Laplacian noise branch
  deartifact → EfficientNet-B0     (better texture/frequency features)
  inpaint    → EfficientNet-B2     (larger receptive field for mask detection)

Key training improvements over v1:
  - Mixup augmentation (alpha=0.3) — prevents memorisation
  - Partial backbone unfreeze (last 3 blocks only) — avoids destroying early features
  - Per-type LR / patience / dropout configs
  - Gradient clipping (max_norm=1.0)
  - More epochs + larger patience

Usage:
    python train_specialist.py --data_dir data_main/degraded --train_all
    python train_specialist.py --data_dir data_main/degraded --type denoise
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_specialist_dataloaders, TYPES, LEVELS
from model   import build_specialist


# ─────────────────────────────────────────────────────────────────────────────
# Per-type hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

TYPE_CFG = {
    # upsample already works — light training
    "upsample": dict(
        lr_head=5e-4, lr_finetune=3e-5, phase_b=5,
        epochs=30,  patience=7,  mixup_alpha=0.2, skip_phase_b=False,
    ),
    # denoise: NO mixup — mixing images corrupts noise statistics→label mapping.
    # std(lam*A+(1-lam)*B) ≠ lam*std(A)+(1-lam)*std(B), so model learns wrong
    # mapping during training and predicts randomly on val (pure images).
    "denoise": dict(
        lr_head=3e-4, lr_finetune=0, phase_b=999,
        epochs=80,  patience=20, mixup_alpha=0.0, skip_phase_b=True,
    ),
    # deartifact: statistics MLP (same reasoning as denoise — frozen backbone
    # suppresses JPEG artifact signals). No Phase B needed, no backbone.
    "deartifact": dict(
        lr_head=3e-4, lr_finetune=0, phase_b=999,
        epochs=80,  patience=20, mixup_alpha=0.0, skip_phase_b=True,
    ),
    # inpaint: EfficientNet-B2 full backbone unfreeze — mask spatial extent
    # requires spatial features, statistics compress away spatial info.
    # Full unfreeze at Phase B with low LR forces mask-size feature learning.
    "inpaint": dict(
        lr_head=3e-4, lr_finetune=1e-5, phase_b=5,
        epochs=60,  patience=12, mixup_alpha=0.0, skip_phase_b=False,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Mixup helpers
# ─────────────────────────────────────────────────────────────────────────────

def mixup_batch(images, labels, alpha):
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam     = float(np.random.beta(alpha, alpha))
    idx     = torch.randperm(images.size(0), device=images.device)
    mixed   = lam * images + (1.0 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def mixup_loss(criterion, logits, labels_a, labels_b, lam):
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, loss_fn, mixup_alpha):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        images, la, lb, lam = mixup_batch(images, labels, mixup_alpha)

        logits = model(images)
        loss   = mixup_loss(loss_fn, logits, la, lb, lam)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        # Accuracy uses original labels (la == labels before mixup)
        correct += (logits.argmax(1) == la).sum().item()
        total   += images.size(0)

    return total_loss / len(loader), correct / total * 100


@torch.no_grad()
def evaluate_epoch(model, loader, device, loss_fn):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / len(loader), correct / total * 100


# ─────────────────────────────────────────────────────────────────────────────
# Train one specialist
# ─────────────────────────────────────────────────────────────────────────────

def train_one_specialist(deg_type, data_dir, save_dir,
                          epochs=None, batch_size=32,
                          patience=None, device=None):

    cfg       = TYPE_CFG[deg_type]
    epochs    = epochs   or cfg["epochs"]
    patience  = patience or cfg["patience"]
    save_path = os.path.join(save_dir, f"{deg_type}_specialist.pth")

    print(f"\n{'='*65}")
    print(f"  Training [{deg_type}] specialist")
    print(f"  backbone  : {_backbone_name(deg_type)}")
    print(f"  epochs    : {epochs}  patience: {patience}")
    print(f"  lr_head   : {cfg['lr_head']}  lr_finetune: {cfg['lr_finetune']}")
    phase_b_label = "disabled" if cfg["skip_phase_b"] else f"epoch {cfg['phase_b']+1}"
    print(f"  phase_b   : {phase_b_label}  mixup_alpha: {cfg['mixup_alpha']}")
    print(f"{'='*65}")

    type_idx = TYPES.index(deg_type)
    train_loader, val_loader, _ = get_specialist_dataloaders(
        data_dir, type_idx, batch_size=batch_size
    )

    model   = build_specialist(deg_type, freeze_backbone=True).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Phase A — head only
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr_head"], weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(cfg["phase_b"])), eta_min=1e-5
    )

    best_val_loss    = float('inf')
    patience_counter = 0
    history          = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])

    for epoch in range(1, epochs + 1):

        # ── Switch to Phase B (skipped for denoise) ────────────────────────
        if not cfg["skip_phase_b"] and epoch == cfg["phase_b"] + 1:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg["lr_finetune"], weight_decay=1e-3
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, epochs - int(cfg["phase_b"])), eta_min=1e-7
            )
            print(f"  Phase B — partial fine-tune at lr={cfg['lr_finetune']}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn, cfg["mixup_alpha"]
        )
        val_loss, val_acc = evaluate_epoch(model, val_loader, device, loss_fn)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"  Ep {epoch:03d}/{epochs}"
              f" | Loss {train_loss:.4f}/{val_loss:.4f}"
              f" | Acc  {train_acc:.1f}%/{val_acc:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"    ✓ saved  val_acc={val_acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    # Training curve
    plot_path = os.path.join(save_dir, f"{deg_type}_training_curve.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["train_loss"], label="Train"); ax1.plot(history["val_loss"], label="Val")
    ax1.set_title(f"{deg_type} — Loss"); ax1.legend()
    ax2.plot(history["train_acc"],  label="Train"); ax2.plot(history["val_acc"],  label="Val")
    ax2.set_title(f"{deg_type} — Accuracy (%)"); ax2.legend()
    plt.tight_layout(); plt.savefig(plot_path, dpi=120); plt.close()

    print(f"\n  [{deg_type}] best val loss: {best_val_loss:.4f}")
    print(f"  [{deg_type}] saved → {save_path}")
    print(f"  [{deg_type}] curve → {plot_path}")
    return best_val_loss


def _backbone_name(deg_type):
    names = {
        "upsample":   "MobileNetV3-Small",
        "denoise":    "Statistics MLP (no backbone)",
        "deartifact": "Statistics MLP (8x8 block features)",
        "inpaint":    "EfficientNet-B2 (full unfreeze at Phase B)",
    }
    return names[deg_type]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Mac MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    os.makedirs(args.save_dir, exist_ok=True)

    types_to_train = TYPES if args.train_all else [args.type]

    if args.type not in TYPES and not args.train_all:
        print(f"Error: --type must be one of {TYPES}")
        return

    results = {}
    for deg_type in types_to_train:
        results[deg_type] = train_one_specialist(
            deg_type   = deg_type,
            data_dir   = args.data_dir,
            save_dir   = args.save_dir,
            batch_size = args.batch_size,
            device     = device,
        )

    if args.train_all:
        print("\n" + "=" * 65)
        print("ALL SPECIALISTS TRAINED")
        print("=" * 65)
        for t, loss in results.items():
            print(f"  {t:12s} ({_backbone_name(t):<38}) best val loss = {loss:.4f}")
        print(f"\nAll models → {args.save_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   required=True)
    parser.add_argument('--save_dir',   default='specialists')
    parser.add_argument('--train_all',  action='store_true')
    parser.add_argument('--type',       default='denoise', choices=TYPES)
    parser.add_argument('--batch_size', type=int, default=32)
    main(parser.parse_args())
