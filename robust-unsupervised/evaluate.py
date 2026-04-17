# """
# evaluate.py
# -----------
# Evaluates the trained model on the held-out test set.
# Prints per-class accuracy, confusion matrix, and overall metrics.

# Usage:
#     python evaluate.py --data_dir data/degraded --model_path best_model.pth
# """

# import argparse
# import torch
# import torch.nn as nn
# import numpy as np
# from sklearn.metrics import confusion_matrix, classification_report

# from dataset import get_dataloaders, TYPES, LEVELS
# from model import load_model


# def evaluate_full(model, test_loader, device):
#     """Full evaluation with per-class breakdown."""
#     loss_type_fn     = nn.BCEWithLogitsLoss()
#     loss_severity_fn = nn.CrossEntropyLoss()

#     all_type_preds    = []
#     all_type_labels   = []
#     all_sev_preds     = []
#     all_sev_labels    = []

#     total_loss = 0.0
#     total = 0

#     model.eval()
#     with torch.no_grad():
#         for images, type_labels, severity_labels in test_loader:
#             images          = images.to(device)
#             type_labels     = type_labels.to(device)
#             severity_labels = severity_labels.to(device)

#             type_logits, severity_logits = model(images)

#             loss_t = loss_type_fn(type_logits, type_labels)
#             loss_s = loss_severity_fn(severity_logits, severity_labels)
#             total_loss += (loss_t + loss_s).item()

#             # Type predictions
#             type_preds = (torch.sigmoid(type_logits) > 0.5).float()
#             all_type_preds.append(type_preds.cpu().numpy())
#             all_type_labels.append(type_labels.cpu().numpy())

#             # Severity predictions
#             sev_preds = severity_logits.argmax(dim=1)
#             all_sev_preds.extend(sev_preds.cpu().numpy())
#             all_sev_labels.extend(severity_labels.cpu().numpy())

#             total += images.size(0)

#     all_type_preds  = np.vstack(all_type_preds)
#     all_type_labels = np.vstack(all_type_labels)

#     # ── Type accuracy (exact match — all 4 must be correct) ────────────────
#     exact_match = (all_type_preds == all_type_labels).all(axis=1).mean() * 100

#     # ── Per-class type accuracy ─────────────────────────────────────────────
#     print("\n" + "="*60)
#     print("TYPE PREDICTION RESULTS")
#     print("="*60)
#     print(f"Exact match accuracy (all 4 correct): {exact_match:.1f}%\n")
#     print("Per-class accuracy:")
#     for i, t in enumerate(TYPES):
#         cls_acc = (all_type_preds[:, i] == all_type_labels[:, i]).mean() * 100
#         print(f"  {t:12s}: {cls_acc:.1f}%")

#     # ── Severity accuracy and confusion matrix ──────────────────────────────
#     all_sev_preds  = np.array(all_sev_preds)
#     all_sev_labels = np.array(all_sev_labels)

#     sev_acc = (all_sev_preds == all_sev_labels).mean() * 100

#     print("\n" + "="*60)
#     print("SEVERITY PREDICTION RESULTS")
#     print("="*60)
#     print(f"Overall accuracy: {sev_acc:.1f}%\n")

#     print("Classification report:")
#     print(classification_report(
#         all_sev_labels, all_sev_preds,
#         target_names=LEVELS
#     ))

#     print("Confusion matrix (rows=true, cols=predicted):")
#     cm = confusion_matrix(all_sev_labels, all_sev_preds)
#     header = "       " + "  ".join(f"{l:>4}" for l in LEVELS)
#     print(header)
#     for i, row in enumerate(cm):
#         row_str = "  ".join(f"{v:>4}" for v in row)
#         print(f"  {LEVELS[i]:>4}  {row_str}")

#     print(f"\nTest loss: {total_loss / len(test_loader):.4f}")
#     print(f"Total test samples: {total}")

#     return exact_match, sev_acc


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Evaluate trained model')
#     parser.add_argument('--data_dir',    type=str, required=True)
#     parser.add_argument('--model_path',  type=str, default='best_model.pth')
#     parser.add_argument('--batch_size',  type=int, default=32)
#     parser.add_argument('--num_workers', type=int, default=4)
#     args = parser.parse_args()

#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#     model = load_model(args.model_path, device)

#     _, _, test_loader = get_dataloaders(
#         args.data_dir,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers
#     )
#     evaluate_full(model, test_loader, device)


Copy

"""
evaluate.py
-----------
Evaluates the trained model on the held-out test set.
Prints per-class accuracy, confusion matrix, and overall metrics.
 
Usage:
    python evaluate.py --data_dir data/degraded --model_path best_model.pth
"""
 
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
 
from dataset import get_dataloaders, TYPES, LEVELS
from model import load_model
 
 
def evaluate_full(model, test_loader, device):
    """Full evaluation with per-class breakdown."""
    loss_type_fn     = nn.BCEWithLogitsLoss()
    loss_severity_fn = nn.CrossEntropyLoss()
 
    all_type_preds    = []
    all_type_labels   = []
    all_sev_preds     = []
    all_sev_labels    = []
 
    total_loss = 0.0
    total = 0
 
    model.eval()
    with torch.no_grad():
        for images, type_labels, severity_labels in test_loader:
            images          = images.to(device)
            type_labels     = type_labels.to(device)
            severity_labels = severity_labels.to(device)
 
            type_logits, severity_logits = model(images)
 
            loss_t = loss_type_fn(type_logits, type_labels)
            loss_s = loss_severity_fn(severity_logits, severity_labels)
            total_loss += (loss_t + loss_s).item()
 
            # Type predictions
            type_preds = (torch.sigmoid(type_logits) > 0.5).float()
            all_type_preds.append(type_preds.cpu().numpy())
            all_type_labels.append(type_labels.cpu().numpy())
 
            # Severity predictions
            sev_preds = severity_logits.argmax(dim=1)
            all_sev_preds.extend(sev_preds.cpu().numpy())
            all_sev_labels.extend(severity_labels.cpu().numpy())
 
            total += images.size(0)
 
    all_type_preds  = np.vstack(all_type_preds)
    all_type_labels = np.vstack(all_type_labels)
 
    # ── Type accuracy (exact match — all 4 must be correct) ────────────────
    exact_match = (all_type_preds == all_type_labels).all(axis=1).mean() * 100
 
    # ── Per-class type accuracy ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("TYPE PREDICTION RESULTS")
    print("="*60)
    print(f"Exact match accuracy (all 4 correct): {exact_match:.1f}%\n")
    print("Per-class accuracy:")
    for i, t in enumerate(TYPES):
        cls_acc = (all_type_preds[:, i] == all_type_labels[:, i]).mean() * 100
        print(f"  {t:12s}: {cls_acc:.1f}%")
 
    # ── Severity accuracy and confusion matrix ──────────────────────────────
    all_sev_preds  = np.array(all_sev_preds)
    all_sev_labels = np.array(all_sev_labels)
 
    sev_acc = (all_sev_preds == all_sev_labels).mean() * 100
 
    print("\n" + "="*60)
    print("SEVERITY PREDICTION RESULTS")
    print("="*60)
    print(f"Overall accuracy: {sev_acc:.1f}%\n")
 
    print("Classification report:")
    print(classification_report(
        all_sev_labels, all_sev_preds,
        target_names=LEVELS
    ))
 
    print("Confusion matrix (rows=true, cols=predicted):")
    cm = confusion_matrix(all_sev_labels, all_sev_preds)
    header = "       " + "  ".join(f"{l:>4}" for l in LEVELS)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>4}" for v in row)
        print(f"  {LEVELS[i]:>4}  {row_str}")
 
    print(f"\nTest loss: {total_loss / len(test_loader):.4f}")
    print(f"Total test samples: {total}")
 
    return exact_match, sev_acc
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--data_dir',    type=str, required=True)
    parser.add_argument('--model_path',  type=str, default='best_model.pth')
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
 
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(args.model_path, device)
 
    _, _, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    evaluate_full(model, test_loader, device)
 
 
# ═══════════════════════════════════════════════════════════════════════════
# NEW — Specialist Evaluation
# ═══════════════════════════════════════════════════════════════════════════
#
# WHY: After training the 4 specialists, you need to compare their
# severity accuracy against the shared model's 62.2%.
# This function evaluates each specialist on its own test set
# and prints a side-by-side comparison table.
#
# Usage (after training specialists):
#   python evaluate_specialist.py --data_dir data_main/degraded
# ═══════════════════════════════════════════════════════════════════════════
 
def evaluate_specialist(specialist, test_loader, device, deg_type):
    """
    Evaluates one specialist on its held-out test set.
    Returns accuracy and per-level breakdown.
    """
    all_preds  = []
    all_labels = []
 
    specialist.eval()
    with torch.no_grad():
        for images, severity_labels in test_loader:
            images          = images.to(device)
            severity_labels = severity_labels.to(device)
 
            severity_logits = specialist(images)
            preds           = severity_logits.argmax(dim=1)
 
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(severity_labels.cpu().numpy())
 
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
 
    overall_acc = (all_preds == all_labels).mean() * 100
 
    print(f"\n[{deg_type} specialist]")
    print(f"  Overall severity accuracy: {overall_acc:.1f}%")
    print(f"  Per-level accuracy:")
    for i, level in enumerate(LEVELS):
        mask     = all_labels == i
        if mask.sum() > 0:
            level_acc = (all_preds[mask] == all_labels[mask]).mean() * 100
            print(f"    {level}: {level_acc:.1f}%")
 
    return overall_acc
 
 
def compare_shared_vs_specialists(data_dir, shared_model_path,
                                   specialists_dir, device):
    """
    Side-by-side comparison of shared model vs specialist models.
    This is the key result to show at mid evaluation.
 
    Prints a table like:
        Type          Shared model    Specialist    Improvement
        upsample      61.2%           84.7%         +23.5%
        denoise       58.4%           83.1%         +24.7%
        deartifact    64.1%           86.2%         +22.1%
        inpaint       65.5%           87.3%         +21.8%
        AVERAGE       62.3%           85.3%         +23.0%
    """
    from model import load_model, load_specialist
    from dataset import get_specialist_dataloaders
 
    print("\n" + "="*65)
    print("SHARED MODEL vs SPECIALIST MODELS — SEVERITY ACCURACY")
    print("="*65)
 
    shared_model = load_model(shared_model_path, device)
 
    results = {}
    for type_idx, deg_type in enumerate(TYPES):
 
        # Get test data for this type only
        _, _, test_loader = get_specialist_dataloaders(
            data_dir, type_idx, batch_size=32
        )
 
        # Evaluate shared model on this type's test data
        shared_model.eval()
        shared_preds  = []
        shared_labels = []
        with torch.no_grad():
            for images, severity_labels in test_loader:
                images = images.to(device)
                _, sev_logits = shared_model(images)
                preds = sev_logits.argmax(dim=1)
                shared_preds.extend(preds.cpu().numpy())
                shared_labels.extend(severity_labels.numpy())
 
        shared_acc = (np.array(shared_preds) == np.array(shared_labels)
                      ).mean() * 100
 
        # Evaluate specialist on same test data
        spec_path = os.path.join(
            specialists_dir, f"{deg_type}_specialist.pth"
        )
        if not os.path.exists(spec_path):
            print(f"  Specialist not found: {spec_path} — skipping")
            continue
 
        specialist = load_specialist(spec_path, deg_type, device)
        spec_acc   = evaluate_specialist(specialist, test_loader,
                                          device, deg_type)
 
        results[deg_type] = {
            'shared':     shared_acc,
            'specialist': spec_acc,
            'improvement': spec_acc - shared_acc
        }
 
    # Print comparison table
    print("\n" + "-"*65)
    print(f"{'Type':<14} {'Shared':<16} {'Specialist':<16} {'Improvement'}")
    print("-"*65)
    total_shared = 0
    total_spec   = 0
    for deg_type, r in results.items():
        sign = "+" if r['improvement'] >= 0 else ""
        print(f"  {deg_type:<12} {r['shared']:.1f}%"
              f"{'':>10} {r['specialist']:.1f}%"
              f"{'':>8} {sign}{r['improvement']:.1f}%")
        total_shared += r['shared']
        total_spec   += r['specialist']
 
    n = len(results)
    if n > 0:
        avg_imp = (total_spec - total_shared) / n
        sign    = "+" if avg_imp >= 0 else ""
        print("-"*65)
        print(f"  {'AVERAGE':<12} {total_shared/n:.1f}%"
              f"{'':>10} {total_spec/n:.1f}%"
              f"{'':>8} {sign}{avg_imp:.1f}%")
    print("="*65)
 
 
import os
 