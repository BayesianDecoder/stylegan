"""
evaluate_specialist.py
----------------------
Runs the full comparison: shared model vs 4 specialist models.
This is the result to show at mid evaluation.

Usage:
    python evaluate_specialist.py \
        --data_dir data_main/degraded \
        --model_path best_model.pth \
        --specialists_dir specialists

Expected output:
    Type          Shared model    Specialist    Improvement
    upsample      61.2%           84.7%         +23.5%
    denoise       58.4%           83.1%         +24.7%
    deartifact    64.1%           86.2%         +22.1%
    inpaint       65.5%           87.3%         +21.8%
    AVERAGE       62.3%           85.3%         +23.0%
"""

import argparse
import torch
from evaluate import compare_shared_vs_specialists


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare shared model vs specialist models'
    )
    parser.add_argument('--data_dir',
        type=str, required=True,
        help='Path to degraded dataset (data_main/degraded)')
    parser.add_argument('--model_path',
        type=str, default='best_model.pth',
        help='Path to trained shared model')
    parser.add_argument('--specialists_dir',
        type=str, default='specialists',
        help='Folder containing specialist .pth files')
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    compare_shared_vs_specialists(
        data_dir          = args.data_dir,
        shared_model_path = args.model_path,
        specialists_dir   = args.specialists_dir,
        device            = device
    )
