"""
auto_restore_e4e.py
-------------------
Variant of auto_restore.py with e4e/pSp encoder warm start for W initialization.

The ONLY change vs auto_restore.py:
    W_variable is initialized from the e4e encoder output (mapped to W space)
    plus adaptive Gaussian noise scaled by the predicted degradation severity,
    instead of the plain w_avg starting point.

Noise scales per severity (larger degradation → more noise for exploration):
    XS: 0.005  S: 0.010  M: 0.020  L: 0.050  XL: 0.100

Everything else is identical to auto_restore.py:
    - Same CNN degradation prediction
    - Same NGD optimizers for all 3 phases
    - Same LRs (0.08 / 0.02 / 0.005)
    - Same loss function (MultiscaleLPIPS)
    - Same output saving

Usage:
    python auto_restore_e4e.py --image path/to/degraded.png \\
        --pkl_path pretrained_networks/ffhq.pkl \\
        --e4e_path /path/to/e4e_ffhq_encode.pt

Arguments:
    --image           Path to the degraded input image (required)
    --pkl_path        StyleGAN2-ADA generator checkpoint (.pkl)
    --e4e_path        e4e/pSp encoder checkpoint (.pt). If not provided or
                      missing, falls back to w_avg (same as auto_restore.py).
    --model_path      Main CNN checkpoint (best_model.pth)
    --specialists_dir Directory containing the 4 specialist .pth files
    --out_dir         Where to save restored images (default: restored_e4e/)
    --resolution      Generator output resolution (default: 1024)
    --steps           Optimization steps per phase (default: 150)
    --lr_scale        Global learning-rate scale factor (default: 1.0)
    --threshold       CNN type-detection probability threshold (default: 0.5)
"""

import argparse
import datetime
import os
import sys

# ── Make stylegan2_ada importable ────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "stylegan2_ada"))

import torch
import torch.nn.functional as F
import tqdm
from PIL import Image as PilImage
from torchvision import transforms
from torchvision.utils import save_image

import benchmark
import benchmark.config
from benchmark.degradations import (
    ComposedDegradation,
    ResizePrediction,
)
from benchmark.tasks import (
    degradation_levels,
    degradation_types,
)
from robust_unsupervised import (
    MultiscaleLPIPS,
    NGD,
    WVariable,
    WpVariable,
    WppVariable,
    open_generator,
    resize_for_logging,
)
from model import load_all_specialists, load_model
from predict import predict_with_specialists

# Reuse the same device logic as robust_unsupervised/prelude.py
_DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

_TO_TENSOR = transforms.ToTensor()


def load_image_natural(path: str) -> torch.Tensor:
    """
    Loads an image at its natural (saved) resolution without forcing a resize.
    Returns a (1, 3, H, W) tensor on the correct device.
    """
    img = PilImage.open(path).convert("RGB")
    return _TO_TENSOR(img).unsqueeze(0).to(_DEVICE)

# ── CNN type names → benchmark task names ────────────────────────────────────
_TYPE_TO_TASK = {
    "upsample":   "upsampling",
    "denoise":    "denoising",
    "deartifact": "deartifacting",
    "inpaint":    "inpainting",
}


def build_degradation(ordered_types: list, severity: str, resolution: int):
    """
    Constructs a ComposedDegradation from CNN predictions.

    Prepends ResizePrediction so the generator output is always resized to
    match the target resolution before the loss is computed.

    Args:
        ordered_types : CNN type names in paper's fixed composition order
        severity      : predicted severity level e.g. 'M'
        resolution    : generator output resolution

    Returns:
        ComposedDegradation ready for the optimization loop
    """
    task_names   = [_TYPE_TO_TASK[t] for t in ordered_types]
    degradations = [ResizePrediction(resolution)]
    for task_name in task_names:
        deg_cls = degradation_types[task_name]
        deg_arg = degradation_levels[task_name][severity]
        degradations.append(deg_cls(deg_arg))
    return ComposedDegradation(degradations)


def run_phase(label, variable, lr, target, degradation, loss_fn, steps,
              optimizer_cls=NGD):
    """
    Single optimization phase.

    Mirrors the run_phase function in run.py but takes all state explicitly
    (no globals) so it can be called from outside run.py.
    """
    optimizer = optimizer_cls(variable.parameters(), lr=lr)
    try:
        for _ in tqdm.tqdm(range(steps), desc=label):
            def closure():
                optimizer.zero_grad()
                x    = variable.to_image()
                loss = loss_fn(
                    degradation.degrade_prediction, x, target, degradation.mask
                ).mean()
                loss.backward()
                for p in variable.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                return loss

            if isinstance(optimizer, torch.optim.LBFGS):
                backup = variable.data.detach().clone()
                optimizer.step(closure)
                if variable.data.isnan().any() or variable.data.isinf().any():
                    with torch.no_grad():
                        variable.data.copy_(backup)
            else:
                closure()
                optimizer.step()
    except KeyboardInterrupt:
        print(f"\n  Interrupted during {label} — using current state.")
    return variable


def main(args):
    # ── Device ───────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device: CUDA")
    else:
        device = torch.device("cpu")
        print("Device: CPU — this will be slow")

    benchmark.config.resolution = args.resolution

    # ── Step 1: Load CNN models ───────────────────────────────────────────────
    print("\n[1/4] Loading CNN degradation estimator...")
    main_model  = load_model(args.model_path, device)
    specialists = load_all_specialists(args.specialists_dir, device)
    print(f"  Main model    : {args.model_path}")
    print(f"  Specialists   : {list(specialists.keys())}")

    # ── Step 2: Predict degradation ───────────────────────────────────────────
    print("\n[2/4] Predicting degradation type and severity...")
    (predicted_types, predicted_severity, ordered_types,
     type_probs, severity_probs, used_specialist) = predict_with_specialists(
        args.image, main_model, specialists, device, threshold=args.threshold
    )

    print(f"  Detected types     : {predicted_types}")
    print(f"  Severity           : {predicted_severity}")
    print(f"  Composition order  : {ordered_types}")
    print(f"  Used specialist    : {used_specialist}")

    if not ordered_types:
        print("\nERROR: No degradation type detected above threshold. "
              "Try lowering --threshold.")
        sys.exit(1)

    # ── Step 3: Build degradation & load generator ────────────────────────────
    print("\n[3/4] Setting up degradation model and StyleGAN generator...")

    # Load the input at its NATURAL (saved) resolution — do NOT force to 1024.
    # The training dataset creates degraded images via downsample→upsample-back,
    # so they stay at the original resolution (e.g. 256×256) rather than shrinking.
    # Forcing to 1024 would make the benchmark's Downsample output (1024/factor)
    # mismatch the target, causing the tensor size crash.
    target_natural = load_image_natural(args.image)
    natural_size   = target_natural.shape[-1]   # e.g. 256

    # For upsampling: the benchmark optimization works in the low-res domain.
    # Downsample.degrade_prediction = avg_pool2d(G(w), factor) → natural_size/factor.
    # We must match that by extracting the actual low-res content from the blurry input.
    # For all other types (denoise / deartifact / inpaint) the degradation preserves
    # spatial size, so the target stays at natural_size.
    if "upsample" in ordered_types:
        factor = degradation_levels["upsampling"][predicted_severity]  # e.g. 32
        target = F.avg_pool2d(target_natural, factor)                  # → natural_size/factor
        print(f"  Upsample factor    : {factor}  →  target size: {target.shape[-2]}×{target.shape[-1]}")
    else:
        target = target_natural

    # Use the natural image size for ResizePrediction so degrade_prediction
    # and the target are always in the same spatial domain.
    benchmark.config.resolution = natural_size
    degradation = build_degradation(ordered_types, predicted_severity, natural_size)

    G       = open_generator(args.pkl_path)
    loss_fn = MultiscaleLPIPS()
    print(f"  Input image loaded : {args.image}  (natural {natural_size}×{natural_size})")

    # ── Step 4: 3-phase GAN inversion ─────────────────────────────────────────
    print("\n[4/4] Running 3-phase GAN inversion...")
    s = args.steps
    lr = args.lr_scale

    # ── e4e warm start ────────────────────────────────────────────────────────
    # Noise added on top of the encoder output so the optimizer doesn't start
    # exactly on the encoder's solution — larger degradation needs more exploration.
    _NOISE_SCALE = {"XS": 0.005, "S": 0.010, "M": 0.020, "L": 0.050, "XL": 0.100}
    noise_std = _NOISE_SCALE.get(predicted_severity, 0.020)

    _e4e_encoder = None
    if args.e4e_path and os.path.exists(args.e4e_path):
        try:
            for _e4e_dir in [
                os.path.join(_HERE, "..", "encoder4editing"),
                os.path.join(_HERE, "encoder4editing"),
                "/kaggle/working/encoder4editing",
            ]:
                if os.path.isdir(_e4e_dir) and _e4e_dir not in sys.path:
                    sys.path.insert(0, os.path.abspath(_e4e_dir))
                    break
            from models.psp import pSp as _E4eModel
            import argparse as _ap
            _opts = _ap.Namespace(
                encoder_type="Encoder4Editing",
                start_from_latent_avg=True,
                learn_in_w=False,
                output_size=1024,
                checkpoint_path=args.e4e_path,
            )
            _e4e_encoder = _E4eModel(_opts).to(device).eval()
            print(f"  e4e encoder loaded from {args.e4e_path}")
        except Exception as _e:
            print(f"  [e4e] Could not load encoder ({_e}); using w_avg fallback.")

    if _e4e_encoder is not None:
        with torch.no_grad():
            # e4e expects [-1, 1] input at 256×256
            img_for_e4e = F.interpolate(
                (target * 2.0 - 1.0).clamp(-1, 1), size=(256, 256),
                mode="bilinear", align_corners=False,
            )
            w_init = _e4e_encoder(img_for_e4e)          # (1, num_ws, 512) or (1, 512)
            if w_init.dim() == 3:
                w_init = w_init[:, 0, :]                 # take first layer → (1, 512)
        w_init = w_init + noise_std * torch.randn_like(w_init)
        W_variable = WVariable(G, torch.nn.Parameter(w_init.to(device)))
        print(f"  W init : e4e warm start  (noise_std={noise_std}  severity={predicted_severity})")
    else:
        w_init = G.mapping.w_avg.reshape(1, G.w_dim).clone().to(device)
        w_init = w_init + noise_std * torch.randn_like(w_init)
        W_variable = WVariable(G, torch.nn.Parameter(w_init))
        print(f"  W init : w_avg + noise   (noise_std={noise_std}  severity={predicted_severity})")

    # NGD for all 3 phases — same as auto_restore.py
    run_phase("Phase 1 — W   (NGD)",
              W_variable,  lr * 0.08,  target, degradation, loss_fn, s)

    Wp_variable = WpVariable.from_W(W_variable)
    run_phase("Phase 2 — W+  (NGD)",
              Wp_variable, lr * 0.02, target, degradation, loss_fn, s)

    Wpp_variable = WppVariable.from_Wp(Wp_variable)
    run_phase("Phase 3 — W++ (NGD)",
              Wpp_variable, lr * 0.005, target, degradation, loss_fn, s)

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)

    restored         = resize_for_logging(Wpp_variable.to_image(), args.resolution)
    degraded_pred    = degradation.degrade_prediction(restored)
    input_resized    = resize_for_logging(target, args.resolution)

    base      = os.path.splitext(os.path.basename(args.image))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    prefix    = os.path.join(args.out_dir, f"{base}_{timestamp}")

    save_image(restored,                    f"{prefix}_restored.png",          padding=0)
    save_image(input_resized,               f"{prefix}_input.png",             padding=0)
    save_image(degraded_pred,               f"{prefix}_degraded_restored.png", padding=0)
    save_image(
        torch.cat([input_resized, restored], dim=3),
        f"{prefix}_side_by_side.jpg",
        padding=0,
    )

    print(f"\nRestoration complete.")
    print(f"  Prediction : types={predicted_types}, severity={predicted_severity}")
    print(f"  Restored   : {prefix}_restored.png")
    print(f"  Comparison : {prefix}_side_by_side.jpg")
    print(f"  All outputs: {args.out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Blind image restoration using CNN degradation prediction + GAN inversion"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the degraded input image",
    )
    parser.add_argument(
        "--pkl_path", type=str, default="pretrained_networks/ffhq.pkl",
        help="StyleGAN2-ADA generator checkpoint (.pkl)",
    )
    parser.add_argument(
        "--model_path", type=str, default="best_model.pth",
        help="Main CNN model checkpoint",
    )
    parser.add_argument(
        "--specialists_dir", type=str, default="specialists",
        help="Directory containing the 4 specialist .pth files",
    )
    parser.add_argument(
        "--out_dir", type=str, default="restored_e4e",
        help="Output directory for restored images",
    )
    parser.add_argument(
        "--e4e_path", type=str, default=None,
        help="Path to e4e/pSp encoder checkpoint (.pt). "
             "If not provided, falls back to w_avg + noise.",
    )
    parser.add_argument(
        "--resolution", type=int, default=1024,
        help="Generator output resolution (must match the .pkl)",
    )
    parser.add_argument(
        "--steps", type=int, default=150,
        help="Optimization steps per phase",
    )
    parser.add_argument(
        "--lr_scale", type=float, default=1.0,
        help="Global learning-rate scale factor",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="CNN type-detection probability threshold",
    )
    main(parser.parse_args())
