"""
run_e4e.py
----------
Variant of run.py with e4e/pSp encoder warm start for W initialization.

The ONLY change vs run.py:
    W_variable is initialized from the e4e encoder output + adaptive Gaussian
    noise scaled by degradation severity, instead of the plain w_avg.

Noise scales per severity (larger degradation → more noise for exploration):
    XS: 0.005  S: 0.010  M: 0.020  L: 0.050  XL: 0.100

Everything else is identical to run.py:
    - Same NGD optimizer for all 3 phases
    - Same LRs: W=0.08  W+=0.02  W++=0.005  (× global_lr_scale)
    - Same MultiscaleLPIPS loss
    - Same output files (pred_W.png, side_by_side_W++.jpg, etc.)
    - Same benchmark task loop

Usage (Kaggle):
    python run_e4e.py \\
        --name e4e_run \\
        --pkl_path pretrained_networks/ffhq.pkl \\
        --dataset_path /kaggle/input/ffhq512 \\
        --tasks single --levels L,XL --n_images 10 --steps 150 \\
        --e4e_path /kaggle/working/e4e_ffhq_encode.pt

    If --e4e_path is not provided or file is missing, falls back to
    w_avg + noise (still adaptive, no encoder needed).
"""

from cli import parse_config
import glob
import subprocess

import benchmark
from benchmark import Task, Degradation
from robust_unsupervised import *

import torchmetrics
import torchmetrics.image.lpip as lpips_metrics
import torchvision.transforms as TVT
from torchvision.io import read_image, write_png

# ── Config ────────────────────────────────────────────────────────────────────
config = parse_config()
benchmark.config.resolution = config.resolution

print(config.name)
timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")

G       = open_generator(config.pkl_path)
loss_fn = MultiscaleLPIPS()

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FID_REF    = os.path.join(_SCRIPT_DIR, "benchmark", "FFHQ-X_crops128_ncrops1000.npz")

# ── Inline scoring metrics ────────────────────────────────────────────────────
_SCORE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_psnr_metric  = torchmetrics.PeakSignalNoiseRatio(data_range=2.0).to(_SCORE_DEVICE)
_lpips_metric = lpips_metrics.LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(_SCORE_DEVICE)


def compute_scores(pred: torch.Tensor, gt: torch.Tensor):
    with torch.no_grad():
        pred_n = (pred * 2.0 - 1.0).clamp(-1.0, 1.0).to(_SCORE_DEVICE)
        gt_n   = (gt   * 2.0 - 1.0).clamp(-1.0, 1.0).to(_SCORE_DEVICE)
        return _psnr_metric(pred_n, gt_n).item(), _lpips_metric(pred_n, gt_n).item()


# ── pFID helpers ──────────────────────────────────────────────────────────────
_CROP_RES = 128
_CROP_NUM = 1000


def make_pfid_crops(image_paths, out_dir):
    if os.path.exists(out_dir):
        import shutil; shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    crop = TVT.RandomCrop(_CROP_RES)
    n = max(1, _CROP_NUM // len(image_paths))
    idx = 0
    for p in image_paths:
        img = read_image(p)
        if img.shape[1] < _CROP_RES or img.shape[2] < _CROP_RES: continue
        for _ in range(n):
            write_png(crop(img), os.path.join(out_dir, f"{idx:05d}.png")); idx += 1


def compute_pfid(pred_paths, work_dir, suffix="_W++"):
    if not os.path.exists(_FID_REF): return None
    crops_dir = os.path.join(work_dir, f"pfid_crops{suffix.replace('/','_')}")
    make_pfid_crops(pred_paths, crops_dir)
    try:
        out = subprocess.check_output(
            ["python", "-m", "pytorch_fid", _FID_REF, crops_dir],
            cwd=_SCRIPT_DIR, stderr=subprocess.STDOUT,
        )
        return float(out.decode().strip().splitlines()[-1].replace("FID:  ","").strip())
    except Exception: return None


# ── e4e encoder warm start ────────────────────────────────────────────────────
# Noise std per severity — larger degradation needs more initial exploration
_NOISE_SCALE = {"XS": 0.005, "S": 0.010, "M": 0.020, "L": 0.050, "XL": 0.100}
_INT_TO_LEVEL = {0: "XS", 1: "S", 2: "M", 3: "L", 4: "XL"}

# Load e4e encoder once at startup (optional — falls back to w_avg if missing)
_e4e_encoder = None
if hasattr(config, "e4e_path") and config.e4e_path and os.path.exists(config.e4e_path):
    try:
        _here = os.path.dirname(os.path.abspath(__file__))
        for _e4e_dir in [
            os.path.join(_here, "..", "encoder4editing"),
            os.path.join(_here, "encoder4editing"),
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
            checkpoint_path=config.e4e_path,
        )
        _e4e_device = "cuda" if torch.cuda.is_available() else "cpu"
        _e4e_encoder = _E4eModel(_opts).to(_e4e_device).eval()
        print(f"[e4e] Encoder loaded from {config.e4e_path}")
    except Exception as _e:
        print(f"[e4e] Could not load encoder ({_e}); using w_avg + noise fallback.")


def warm_start_W(image_01: torch.Tensor, level_str: str) -> "WVariable":
    """
    Returns a WVariable initialized from e4e encoder output (if available)
    or w_avg, both perturbed with adaptive Gaussian noise.

    Args:
        image_01  : (1, 3, H, W) tensor in [0, 1] — the degraded target image
        level_str : severity level string e.g. 'L'
    """
    noise_std = _NOISE_SCALE.get(level_str, 0.020)
    _dev = next(G.parameters()).device

    if _e4e_encoder is not None:
        try:
            with torch.no_grad():
                img = F.interpolate(
                    (image_01 * 2.0 - 1.0).clamp(-1, 1),
                    size=(256, 256), mode="bilinear", align_corners=False,
                ).to(next(_e4e_encoder.parameters()).device)
                w = _e4e_encoder(img)           # (1, num_ws, 512) or (1, 512)
                if w.dim() == 3:
                    w = w[:, 0, :]              # first layer → (1, 512)
            w = w.to(_dev)
        except Exception as _e:
            print(f"  [e4e] Inference failed ({_e}); falling back to w_avg.")
            w = G.mapping.w_avg.reshape(1, G.w_dim).clone().to(_dev)
    else:
        w = G.mapping.w_avg.reshape(1, G.w_dim).clone().to(_dev)

    w = w + noise_std * torch.randn_like(w)
    return WVariable(G, torch.nn.Parameter(w))


# ── Optimization phase — identical to run.py ─────────────────────────────────
_BATCH_SIZE = 5


def run_phase(label: str, variable: Variable, lr: float):
    optimizer = NGD(variable.parameters(), lr=lr)
    try:
        for _ in tqdm.tqdm(range(config.steps), desc=label):
            x    = variable.to_image()
            loss = loss_fn(degradation.degrade_prediction, x, target, degradation.mask).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    except KeyboardInterrupt:
        pass

    suffix = "_" + label
    pred   = resize_for_logging(variable.to_image(), config.resolution)

    approx_degraded_pred = degradation.degrade_prediction(pred)
    degraded_pred        = degradation.degrade_ground_truth(pred)

    save_image(pred,                   f"pred{suffix}.png",               padding=0)
    save_image(degraded_pred,          f"degraded_pred{suffix}.png",      padding=0)
    save_image(
        torch.cat([approx_degraded_pred, degraded_pred]),
        f"degradation_approximation{suffix}.jpg", padding=0,
    )
    save_image(
        torch.cat([ground_truth, resize_for_logging(target, config.resolution),
                   resize_for_logging(degraded_pred, config.resolution), pred]),
        f"side_by_side{suffix}.jpg", padding=0,
    )
    save_image(
        torch.cat([resize_for_logging(target, config.resolution), pred]),
        f"result{suffix}.jpg", padding=0,
    )
    save_image(
        torch.cat([target, degraded_pred, (target - degraded_pred).abs()]),
        f"fidelity{suffix}.jpg", padding=0,
    )
    save_image(
        torch.cat([ground_truth, pred, (ground_truth - pred).abs()]),
        f"accuracy{suffix}.jpg", padding=0,
    )
    psnr, lpips = compute_scores(pred, ground_truth)
    return {"PSNR": psnr, "LPIPS": lpips}


# ── Summary helpers ───────────────────────────────────────────────────────────
def _mean(vals): return sum(vals) / len(vals) if vals else float("nan")


def print_scores_table(scores_by_task):
    phases = ["W", "W+", "W++"]
    sep    = "=" * 90
    print(f"\n{sep}")
    print(f"{'DEGRADATION':<22} {'PHASE':<6} {'PSNR (dB)':>10}  {'LPIPS':>8}  {'pFID':>8}  {'N':>4}")
    print(sep)
    for task_key in sorted(scores_by_task):
        d, first = scores_by_task[task_key], True
        for phase in phases:
            entries = d.get(phase, [])
            if not entries: continue
            pfid_str = f"{d['pFID']:.2f}" if phase == "W++" and d.get("pFID") else ""
            label    = task_key if first else ""
            print(f"  {label:<20} {phase:<6} {_mean([s['PSNR']  for s in entries]):>10.2f}"
                  f"  {_mean([s['LPIPS'] for s in entries]):>8.4f}  {pfid_str:>8}  {len(entries):>4}")
            first = False
        print("-" * 90)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if config.tasks == "single":      tasks = benchmark.single_tasks
    elif config.tasks == "composed":  tasks = benchmark.composed_tasks
    elif config.tasks == "all":       tasks = benchmark.all_tasks
    else:                             raise Exception("Invalid task name")

    if config.task_name is not None:
        tasks = [t for t in tasks if t.name == config.task_name]
    if config.levels is not None:
        allowed = set(config.levels.split(","))
        tasks   = [t for t in tasks if str(t.level) in allowed]

    scores_by_task = {}

    for task in tasks:
        experiment_path = f"out/{config.name}/{timestamp}/{task.category}/{task.name}/{task.level}/"
        abs_experiment  = os.path.abspath(experiment_path)
        task_key        = f"{task.name}/{task.level}"

        # Map level to string (handles both str and int level values)
        level_str = task.level if isinstance(task.level, str) \
                    else _INT_TO_LEVEL.get(int(task.level), "M")

        image_paths = sorted([
            os.path.abspath(p) for p in (
                glob.glob(config.dataset_path + "/**/*.png",  recursive=True) +
                glob.glob(config.dataset_path + "/**/*.jpg",  recursive=True) +
                glob.glob(config.dataset_path + "/**/*.jpeg", recursive=True) +
                glob.glob(config.dataset_path + "/**/*.tif",  recursive=True)
            )
        ])
        assert len(image_paths) > 0, "No images found!"
        image_paths = image_paths[config.start_idx:]
        if config.n_images is not None:
            image_paths = image_paths[:config.n_images]

        print(f"  Running images {config.start_idx}–{config.start_idx + len(image_paths) - 1}"
              f" ({len(image_paths)} total)")

        if task_key not in scores_by_task:
            scores_by_task[task_key] = {"W": [], "W+": [], "W++": [], "pFID": None}

        try:
            with directory(experiment_path):
                for batch_start in range(0, len(image_paths), _BATCH_SIZE):
                    batch = image_paths[batch_start:batch_start + _BATCH_SIZE]

                    for j_rel, image_path in enumerate(batch):
                        j = batch_start + j_rel
                        try:
                            with directory(f"inversions/{j:04d}"):
                                print(f"- {j:04d}")

                                ground_truth = open_image(image_path, config.resolution)
                                degradation  = task.init_degradation()
                                save_image(ground_truth, "ground_truth.png")
                                target = degradation.degrade_ground_truth(ground_truth)
                                save_image(target, "target.png")

                                # ── e4e warm start (only change vs run.py) ──
                                W_variable = warm_start_W(target, level_str)

                                w_scores = run_phase("W", W_variable,
                                                     config.global_lr_scale * 0.08)
                                scores_by_task[task_key]["W"].append(w_scores)

                                Wp_variable  = WpVariable.from_W(W_variable)
                                wp_scores    = run_phase("W+", Wp_variable,
                                                         config.global_lr_scale * 0.02)
                                scores_by_task[task_key]["W+"].append(wp_scores)

                                Wpp_variable = WppVariable.from_Wp(Wp_variable)
                                wpp_scores   = run_phase("W++", Wpp_variable,
                                                         config.global_lr_scale * 0.005)
                                scores_by_task[task_key]["W++"].append(wpp_scores)

                                print(
                                    f"  [{task_key}] img {j:04d}"
                                    f" | W   PSNR {w_scores['PSNR']:6.2f} dB  LPIPS {w_scores['LPIPS']:.4f}"
                                    f" | W+  PSNR {wp_scores['PSNR']:6.2f} dB  LPIPS {wp_scores['LPIPS']:.4f}"
                                    f" | W++ PSNR {wpp_scores['PSNR']:6.2f} dB  LPIPS {wpp_scores['LPIPS']:.4f}"
                                )
                        except Exception as img_err:
                            print(f"  [SKIP] img {j:04d} — {img_err}")

                    # Batch summary
                    n_done = min(batch_start + _BATCH_SIZE, len(image_paths))
                    print(f"\n--- Batch summary (imgs {batch_start}–{n_done-1}, n={n_done}) ---")
                    for ph in ["W", "W+", "W++"]:
                        e = scores_by_task[task_key][ph]
                        if e:
                            print(f"  {ph:<4} avg PSNR {_mean([s['PSNR'] for s in e]):5.2f} dB"
                                  f"  avg LPIPS {_mean([s['LPIPS'] for s in e]):.4f}"
                                  f"  (n={len(e)})")
                    print()

            # pFID
            pred_paths = [p for p in [
                os.path.join(abs_experiment, "inversions", f"{j:04d}", "pred_W++.png")
                for j in range(len(image_paths))
            ] if os.path.exists(p)]
            if pred_paths:
                pfid = compute_pfid(pred_paths, abs_experiment)
                scores_by_task[task_key]["pFID"] = pfid
                print(f"\n  pFID ({task_key}): {pfid:.2f}" if pfid else "  pFID: failed")

        except Exception as task_err:
            print(f"\n  [SKIP] '{task_key}' — {task_err}")

        print(f"\n--- Scores so far for {task_key} ---")
        for phase in ["W", "W+", "W++"]:
            entries = scores_by_task[task_key][phase]
            if entries:
                print(f"  {phase:<4}  avg PSNR {_mean([s['PSNR']  for s in entries]):6.2f} dB"
                      f"  avg LPIPS {_mean([s['LPIPS'] for s in entries]):.4f}"
                      f"  (n={len(entries)})")

    print("\n\n=== FINAL SCORES (e4e warm start) ===")
    print_scores_table(scores_by_task)
