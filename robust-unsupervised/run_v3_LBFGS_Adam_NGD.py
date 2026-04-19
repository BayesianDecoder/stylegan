"""
run_v3_LBFGS_Adam_NGD.py  —  Ablation Variant 3
================================================
Optimizer order:
    Phase 1  W    LBFGS  adaptive LR scaled by degradation severity
    Phase 2  W+   Adam   adaptive LR scaled by degradation severity
    Phase 3  W++  NGD    fixed LR  (gradient normalisation handles scale)

Rationale:
    LBFGS in Phase 1 uses curvature for a high-quality initial W.
    Adam in Phase 2 uses momentum for stable per-layer W+ refinement.
    NGD in Phase 3 applies scale-invariant gradient descent in W++ space,
    preventing the very-high-DOF W++ optimisation from diverging.

    This is the closest variant to the paper's original configuration.
    The adaptive LRs improve on the fixed values by tuning down for
    severe degradations where the loss landscape is rougher.

Usage (Kaggle):
    python run_v3_LBFGS_Adam_NGD.py --name v3_LBFGS_Adam_NGD \
        --dataset_path /kaggle/input/ffhq-dataset/images \
        --pkl_path /kaggle/input/ffhq-stylegan/ffhq.pkl \
        --tasks single --levels M,L,XL --n_images 50
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


# ── Adaptive LR tables ────────────────────────────────────────────────────────
LBFGS_SCALE = {
    "XS": 1.00,
    "S":  0.80,
    "M":  0.50,
    "L":  0.30,
    "XL": 0.10,
}

ADAM_SCALE = {
    "XS": 1.00,
    "S":  0.80,
    "M":  0.60,
    "L":  0.40,
    "XL": 0.20,
}

_INT_TO_LEVEL = {2: "M", 3: "L", 4: "XL"}


def get_level_str(level) -> str:
    if isinstance(level, str):
        return level
    return _INT_TO_LEVEL.get(int(level), "M")


# ── Globals ───────────────────────────────────────────────────────────────────
config = parse_config()
benchmark.config.resolution = config.resolution

print(config.name)
timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")

G       = open_generator(config.pkl_path)
loss_fn = MultiscaleLPIPS()

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FID_REF    = os.path.join(_SCRIPT_DIR, "benchmark", "FFHQ-X_crops128_ncrops1000.npz")

_SCORE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_psnr_metric  = torchmetrics.PeakSignalNoiseRatio(data_range=2.0).to(_SCORE_DEVICE)
_lpips_metric = lpips_metrics.LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(_SCORE_DEVICE)


def compute_scores(pred: torch.Tensor, gt: torch.Tensor):
    with torch.no_grad():
        pred_n = (pred * 2.0 - 1.0).clamp(-1.0, 1.0).to(_SCORE_DEVICE)
        gt_n   = (gt   * 2.0 - 1.0).clamp(-1.0, 1.0).to(_SCORE_DEVICE)
        psnr_val  = _psnr_metric(pred_n, gt_n).item()
        lpips_val = _lpips_metric(pred_n, gt_n).item()
    return psnr_val, lpips_val


_CROP_RES = 128
_CROP_NUM = 1000


def make_pfid_crops(image_paths: list, out_dir: str):
    if os.path.exists(out_dir):
        import shutil; shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    cropping = TVT.RandomCrop(_CROP_RES)
    crops_per_image = max(1, _CROP_NUM // len(image_paths))
    idx = 0
    for path in image_paths:
        img = read_image(path)
        if img.shape[1] < _CROP_RES or img.shape[2] < _CROP_RES:
            continue
        for _ in range(crops_per_image):
            write_png(cropping(img), os.path.join(out_dir, f"{idx:05d}.png"))
            idx += 1


def compute_pfid(pred_paths: list, work_dir: str, suffix: str = "_W++"):
    if not os.path.exists(_FID_REF):
        return None
    crops_dir = os.path.join(work_dir, f"pfid_crops{suffix.replace('/', '_')}")
    make_pfid_crops(pred_paths, crops_dir)
    try:
        result = subprocess.check_output(
            ["python", "-m", "pytorch_fid", _FID_REF, crops_dir],
            cwd=_SCRIPT_DIR, stderr=subprocess.STDOUT,
        )
        return float(result.decode("utf8").strip().splitlines()[-1].replace("FID:  ", "").strip())
    except Exception:
        return None


def run_phase(label: str, variable: Variable, lr: float, optimizer_cls=NGD):  # type: ignore[assignment]
    optimizer = optimizer_cls(variable.parameters(), lr=lr)
    try:
        for _ in tqdm.tqdm(range(config.steps), desc=label):
            def closure():
                optimizer.zero_grad()
                x    = variable.to_image()
                loss = loss_fn(degradation.degrade_prediction, x, target, degradation.mask).mean()
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
        pass

    suffix = "_" + label
    pred = resize_for_logging(variable.to_image(), config.resolution)
    approx_degraded_pred = degradation.degrade_prediction(pred)
    degraded_pred = degradation.degrade_ground_truth(pred)

    save_image(pred, f"pred{suffix}.png", padding=0)
    save_image(degraded_pred, f"degraded_pred{suffix}.png", padding=0)
    save_image(torch.cat([approx_degraded_pred, degraded_pred]),
               f"degradation_approximation{suffix}.jpg", padding=0)
    save_image(torch.cat([ground_truth,
                           resize_for_logging(target, config.resolution),
                           resize_for_logging(degraded_pred, config.resolution),
                           pred]),
               f"side_by_side{suffix}.jpg", padding=0)
    save_image(torch.cat([resize_for_logging(target, config.resolution), pred]),
               f"result{suffix}.jpg", padding=0)
    save_image(torch.cat([target, degraded_pred, (target - degraded_pred).abs()]),
               f"fidelity{suffix}.jpg", padding=0)
    save_image(torch.cat([ground_truth, pred, (ground_truth - pred).abs()]),
               f"accuracy{suffix}.jpg", padding=0)

    psnr_val, lpips_val = compute_scores(pred, ground_truth)
    return {"PSNR": psnr_val, "LPIPS": lpips_val}


def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def print_scores_table(scores_by_task: dict):
    phases = ["W", "W+", "W++"]
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"{'DEGRADATION':<22} {'PHASE':<6} {'PSNR (dB)':>10}  {'LPIPS':>8}  {'pFID':>8}  {'N':>4}")
    print(sep)
    for task_key in sorted(scores_by_task):
        d = scores_by_task[task_key]
        first = True
        for phase in phases:
            entries = d.get(phase, [])
            if not entries:
                continue
            psnr_avg  = _mean([s["PSNR"]  for s in entries])
            lpips_avg = _mean([s["LPIPS"] for s in entries])
            pfid_str = f"{d['pFID']:.2f}" if phase == "W++" and d.get("pFID") is not None else ""
            label = task_key if first else ""
            print(f"  {label:<20} {phase:<6} {psnr_avg:>10.2f}  {lpips_avg:>8.4f}  {pfid_str:>8}  {len(entries):>4}")
            first = False
        print("-" * 90)
    print()


if __name__ == '__main__':
    if config.tasks == "single":
        tasks = benchmark.single_tasks
    elif config.tasks == "composed":
        tasks = benchmark.composed_tasks
    elif config.tasks == "all":
        tasks = benchmark.all_tasks
    else:
        raise Exception("Invalid task name")

    if config.task_name is not None:
        tasks = [t for t in tasks if t.name == config.task_name]
    if config.levels is not None:
        allowed = set(config.levels.split(","))
        tasks = [t for t in tasks if str(t.level) in allowed]

    scores_by_task = {}

    for task in tasks:
        experiment_path     = f"out/{config.name}/{timestamp}/{task.category}/{task.name}/{task.level}/"
        abs_experiment_path = os.path.abspath(experiment_path)
        task_key            = f"{task.name}/{task.level}"
        level_str           = get_level_str(task.level)

        # ── Adaptive LRs for this task's severity level ───────────────────
        # Phase 1: LBFGS — scale down for severe degradations (rough landscape)
        lr_W   = config.global_lr_scale * 0.10 * LBFGS_SCALE[level_str]
        # Phase 2: Adam — scale down for severe degradations (noisy gradients)
        lr_Wp  = config.global_lr_scale * 0.025 * ADAM_SCALE[level_str]
        # Phase 3: NGD — fixed (gradient normalisation already scale-invariant)
        lr_Wpp = config.global_lr_scale * 0.008

        print(f"\n[V3] {task_key}  level_str={level_str}"
              f"  lr_W={lr_W:.4f}  lr_W+={lr_Wp:.5f}  lr_W++={lr_Wpp:.5f}")

        image_paths = sorted([
            os.path.abspath(p)
            for p in (
                glob.glob(config.dataset_path + "/**/*.png",  recursive=True)
                + glob.glob(config.dataset_path + "/**/*.jpg",  recursive=True)
                + glob.glob(config.dataset_path + "/**/*.jpeg", recursive=True)
                + glob.glob(config.dataset_path + "/**/*.tif",  recursive=True)
            )
        ])
        assert len(image_paths) > 0, "No images found!"
        image_paths = image_paths[config.start_idx:]
        if config.n_images is not None:
            image_paths = image_paths[:config.n_images]

        if task_key not in scores_by_task:
            scores_by_task[task_key] = {"W": [], "W+": [], "W++": [], "pFID": None}

        try:
            with directory(experiment_path):
                for j, image_path in enumerate(image_paths):
                    try:
                        with directory(f"inversions/{j:04d}"):
                            print(f"- {j:04d}")
                            ground_truth = open_image(image_path, config.resolution)
                            degradation  = task.init_degradation()
                            save_image(ground_truth, "ground_truth.png")
                            target = degradation.degrade_ground_truth(ground_truth)
                            save_image(target, "target.png")

                            # ── V3: LBFGS → Adam → NGD ───────────────────
                            W_variable = WVariable.sample_from(G)
                            w_scores = run_phase("W", W_variable, lr_W,
                                                 optimizer_cls=LBFGSPhase)
                            scores_by_task[task_key]["W"].append(w_scores)

                            Wp_variable = WpVariable.from_W(W_variable)
                            wp_scores = run_phase("W+", Wp_variable, lr_Wp,
                                                  optimizer_cls=torch.optim.Adam)
                            scores_by_task[task_key]["W+"].append(wp_scores)

                            Wpp_variable = WppVariable.from_Wp(Wp_variable)
                            wpp_scores = run_phase("W++", Wpp_variable, lr_Wpp,
                                                   optimizer_cls=NGD)
                            scores_by_task[task_key]["W++"].append(wpp_scores)

                            print(f"  [{task_key}] img {j:04d}"
                                  f" | W   PSNR {w_scores['PSNR']:6.2f} dB  LPIPS {w_scores['LPIPS']:.4f}"
                                  f" | W+  PSNR {wp_scores['PSNR']:6.2f} dB  LPIPS {wp_scores['LPIPS']:.4f}"
                                  f" | W++ PSNR {wpp_scores['PSNR']:6.2f} dB  LPIPS {wpp_scores['LPIPS']:.4f}")
                    except Exception as img_err:
                        print(f"  [SKIP] img {j:04d} — {img_err}")

            pred_paths = [
                p for p in [
                    os.path.join(abs_experiment_path, "inversions", f"{j:04d}", "pred_W++.png")
                    for j in range(len(image_paths))
                ] if os.path.exists(p)
            ]
            if pred_paths:
                pfid = compute_pfid(pred_paths, abs_experiment_path)
                scores_by_task[task_key]["pFID"] = pfid
                print(f"\n  pFID ({task_key}): {pfid:.2f}" if pfid else f"\n  pFID: failed")

        except Exception as task_err:
            print(f"\n  [SKIP] '{task_key}' — {task_err}")

        for phase in ["W", "W+", "W++"]:
            entries = scores_by_task[task_key][phase]
            if entries:
                print(f"  {phase:<4}  avg PSNR {_mean([s['PSNR'] for s in entries]):6.2f} dB"
                      f"  avg LPIPS {_mean([s['LPIPS'] for s in entries]):.4f}"
                      f"  (n={len(entries)})")

    print("\n\n=== V3 (LBFGS → Adam → NGD) FINAL SCORES ===")
    print_scores_table(scores_by_task)
