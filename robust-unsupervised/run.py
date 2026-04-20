
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


config = parse_config()
benchmark.config.resolution = config.resolution

print(config.name)
timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")

G = open_generator(config.pkl_path)
loss_fn = MultiscaleLPIPS()

# ── Degradation-adaptive LR scaling ──────────────────────────────────────────
_LR_SCALE = {"XS": 1.00, "S": 0.80, "M": 0.60, "L": 0.40, "XL": 0.20}
_INT_TO_LEVEL = {2: "M", 3: "L", 4: "XL"}

# ── Optional e4e encoder warm-start ──────────────────────────────────────────
_e4e_encoder = None
if config.e4e_path and os.path.exists(config.e4e_path):
    try:
        from encoder4editing.models.psp import pSp as _E4eModel
        _ckpt = torch.load(config.e4e_path, map_location="cpu")
        _opts = _ckpt["opts"]
        _opts["checkpoint_path"] = config.e4e_path
        _e4e_model = _E4eModel(**_opts).to(DEVICE).eval()
        _e4e_encoder = lambda img: _e4e_model.encoder(img)[:, 0, :]
        print(f"[e4e] Loaded encoder from {config.e4e_path}")
    except Exception as _e:
        print(f"[e4e] Could not load encoder ({_e}); using w_avg fallback.")

# Absolute path to reference FID stats (stays valid even after os.chdir)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FID_REF    = os.path.join(_SCRIPT_DIR, "benchmark", "FFHQ-X_crops128_ncrops1000.npz")

# ---------------------------------------------------------------------------
# Inline scoring metrics
# ---------------------------------------------------------------------------
_SCORE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_psnr_metric  = torchmetrics.PeakSignalNoiseRatio(data_range=2.0).to(_SCORE_DEVICE)
_lpips_metric = lpips_metrics.LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(_SCORE_DEVICE)


def compute_scores(pred: torch.Tensor, gt: torch.Tensor):
    """Return (PSNR, LPIPS) for pred vs gt; both tensors in [0, 1]."""
    with torch.no_grad():
        pred_n = (pred * 2.0 - 1.0).clamp(-1.0, 1.0).to(_SCORE_DEVICE)
        gt_n   = (gt   * 2.0 - 1.0).clamp(-1.0, 1.0).to(_SCORE_DEVICE)
        psnr_val  = _psnr_metric(pred_n, gt_n).item()
        lpips_val = _lpips_metric(pred_n, gt_n).item()
    return psnr_val, lpips_val


# ---------------------------------------------------------------------------
# pFID helpers
# ---------------------------------------------------------------------------
_CROP_RES  = 128
_CROP_NUM  = 1000   # total crops across all images


def make_pfid_crops(image_paths: list, out_dir: str):
    """Random-crop each pred image and save to out_dir for pytorch_fid."""
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    cropping = TVT.RandomCrop(_CROP_RES)
    crops_per_image = max(1, _CROP_NUM // len(image_paths))
    idx = 0
    for path in image_paths:
        img = read_image(path)          # [C, H, W] uint8
        if img.shape[1] < _CROP_RES or img.shape[2] < _CROP_RES:
            print(f"  [pFID] skipping {path} — smaller than {_CROP_RES}px")
            continue
        for _ in range(crops_per_image):
            write_png(cropping(img), os.path.join(out_dir, f"{idx:05d}.png"))
            idx += 1


def compute_pfid(pred_paths: list, work_dir: str, suffix: str = "_W++"):
    """Make crops from pred_paths and return pFID vs FFHQ reference."""
    if not os.path.exists(_FID_REF):
        print(f"  [pFID] reference file not found: {_FID_REF}")
        return None

    crops_dir = os.path.join(work_dir, f"pfid_crops{suffix.replace('/', '_')}")
    print(f"  Making {_CROP_NUM} crops for pFID...")
    make_pfid_crops(pred_paths, crops_dir)

    try:
        result = subprocess.check_output(
            ["python", "-m", "pytorch_fid", _FID_REF, crops_dir],
            cwd=_SCRIPT_DIR,
            stderr=subprocess.STDOUT,
        )
        last_line = result.decode("utf8").strip().splitlines()[-1]
        return float(last_line.replace("FID:  ", "").strip())
    except subprocess.CalledProcessError as e:
        print(f"  [pFID] pytorch_fid failed:\n{e.output.decode()}")
        return None
    except Exception as e:
        print(f"  [pFID] unexpected error: {e}")
        return None


# ---------------------------------------------------------------------------
# Optimization phases
# ---------------------------------------------------------------------------

def run_phase(label: str, variable: Variable, lr: float, optimizer_cls=NGD):  # type: ignore[assignment]
    optimizer = optimizer_cls(variable.parameters(), lr=lr)
    try:
        for _ in tqdm.tqdm(range(config.steps), desc=label):
            def closure():
                optimizer.zero_grad()
                x = variable.to_image()
                loss = loss_fn(degradation.degrade_prediction, x, target, degradation.mask).mean()
                loss.backward()
                # Sanitize gradients so LBFGS line search never sees NaN/inf
                for p in variable.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                return loss

            if isinstance(optimizer, torch.optim.LBFGS):
                # Save params before step; restore if LBFGS produces NaN
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

    save_image(
        torch.cat([approx_degraded_pred, degraded_pred]),
        f"degradation_approximation{suffix}.jpg",
        padding=0,
    )
    save_image(
        torch.cat(
            [
                ground_truth,
                resize_for_logging(target, config.resolution),
                resize_for_logging(degraded_pred, config.resolution),
                pred,
            ]
        ),
        f"side_by_side{suffix}.jpg",
        padding=0,
    )
    save_image(
        torch.cat([resize_for_logging(target, config.resolution), pred]),
        f"result{suffix}.jpg",
        padding=0,
    )
    save_image(
        torch.cat([target, degraded_pred, (target - degraded_pred).abs()]),
        f"fidelity{suffix}.jpg",
        padding=0,
    )
    save_image(
        torch.cat([ground_truth, pred, (ground_truth - pred).abs()]),
        f"accuracy{suffix}.jpg",
        padding=0,
    )

    psnr_val, lpips_val = compute_scores(pred, ground_truth)
    return {"PSNR": psnr_val, "LPIPS": lpips_val}


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

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
            # pFID is per-task (not per-phase), show only on W++ row
            pfid_str = f"{d['pFID']:.2f}" if phase == "W++" and d.get("pFID") is not None else ""
            label = task_key if first else ""
            print(f"  {label:<20} {phase:<6} {psnr_avg:>10.2f}  {lpips_avg:>8.4f}  {pfid_str:>8}  {len(entries):>4}")
            first = False
        print("-" * 90)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if config.tasks == "single":
        tasks = benchmark.single_tasks
    elif config.tasks == "composed":
        tasks = benchmark.composed_tasks
    elif config.tasks == "all":
        tasks = benchmark.all_tasks
    elif config.tasks == "custom":
        class YourDegradation:
            def degrade_ground_truth(self, x):
                raise NotImplementedError
            def degrade_prediction(self, x):
                raise NotImplementedError
        tasks = [
            benchmark.Task(
                constructor=YourDegradation,
                name="your_degradation",
                category="single",
                level="M",
            )
        ]
    else:
        raise Exception("Invalid task name")

    if config.task_name is not None:
        tasks = [t for t in tasks if t.name == config.task_name]
    if config.levels is not None:
        allowed = set(config.levels.split(","))
        tasks = [t for t in tasks if t.level in allowed]

    scores_by_task = {}   # task_key -> {"W": [...], "W+": [...], "W++": [...], "pFID": float}

    for task in tasks:
        experiment_path = f"out/{config.name}/{timestamp}/{task.category}/{task.name}/{task.level}/"
        abs_experiment_path = os.path.abspath(experiment_path)
        task_key  = f"{task.name}/{task.level}"
        level_str = task.level if isinstance(task.level, str) else _INT_TO_LEVEL.get(int(task.level), "M")
        lr_scale  = _LR_SCALE.get(level_str, 0.60)

        image_paths = sorted(
            [
                os.path.abspath(path)
                for path in (
                    glob.glob(config.dataset_path + "/**/*.png", recursive=True)
                    + glob.glob(config.dataset_path + "/**/*.jpg", recursive=True)
                    + glob.glob(config.dataset_path + "/**/*.jpeg", recursive=True)
                    + glob.glob(config.dataset_path + "/**/*.tif", recursive=True)
                )
            ]
        )
        assert len(image_paths) > 0, "No images found!"
        image_paths = image_paths[config.start_idx:]
        if config.n_images is not None:
            image_paths = image_paths[:config.n_images]
        print(f"  Running images {config.start_idx} – {config.start_idx + len(image_paths) - 1} ({len(image_paths)} total)")

        if task_key not in scores_by_task:
            scores_by_task[task_key] = {"W": [], "W+": [], "W++": [], "pFID": None}

        try:
            with directory(experiment_path):
                print(experiment_path)
                print(os.path.abspath(config.dataset_path))

                for j, image_path in enumerate(image_paths):
                    try:
                        with directory(f"inversions/{j:04d}"):
                            print(f"- {j:04d}")

                            ground_truth = open_image(image_path, config.resolution)
                            degradation = task.init_degradation()
                            save_image(ground_truth, f"ground_truth.png")
                            target = degradation.degrade_ground_truth(ground_truth)
                            save_image(target, f"target.png")

                            # ── Stochastic encoder warm-start ────────────────
                            W_variable = warm_start_W(G, target, level_str, _e4e_encoder)

                            # ── Phase 1: LBFGS (W) ───────────────────────────
                            lr_W = config.global_lr_scale * 0.10 * lr_scale
                            w_scores = run_phase("W", W_variable, lr_W, optimizer_cls=LBFGSPhase)  # type: ignore[arg-type]
                            scores_by_task[task_key]["W"].append(w_scores)

                            # ── Phase 2: NGD (W+) ─────────────────────────────
                            lr_Wp = config.global_lr_scale * 0.06 * lr_scale
                            Wp_variable = WpVariable.from_W(W_variable)
                            wp_scores = run_phase("W+", Wp_variable, lr_Wp, optimizer_cls=NGD)
                            scores_by_task[task_key]["W+"].append(wp_scores)

                            # ── Phase 3: NGD (W++) ────────────────────────────
                            lr_Wpp = config.global_lr_scale * 0.008 * lr_scale
                            Wpp_variable = WppVariable.from_Wp(Wp_variable)
                            wpp_scores = run_phase("W++", Wpp_variable, lr_Wpp, optimizer_cls=NGD)
                            scores_by_task[task_key]["W++"].append(wpp_scores)

                            print(
                                f"  [{task_key}] img {j:04d}"
                                f" | W   PSNR {w_scores['PSNR']:6.2f} dB  LPIPS {w_scores['LPIPS']:.4f}"
                                f" | W+  PSNR {wp_scores['PSNR']:6.2f} dB  LPIPS {wp_scores['LPIPS']:.4f}"
                                f" | W++ PSNR {wpp_scores['PSNR']:6.2f} dB  LPIPS {wpp_scores['LPIPS']:.4f}"
                            )
                    except Exception as img_err:
                        print(f"  [SKIP] img {j:04d} failed — {img_err}")

            # ---- pFID: collect all W++ predictions for this task ----
            abs_experiment = abs_experiment_path
            all_pred_paths = [
                os.path.join(abs_experiment, "inversions", f"{j:04d}", "pred_W++.png")
                for j in range(len(image_paths))
            ]
            pred_paths = [p for p in all_pred_paths if os.path.exists(p)]
            missing = [p for p in all_pred_paths if not os.path.exists(p)]
            if missing:
                print(f"  [pFID] {len(missing)} pred_W++.png file(s) missing — skipped images above")

            if pred_paths:
                print(f"  [pFID] computing over {len(pred_paths)} images...")
                pfid = compute_pfid(pred_paths, abs_experiment, suffix="_W++")
                scores_by_task[task_key]["pFID"] = pfid
                pfid_str = f"{pfid:.2f}" if pfid is not None else "failed"
                print(f"\n  pFID ({task_key}): {pfid_str}")
            else:
                print(f"  [pFID] no pred_W++.png files found at {abs_experiment}/inversions/")

        except Exception as task_err:
            print(f"\n  [SKIP] degradation '{task_key}' failed — {task_err}")

        # Per-task running summary
        print(f"\n--- Scores so far for {task_key} ---")
        for phase in ["W", "W+", "W++"]:
            entries = scores_by_task[task_key][phase]
            if entries:
                print(
                    f"  {phase:<4}  avg PSNR {_mean([s['PSNR']  for s in entries]):6.2f} dB"
                    f"  avg LPIPS {_mean([s['LPIPS'] for s in entries]):.4f}"
                    f"  (n={len(entries)})"
                )
        pfid = scores_by_task[task_key].get("pFID")
        if pfid is not None:
            print(f"  pFID  {pfid:.2f}  (W++ predictions vs FFHQ reference)")

    # Final table
    print("\n\n=== FINAL DEGRADATION SCORES SUMMARY ===")
    print_scores_table(scores_by_task)
