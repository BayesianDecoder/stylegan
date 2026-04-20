from cli import parse_config
import glob

import benchmark
from benchmark import Task, Degradation
from robust_unsupervised import *

import torchmetrics
import torchmetrics.image.lpip as lpips_metrics


config = parse_config()
benchmark.config.resolution = config.resolution

print(config.name)
timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")

G = open_generator(config.pkl_path)
loss_fn = MultiscaleLPIPS()

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
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


def _mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


# ---------------------------------------------------------------------------
# Optimization — NGD for all phases
# ---------------------------------------------------------------------------

def run_phase(label: str, variable: Variable, lr: float):
    optimizer = NGD(variable.parameters(), lr=lr)
    try:
        for _ in tqdm.tqdm(range(config.steps), desc=label):
            x = variable.to_image()
            loss = loss_fn(degradation.degrade_prediction, x, target, degradation.mask).mean()
            optimizer.zero_grad()
            loss.backward()
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
        f"degradation_approximation{suffix}.jpg", padding=0,
    )
    save_image(
        torch.cat([
            ground_truth,
            resize_for_logging(target, config.resolution),
            resize_for_logging(degraded_pred, config.resolution),
            pred,
        ]),
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

    psnr_val, lpips_val = compute_scores(pred, ground_truth)
    return {"PSNR": psnr_val, "LPIPS": lpips_val}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_scores_table(scores_by_task: dict):
    phases = ["W", "W+", "W++"]
    sep = "=" * 75
    print(f"\n{sep}")
    print(f"{'DEGRADATION':<22} {'PHASE':<6} {'PSNR (dB)':>10}  {'LPIPS':>8}  {'N':>4}")
    print(sep)
    for task_key in sorted(scores_by_task):
        d = scores_by_task[task_key]
        first = True
        for phase in phases:
            entries = d.get(phase, [])
            if not entries:
                continue
            label = task_key if first else ""
            print(
                f"  {label:<20} {phase:<6}"
                f" {_mean([s['PSNR']  for s in entries]):>10.2f}"
                f"  {_mean([s['LPIPS'] for s in entries]):>8.4f}"
                f"  {len(entries):>4}"
            )
            first = False
        print("-" * 75)
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

    scores_by_task = {}

    for task in tasks:
        experiment_path = f"out/{config.name}/{timestamp}/{task.category}/{task.name}/{task.level}/"
        task_key = f"{task.name}/{task.level}"

        image_paths = sorted([
            os.path.abspath(path)
            for path in (
                glob.glob(config.dataset_path + "/**/*.png",  recursive=True)
                + glob.glob(config.dataset_path + "/**/*.jpg",  recursive=True)
                + glob.glob(config.dataset_path + "/**/*.jpeg", recursive=True)
                + glob.glob(config.dataset_path + "/**/*.tif",  recursive=True)
            )
        ])
        assert len(image_paths) > 0, "No images found!"
        if config.n_images is not None:
            image_paths = image_paths[config.start_idx : config.start_idx + config.n_images]
        elif config.start_idx > 0:
            image_paths = image_paths[config.start_idx:]

        if task_key not in scores_by_task:
            scores_by_task[task_key] = {"W": [], "W+": [], "W++": []}

        with directory(experiment_path):
            print(experiment_path)

            for j, image_path in enumerate(image_paths):
                with directory(f"inversions/{j:04d}"):
                    print(f"- {j:04d}")

                    ground_truth = open_image(image_path, config.resolution)
                    degradation  = task.init_degradation()
                    save_image(ground_truth, "ground_truth.png")
                    target = degradation.degrade_ground_truth(ground_truth)
                    save_image(target, "target.png")

                    W_variable  = WVariable.sample_from(G)
                    w_scores    = run_phase("W",   W_variable,  config.global_lr_scale * 0.08)

                    Wp_variable = WpVariable.from_W(W_variable)
                    wp_scores   = run_phase("W+",  Wp_variable, config.global_lr_scale * 0.02)

                    Wpp_variable = WppVariable.from_Wp(Wp_variable)
                    wpp_scores   = run_phase("W++", Wpp_variable, config.global_lr_scale * 0.005)

                    scores_by_task[task_key]["W"].append(w_scores)
                    scores_by_task[task_key]["W+"].append(wp_scores)
                    scores_by_task[task_key]["W++"].append(wpp_scores)

                    print(
                        f"  [{task_key}] img {j:04d}"
                        f" | W   PSNR {w_scores['PSNR']:6.2f} dB  LPIPS {w_scores['LPIPS']:.4f}"
                        f" | W+  PSNR {wp_scores['PSNR']:6.2f} dB  LPIPS {wp_scores['LPIPS']:.4f}"
                        f" | W++ PSNR {wpp_scores['PSNR']:6.2f} dB  LPIPS {wpp_scores['LPIPS']:.4f}"
                    )

        print(f"\n--- Scores so far: {task_key} ---")
        for phase in ["W", "W+", "W++"]:
            entries = scores_by_task[task_key][phase]
            if entries:
                print(
                    f"  {phase:<4}  avg PSNR {_mean([s['PSNR']  for s in entries]):6.2f} dB"
                    f"  avg LPIPS {_mean([s['LPIPS'] for s in entries]):.4f}"
                    f"  (n={len(entries)})"
                )

    print("\n\n=== FINAL SCORES SUMMARY (NGD x3) ===")
    print_scores_table(scores_by_task)
