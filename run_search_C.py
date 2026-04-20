"""
run_search_C.py  —  LR Search Config C: Low LRs (realism-focused)
==================================================================
NGD → Adam → LBFGS  with LOW adaptive LRs.

Hypothesis: lower LRs (especially LBFGS) keep the reconstruction closer to
the GAN manifold → better pFID realism, may trade off some Accuracy LPIPS.

L  → Adam lr = 0.02 × 0.30 = 0.0060 | LBFGS lr = 0.005 × 0.10 = 0.00050
XL → Adam lr = 0.02 × 0.15 = 0.0030 | LBFGS lr = 0.005 × 0.05 = 0.00025

Usage (Kaggle Notebook 3):
    python run_search_C.py \
        --name search_C \
        --pkl_path /kaggle/working/stylegan/robust-unsupervised/pretrained_networks/ffhq.pkl \
        --dataset_path /kaggle/input/datasets/chelove4draste/ffhq-512x512/ffhq512 \
        --tasks single --levels L,XL --n_images 10 --steps 150
"""

from cli import parse_config
import glob, subprocess

import benchmark
from benchmark import Task, Degradation
from robust_unsupervised import *

import torchmetrics
import torchmetrics.image.lpip as lpips_metrics
import torchvision.transforms as TVT
from torchvision.io import read_image, write_png

# ── Config C: LOW LRs — stay close to GAN manifold for realism ───────────────
ADAM_SCALE = {
    "XS": 1.00,
    "S":  0.70,
    "M":  0.50,
    "L":  0.30,   # <-- lower than baseline
    "XL": 0.15,   # <-- lower than baseline
}
LBFGS_SCALE = {
    "XS": 1.00,
    "S":  0.60,
    "M":  0.30,
    "L":  0.10,   # <-- much lower than baseline
    "XL": 0.05,   # <-- much lower than baseline
}

_BATCH_SIZE = 5   # print intermediate summary every N images

_INT_TO_LEVEL = {2: "M", 3: "L", 4: "XL"}

def get_level_str(level) -> str:
    return level if isinstance(level, str) else _INT_TO_LEVEL.get(int(level), "M")

# ── Setup ─────────────────────────────────────────────────────────────────────
config = parse_config()
benchmark.config.resolution = config.resolution
print(f"[Search C] {config.name}")
timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")
G       = open_generator(config.pkl_path)
loss_fn = MultiscaleLPIPS()
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_FID_REF      = os.path.join(_SCRIPT_DIR, "benchmark", "FFHQ-X_crops128_ncrops1000.npz")
_SCORE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_psnr_metric  = torchmetrics.PeakSignalNoiseRatio(data_range=2.0).to(_SCORE_DEVICE)
_lpips_metric = lpips_metrics.LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(_SCORE_DEVICE)

def compute_scores(pred, gt):
    with torch.no_grad():
        p = (pred * 2.0 - 1.0).clamp(-1, 1).to(_SCORE_DEVICE)
        g = (gt   * 2.0 - 1.0).clamp(-1, 1).to(_SCORE_DEVICE)
        return _psnr_metric(p, g).item(), _lpips_metric(p, g).item()

_CROP_RES, _CROP_NUM = 128, 1000

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
        out = subprocess.check_output(["python", "-m", "pytorch_fid", _FID_REF, crops_dir],
                                       cwd=_SCRIPT_DIR, stderr=subprocess.STDOUT)
        return float(out.decode().strip().splitlines()[-1].replace("FID:  ","").strip())
    except Exception: return None

def run_phase(label, variable, lr, optimizer_cls=NGD):  # type: ignore[assignment]
    optimizer = optimizer_cls(variable.parameters(), lr=lr)
    try:
        for _ in tqdm.tqdm(range(config.steps), desc=label):
            def closure():
                optimizer.zero_grad()
                x = variable.to_image()
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
                    with torch.no_grad(): variable.data.copy_(backup)
            else:
                closure(); optimizer.step()
    except KeyboardInterrupt: pass

    sfx  = "_" + label
    pred = resize_for_logging(variable.to_image(), config.resolution)
    dp   = degradation.degrade_ground_truth(pred)
    save_image(pred, f"pred{sfx}.png", padding=0)
    save_image(dp,   f"degraded_pred{sfx}.png", padding=0)
    save_image(torch.cat([degradation.degrade_prediction(pred), dp]),
               f"degradation_approximation{sfx}.jpg", padding=0)
    save_image(torch.cat([ground_truth, resize_for_logging(target, config.resolution),
                           resize_for_logging(dp, config.resolution), pred]),
               f"side_by_side{sfx}.jpg", padding=0)
    save_image(torch.cat([resize_for_logging(target, config.resolution), pred]),
               f"result{sfx}.jpg", padding=0)
    save_image(torch.cat([target, dp, (target-dp).abs()]), f"fidelity{sfx}.jpg", padding=0)
    save_image(torch.cat([ground_truth, pred, (ground_truth-pred).abs()]),
               f"accuracy{sfx}.jpg", padding=0)
    psnr, lpips = compute_scores(pred, ground_truth)
    return {"PSNR": psnr, "LPIPS": lpips}

def _mean(v): return sum(v)/len(v) if v else float("nan")

def print_table(scores_by_task):
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"{'DEGRADATION':<22} {'PHASE':<6} {'PSNR (dB)':>10}  {'LPIPS':>8}  {'pFID':>8}  {'N':>4}")
    print(sep)
    for k in sorted(scores_by_task):
        d, first = scores_by_task[k], True
        for ph in ["W","W+","W++"]:
            e = d.get(ph, [])
            if not e: continue
            pfid_str = f"{d['pFID']:.2f}" if ph=="W++" and d.get("pFID") else ""
            lbl = k if first else ""
            print(f"  {lbl:<20} {ph:<6} {_mean([s['PSNR'] for s in e]):>10.2f}"
                  f"  {_mean([s['LPIPS'] for s in e]):>8.4f}  {pfid_str:>8}  {len(e):>4}")
            first = False
        print("-" * 90)

if __name__ == '__main__':
    if config.tasks == "single":      tasks = benchmark.single_tasks
    elif config.tasks == "composed":  tasks = benchmark.composed_tasks
    else:                             tasks = benchmark.all_tasks

    if config.task_name is not None:
        tasks = [t for t in tasks if t.name == config.task_name]
    if config.levels is not None:
        allowed = set(config.levels.split(","))
        tasks = [t for t in tasks if str(t.level) in allowed]

    scores = {}
    for task in tasks:
        exp     = f"out/{config.name}/{timestamp}/{task.category}/{task.name}/{task.level}/"
        abs_exp = os.path.abspath(exp)
        key     = f"{task.name}/{task.level}"
        lvl     = get_level_str(task.level)

        lr_W   = config.global_lr_scale * 0.08                         # NGD fixed
        lr_Wp  = config.global_lr_scale * 0.02  * ADAM_SCALE[lvl]     # Adam adaptive
        lr_Wpp = config.global_lr_scale * 0.005 * LBFGS_SCALE[lvl]    # LBFGS adaptive

        print(f"\n[C] {key}  lr_W={lr_W:.4f}  lr_W+={lr_Wp:.5f}  lr_W++={lr_Wpp:.5f}")

        imgs = sorted([os.path.abspath(p) for p in (
            glob.glob(config.dataset_path+"/**/*.png", recursive=True) +
            glob.glob(config.dataset_path+"/**/*.jpg", recursive=True) +
            glob.glob(config.dataset_path+"/**/*.jpeg",recursive=True))])
        assert imgs, "No images found!"
        imgs = imgs[config.start_idx:]
        if config.n_images: imgs = imgs[:config.n_images]

        scores[key] = {"W":[], "W+":[], "W++":[], "pFID": None}
        batch_sz = _BATCH_SIZE
        try:
            with directory(exp):
                for batch_start in range(0, len(imgs), batch_sz):
                    batch = imgs[batch_start:batch_start + batch_sz]
                    for j_rel, img_path in enumerate(batch):
                        j = batch_start + j_rel
                        try:
                            with directory(f"inversions/{j:04d}"):
                                print(f"- {j:04d}")
                                ground_truth = open_image(img_path, config.resolution)
                                degradation  = task.init_degradation()
                                save_image(ground_truth, "ground_truth.png")
                                target = degradation.degrade_ground_truth(ground_truth)
                                save_image(target, "target.png")

                                W_var = WVariable.sample_from(G)
                                scores[key]["W"].append(run_phase("W", W_var, lr_W, NGD))

                                Wp_var = WpVariable.from_W(W_var)
                                scores[key]["W+"].append(run_phase("W+", Wp_var, lr_Wp, torch.optim.Adam))

                                Wpp_var = WppVariable.from_Wp(Wp_var)
                                scores[key]["W++"].append(run_phase("W++", Wpp_var, lr_Wpp, LBFGSPhase))

                                s = scores[key]
                                print(f"  W   PSNR {s['W'][-1]['PSNR']:5.2f} dB  LPIPS {s['W'][-1]['LPIPS']:.4f}"
                                      f" | W+  PSNR {s['W+'][-1]['PSNR']:5.2f} dB  LPIPS {s['W+'][-1]['LPIPS']:.4f}"
                                      f" | W++ PSNR {s['W++'][-1]['PSNR']:5.2f} dB  LPIPS {s['W++'][-1]['LPIPS']:.4f}")
                        except Exception as e: print(f"  [SKIP] {j:04d} — {e}")

                    # ── Batch summary ─────────────────────────────────────────
                    n_done = min(batch_start + batch_sz, len(imgs))
                    print(f"\n--- Batch {batch_start//batch_sz + 1} summary  "
                          f"(imgs {batch_start}–{n_done-1}, cumulative n={n_done}) ---")
                    for ph in ["W","W+","W++"]:
                        e = scores[key][ph]
                        if e: print(f"  {ph:<4} avg PSNR {_mean([s['PSNR'] for s in e]):5.2f} dB"
                                    f"  avg LPIPS {_mean([s['LPIPS'] for s in e]):.4f}"
                                    f"  (n={len(e)})")
                    print()

            preds = [p for p in [os.path.join(abs_exp,"inversions",f"{j:04d}","pred_W++.png")
                                  for j in range(len(imgs))] if os.path.exists(p)]
            if preds:
                pfid = compute_pfid(preds, abs_exp)
                scores[key]["pFID"] = pfid
                print(f"  pFID: {pfid:.2f}" if pfid else "  pFID: failed")
        except Exception as e: print(f"  [SKIP task] {key} — {e}")

        for ph in ["W","W+","W++"]:
            e = scores[key][ph]
            if e: print(f"  {ph}  PSNR {_mean([s['PSNR'] for s in e]):5.2f}  LPIPS {_mean([s['LPIPS'] for s in e]):.4f}")

    print("\n\n=== SEARCH C (Low LRs) FINAL SCORES ===")
    print_table(scores)
