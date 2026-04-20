"""
Display all eval scores from the output directory.
Usage: python show_scores.py out/restored_samples
"""
import os
import sys
import json
import glob


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_scores(base_path):
    results = []

    for gt_score_path in sorted(glob.glob(f"{base_path}/**/ground_truth_scores*.json", recursive=True)):
        expr_path = os.path.dirname(gt_score_path)

        # Parse task/level from path
        parts = gt_score_path.replace("\\", "/").split("/")
        try:
            level = parts[-2]
            task  = parts[-3]
            category = parts[-4]
        except IndexError:
            task = level = category = "unknown"

        gt_scores = load_json(gt_score_path)

        # Degraded scores (how bad the input is)
        deg_score_path = gt_score_path.replace("ground_truth_scores", "degraded_scores")
        deg_scores = load_json(deg_score_path) if os.path.exists(deg_score_path) else {}

        # FID score
        fid_score = None
        fid_paths = glob.glob(f"{expr_path}/fid*.json")
        if fid_paths:
            fid_score = load_json(fid_paths[0])

        results.append({
            "category": category,
            "task": task,
            "level": level,
            "gt_scores": gt_scores,
            "deg_scores": deg_scores,
            "fid": fid_score,
        })

    return results


def display(results):
    if not results:
        print("No score files found. Make sure you ran: python -m benchmark.eval <out_path>")
        return

    # Header
    print(f"\n{'='*80}")
    print(f"{'TASK':<20} {'LEVEL':<8} {'LPIPS (restored)':<20} {'PSNR (restored)':<20} {'pFID':<12}")
    print(f"{'':20} {'':8} {'LPIPS (degraded)':<20} {'PSNR (degraded)':<20}")
    print(f"{'='*80}")

    for r in results:
        gt  = r["gt_scores"]
        deg = r["deg_scores"]
        fid = r["fid"]

        lpips_gt  = gt.get("LPIPS",  gt.get("LPIS",  None))
        psnr_gt   = gt.get("PSNR",   gt.get("PSNR",  None))
        lpips_deg = deg.get("LPIPS", deg.get("LPIS",  None))
        psnr_deg  = deg.get("PSNR",  deg.get("PSNR",  None))

        lpips_gt_str  = f"{lpips_gt:.4f}"  if lpips_gt  is not None else "N/A"
        psnr_gt_str   = f"{psnr_gt:.2f}"   if psnr_gt   is not None else "N/A"
        lpips_deg_str = f"{lpips_deg:.4f}" if lpips_deg is not None else "N/A"
        psnr_deg_str  = f"{psnr_deg:.2f}"  if psnr_deg  is not None else "N/A"
        fid_str       = f"{fid:.2f}"       if fid       is not None else "N/A"

        print(f"{r['task']:<20} {r['level']:<8} {lpips_gt_str:<20} {psnr_gt_str:<20} {fid_str:<12}")
        print(f"{'':20} {'':8} {lpips_deg_str:<20} {psnr_deg_str:<20}")
        print(f"{'-'*80}")

    # Summary averages
    all_lpips = [r["gt_scores"].get("LPIPS") for r in results if r["gt_scores"].get("LPIPS") is not None]
    all_psnr  = [r["gt_scores"].get("PSNR")  for r in results if r["gt_scores"].get("PSNR")  is not None]
    all_fid   = [r["fid"] for r in results if r["fid"] is not None]

    if all_lpips:
        print(f"\nAverage LPIPS (restored vs ground truth): {sum(all_lpips)/len(all_lpips):.4f}  [lower = better]")
    if all_psnr:
        print(f"Average PSNR  (restored vs ground truth): {sum(all_psnr)/len(all_psnr):.2f} dB  [higher = better]")
    if all_fid:
        print(f"Average pFID:                             {sum(all_fid)/len(all_fid):.2f}  [lower = better]")
    print()


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "out/restored_samples"
    results = find_scores(base)
    display(results)
