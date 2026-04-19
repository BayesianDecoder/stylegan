import tyro
from dataclasses import dataclass
from typing import * 

import sys 
sys.path.append("stylegan2_ada")


@dataclass
class Config:
    name: str = f"restored_samples"
    "A name used to group log files."

    pkl_path: str = "pretrained_networks/ffhq.pkl"
    "The location of the pretrained StyleGAN."

    dataset_path: str = "datasets/FFHQ-X"
    "The location of the images to process."
    
    resolution: int = 1024
    "The resolution of your images. Images which are smaller or larger will be resized."

    global_lr_scale: float = 1.0
    "A global factor which scales up and down all learning rates. This may need adjustment for datasets other than faces."

    tasks: Literal["all", "single", "composed", "custom"] = "all"
    "Selects which tasks to run."

    task_name: Optional[str] = None
    "Filter to a specific task e.g. upsampling, denoising, deartifacting, inpainting."

    levels: Optional[str] = None
    "Comma-separated levels to run e.g. M or XL,L,M."

    start_idx: int = 0
    "Index of the first image to process (for batched runs)."

    n_images: Optional[int] = None
    "Number of images to process starting from start_idx."

    steps: int = 150
    "Number of optimization steps per phase."

    # ── Adaptive LR scales for NGD → Adam → LBFGS (run_v1) ───────────────────
    # Each scale multiplies the base LR for that optimizer at that severity level.
    # NGD is always fixed (gradient normalisation makes it scale-invariant).
    # Vary these across the 3 parallel Kaggle runs to find the best combination.

    adam_scale_xs:  float = 1.00
    "Adam LR scale for XS degradation level."
    adam_scale_s:   float = 0.80
    "Adam LR scale for S degradation level."
    adam_scale_m:   float = 0.60
    "Adam LR scale for M degradation level."
    adam_scale_l:   float = 0.40
    "Adam LR scale for L degradation level."
    adam_scale_xl:  float = 0.20
    "Adam LR scale for XL degradation level."

    lbfgs_scale_xs: float = 1.00
    "LBFGS LR scale for XS degradation level."
    lbfgs_scale_s:  float = 0.80
    "LBFGS LR scale for S degradation level."
    lbfgs_scale_m:  float = 0.50
    "LBFGS LR scale for M degradation level."
    lbfgs_scale_l:  float = 0.30
    "LBFGS LR scale for L degradation level."
    lbfgs_scale_xl: float = 0.10
    "LBFGS LR scale for XL degradation level."


def parse_config() -> Config:
    return tyro.cli(Config)
