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

    n_images: Optional[int] = None
    "Limit the number of images to process."

    steps: int = 150
    "Number of optimization steps per phase."


def parse_config() -> Config:
    return tyro.cli(Config)
