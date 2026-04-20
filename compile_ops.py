"""
Run this once before run.py to pre-compile the custom CUDA ops.
Usage: python compile_ops.py
"""
import subprocess
import sys
import os

# Install ninja (required for JIT compilation of CUDA extensions)
subprocess.check_call([sys.executable, "-m", "pip", "install", "ninja", "-q"])

# Add stylegan2_ada to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stylegan2_ada"))

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Force compile bias_act plugin
print("\nCompiling bias_act CUDA kernel...")
from torch_utils.ops import bias_act
bias_act._init()
if bias_act._plugin is not None:
    print("bias_act_plugin compiled successfully.")
else:
    print("bias_act_plugin failed to compile — will use reference implementation.")

# Force compile upfirdn2d plugin
print("\nCompiling upfirdn2d CUDA kernel...")
from torch_utils.ops import upfirdn2d
upfirdn2d._init()
if upfirdn2d._plugin is not None:
    print("upfirdn2d_plugin compiled successfully.")
else:
    print("upfirdn2d_plugin failed to compile — will use reference implementation.")

print("\nDone. You can now run: python run.py --dataset_path datasets/samples")
