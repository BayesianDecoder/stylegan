import copy
import os

import functools
from functools import partial
from time import perf_counter
import sys
import torch.optim as optim
import tqdm
import click
import dataclasses
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from xml.dom import minidom
import shutil
import itertools
from warnings import warn

from torchvision.utils import save_image, make_grid

from abc import ABC, abstractmethod, abstractstaticmethod, abstractclassmethod
from dataclasses import dataclass, field

from typing import Optional, Type, List, final, Tuple, Callable, Iterator, Iterable, Dict, ClassVar, Union, Any

try:
    from torchvision.io import write_video
except ImportError:
    write_video = None

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
