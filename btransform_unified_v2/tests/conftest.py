"""CPU-only test discipline for the RIFT implementation."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
