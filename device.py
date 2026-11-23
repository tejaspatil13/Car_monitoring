"""Pick the best available device: NVIDIA GPU > Apple GPU > CPU."""

import torch


def best_device(requested="auto"):
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
