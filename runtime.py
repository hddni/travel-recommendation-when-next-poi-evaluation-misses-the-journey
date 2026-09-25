"""Shared runtime settings for exp1 / exp2: seed list, automatic GPU selection, seed averaging.

GPU
    The index is not fixed. If CUDA_VISIBLE_DEVICES is already set it is followed as-is;
    otherwise nvidia-smi is used to pick the one GPU with the most free memory. Without a
    GPU it runs on CPU. The choice must be made before CUDA is initialized, so nvidia-smi
    is used instead of torch.
"""

import os
import subprocess

import numpy as np

SEEDS = list(range(556, 566))          # 556 ~ 565, 10 seeds


def parse_seeds(text):
    """'556,557' or '556-565' -> [int]. Default SEEDS if empty."""
    if not text:
        return list(SEEDS)
    out = []
    for part in str(text).replace(" ", ",").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out


def pick_gpu():
    """Value for CUDA_VISIBLE_DEVICES. Empty string (CPU) if there is no GPU."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return env
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True)
        rows = [line.split(",") for line in res.stdout.strip().splitlines() if line]
        best = max(rows, key=lambda r: int(r[1]))[0].strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""
    #  Align the index ordering of nvidia-smi and CUDA.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return best


def use_device():
    """Pick a GPU, set CUDA_VISIBLE_DEVICES, and return the torch device string."""
    os.environ["CUDA_VISIBLE_DEVICES"] = pick_gpu()
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def mean_std(runs):
    """List of per-seed metric dicts -> (mean dict, std dict). Handles nested dicts too.

    Non-numeric values (None, etc.) keep the first seed's value as-is.
    """
    first = runs[0]
    mean, std = {}, {}
    for k, v in first.items():
        vals = [r.get(k) for r in runs]
        if isinstance(v, dict):
            mean[k], std[k] = mean_std(vals)
        elif isinstance(v, bool) or not isinstance(v, (int, float)) \
                or any(x is None for x in vals):
            mean[k] = v
        else:
            arr = np.asarray(vals, dtype=np.float64)
            mean[k] = round(float(np.nanmean(arr)), 6)
            std[k] = round(float(np.nanstd(arr)), 6)
    return mean, std
