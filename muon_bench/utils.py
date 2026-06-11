"""Shared utilities: seeding, timing, environment capture."""

from __future__ import annotations

import json
import os
import random
import time

import torch

__all__ = ["set_seed", "CudaBlockTimer", "ModelEMA", "env_info", "save_json"]


def set_seed(seed: int) -> None:
    """Seed Python, (optionally) NumPy, and torch RNGs."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CudaBlockTimer:
    """Accumulates wall-clock time over training blocks, synchronizing the
    GPU only at block boundaries (per-step synchronization would serialize
    the CUDA stream and distort the measurement).

    Usage:
        timer = CudaBlockTimer(device)
        timer.start()
        ... run N training steps (async) ...
        timer.stop()            # syncs, adds elapsed to .total
        ... evaluation (not timed) ...
        timer.start()
    """

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self.total = 0.0
        self._t0 = None

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self) -> None:
        self._sync()
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        self._sync()
        self.total += time.perf_counter() - self._t0
        self._t0 = None


class ModelEMA:
    """Exponential moving average of a model's parameters and float buffers.

    Standard CIFAR-10 speedrun ingredient: the EMA weights are evaluated
    instead of (well, here: alongside) the raw weights, which both smooths
    eval noise and adds a few tenths of accuracy late in training. Applied
    identically to every optimizer, so benchmark comparisons stay fair.

    Usage:
        ema = ModelEMA(model, decay=0.995)
        ... after every optimizer step:  ema.update(model)
        ... to evaluate:  ema.swap_in(model); evaluate(model); ema.swap_out(model)
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"invalid EMA decay: {decay}")
        self.decay = decay
        self.shadow = [t.detach().clone() for t in self._tensors(model)]
        self._backup = None

    @staticmethod
    def _tensors(model: torch.nn.Module):
        # float buffers = BatchNorm running stats; the int step counters are
        # irrelevant at eval time (BN momentum is fixed).
        return list(model.parameters()) + [
            b for b in model.buffers() if b.dtype.is_floating_point
        ]

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for s, t in zip(self.shadow, self._tensors(model)):
            s.lerp_(t.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model: torch.nn.Module) -> None:
        live = self._tensors(model)
        self._backup = [t.detach().clone() for t in live]
        for t, s in zip(live, self.shadow):
            t.copy_(s)

    @torch.no_grad()
    def swap_out(self, model: torch.nn.Module) -> None:
        for t, b in zip(self._tensors(model), self._backup):
            t.copy_(b)
        self._backup = None


def env_info(device: torch.device | str) -> dict:
    device = torch.device(device)
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "platform": os.name,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        info["gpu"] = props.name
        info["vram_gb"] = round(props.total_memory / 2**30, 2)
        info["cuda"] = torch.version.cuda
    return info


def save_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
