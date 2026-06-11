"""GPU-resident CIFAR-10 augmentation pipeline.

Design: the entire dataset lives on the GPU as uint8 for the whole run, and
every augmentation (random crop, horizontal flip, cutout, normalization) is
a batched tensor op executed on-device in a few hundred microseconds. There
are no DataLoader workers, no CPU->GPU copies per batch, no PIL, and no
per-sample Python loops.

Why: on small models like ResNet-9 a conventional CPU dataloader is the
bottleneck -- the GPU finishes a step faster than the CPU can decode and
augment the next batch, so any *optimizer* overhead disappears into data
stalls. With the pipeline GPU-resident, step time is pure compute, which is
what lets this repo measure Muon's per-step overhead directly and keep the
GPU saturated.

Memory budget (uint8, batch 512):
    train images padded to 40x40:  50,000 * 3 * 40 * 40 = 229 MB
    test images 32x32:             10,000 * 3 * 32 * 32 =  30 MB
    labels:                                              < 1 MB
    ----------------------------------------------------------
    total pipeline residency:                           ~260 MB

which fits comfortably alongside the ~6.5M-param model and its activations
within an 8 GB VRAM budget.
"""

from __future__ import annotations

import os
from typing import Iterator, Tuple

import torch
import torch.nn.functional as F

__all__ = ["CIFAR10_MEAN", "CIFAR10_STD", "load_cifar10", "GPUCifar10"]

# Standard CIFAR-10 channel statistics (computed on the training set, [0,1] scale).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(root: str = "./data") -> Tuple[torch.Tensor, ...]:
    """Download (once) and load CIFAR-10 as CPU uint8 tensors.

    Returns:
        train_x (50000,3,32,32) uint8, train_y (50000,) int64,
        test_x  (10000,3,32,32) uint8, test_y  (10000,) int64
    """
    from torchvision.datasets import CIFAR10  # imported lazily; only needed here

    os.makedirs(root, exist_ok=True)
    train = CIFAR10(root=root, train=True, download=True)
    test = CIFAR10(root=root, train=False, download=True)

    # .data is (N, 32, 32, 3) uint8 numpy -> (N, 3, 32, 32) torch uint8
    train_x = torch.from_numpy(train.data).permute(0, 3, 1, 2).contiguous()
    test_x = torch.from_numpy(test.data).permute(0, 3, 1, 2).contiguous()
    train_y = torch.tensor(train.targets, dtype=torch.int64)
    test_y = torch.tensor(test.targets, dtype=torch.int64)
    return train_x, train_y, test_x, test_y


class GPUCifar10:
    """Holds CIFAR-10 on-device and serves augmented batches with tensor ops.

    Train-time augmentation (the standard 94%-recipe):
      * pad 4 -> random 32x32 crop  (zero padding, done once up front:
        images are stored pre-padded at 40x40 so a crop is a pure gather)
      * random horizontal flip (p=0.5)
      * normalize to zero mean / unit variance, cast to `dtype` (bf16)
      * cutout: one `cutout x cutout` square erased per image (post-
        normalization, filled with 0 = the channel mean)

    Test batches are only normalized + cast.

    All randomness is drawn from a private `torch.Generator` on `device`,
    so epochs are reproducible given a seed and independent of global RNG use
    elsewhere in the program.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        test_x: torch.Tensor,
        test_y: torch.Tensor,
        device: torch.device | str = "cuda",
        batch_size: int = 512,
        pad: int = 4,
        cutout: int = 8,
        flip: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        seed: int = 0,
        channels_last: bool = True,
    ):
        assert train_x.dtype == torch.uint8 and test_x.dtype == torch.uint8
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.pad = pad
        self.cutout = cutout
        self.flip = flip
        self.dtype = dtype
        self.channels_last = channels_last
        self.crop_size = train_x.size(-1)  # 32

        # Pre-pad the whole training set once; a random crop then becomes a
        # cheap advanced-indexing gather with per-image offsets.
        self.train_x = F.pad(train_x, (pad, pad, pad, pad)).to(self.device, non_blocking=True)
        self.train_y = train_y.to(self.device, non_blocking=True)
        self.test_x = test_x.to(self.device, non_blocking=True)
        self.test_y = test_y.to(self.device, non_blocking=True)

        # Normalization constants pre-scaled to uint8 range so we can go
        # uint8 -> normalized `dtype` in two fused ops: (x - mean) * inv_std.
        mean = torch.tensor(CIFAR10_MEAN, device=self.device).view(1, 3, 1, 1) * 255.0
        std = torch.tensor(CIFAR10_STD, device=self.device).view(1, 3, 1, 1) * 255.0
        self._mean = mean.to(dtype)
        self._inv_std = (1.0 / std).to(dtype)

        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)
        self._arange_crop = torch.arange(self.crop_size, device=self.device)
        self._arange_c = torch.arange(3, device=self.device).view(1, 3, 1, 1)

    # ---------------------------------------------------------------- train

    @property
    def steps_per_epoch(self) -> int:
        return self.train_x.size(0) // self.batch_size  # drop_last

    def _normalize(self, x_uint8: torch.Tensor) -> torch.Tensor:
        x = x_uint8.to(self.dtype)
        x = (x - self._mean) * self._inv_std
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    def train_batches(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """One epoch of shuffled, augmented training batches (drop_last)."""
        n = self.train_x.size(0)
        s = self.crop_size
        perm = torch.randperm(n, device=self.device, generator=self._gen)

        for i in range(self.steps_per_epoch):
            idx = perm[i * self.batch_size : (i + 1) * self.batch_size]
            b = idx.size(0)
            imgs = self.train_x[idx]  # (B, 3, 40, 40) uint8 gather

            # Random crop: per-image (dy, dx) offsets into the padded image,
            # gathered with broadcasted advanced indexing -> (B, 3, 32, 32).
            dy = torch.randint(0, 2 * self.pad + 1, (b,), device=self.device, generator=self._gen)
            dx = torch.randint(0, 2 * self.pad + 1, (b,), device=self.device, generator=self._gen)
            rows = (dy[:, None] + self._arange_crop[None, :]).view(b, 1, s, 1)
            cols = (dx[:, None] + self._arange_crop[None, :]).view(b, 1, 1, s)
            imgs = imgs[torch.arange(b, device=self.device).view(b, 1, 1, 1), self._arange_c, rows, cols]

            # Random horizontal flip.
            if self.flip:
                do_flip = torch.rand(b, device=self.device, generator=self._gen) < 0.5
                imgs = torch.where(do_flip.view(b, 1, 1, 1), imgs.flip(-1), imgs)

            x = self._normalize(imgs)

            # Cutout: erase one square per image. Done after normalization so
            # the fill value 0 equals the per-channel mean.
            if self.cutout > 0:
                half = self.cutout // 2
                cy = torch.randint(0, s, (b,), device=self.device, generator=self._gen)
                cx = torch.randint(0, s, (b,), device=self.device, generator=self._gen)
                r = self._arange_crop
                row_in = (r[None, :] >= (cy - half)[:, None]) & (r[None, :] < (cy + half)[:, None])
                col_in = (r[None, :] >= (cx - half)[:, None]) & (r[None, :] < (cx + half)[:, None])
                mask = row_in.view(b, 1, s, 1) & col_in.view(b, 1, 1, s)
                x.masked_fill_(mask, 0.0)  # in-place: keeps channels-last layout

            yield x, self.train_y[idx]

    # ----------------------------------------------------------------- test

    def test_batches(self, batch_size: int = 1000) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Deterministic, augmentation-free test batches (normalized only)."""
        n = self.test_x.size(0)
        for i in range(0, n, batch_size):
            x = self._normalize(self.test_x[i : i + batch_size])
            yield x, self.test_y[i : i + batch_size]

    # ---------------------------------------------------------------- misc

    def vram_mb(self) -> float:
        """Approximate on-device residency of the pipeline's tensors, in MB."""
        tensors = [self.train_x, self.train_y, self.test_x, self.test_y]
        return sum(t.numel() * t.element_size() for t in tensors) / 2**20
