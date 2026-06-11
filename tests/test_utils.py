"""ModelEMA: shadow math, swap-in/out integrity, buffer handling."""

import torch
from torch import nn

from muon_bench import ModelEMA


def small_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4))


def test_shadow_initialized_to_model():
    m = small_model()
    ema = ModelEMA(m, decay=0.9)
    for s, t in zip(ema.shadow, ModelEMA._tensors(m)):
        assert torch.equal(s, t)


def test_update_is_lerp():
    m = small_model()
    ema = ModelEMA(m, decay=0.9)
    old = [t.detach().clone() for t in ModelEMA._tensors(m)]
    with torch.no_grad():
        for t in ModelEMA._tensors(m):
            t.add_(1.0)
    ema.update(m)
    for s, o, t in zip(ema.shadow, old, ModelEMA._tensors(m)):
        assert torch.allclose(s, 0.9 * o + 0.1 * t, atol=1e-6)


def test_swap_roundtrip_restores_weights():
    m = small_model()
    ema = ModelEMA(m, decay=0.5)
    with torch.no_grad():
        for t in ModelEMA._tensors(m):
            t.mul_(2.0).add_(0.3)
    ema.update(m)
    live_before = [t.detach().clone() for t in ModelEMA._tensors(m)]
    ema.swap_in(m)
    for s, t in zip(ema.shadow, ModelEMA._tensors(m)):
        assert torch.equal(s, t)  # model now holds the EMA weights
    ema.swap_out(m)
    for b, t in zip(live_before, ModelEMA._tensors(m)):
        assert torch.equal(b, t)  # original weights restored exactly


def test_int_buffers_excluded():
    m = small_model()
    ema = ModelEMA(m)
    # BatchNorm's num_batches_tracked (int64) must not be EMA-tracked.
    assert all(t.dtype.is_floating_point for t in ema.shadow)
    # params (2: conv W, BN gain+bias) + float buffers (running mean/var)
    assert len(ema.shadow) == 3 + 2
