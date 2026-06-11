"""ResNet-9: architecture invariants (shape, parameter count, layer count,
logit scaling)."""

import torch
from torch import nn

from muon_bench import ResNet9

# Exact parameter budget of the standard CIFAR-10 ResNet-9:
#   conv/linear weights: 6,568,640   BN gains+biases: 4,480
EXPECTED_PARAMS = 6_573_120


def test_forward_shape():
    model = ResNet9()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)
    assert torch.isfinite(out).all()


def test_param_count():
    model = ResNet9()
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMS


def test_nine_weight_layers():
    model = ResNet9()
    weighted = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    assert len(weighted) == 9
    assert all(m.bias is None for m in weighted)  # all biases live in BN


def test_logit_scale():
    a = ResNet9(logit_scale=0.125)
    b = ResNet9(logit_scale=0.25)
    b.load_state_dict(a.state_dict())
    a.eval(), b.eval()
    x = torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.allclose(b(x), 2 * a(x), atol=1e-5)


def test_residual_blocks_are_identity_plus_f():
    """Zeroing a residual branch's final BN gain must make it an identity."""
    model = ResNet9()
    res = model.layer1[1]
    nn.init.zeros_(res.inner[1][1].weight)  # second block's BN gain
    res.eval()
    x = torch.randn(2, 128, 16, 16)
    with torch.no_grad():
        assert torch.allclose(res(x), x)
