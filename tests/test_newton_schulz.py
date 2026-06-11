"""Newton-Schulz orthogonalization: parity with the reference to <= 1e-4,
plus mathematical sanity (orthogonality, singular-vector preservation,
transpose equivariance)."""

import pytest
import torch

from muon_bench import newton_schulz_orthogonalize
from tests.reference_muon import zeropower_via_newtonschulz5

TOL = 1e-4  # the headline claim: from-scratch impl matches the reference to 1e-4

# Shapes that occur in ResNet-9 (conv filters flattened to 2D, head, stem)
# plus square / tall / wide edge cases.
SHAPES = [
    (64, 64),
    (64, 27),       # stem conv 3x3x3 -> 64, flattened
    (128, 1152),    # conv 128 <- 128*3*3
    (512, 4608),    # conv 512 <- 512*3*3
    (10, 512),      # classifier head
    (512, 10),      # tall version of the head
    (256, 64),
    (3, 3),
]


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("steps", [1, 5, 10])
def test_matches_reference_bf16(shape, steps):
    """Default (bfloat16) path agrees with the reference implementation."""
    g = torch.randn(*shape, generator=torch.Generator().manual_seed(hash(shape) % 2**31))
    ours = newton_schulz_orthogonalize(g, steps=steps)
    ref = zeropower_via_newtonschulz5(g, steps=steps)
    assert ours.shape == g.shape
    assert ours.dtype == torch.bfloat16
    assert max_diff(ours, ref) <= TOL


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_reference_across_scales(shape):
    """Parity holds regardless of the input's overall scale (the iteration
    normalizes by the Frobenius norm first)."""
    gen = torch.Generator().manual_seed(0)
    for scale in (1e-4, 1.0, 1e4):
        g = torch.randn(*shape, generator=gen) * scale
        assert max_diff(
            newton_schulz_orthogonalize(g), zeropower_via_newtonschulz5(g, steps=5)
        ) <= TOL


def test_matches_reference_float32():
    """Parity also holds when the iteration is run in float32."""
    g = torch.randn(128, 512, generator=torch.Generator().manual_seed(1))
    ours = newton_schulz_orthogonalize(g, dtype=torch.float32)
    # Reference is bf16-only; rerun its exact op sequence in fp32.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = g.float()
    X = X / (X.norm() + 1e-7)
    for _ in range(5):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    assert max_diff(ours, X) <= TOL


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("shape", [(64, 64), (128, 1152), (512, 10)])
def test_matches_reference_bf16_cuda(shape):
    """Parity on the GPU (the dtype/device combination used in training)."""
    g = torch.randn(*shape, generator=torch.Generator().manual_seed(2)).cuda()
    ours = newton_schulz_orthogonalize(g)
    ref = zeropower_via_newtonschulz5(g, steps=5)
    assert max_diff(ours, ref) <= TOL


def _well_conditioned(m, n, s_lo=0.5, s_hi=1.0, seed=3):
    """Matrix with known singular values uniform in [s_lo, s_hi]."""
    gen = torch.Generator().manual_seed(seed)
    k = min(m, n)
    u, _ = torch.linalg.qr(torch.randn(m, k, generator=gen, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(n, k, generator=gen, dtype=torch.float64))
    s = torch.linspace(s_lo, s_hi, k, dtype=torch.float64)
    return (u * s) @ v.mT


@pytest.mark.parametrize("shape", [(64, 64), (64, 256), (256, 64)])
def test_orthogonalizes(shape):
    """After 5 iterations on a well-conditioned input, all singular values of
    the output are close to 1 (the documented [~0.68, ~1.13] band)."""
    g = _well_conditioned(*shape).float()
    out = newton_schulz_orthogonalize(g, dtype=torch.float32).float()
    s = torch.linalg.svdvals(out)
    assert s.min().item() > 0.6
    assert s.max().item() < 1.25


def test_preserves_singular_vectors():
    """The output approximates the orthogonal polar factor U V^T of the input
    (the iteration only reshapes singular values, never rotates directions)."""
    g = _well_conditioned(96, 192)
    u, _, vh = torch.linalg.svd(g, full_matrices=False)
    polar = (u @ vh).float()
    out = newton_schulz_orthogonalize(g.float(), dtype=torch.float32).float()
    rel_err = (out - polar).norm() / polar.norm()
    assert rel_err.item() < 0.25


def test_transpose_equivariance():
    """Ortho(G^T) == Ortho(G)^T (guaranteed by the tall-matrix transpose trick)."""
    g = torch.randn(48, 160, generator=torch.Generator().manual_seed(4))
    a = newton_schulz_orthogonalize(g.T)
    b = newton_schulz_orthogonalize(g).T
    assert max_diff(a, b) <= TOL


def test_zero_input_is_fixed_point():
    """NS(0) = 0 -- the eps guard prevents division by zero."""
    out = newton_schulz_orthogonalize(torch.zeros(32, 64))
    assert torch.all(out == 0)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        newton_schulz_orthogonalize(torch.randn(8))
    with pytest.raises(ValueError):
        newton_schulz_orthogonalize(torch.randn(2, 3, 4))
    with pytest.raises(ValueError):
        newton_schulz_orthogonalize(torch.randn(4, 4), steps=0)
