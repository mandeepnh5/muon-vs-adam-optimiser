"""Full Muon optimizer: trajectory parity with the reference port to <= 1e-4,
weight decay semantics, parameter routing, and input validation."""

import pytest
import torch

from muon_bench import Muon, split_muon_params
from muon_bench.resnet9 import ResNet9
from tests.reference_muon import ReferenceMuon

TOL = 1e-4

# A mix of linear-shaped (2D) and conv-shaped (4D) parameters, wide and tall.
PARAM_SHAPES = [(32, 16), (16, 64), (16, 8, 3, 3), (40, 40)]


def make_params(seed):
    gen = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(*s, generator=gen) * 0.1) for s in PARAM_SHAPES]


def synthetic_grads(step, seed=123):
    gen = torch.Generator().manual_seed(seed + 1000 * step)
    return [torch.randn(*s, generator=gen) * 0.05 for s in PARAM_SHAPES]


@pytest.mark.parametrize("nesterov", [True, False])
@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
def test_trajectory_matches_reference(nesterov, weight_decay):
    """Run 10 optimization steps with identical gradients through our Muon and
    the reference; parameters must stay within 1e-4 elementwise throughout."""
    ours_params = make_params(seed=7)
    ref_params = make_params(seed=7)

    kwargs = dict(lr=0.02, weight_decay=weight_decay, momentum=0.95, nesterov=nesterov, ns_steps=5)
    ours = Muon(ours_params, **kwargs)
    ref = ReferenceMuon(ref_params, **kwargs)

    for step in range(10):
        grads = synthetic_grads(step)
        # The reference mutates p.grad in place (nesterov lerp), so each
        # optimizer gets its own clone.
        for p, g in zip(ours_params, grads):
            p.grad = g.clone()
        for p, g in zip(ref_params, grads):
            p.grad = g.clone()
        ours.step()
        ref.step()
        for i, (a, b) in enumerate(zip(ours_params, ref_params)):
            diff = (a.detach() - b.detach()).abs().max().item()
            assert diff <= TOL, f"param {i} diverged at step {step}: max diff {diff:.2e}"


def test_zero_grad_only_applies_weight_decay():
    """With zero gradients the orthogonalized update is zero, so a step must
    reduce to pure decoupled weight decay: p <- p * (1 - lr*wd)."""
    p = torch.nn.Parameter(torch.randn(24, 24, generator=torch.Generator().manual_seed(0)))
    before = p.detach().clone()
    opt = Muon([p], lr=0.1, weight_decay=0.5)
    p.grad = torch.zeros_like(p)
    opt.step()
    expected = before * (1 - 0.1 * 0.5)
    assert torch.allclose(p.detach(), expected, atol=1e-7)


def test_update_is_orthogonal():
    """The applied update (ignoring weight decay) must be approximately
    semi-orthogonal: its singular values should sit near 1."""
    p = torch.nn.Parameter(torch.zeros(64, 96))
    opt = Muon([p], lr=1.0, weight_decay=0.0, momentum=0.0, nesterov=False)
    gen = torch.Generator().manual_seed(5)
    p.grad = torch.randn(64, 96, generator=gen)
    opt.step()
    update = -p.detach()  # started from zero, lr=1, scale=max(1,64/96)**0.5=1
    s = torch.linalg.svdvals(update.float())
    assert s.min().item() > 0.3
    assert s.max().item() < 1.35


def test_momentum_buffer_is_ema():
    """buf <- (1-beta)*g for the first step from a zero buffer."""
    p = torch.nn.Parameter(torch.zeros(8, 8))
    opt = Muon([p], lr=0.01, momentum=0.9)
    g = torch.randn(8, 8, generator=torch.Generator().manual_seed(6))
    p.grad = g.clone()
    opt.step()
    buf = opt.state[p]["momentum_buffer"]
    assert torch.allclose(buf, 0.1 * g, atol=1e-6)


def test_rejects_vector_params():
    bias = torch.nn.Parameter(torch.zeros(10))
    with pytest.raises(ValueError):
        Muon([bias])


def test_split_muon_params_resnet9():
    """Stem conv + head linear + all 1D params go to AdamW; the 7 hidden conv
    weight matrices go to Muon; every parameter is covered exactly once."""
    model = ResNet9()
    muon_params, adamw_params = split_muon_params(model)

    assert len(muon_params) == 7
    assert all(p.ndim == 4 for p in muon_params)

    all_ids = {id(p) for p in model.parameters()}
    routed = [id(p) for p in muon_params + adamw_params]
    assert len(routed) == len(all_ids)
    assert set(routed) == all_ids

    stem = model.prep[0].weight
    head = model.head[2].weight
    adamw_ids = {id(p) for p in adamw_params}
    assert id(stem) in adamw_ids and id(head) in adamw_ids
    assert all(id(p) in adamw_ids for p in model.parameters() if p.ndim < 2)


def test_state_dict_roundtrip():
    """Optimizer state (momentum buffers) survives save/load."""
    params = make_params(seed=9)
    opt = Muon(params, lr=0.02)
    for p, g in zip(params, synthetic_grads(0)):
        p.grad = g.clone()
    opt.step()
    state = opt.state_dict()

    params2 = make_params(seed=9)
    opt2 = Muon(params2, lr=0.02)
    opt2.load_state_dict(state)
    bufs1 = [opt.state[p]["momentum_buffer"] for p in params]
    bufs2 = [opt2.state[p]["momentum_buffer"] for p in params2]
    for b1, b2 in zip(bufs1, bufs2):
        assert torch.equal(b1, b2)
