"""Reference Muon implementation, used ONLY as ground truth in the tests.

This is the reference algorithm from Keller Jordan's Muon repository
(https://github.com/KellerJordan/Muon, MIT license), reproduced faithfully
(single-GPU form, distributed plumbing removed) so the test suite can assert
that this repo's independent implementation in `muon_bench/muon.py` matches
it to <= 1e-4. Do not import this module from library code.
"""

import torch


def zeropower_via_newtonschulz5(G, steps: int):
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization
    of G. Quintic iteration whose coefficients are selected to maximize the
    slope at zero; the iteration produces US'V^T with S' ~ Uniform(0.68, 1.13),
    which empirically does not hurt model performance relative to UV^T.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.flatten(1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class ReferenceMuon(torch.optim.Optimizer):
    """Single-GPU port of the reference Muon optimizer."""

    def __init__(self, params, lr=0.02, weight_decay=0.0, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(
                    p.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                    ns_steps=group["ns_steps"],
                    nesterov=group["nesterov"],
                )
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape).to(p.dtype), alpha=-group["lr"])
        return loss
