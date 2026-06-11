"""Muon optimizer implemented from scratch in PyTorch.

Muon (MomentUm Orthogonalized by Newton-Schulz) is an optimizer for the
hidden weight *matrices* of a neural network. Each step it:

  1. updates an SGD-style momentum buffer (EMA form) with the raw gradient,
  2. (optionally) applies a Nesterov-style correction,
  3. replaces the resulting update matrix M with an approximate
     orthogonalization Ortho(M) = U V^T, where M = U S V^T is the SVD --
     i.e. it keeps the "directions" of the update but whitens its spectrum,
  4. scales by sqrt(max(1, rows/cols)) and applies decoupled weight decay.

The orthogonalization is computed cheaply on-GPU with a quintic
Newton-Schulz iteration run in bfloat16 (no SVD, no eigendecomposition).

Reference: Keller Jordan et al., "Muon: An optimizer for hidden layers in
neural networks" (https://kellerjordan.github.io/posts/muon/) and the
reference implementation at https://github.com/KellerJordan/Muon (MIT).
This file is an independent from-scratch implementation; the test suite
verifies it matches the reference to <= 1e-4 (see tests/test_newton_schulz.py
and tests/test_muon_step.py).

Muon is intended ONLY for parameters that are weight matrices of hidden
layers (2D linear weights, or conv filters flattened to 2D). Embeddings,
classifier heads, biases and norm gains should be handled by AdamW --
see `split_muon_params` below.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["newton_schulz_orthogonalize", "Muon", "split_muon_params"]


# Coefficients of the quintic iteration X <- a*X + b*(X X^T) X + c*(X X^T)^2 X.
# These are the "tuned for speed" coefficients from the Muon write-up: they
# maximize the slope of the polynomial at zero (so small singular values are
# inflated as fast as possible) at the cost of converging to singular values
# that oscillate in roughly [0.68, 1.13] rather than exactly 1. Empirically
# this does not hurt the optimizer.
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz_orthogonalize(
    grad: Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Approximately orthogonalize a matrix with a quintic Newton-Schulz iteration.

    Given G with reduced SVD G = U S V^T, returns an approximation of U V^T
    (the nearest semi-orthogonal matrix / the orthogonal polar factor).

    How it works: every iterate is of the form p(X X^T) X for a polynomial p,
    so it shares singular vectors with G and only the singular values change.
    After normalizing by the Frobenius norm (which puts every singular value
    in [0, 1]), each iteration applies the odd quintic
        s <- a*s + b*s^3 + c*s^5
    to every singular value s. With `NS_COEFFS` this quintic is tuned so that
    after `steps` applications every singular value lands near 1, i.e.
    S -> I and p(X X^T) X -> U V^T.

    The iteration is matmul-only, so it runs entirely in `dtype`
    (bfloat16 by default) on the GPU -- this is what keeps Muon's per-step
    overhead small. bfloat16 is safe here because the iteration is
    self-correcting: it converges to the same fixed point regardless of small
    perturbations along the way.

    NOTE: the operation order below deliberately mirrors the reference
    implementation (including `b * A + c * A @ A`, which groups as
    (c*A) @ A) so that results in bfloat16 are bit-for-bit reproducible
    against it. The test suite asserts agreement to <= 1e-4.

    Args:
        grad: a 2D tensor (rows x cols). Conv filters must be flattened to 2D
            by the caller (the `Muon` optimizer does this).
        steps: number of Newton-Schulz iterations (5 is the standard choice).
        eps: numerical fuzz added to the Frobenius norm before dividing.
        dtype: compute dtype for the iteration.

    Returns:
        A tensor with the same shape as `grad` (in `dtype`) approximating the
        orthogonal polar factor of `grad`.
    """
    if grad.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {tuple(grad.shape)}")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    a, b, c = NS_COEFFS
    X = grad.to(dtype)

    # The iteration multiplies X by polynomials of A = X X^T (rows x rows).
    # If the matrix is tall, work with its transpose so A is the *smaller*
    # Gram matrix -- cheaper, and mathematically equivalent.
    transposed = grad.size(0) > grad.size(1)
    if transposed:
        X = X.T

    # Normalize so all singular values are <= 1 (||X||_F >= ||X||_2).
    # The quintic only converges to 1 from inside [0, ~1.13]. The norm is
    # computed with the same dim-reduction call as the reference so the
    # bf16 result is bit-for-bit identical to it.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # quintic term; grouping matches the reference
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon: momentum SGD whose update matrix is orthogonalized each step.

    Update rule for a weight matrix W with gradient G:

        buf  <- (1 - momentum) * G + momentum * buf          (EMA momentum)
        M    <- lerp(G, buf, momentum)  if nesterov else buf
        O    <- NewtonSchulz(M.flatten(1) if conv else M)
        W    <- W * (1 - lr * weight_decay)                  (decoupled WD)
        W    <- W - lr * sqrt(max(1, rows/cols)) * O

    The sqrt(rows/cols) factor keeps the per-row update RMS roughly constant
    across differently-shaped layers, so a single learning rate transfers.

    Only pass hidden-layer weight tensors with ndim >= 2 to this optimizer
    (conv filters are flattened internally). Use AdamW for everything else.

    Args:
        params: iterable of 2D/4D weight tensors.
        lr: learning rate (0.02 is a good starting point; this repo's
            CIFAR-10 ResNet-9 benchmark uses a tuned value, see README).
        weight_decay: decoupled weight decay (AdamW-style).
        momentum: EMA momentum coefficient (default 0.95).
        nesterov: apply Nesterov-style lookahead to the momentum (default True).
        ns_steps: Newton-Schulz iterations per step (default 5).
        ns_dtype: compute dtype for Newton-Schulz (default bfloat16).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_dtype: torch.dtype = torch.bfloat16,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"invalid momentum: {momentum}")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_dtype=ns_dtype,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 2:
                    raise ValueError(
                        "Muon only accepts matrix-shaped parameters (ndim >= 2); "
                        f"got a parameter with shape {tuple(p.shape)}. "
                        "Route biases/norm gains to AdamW (see split_muon_params)."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]

                # EMA momentum: buf <- buf + (1-beta) * (grad - buf)
                buf.lerp_(grad, 1.0 - beta)
                # Nesterov lookahead: blend the fresh gradient back in.
                update = torch.lerp(grad, buf, beta) if group["nesterov"] else buf

                # Conv filters (out, in, kh, kw) act as linear maps from the
                # flattened patch space -> orthogonalize them as (out, in*kh*kw).
                update_2d = update.flatten(1) if update.ndim > 2 else update
                update_2d = newton_schulz_orthogonalize(
                    update_2d,
                    steps=group["ns_steps"],
                    dtype=group["ns_dtype"],
                )

                # Shape-aware LR scaling (matches the reference implementation:
                # computed on the *unflattened* gradient shape, and multiplied
                # into the low-precision update -- NOT folded into fp32 alpha --
                # so the bf16 rounding matches the reference bit-for-bit).
                update_2d *= max(1.0, grad.size(-2) / grad.size(-1)) ** 0.5

                p.mul_(1.0 - lr * wd)
                p.add_(update_2d.reshape(p.shape).to(p.dtype), alpha=-lr)

        return loss


def split_muon_params(model: torch.nn.Module):
    """Split a model's parameters into (muon_params, adamw_params).

    Muon gets the hidden weight matrices: every parameter with ndim >= 2
    EXCEPT the first ("stem"/embedding-like) and last ("head") matrix
    parameters of the network, which the Muon authors recommend leaving to
    AdamW. Biases and norm gains (ndim < 2) always go to AdamW.

    Returns:
        (muon_params, adamw_params): two lists of parameters covering every
        trainable parameter of the model exactly once.
    """
    matrix_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    other_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]

    if len(matrix_params) <= 2:
        # Degenerate tiny model: nothing left for Muon after stem+head.
        return [], matrix_params + other_params

    muon_params = matrix_params[1:-1]
    adamw_params = [matrix_params[0], matrix_params[-1]] + other_params
    return muon_params, adamw_params
