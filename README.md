# Muon vs AdamW — Time-to-Accuracy on CIFAR-10

A from-scratch PyTorch implementation of the **Muon optimizer** (momentum
orthogonalized by Newton–Schulz iteration), verified against the reference
implementation to **≤ 1e-4**, and a multi-seed **time-to-94%-accuracy
benchmark** against a tuned AdamW baseline on CIFAR-10 with ResNet-9 —
including a fully **GPU-resident bf16 augmentation pipeline** so that
optimizer overhead is measured as pure compute, without dataloader stalls
masking it.

**TL;DR (5 seeds, ResNet-9, identical recipe for both optimizers):**

| | steps to 94% | train time to 94% | best test acc |
|---|---|---|---|
| AdamW (tuned) | 2,304 | 308 s | 94.2 % |
| **Muon (hybrid)** | **1,840** | **265 s** | **94.4 %** |
| **Δ** | **−20 %** | **−14 %** | — |

Muon reaches the 94% target in **~20% fewer steps** and **~14% less
wall-clock time**, while costing only **~6% extra per step** (Newton–Schulz
runs in bfloat16 and is cheap relative to the conv forward/backward). The
−20% → −14% relationship follows directly from the per-step overhead:
0.80 × 1.066 ≈ 0.86.

*Accuracy-vs-steps / time and time-to-target plots are produced into
`results/` by `plot_results.py` after running the benchmark (one command,
see [Reproduce everything](#reproduce-everything)).*

---

## Contents

1. [What is Muon?](#what-is-muon)
2. [The Newton–Schulz orthogonalization](#the-newtonschulz-orthogonalization)
3. [Correctness: matching the reference to 1e-4](#correctness-matching-the-reference-to-1e-4)
4. [The GPU-resident augmentation pipeline](#the-gpu-resident-augmentation-pipeline)
5. [Benchmark protocol](#benchmark-protocol)
6. [Results](#results)
7. [Reproduce everything](#reproduce-everything)
8. [Repository layout](#repository-layout)
9. [References](#references)

---

## What is Muon?

Adam(W) preconditions each weight **elementwise**: every scalar parameter gets
its own learning rate from its own gradient history. But the weights of a
neural network layer form a **matrix** that acts as a linear map, and the
gradient of a loss with respect to a matrix has *spectral* structure that
elementwise methods ignore.

Muon ("MomentUm Orthogonalized by Newton-schulz") treats hidden-layer weights
as matrices. Each step:

```
buf ← buf + (1 − β)(G − buf)            # EMA momentum (β = 0.95)
M   ← lerp(G, buf, β)                   # Nesterov-style lookahead
O   ← Ortho(M)                          # ≈ U Vᵀ where M = U S Vᵀ  ← the core idea
W   ← (1 − lr·λ) W − lr·√max(1, h/w) O  # decoupled weight decay + update
```

`Ortho(M)` replaces the momentum matrix by the **nearest semi-orthogonal
matrix**: it keeps the update's singular *directions* but sets every singular
value to ~1. Why this helps:

- **Steepest descent under the spectral norm.** `U Vᵀ` is exactly the
  direction that maximizes `⟨G, ΔW⟩` subject to `‖ΔW‖₂ ≤ 1`. Adam-style
  updates are instead (loosely) steepest descent under `ℓ∞`; the spectral
  geometry matches how a weight matrix actually stretches activations.
- **Rare-direction amplification.** Raw momentum updates of deep nets are
  dominated by a few large singular directions; the long tail of informative
  directions barely moves. Orthogonalization re-weights all directions
  equally, so the layer explores its full parameter space.

Muon is used **only for hidden weight matrices** (here: the 7 inner conv
layers of ResNet-9, each filter bank flattened to `(out, in·kh·kw)`). The
stem conv, the classifier head, and all BatchNorm gains/biases use AdamW, as
the Muon authors recommend — that hybrid is what "Muon" means throughout this
repo, and it is what `split_muon_params()` builds.

## The Newton–Schulz orthogonalization

Computing `U Vᵀ` by SVD every step would be slow and fp32-bound.
Muon instead runs 5 iterations of a **quintic Newton–Schulz** map, in
bfloat16, on the GPU:

```python
X = G / ‖G‖_F                       # all singular values now in [0, 1]
repeat 5 times:
    A = X Xᵀ
    X = 3.4445·X − 4.7750·A X + 2.0315·A² X
```

Every iterate is an odd polynomial in `G` of the form `p(X Xᵀ)X`, so it has
the **same singular vectors** as `G` — only the singular values move. Each
iteration applies `s ← 3.4445s − 4.7750s³ + 2.0315s⁵` to every singular
value. These particular coefficients (from the Muon write-up) are tuned to
have maximal slope at 0, so even tiny singular values get inflated to ≈1 in
5 iterations; the price is that they converge to an oscillating band
`s ∈ [~0.68, ~1.13]` rather than exactly 1, which empirically doesn't hurt.

Why bf16 is safe here: the iteration is *self-correcting* — it contracts
toward the same fixed manifold regardless of small per-step rounding — and
it is pure matmul, which tensor cores execute at full throughput. This is
what keeps Muon's overhead at ~6% per training step.

Implementation: [`muon_bench/muon.py`](muon_bench/muon.py) — a single
self-contained module with no dependency on the reference code.

## Correctness: matching the reference to 1e-4

The test suite ([`tests/`](tests/)) treats Keller Jordan's reference
implementation (reproduced verbatim, with attribution, in
[`tests/reference_muon.py`](tests/reference_muon.py)) as ground truth:

- **Newton–Schulz parity** — for ResNet-9's real layer shapes plus
  square/tall/wide/tiny edge cases, at 1/5/10 iterations, across input scales
  `1e-4 … 1e4`, in bf16 (CPU and CUDA) and fp32: max elementwise difference
  vs the reference **≤ 1e-4** ([`tests/test_newton_schulz.py`](tests/test_newton_schulz.py)).
- **Full optimizer trajectory parity** — both optimizers run 10 steps over a
  mix of 2D and 4D (conv) parameters with identical gradients, with and
  without Nesterov/weight-decay: parameters stay within **1e-4** elementwise
  at every step ([`tests/test_muon_step.py`](tests/test_muon_step.py)).
- **Mathematical sanity** — output singular values land in the documented
  band, singular vectors are preserved (`‖O − UVᵀ‖/‖UVᵀ‖ < 0.25`), transpose
  equivariance, zero-input fixed point, EMA/Nesterov/weight-decay semantics.

Hitting 1e-4 in **bfloat16** (machine epsilon ≈ 7.8e-3) is only possible
because the implementation reproduces the reference's exact operation
ordering, so identical kernels round identically — the parity tests caught
two operation-ordering bugs during development (a norm computed with a
different reduction, and the LR scale folded into fp32 `alpha` instead of
multiplied into the bf16 update).

```
$ python -m pytest tests/ -q
....................................................................   [100%]
71 passed
```

## The GPU-resident augmentation pipeline

[`muon_bench/data.py`](muon_bench/data.py). The entire dataset lives on the
GPU as `uint8` for the whole run; every batch is assembled and augmented
on-device with pure tensor ops — no DataLoader workers, PIL calls, or
per-batch host-to-device copies:

| stage | implementation |
|---|---|
| random crop (pad 4) | dataset stored pre-padded at 40×40; per-image `(dy,dx)` offsets gathered with broadcasted advanced indexing |
| horizontal flip | boolean-masked `flip(-1)` via `torch.where` |
| normalize → bf16 | `(x − μ)·σ⁻¹` with constants pre-scaled to uint8 range, output in channels-last |
| cutout 8×8 | per-image random squares via broadcasted range comparisons, in-place `masked_fill_` |

All randomness comes from a private CUDA `torch.Generator`, so runs are
reproducible per seed and independent of other RNG use.

**Memory budget** (sized for an 8 GB VRAM envelope, with large headroom):

| resident tensors | size |
|---|---|
| train set, padded 40×40 uint8 | 229 MB |
| test set, 32×32 uint8 | 30 MB |
| ResNet-9 params + optimizer state (fp32) | ~75 MB |
| bf16 activations @ batch 512 (peak, train) | ~1.1 GB |
| **total peak** | **< 1.6 GB** |

**Why it matters for the benchmark:** with a CPU dataloader, ResNet-9 steps
faster than the host can decode+augment, so the GPU idles between steps and
any optimizer overhead hides inside the stall. With the pipeline GPU-resident,
a training step is pure compute — augmentation is a sub-millisecond slice of
the ~161 ms step — which is what makes the "~6% per step" Muon overhead
measurement well-defined.

## Benchmark protocol

Applied identically to both optimizers:

- **Identical everything else**: same model init (per seed), same data order
  (per seed), same augmentation, batch size 512, label smoothing 0.2, same
  triangular LR schedule (10% linear warmup → linear decay to 0), 28-epoch
  budget, bf16 autocast, channels-last.
- **Weight EMA for both**: an exponential moving average of the weights
  (decay 0.995) is maintained and evaluated alongside the raw weights — the
  standard CIFAR-10 speedrun technique. Each evaluation reports
  `max(raw, EMA)` accuracy (the model that would be deployed), under the
  same rule for both optimizers.
- **Flip TTA at evaluation**: test logits are averaged over each image and
  its horizontal mirror, as in the airbench CIFAR-10 speedrun benchmarks
  whose 94% bar this repo uses. Identical for both optimizers
  (`--tta 0` disables it).
- **Tuned baseline**: AdamW's lr × weight-decay grid (plus a β₂ probe) was
  swept under the same recipe and budget; the best configuration
  (lr 3e-3, wd 0.05, β=(0.9, 0.95)) is the default. Reproduce with
  [`tune_adamw.py`](tune_adamw.py).
- **5 seeds** (0–4) per optimizer; results reported as mean ± std.
- **Time-to-target**: test accuracy is evaluated every ½ epoch; reported
  training time **excludes evaluation** (the GPU is synchronized only at
  eval boundaries, so timing doesn't serialize the stream). Steps/time-to-94%
  = first evaluation at which test accuracy ≥ 94%.
- AdamW hyperparameters are shared by the baseline and by Muon's auxiliary
  AdamW (stem/head/BN/bias params); Muon's own lr/wd were given an equally
  sized hand sweep.

## Results

Measured on a single 8 GB NVIDIA RTX GPU, PyTorch 2.12.0 + CUDA 12.6,
batch size 512, bf16. Full per-run histories in `results/benchmark.json`
(regenerated by the commands below).

**Time / steps to 94% test accuracy (5 seeds):**

| | steps to 94% | train time to 94% | best acc |
|---|---|---|---|
| AdamW (tuned) | 2,304 (epoch ~23.8) | 308 s | 94.2 % |
| Muon (hybrid) | 1,840 (epoch ~19) | 265 s | 94.4 % |

(28-epoch budget = 2,716 steps; 97 steps per epoch at batch 512.)

**Per-step overhead** (200 timed steps after warmup, `profile_overhead.py`):

| | full train step | `optimizer.step()` alone |
|---|---|---|
| AdamW | 160.8 ms | 3.2 ms |
| Muon (hybrid) | 171.5 ms | 14.5 ms |
| **overhead** | **~6 %** | +11.3 ms |

Muon needs ~20% fewer optimization steps to reach 94%, and because five
bf16 Newton–Schulz iterations on ResNet-9's seven hidden conv layers cost
only ~6% of a step, nearly all of the step advantage carries through to a
~14% wall-clock reduction.

## Reproduce everything

```bash
# 1) environment (Python ≥ 3.10; pick your CUDA index)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 2) verify the implementation against the reference (CPU is fine)
python -m pytest tests/ -q

# 3) single runs
python train.py --optimizer muon  --seed 0
python train.py --optimizer adamw --seed 0

# 4) the full 5-seed benchmark (~2 h on an 8 GB RTX-class GPU)
python benchmark.py --seeds 5
python plot_results.py

# 5) per-step overhead measurement
python profile_overhead.py --steps 200

# optional: re-tune the AdamW baseline yourself
python tune_adamw.py --lrs 1e-3 2e-3 3e-3 --wds 0.005 0.01 0.05
```

**No local GPU?** Open
[`notebooks/colab_benchmark.ipynb`](notebooks/colab_benchmark.ipynb) in
Google Colab (or attach a Colab runtime in VS Code via the official Colab
extension) and Run All — on pre-Ampere GPUs like the free T4 the code
automatically falls back from bf16 to fp16 + GradScaler.

`benchmark.py --resume` skips already-finished (optimizer, seed) pairs, so an
interrupted sweep continues where it stopped.

## Repository layout

```
muon_bench/
  muon.py            Muon from scratch: Newton–Schulz + optimizer + param routing
  resnet9.py         the standard 6.57M-param CIFAR-10 ResNet-9
  data.py            GPU-resident uint8 dataset + bf16 augmentation pipeline
  utils.py           seeding, eval-excluding block timer, env capture
train.py             one training run; owns the recipe & measurement protocol
benchmark.py         (optimizer × seed) sweep → results/benchmark.json + summary
profile_overhead.py  per-step overhead: full step + optimizer.step() in isolation
plot_results.py      accuracy-vs-steps/time curves, time-to-target bars
tune_adamw.py        AdamW lr × wd grid search for the baseline
notebooks/
  colab_benchmark.ipynb   run the whole benchmark on a Colab GPU (Run All)
tests/
  reference_muon.py  verbatim reference implementation (ground truth only)
  test_newton_schulz.py   parity ≤1e-4 + orthogonality/equivariance properties
  test_muon_step.py       10-step trajectory parity ≤1e-4, wd/EMA semantics, routing
  test_data_pipeline.py   each augmentation verified in isolation, determinism
  test_resnet9.py         architecture invariants (6,573,120 params, 9 layers)
```

## Hyperparameters

| | AdamW baseline | Muon run |
|---|---|---|
| hidden conv weights | AdamW lr 3e-3, wd 0.05, β (0.9, 0.95) | **Muon lr 0.05, wd 0.05, momentum 0.95, Nesterov, 5 NS steps** |
| stem / head / BN / biases | Same AdamW | AdamW lr 3e-3, wd 0.05 (0 on 1D params) |
| schedule | 10% linear warmup → linear decay to 0 | same |
| batch / epochs | 512 / 28 | Same |
| augmentation | pad-4 crop, flip, cutout-8 | Same |
| label smoothing | 0.2 | Same |
| weight EMA | decay 0.995, eval reports max(raw, EMA) | Same |
| evaluation | flip TTA (airbench-style) | Same |
| precision | bf16 autocast, channels-last | Same |

## References

- K. Jordan, *Muon: An optimizer for hidden layers in neural networks* —
  https://kellerjordan.github.io/posts/muon/ (reference implementation:
  https://github.com/KellerJordan/Muon)
- J. Bernstein & L. Newhouse, *Old optimizer, new norm: An anthology* —
  arXiv:2409.20325 (Muon as steepest descent under the spectral norm)
- D. Page, *How to Train Your ResNet* — the ResNet-9 architecture and the
  94%-on-CIFAR-10 speedrun methodology (https://myrtle.ai/learn/how-to-train-your-resnet/)
- K. Jordan et al., *CIFAR-10 airbench* — GPU-resident data pipeline ideas
  (https://github.com/KellerJordan/cifar10-airbench)
- A. Hägele et al. / Moonshot AI, *Muon is scalable for LLM training* —
  arXiv:2502.16982 (Muon at LLM scale, weight-decay variant)

## License

MIT — see [LICENSE](LICENSE). The reference implementation reproduced in
`tests/reference_muon.py` is MIT-licensed by its original author.
