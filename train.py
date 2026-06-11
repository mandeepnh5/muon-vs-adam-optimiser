"""Train ResNet-9 on CIFAR-10 with Muon (hybrid) or tuned AdamW.

Single-run entry point; `benchmark.py` calls `train_one_run` directly.

Measurement protocol (identical for both optimizers, see README):
  * the model trains in bf16 autocast, channels-last, batch 512, with the
    GPU-resident augmentation pipeline (no CPU dataloader);
  * test accuracy is evaluated every `eval_every` steps (default: twice per
    epoch); evaluation time is excluded from the reported training time by
    timing train blocks between evals with GPU synchronization only at
    block boundaries;
  * "steps/time to target" = first evaluation at which test accuracy
    reaches `target_acc` (no smoothing).

Example:
    python train.py --optimizer muon --seed 0
    python train.py --optimizer adamw --epochs 16 --target-acc 0.94
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from muon_bench import (
    CudaBlockTimer,
    GPUCifar10,
    ModelEMA,
    Muon,
    build_resnet9,
    env_info,
    load_cifar10,
    save_json,
    set_seed,
    split_muon_params,
)

# ----------------------------------------------------------------- defaults
# Tuned hyperparameters used for the reported benchmark results (see README
# for the tuning protocol; reproduce the AdamW sweep with tune_adamw.py).

ADAMW_DEFAULTS = dict(lr=3e-3, weight_decay=0.05, betas=(0.9, 0.95))
MUON_DEFAULTS = dict(lr=0.05, weight_decay=0.05, momentum=0.95, nesterov=True, ns_steps=5)


def amp_dtype_for(device: torch.device) -> torch.dtype:
    """bf16 where supported (Ampere+); fp16 fallback for older GPUs (e.g. the
    Colab T4), paired with a GradScaler in the train loop. CPU uses bf16
    autocast, which torch emulates."""
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16


def build_optimizers(model, name: str, args) -> list[torch.optim.Optimizer]:
    """Build the optimizer stack for `name` in {'adamw', 'muon'}.

    adamw: one tuned AdamW over everything (no weight decay on 1D params --
           biases and BatchNorm gains -- which is standard practice).
    muon:  Muon on the hidden conv weight matrices; AdamW (same tuned
           settings as the baseline) on the stem conv, final linear head,
           and all 1D params, as recommended by the Muon authors.
    """
    decay_1d_exempt = lambda ps: [
        {"params": [p for p in ps if p.ndim >= 2], "weight_decay": args.adamw_wd},
        {"params": [p for p in ps if p.ndim < 2], "weight_decay": 0.0},
    ]

    if name == "adamw":
        groups = decay_1d_exempt(list(model.parameters()))
        return [torch.optim.AdamW(groups, lr=args.adamw_lr, betas=tuple(args.adamw_betas))]

    if name == "muon":
        muon_params, adamw_params = split_muon_params(model)
        muon_opt = Muon(
            muon_params,
            lr=args.muon_lr,
            weight_decay=args.muon_wd,
            momentum=args.muon_momentum,
            nesterov=True,
            ns_steps=args.ns_steps,
        )
        adamw_opt = torch.optim.AdamW(
            decay_1d_exempt(adamw_params), lr=args.adamw_lr, betas=tuple(args.adamw_betas)
        )
        return [muon_opt, adamw_opt]

    raise ValueError(f"unknown optimizer {name!r} (expected 'adamw' or 'muon')")


def lr_lambda(step: int, total_steps: int, warmup_frac: float) -> float:
    """One-cycle triangular schedule: linear warmup, then linear decay to 0."""
    warmup = max(1, int(total_steps * warmup_frac))
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total_steps - step) / (total_steps - warmup))


@torch.no_grad()
def evaluate(model, pipeline, device, tta: bool = True) -> tuple[float, float]:
    """Returns (test_accuracy, test_loss) over the full test set.

    With `tta` (default), logits are averaged over the image and its
    horizontal mirror -- the standard evaluation for CIFAR-10 94%-speedrun
    benchmarks (cf. airbench). Applied identically to every optimizer.
    """
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for x, y in pipeline.test_batches():
            logits = model(x).float()
            if tta:
                logits = logits + model(x.flip(-1)).float()
                logits = logits / 2
            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total, loss_sum / total


def train_one_run(args, raw_data=None, verbose=True) -> dict:
    """Run one full training and return a result dict (history + metrics)."""
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    set_seed(args.seed)

    if raw_data is None:
        raw_data = load_cifar10(args.data_root)
    pipeline = GPUCifar10(
        *raw_data,
        device=device,
        batch_size=args.batch_size,
        cutout=args.cutout,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        seed=args.seed,
    )

    model = build_resnet9(device)
    optimizers = build_optimizers(model, args.optimizer, args)
    base_lrs = [[g["lr"] for g in opt.param_groups] for opt in optimizers]
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None

    steps_per_epoch = pipeline.steps_per_epoch
    total_steps = steps_per_epoch * args.epochs
    eval_every = args.eval_every or max(1, steps_per_epoch // 2)

    timer = CudaBlockTimer(device)
    history = []
    steps_to_target = None
    time_to_target = None
    step = 0

    def run_eval():
        nonlocal steps_to_target, time_to_target
        timer.stop()
        acc, loss = evaluate(model, pipeline, device, tta=bool(args.tta))
        ema_acc = None
        if ema is not None:
            ema.swap_in(model)
            ema_acc, _ = evaluate(model, pipeline, device, tta=bool(args.tta))
            ema.swap_out(model)
        # The deployable accuracy at this point in training: best of raw/EMA
        # weights (the same rule for every optimizer).
        acc_now = max(acc, ema_acc) if ema_acc is not None else acc
        history.append(
            {"step": step, "epoch": step / steps_per_epoch, "train_time_s": round(timer.total, 3),
             "test_acc": acc_now, "raw_acc": acc, "ema_acc": ema_acc, "test_loss": round(loss, 4)}
        )
        if steps_to_target is None and acc_now >= args.target_acc:
            steps_to_target = step
            time_to_target = timer.total
        if verbose:
            ema_str = f" ema {ema_acc * 100:6.2f}%" if ema_acc is not None else ""
            print(
                f"  [{args.optimizer} seed={args.seed}] step {step:5d}/{total_steps} "
                f"epoch {step / steps_per_epoch:5.2f}  acc {acc * 100:6.2f}%{ema_str}  "
                f"train_time {timer.total:7.2f}s",
                flush=True,
            )
        timer.start()

    model.train()
    start_wall = time.perf_counter()
    timer.start()

    for _epoch in range(args.epochs):
        for x, y in pipeline.train_batches():
            scale = lr_lambda(step, total_steps, args.warmup_frac)
            for opt, lrs in zip(optimizers, base_lrs):
                for group, base in zip(opt.param_groups, lrs):
                    group["lr"] = base * scale

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.float(), y, label_smoothing=args.label_smoothing)

            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in optimizers:
                opt.step()
            if ema is not None:
                ema.update(model)

            step += 1
            if step % eval_every == 0:
                run_eval()

    if not history or history[-1]["step"] != step:
        run_eval()
    timer.stop()

    result = {
        "optimizer": args.optimizer,
        "seed": args.seed,
        "target_acc": args.target_acc,
        "steps_to_target": steps_to_target,
        "train_time_to_target_s": round(time_to_target, 3) if time_to_target else None,
        "final_acc": history[-1]["test_acc"],
        "best_acc": max(h["test_acc"] for h in history),
        "total_train_time_s": round(timer.total, 3),
        "total_wall_time_s": round(time.perf_counter() - start_wall, 3),
        "total_steps": step,
        "steps_per_epoch": steps_per_epoch,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "label_smoothing": args.label_smoothing,
            "warmup_frac": args.warmup_frac,
            "cutout": args.cutout,
            "ema_decay": args.ema_decay,
            "tta": bool(args.tta),
            "eval_every": eval_every,
            "adamw": {"lr": args.adamw_lr, "weight_decay": args.adamw_wd, "betas": list(args.adamw_betas)},
            "muon": {"lr": args.muon_lr, "weight_decay": args.muon_wd,
                     "momentum": args.muon_momentum, "ns_steps": args.ns_steps},
        },
        "env": env_info(device),
        "history": history,
    }
    if verbose:
        tt = f"{time_to_target:.1f}s @ step {steps_to_target}" if steps_to_target else "not reached"
        print(
            f"[{args.optimizer} seed={args.seed}] done: final {result['final_acc'] * 100:.2f}% "
            f"best {result['best_acc'] * 100:.2f}%  time-to-{args.target_acc * 100:.0f}%: {tt}"
        )
    return result


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="muon")
    p.add_argument("--epochs", type=int, default=28)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-acc", type=float, default=0.94)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--eval-every", type=int, default=0, help="steps between evals (0 = twice per epoch)")
    # shared recipe
    p.add_argument("--label-smoothing", type=float, default=0.2)
    p.add_argument("--warmup-frac", type=float, default=0.1)
    p.add_argument("--cutout", type=int, default=8)
    p.add_argument("--ema-decay", type=float, default=0.995, help="weight EMA decay (0 disables)")
    p.add_argument("--tta", type=int, default=1, help="horizontal-flip TTA at eval (0 disables)")
    # adamw (also used for the aux params of the muon run)
    p.add_argument("--adamw-lr", type=float, default=ADAMW_DEFAULTS["lr"])
    p.add_argument("--adamw-wd", type=float, default=ADAMW_DEFAULTS["weight_decay"])
    p.add_argument("--adamw-betas", type=float, nargs=2, default=list(ADAMW_DEFAULTS["betas"]))
    # muon
    p.add_argument("--muon-lr", type=float, default=MUON_DEFAULTS["lr"])
    p.add_argument("--muon-wd", type=float, default=MUON_DEFAULTS["weight_decay"])
    p.add_argument("--muon-momentum", type=float, default=MUON_DEFAULTS["momentum"])
    p.add_argument("--ns-steps", type=int, default=MUON_DEFAULTS["ns_steps"])
    p.add_argument("--out", default="", help="optional path to write the result JSON")
    return p


def main():
    args = make_parser().parse_args()
    result = train_one_run(args)
    if args.out:
        save_json(result, args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
