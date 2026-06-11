"""Measure Muon's per-step overhead over AdamW on the real training step.

For each optimizer this script times, after warmup (which also lets
cudnn.benchmark autotune), `--steps` full training steps -- GPU-resident
augmentation + bf16 forward + backward + optimizer step -- and additionally
brackets `optimizer.step()` with CUDA events to isolate the optimizer's own
GPU time (for Muon, that includes all Newton-Schulz iterations).

Reported numbers:
  * mean full step time per optimizer (ms)
  * mean optimizer.step() GPU time per optimizer (ms)
  * Muon overhead per step = muon_full_step / adamw_full_step - 1  (%)

Because the data pipeline is GPU-resident, the step time is pure compute --
there are no dataloader stalls to hide (or fake) the overhead.

    python profile_overhead.py --steps 200
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from muon_bench import GPUCifar10, build_resnet9, env_info, load_cifar10, save_json, set_seed
from train import build_optimizers, make_parser


def profile_optimizer(name: str, args, raw_data, steps: int, warmup: int) -> dict:
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    set_seed(0)

    pipeline = GPUCifar10(*raw_data, device=device, batch_size=args.batch_size, seed=0)
    model = build_resnet9(device)
    optimizers = build_optimizers(model, name, args)

    def batches():
        while True:
            yield from pipeline.train_batches()

    batch_iter = batches()
    opt_events = []  # (start, end) CUDA event pairs around optimizer.step()
    use_events = device.type == "cuda"

    def one_step(record: bool):
        x, y = next(batch_iter)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).float(), y, label_smoothing=0.1)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        if record and use_events:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            ev0.record()
            for opt in optimizers:
                opt.step()
            ev1.record()
            opt_events.append((ev0, ev1))
        else:
            for opt in optimizers:
                opt.step()

    model.train()
    for _ in range(warmup):
        one_step(record=False)

    if use_events:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        one_step(record=True)
    if use_events:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    full_step_ms = elapsed / steps * 1000
    opt_ms = statistics.mean(e0.elapsed_time(e1) for e0, e1 in opt_events) if use_events else float("nan")
    return {"optimizer": name, "full_step_ms": round(full_step_ms, 3), "opt_step_ms": round(opt_ms, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--out", default="results/overhead.json")
    prof_args, rest = parser.parse_known_args()
    args = make_parser().parse_args(rest)

    print("loading CIFAR-10 ...")
    raw_data = load_cifar10(args.data_root)

    results = {}
    for name in ("adamw", "muon"):
        print(f"profiling {name} ({prof_args.warmup} warmup + {prof_args.steps} timed steps) ...")
        results[name] = profile_optimizer(name, args, raw_data, prof_args.steps, prof_args.warmup)
        print(
            f"  {name}: full step {results[name]['full_step_ms']:.2f} ms | "
            f"optimizer.step() {results[name]['opt_step_ms']:.2f} ms"
        )

    overhead_pct = 100 * (results["muon"]["full_step_ms"] / results["adamw"]["full_step_ms"] - 1)
    opt_only_pct = 100 * (
        (results["muon"]["opt_step_ms"] - results["adamw"]["opt_step_ms"]) / results["adamw"]["full_step_ms"]
    )
    results["muon_overhead_pct_per_step"] = round(overhead_pct, 2)
    results["muon_extra_opt_time_pct_of_step"] = round(opt_only_pct, 2)
    results["batch_size"] = args.batch_size
    results["env"] = env_info(args.device)

    print(f"\nMuon per-step overhead vs AdamW: {overhead_pct:+.2f}% "
          f"(optimizer-kernel delta alone: {opt_only_pct:+.2f}% of an AdamW step)")
    save_json(results, prof_args.out)
    print(f"wrote {prof_args.out}")


if __name__ == "__main__":
    main()
