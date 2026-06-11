"""Multi-seed benchmark: Muon (hybrid) vs tuned AdamW on CIFAR-10 / ResNet-9.

Runs every (optimizer, seed) combination with an identical recipe (same
model, data order seed, augmentation, schedule shape, label smoothing,
batch size) and reports mean +/- std of:

  * steps to reach the target test accuracy (default 94%)
  * training wall-clock time to reach it (eval time excluded)
  * final / best accuracy

Results are written incrementally to <out-dir>/benchmark.json so an
interrupted sweep keeps its finished runs. Render plots afterwards with
`python plot_results.py`.

Example:
    python benchmark.py --seeds 5                  # full 5-seed benchmark
    python benchmark.py --seeds 2 --epochs 12      # quicker sanity pass
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

from muon_bench import load_cifar10, save_json
from train import make_parser, train_one_run


def summarize(runs: list[dict], target_acc: float) -> dict:
    """Aggregate per-optimizer stats across seeds."""
    summary = {}
    for name in ("adamw", "muon"):
        rs = [r for r in runs if r["optimizer"] == name]
        if not rs:
            continue
        reached = [r for r in rs if r["steps_to_target"] is not None]
        stat = lambda xs: (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0)
        entry = {
            "seeds": [r["seed"] for r in rs],
            "n_runs": len(rs),
            "n_reached_target": len(reached),
            "final_acc": stat([r["final_acc"] for r in rs]),
            "best_acc": stat([r["best_acc"] for r in rs]),
        }
        if reached:
            entry["steps_to_target"] = stat([r["steps_to_target"] for r in reached])
            entry["time_to_target_s"] = stat([r["train_time_to_target_s"] for r in reached])
        summary[name] = entry

    if "adamw" in summary and "muon" in summary:
        a, m = summary["adamw"], summary["muon"]
        if "steps_to_target" in a and "steps_to_target" in m:
            summary["muon_vs_adamw"] = {
                "steps_reduction_pct": round(100 * (1 - m["steps_to_target"][0] / a["steps_to_target"][0]), 2),
                "time_reduction_pct": round(100 * (1 - m["time_to_target_s"][0] / a["time_to_target_s"][0]), 2),
                "target_acc": target_acc,
            }
    return summary


def print_summary(summary: dict, target_acc: float) -> None:
    pct = f"{target_acc * 100:.0f}%"
    print("\n" + "=" * 78)
    print(f"  Muon vs AdamW -- CIFAR-10 / ResNet-9 -- time & steps to {pct} test accuracy")
    print("=" * 78)
    header = f"  {'optimizer':<10} {'reached':>8} {'steps-to-' + pct:>14} {'time-to-' + pct + ' (s)':>18} {'best acc':>12}"
    print(header)
    print("  " + "-" * 74)
    for name in ("adamw", "muon"):
        if name not in summary:
            continue
        e = summary[name]
        steps = f"{e['steps_to_target'][0]:7.0f} +/- {e['steps_to_target'][1]:4.0f}" if "steps_to_target" in e else "--"
        tsec = f"{e['time_to_target_s'][0]:7.1f} +/- {e['time_to_target_s'][1]:5.1f}" if "time_to_target_s" in e else "--"
        acc = f"{e['best_acc'][0] * 100:5.2f} +/- {e['best_acc'][1] * 100:4.2f}"
        print(f"  {name:<10} {e['n_reached_target']}/{e['n_runs']:>6} {steps:>14} {tsec:>18} {acc:>12}")
    if "muon_vs_adamw" in summary:
        d = summary["muon_vs_adamw"]
        print("  " + "-" * 74)
        print(
            f"  Muon vs AdamW: {d['steps_reduction_pct']:+.1f}% steps, "
            f"{d['time_reduction_pct']:+.1f}% wall-clock to {pct} "
            f"(negative = Muon needs fewer/less)"
        )
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds (0..n-1)")
    parser.add_argument("--seed-list", type=int, nargs="*", default=None, help="explicit seed list (overrides --seeds)")
    parser.add_argument("--optimizers", nargs="*", default=["adamw", "muon"], choices=["adamw", "muon"])
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--resume", action="store_true", help="skip (optimizer, seed) pairs already in benchmark.json")
    parser.add_argument("--cooldown", type=int, default=30,
                        help="seconds to idle the GPU between runs (thermal consistency)")
    bench_args, rest = parser.parse_known_args()

    train_parser = make_parser()
    seeds = bench_args.seed_list if bench_args.seed_list is not None else list(range(bench_args.seeds))
    out_path = os.path.join(bench_args.out_dir, "benchmark.json")

    runs: list[dict] = []
    if bench_args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            runs = json.load(f)["runs"]
        print(f"resuming: {len(runs)} finished runs loaded from {out_path}")

    done = {(r["optimizer"], r["seed"]) for r in runs}
    print("loading CIFAR-10 ...")
    raw_data = load_cifar10("./data")

    # Interleave optimizers within each seed (rather than all-A-then-all-B):
    # sustained load causes thermal throttling on many GPUs, and interleaving
    # spreads any clock drift evenly across both optimizers.
    target_acc = None
    first = True
    for seed in seeds:
        for opt_name in bench_args.optimizers:
            if (opt_name, seed) in done:
                continue
            if not first and bench_args.cooldown > 0:
                print(f"(cooldown {bench_args.cooldown}s)", flush=True)
                time.sleep(bench_args.cooldown)
            first = False
            args = train_parser.parse_args(rest + ["--optimizer", opt_name, "--seed", str(seed)])
            target_acc = args.target_acc
            print(f"\n--- run: optimizer={opt_name} seed={seed} epochs={args.epochs} ---")
            runs.append(train_one_run(args, raw_data=raw_data))
            save_json({"runs": runs, "summary": summarize(runs, target_acc)}, out_path)

    if target_acc is None:
        target_acc = runs[0]["target_acc"] if runs else 0.94
    summary = summarize(runs, target_acc)
    save_json({"runs": runs, "summary": summary}, out_path)
    print_summary(summary, target_acc)
    print(f"full results: {out_path}")


if __name__ == "__main__":
    main()
