"""Grid-search AdamW hyperparameters for the baseline.

The comparison is only as strong as its baseline, so AdamW is tuned with
this sweep (learning rate x weight decay, optionally betas) under the same
recipe and epoch budget as the benchmark, ranked by time-to-target (ties
broken by best accuracy). The winning configuration becomes the
ADAMW_DEFAULTS in train.py.

    python tune_adamw.py --lrs 1e-3 2e-3 3e-3 --wds 0.005 0.01 0.05 --seed 0
"""

from __future__ import annotations

import argparse

from muon_bench import load_cifar10, save_json
from train import make_parser, train_one_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 2e-3, 3e-3])
    parser.add_argument("--wds", type=float, nargs="+", default=[0.005, 0.01, 0.05])
    parser.add_argument("--out", default="results/adamw_sweep.json")
    sweep_args, rest = parser.parse_known_args()
    train_parser = make_parser()

    print("loading CIFAR-10 ...")
    raw_data = load_cifar10("./data")

    rows = []
    for lr in sweep_args.lrs:
        for wd in sweep_args.wds:
            args = train_parser.parse_args(
                rest + ["--optimizer", "adamw", "--adamw-lr", str(lr), "--adamw-wd", str(wd)]
            )
            print(f"\n--- adamw lr={lr} wd={wd} ---")
            r = train_one_run(args, raw_data=raw_data)
            rows.append(
                {"lr": lr, "wd": wd, "steps_to_target": r["steps_to_target"],
                 "time_to_target_s": r["train_time_to_target_s"],
                 "best_acc": r["best_acc"], "final_acc": r["final_acc"]}
            )
            save_json({"rows": rows}, sweep_args.out)

    # Rank: reached-target first (by time), then by best accuracy.
    rows.sort(key=lambda x: (x["time_to_target_s"] is None,
                             x["time_to_target_s"] if x["time_to_target_s"] is not None else 0,
                             -x["best_acc"]))
    print(f"\n{'lr':>8} {'wd':>8} {'steps@T':>9} {'time@T':>9} {'best acc':>9}")
    for x in rows:
        steps = x["steps_to_target"] if x["steps_to_target"] is not None else "--"
        tsec = f"{x['time_to_target_s']:.1f}" if x["time_to_target_s"] is not None else "--"
        print(f"{x['lr']:>8} {x['wd']:>8} {steps:>9} {tsec:>9} {x['best_acc'] * 100:8.2f}%")
    print(f"\nwrote {sweep_args.out}")


if __name__ == "__main__":
    main()
