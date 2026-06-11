"""Render benchmark plots from results/benchmark.json.

Produces (into --out-dir):
  * acc_vs_steps.png      test accuracy vs optimizer steps (mean over seeds,
                          +/- 1 std band, per-seed curves faint)
  * acc_vs_time.png       test accuracy vs training wall-clock seconds
  * time_to_target.png    bar chart: steps- and time-to-target, +/- 1 std

    python plot_results.py
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

COLORS = {"adamw": "#d1495b", "muon": "#1f7a8c"}
LABELS = {"adamw": "AdamW (tuned)", "muon": "Muon (hybrid)"}


def curves(runs, key):
    """Per-optimizer list of (x, acc) arrays, one per seed."""
    out = {}
    for r in runs:
        xs = np.array([h[key] for h in r["history"]], dtype=float)
        ys = np.array([h["test_acc"] for h in r["history"]], dtype=float)
        out.setdefault(r["optimizer"], []).append((xs, ys))
    return out


def plot_acc(ax, per_opt, target, xlabel):
    for name, seeds in per_opt.items():
        grid = np.linspace(0, min(xs[-1] for xs, _ in seeds), 200)
        interp = np.stack([np.interp(grid, xs, ys) for xs, ys in seeds])
        mean, std = interp.mean(0), interp.std(0)
        for xs, ys in seeds:
            ax.plot(xs, ys * 100, color=COLORS[name], alpha=0.18, lw=0.8)
        ax.plot(grid, mean * 100, color=COLORS[name], lw=2.2, label=f"{LABELS[name]} (n={len(seeds)})")
        ax.fill_between(grid, (mean - std) * 100, (mean + std) * 100, color=COLORS[name], alpha=0.15)
    ax.axhline(target * 100, color="black", ls="--", lw=1, alpha=0.6)
    ax.text(0.99, target * 100 + 0.15, f"target {target * 100:.0f}%", ha="right",
            transform=ax.get_yaxis_transform(), fontsize=9, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(70, 97)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/benchmark.json")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(args.results) as f:
        data = json.load(f)
    runs, summary = data["runs"], data["summary"]
    target = runs[0]["target_acc"]
    os.makedirs(args.out_dir, exist_ok=True)

    for key, xlabel, fname in (
        ("step", "optimizer steps", "acc_vs_steps.png"),
        ("train_time_s", "training time (s, eval excluded)", "acc_vs_time.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
        plot_acc(ax, curves(runs, key), target, xlabel)
        ax.set_title(f"CIFAR-10 / ResNet-9: Muon vs AdamW ({xlabel})")
        fig.tight_layout()
        path = os.path.join(args.out_dir, fname)
        fig.savefig(path)
        plt.close(fig)
        print(f"wrote {path}")

    # Bar chart: steps / time to target.
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), dpi=150)
    for ax, key, title in (
        (axes[0], "steps_to_target", f"steps to {target * 100:.0f}%"),
        (axes[1], "time_to_target_s", f"train time to {target * 100:.0f}% (s)"),
    ):
        names = [n for n in ("adamw", "muon") if n in summary and key in summary[n]]
        means = [summary[n][key][0] for n in names]
        stds = [summary[n][key][1] for n in names]
        bars = ax.bar([LABELS[n] for n in names], means, yerr=stds, capsize=5,
                      color=[COLORS[n] for n in names], width=0.55)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mean:.0f}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
    if "muon_vs_adamw" in summary:
        d = summary["muon_vs_adamw"]
        fig.suptitle(
            f"Muon cuts steps by {d['steps_reduction_pct']:.0f}% and wall-clock by "
            f"{d['time_reduction_pct']:.0f}% (mean of {summary['muon']['n_runs']} seeds)",
            fontsize=11,
        )
    fig.tight_layout()
    path = os.path.join(args.out_dir, "time_to_target.png")
    fig.savefig(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
