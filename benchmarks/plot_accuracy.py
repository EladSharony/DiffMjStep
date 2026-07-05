"""Plot generated transition-FD and whole-rollout-FD gradient components.

Each point compares one component of MuJoCo's transition finite-difference VJP with a
whole-rollout central finite-difference reference. This script only reads benchmark data.

    python benchmarks/plot_accuracy.py --out benchmarks/results/accuracy.png
"""
from __future__ import annotations

import argparse

import numpy as np

from _plot_style import (
    MODEL_PALETTE,
    apply_style,
    finite_float,
    positive_float,
    read_rows,
    require_unique,
    save,
    titled,
)
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="benchmarks/results/gradient_components.csv")
    ap.add_argument("--out", default="benchmarks/results/accuracy.png")
    args = ap.parse_args()
    apply_style()

    rows = read_rows(args.data)
    require_unique(
        rows,
        (
            "model",
            "seed",
            "horizon",
            "reference_eps",
            "transition_eps",
            "component",
        ),
    )
    if not rows:
        raise ValueError("gradient component data is empty")
    by_model = {}
    by_cell = {}
    reference_eps = set()
    transition_eps = set()
    for row in rows:
        reference_eps.add(positive_float(row["reference_eps"], "reference_eps"))
        transition_eps.add(positive_float(row["transition_eps"], "transition_eps"))
        pair = (
            finite_float(row["transition_fd_vjp"], "transition_fd_vjp"),
            finite_float(row["whole_rollout_fd"], "whole_rollout_fd"),
        )
        by_model.setdefault(row["model"], []).append(pair)
        cell = (
            row["model"],
            row["seed"],
            row["horizon"],
            row["reference_eps"],
            row["transition_eps"],
        )
        by_cell.setdefault(cell, []).append(pair)
    models = sorted(by_model)

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    all_a, all_f = [], []
    for i, name in enumerate(models):
        ga = np.asarray([pair[0] for pair in by_model[name]])
        gf = np.asarray([pair[1] for pair in by_model[name]])
        all_a.append(ga); all_f.append(gf)
        ax.scatter(gf, ga, s=26, color=MODEL_PALETTE[i % len(MODEL_PALETTE)], alpha=0.8,
                   edgecolor="white", linewidth=0.4, zorder=3, label=name.replace("dm_", ""))

    a = np.concatenate(all_a); f = np.concatenate(all_f)
    rel_errs = []
    for pairs in by_cell.values():
        cell_a = np.asarray([pair[0] for pair in pairs])
        cell_f = np.asarray([pair[1] for pair in pairs])
        rel_errs.append(np.linalg.norm(cell_a - cell_f) / (np.linalg.norm(cell_f) + 1e-30))
    cosine = float(a @ f / ((np.linalg.norm(a) * np.linalg.norm(f)) + 1e-30))
    lim = max(float(np.max(np.abs(np.concatenate([a, f])))) * 1.1, 1e-12)
    linthresh = max(lim * 1e-4, 1e-12)
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="#888888", lw=1.2, zorder=2, label="y = x")
    ax.set_xscale("symlog", linthresh=linthresh); ax.set_yscale("symlog", linthresh=linthresh)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator([
            tick for tick in axis.get_majorticklocs()
            if tick == 0 or abs(tick) > linthresh
        ]))
    ax.set_aspect("equal")
    ax.set_xlabel("whole-rollout central FD reference")
    ax.set_ylabel("MuJoCo transition-FD VJP")
    ax.text(0.04, 0.96, f"cosine = {cosine:.6f}\nrel. error ≤ {max(rel_errs):.0e}",
            transform=ax.transAxes, va="top", ha="left", fontsize=11, fontweight="bold",
            color="#2f7d4f", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9))
    titled(fig, ax,
           "Transition-FD VJPs agree with a whole-rollout FD reference",
           f"{len(rows)} components across {len(models)} dm_control models · "
           f"horizons = {', '.join(sorted({row['horizon'] for row in rows}, key=int))} · "
           f"reference eps = {', '.join(f'{value:g}' for value in sorted(reference_eps))} · "
           f"transition eps = {', '.join(f'{value:g}' for value in sorted(transition_eps))}")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    save(fig, args.out)


if __name__ == "__main__":
    main()
