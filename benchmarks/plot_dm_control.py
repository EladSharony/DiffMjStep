"""Plot generated raw timings for optimized backends versus the Python reference.

For each dm_control domain, bootstrap the speedup from raw forward and backward samples
and draw a grouped bar with a 95% interval. This script only reads benchmark artifacts.

    python benchmarks/plot_dm_control.py --out benchmarks/results/dm_control_speedup.png
"""
from __future__ import annotations

import argparse

import numpy as np

from _plot_style import (
    BACKWARD,
    BASELINE,
    FORWARD,
    MULTIPLIER,
    apply_style,
    positive_float,
    read_manifest,
    read_rows,
    require_unique,
    save,
    titled,
)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _samples(rows, model, backend):
    samples = np.asarray(
        [
            positive_float(row["elapsed_ms"], "elapsed_ms")
            for row in rows
            if row["model"] == model and row["backend"] == backend
        ]
    )
    if not samples.size:
        raise ValueError(f"no {backend} samples for {model}")
    return samples


def _boot_speedup(base_ms, opt_ms, rng, n_boot=2000):
    """Bootstrap distribution of the speedup mean(base)/mean(opt)."""
    bi = rng.integers(0, len(base_ms), size=(n_boot, len(base_ms)))
    oi = rng.integers(0, len(opt_ms), size=(n_boot, len(opt_ms)))
    return base_ms[bi].mean(1) / opt_ms[oi].mean(1)


def _bars(ax, x, dists, color, label):
    """Grouped bar with 95% CI whisker + median printed on top. Returns the medians."""
    med = np.array([np.median(d) for d in dists])
    lo = np.array([np.percentile(d, 2.5) for d in dists])
    hi = np.array([np.percentile(d, 97.5) for d in dists])
    ax.bar(x, med, width=0.38, color=color, alpha=0.9, edgecolor="#2b2b2b", linewidth=0.8,
           yerr=[med - lo, hi - med], error_kw=dict(ecolor="#2b2b2b", elinewidth=1.0, capsize=2.5),
           label=label, zorder=3)
    for xi, m, h in zip(x, med, hi):
        ax.annotate(f"{m:.0f}×", (xi, h), textcoords="offset points", xytext=(0, 3),
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1a1a1a", zorder=4)
    return med


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forward", default="benchmarks/results/cpu_forward_samples.csv")
    ap.add_argument("--backward", default="benchmarks/results/cpu_backward_samples.csv")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--out", default="benchmarks/results/dm_control_speedup.png")
    args = ap.parse_args()
    apply_style()
    rng = np.random.default_rng(0)

    forward_rows = read_rows(args.forward)
    backward_rows = read_rows(args.backward)
    identity = ("model", "backend", "batch", "horizon", "dtype", "sample")
    require_unique(forward_rows, identity)
    require_unique(backward_rows, identity)
    forward = [
        row
        for row in forward_rows
        if row["backend"] in ("python_ref", "mujoco_rollout")
        and int(row["batch"]) == args.batch
        and int(row["horizon"]) == args.horizon
    ]
    backward = [
        row
        for row in backward_rows
        if row["backend"] in ("python_vjp", "cpp_vjp")
        and int(row["batch"]) == args.batch
        and int(row["horizon"]) == args.horizon
    ]
    rows = forward + backward
    models = {row["model"] for row in rows}
    if not models:
        raise ValueError(f"no rows for batch={args.batch}, horizon={args.horizon}")

    dofs = {
        model: {int(row["dof"]) for row in rows if row["model"] == model}
        for model in models
    }
    if any(len(values) != 1 for values in dofs.values()):
        raise ValueError("each model must have exactly one dof")
    models = sorted(models, key=lambda model: (next(iter(dofs[model])), model))
    fwd_dists, bwd_dists, labels = [], [], []
    for name in models:
        nv = next(iter(dofs[name]))
        fr = _samples(forward, name, "python_ref")
        fo = _samples(forward, name, "mujoco_rollout")
        br = _samples(backward, name, "python_vjp")
        bo = _samples(backward, name, "cpp_vjp")
        fwd_dists.append(_boot_speedup(fr, fo, rng))
        bwd_dists.append(_boot_speedup(br, bo, rng))
        labels.append(f"{name.replace('dm_', '')}\n({nv} dof)")

    dtypes = {row["dtype"] for row in rows}
    if len(dtypes) != 1:
        raise ValueError("selected rows must use one dtype")
    forward_cpu = read_manifest(args.forward).get("cpu") or "unknown CPU"
    backward_cpu = read_manifest(args.backward).get("cpu") or "unknown CPU"
    hardware = forward_cpu if forward_cpu == backward_cpu else f"{forward_cpu} / {backward_cpu}"

    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(1.2 * len(models) + 1.5, 5.2))
    fmed = _bars(ax, x - 0.21, fwd_dists, FORWARD, "forward · mujoco_rollout")
    bmed = _bars(ax, x + 0.21, bwd_dists, BACKWARD, "backward · cpp_vjp")

    ax.axhline(1.0, ls="--", color=BASELINE, lw=1.1, zorder=2)
    ax.text(-0.5, 1.3, "1× = reference", ha="left", va="bottom",
            fontsize=8, color=BASELINE, style="italic")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(models) - 0.4)
    top = max(np.percentile(np.concatenate(fwd_dists + bwd_dists), 99.9), 1.0)
    ax.set_ylim(0, top * 1.13)
    ax.yaxis.set_major_formatter(MULTIPLIER)
    ax.set_ylabel("speedup over the Python reference")
    titled(fig, ax,
           f"Measured rollout speedup: forward {fmed.min():.0f}–{fmed.max():.0f}×, "
           f"backward {bmed.min():.0f}–{bmed.max():.0f}× over Python VJP",
           f"speedup over reference paths across {len(models)} measured dm_control models · "
           f"B×T = {args.batch}×{args.horizon} · "
           f"{next(iter(dtypes))} · {hardware} · bars = median, whiskers = 95% bootstrap CI")
    ax.grid(True, axis="y")
    ax.legend(handles=[Patch(facecolor=FORWARD, alpha=0.9, label="forward · mujoco_rollout"),
                       Patch(facecolor=BACKWARD, alpha=0.9, label="backward · cpp_vjp")], loc="upper right")
    fig.tight_layout()
    save(fig, args.out)


if __name__ == "__main__":
    main()
