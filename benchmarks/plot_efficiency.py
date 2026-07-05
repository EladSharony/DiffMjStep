"""Efficiency figure: differentiable throughput (transitions/sec) vs batch size.

Read generated optimized-backend samples over a batch sweep and plot transitions/sec =
B*T / median wall-clock on log-log axes for one smooth and one contact-rich environment.

    python benchmarks/plot_efficiency.py --out benchmarks/results/efficiency.png
"""
from __future__ import annotations

import argparse

import numpy as np

from _plot_style import (
    BACKWARD,
    ENGINEERING,
    FORWARD,
    apply_style,
    engineering,
    positive_float,
    read_manifest,
    read_rows,
    require_unique,
    save,
    titled,
)
import matplotlib.pyplot as plt

# (env, linestyle, marker) — one smooth, one contact-rich, spanning the DOF range.
ENVS = [("dm_pendulum", "-", "o"), ("dm_cheetah", "--", "s")]


def _throughput(rows, model, batch, horizon):
    samples = [
        positive_float(row["elapsed_ms"], "elapsed_ms")
        for row in rows
        if row["model"] == model and int(row["batch"]) == batch
    ]
    if not samples:
        raise ValueError(f"no samples for {model}, batch={batch}, horizon={horizon}")
    return batch * horizon / (np.median(samples) / 1e3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forward", default="benchmarks/results/cpu_forward_samples.csv")
    ap.add_argument("--backward", default="benchmarks/results/cpu_backward_samples.csv")
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64, 256, 1024, 4096])
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--out", default="benchmarks/results/efficiency.png")
    args = ap.parse_args()
    apply_style()

    forward_rows = read_rows(args.forward)
    backward_rows = read_rows(args.backward)
    identity = ("model", "backend", "batch", "horizon", "dtype", "sample")
    require_unique(forward_rows, identity)
    require_unique(backward_rows, identity)
    forward = [
        row
        for row in forward_rows
        if row["backend"] == "mujoco_rollout" and int(row["horizon"]) == args.horizon
    ]
    backward = [
        row
        for row in backward_rows
        if row["backend"] == "cpp_vjp" and int(row["horizon"]) == args.horizon
    ]
    dtypes = {row["dtype"] for row in forward + backward}
    if len(dtypes) != 1:
        raise ValueError("selected rows must use one dtype")

    batches = np.array(sorted(args.batch))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    peak = []
    for name, ls, mk in ENVS:
        env = name.replace("dm_", "")
        fwd = [_throughput(forward, name, b, args.horizon) for b in batches]
        bwd = [_throughput(backward, name, b, args.horizon) for b in batches]
        ax.plot(batches, fwd, ls=ls, marker=mk, color=FORWARD, lw=2.0, ms=6, zorder=3,
                label=f"forward · {env}")
        ax.plot(batches, bwd, ls=ls, marker=mk, color=BACKWARD, lw=2.0, ms=6, zorder=3,
                label=f"backward · {env}")
        peak += [(batches[-1], fwd[-1]), (batches[-1], bwd[-1])]

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(batches); ax.set_xticklabels([str(b) for b in batches])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_formatter(ENGINEERING)
    ax.set_xlabel("parallel envs (batch size, horizon = %d)" % args.horizon)
    ax.set_ylabel("differentiable throughput  (transitions / sec)")
    # annotate the peak (largest-batch) throughput of each curve
    for bx, ty in peak:
        ax.annotate(engineering(ty) + "/s", (bx, ty), textcoords="offset points",
                    xytext=(6, 0), ha="left", va="center", fontsize=8, fontweight="bold", color="#1a1a1a")
    forward_cpu = read_manifest(args.forward).get("cpu") or "unknown CPU"
    backward_cpu = read_manifest(args.backward).get("cpu") or "unknown CPU"
    hardware = forward_cpu if forward_cpu == backward_cpu else f"{forward_cpu} / {backward_cpu}"
    titled(fig, ax,
           "Differentiable CPU throughput across batch sizes",
           f"throughput = B×T / median wall-clock · {next(iter(dtypes))} · {hardware} · "
           "forward = mujoco_rollout, backward = cpp_vjp")
    ax.grid(True, which="major", axis="both")
    ax.legend(loc="upper left", ncol=2, fontsize=8.5)
    ax.set_xlim(batches[0] * 0.8, batches[-1] * 1.85)
    fig.tight_layout()
    save(fig, args.out)


if __name__ == "__main__":
    main()
