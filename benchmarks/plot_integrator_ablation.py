"""Box-plot generated integrator-ablation gradient agreement data.

Environments on the x-axis (like the speedup figures); for each env, three boxes (Euler /
implicit / implicitfast) show transition-FD-VJP versus whole-rollout-FD relative error
across the generated timestep sweep. The plotter reads only paths supplied on the command
line (or its documented default glob).

    python benchmarks/plot_integrator_ablation.py --out benchmarks/results/integrator_ablation.png
"""
from __future__ import annotations

import argparse
import glob

import numpy as np

from _plot_style import (
    INTEGRATOR,
    apply_style,
    finite_float,
    positive_float,
    read_rows,
    require_unique,
    save,
    titled,
)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INTEGRATORS = ["Euler", "implicit", "implicitfast"]
_ZERO_FLOOR = 1e-16


def _style(bp, color):
    for box in bp["boxes"]:
        box.set(facecolor=color, alpha=0.65, edgecolor="#333333", linewidth=1.0)
    for med in bp["medians"]:
        med.set(color="#111111", linewidth=1.3)
    for w in bp["whiskers"] + bp["caps"]:
        w.set(color="#333333", linewidth=0.9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data",
        nargs="+",
        default=sorted(glob.glob("benchmarks/results/integrator_ablation_dm_*.csv")),
    )
    ap.add_argument("--out", default="benchmarks/results/integrator_ablation.png")
    args = ap.parse_args()
    apply_style()

    if not args.data:
        raise SystemExit("no integrator_ablation_dm_*.csv found; run benchmarks/ablate_integrator.py first")
    rows = []
    for path in args.data:
        rows.extend(read_rows(path))
    require_unique(
        rows,
        (
            "model",
            "integrator",
            "dt",
            "horizon",
            "reference_eps",
            "transition_eps",
        ),
    )

    data = {}
    dofs = {}
    reference_eps = set()
    transition_eps = set()
    for row in rows:
        env = row["model"]
        dofs.setdefault(env, set()).add(int(row["dof"]))
        by_integrator = data.setdefault(env, {name: [] for name in INTEGRATORS})
        if row["integrator"] not in by_integrator:
            raise ValueError(f"unknown integrator {row['integrator']!r}")
        positive_float(row["dt"], "dt")
        reference_eps.add(positive_float(row["reference_eps"], "reference_eps"))
        transition_eps.add(positive_float(row["transition_eps"], "transition_eps"))
        error = finite_float(row["grad_rel_error"], "grad_rel_error")
        if error < 0:
            raise ValueError("grad_rel_error must be nonnegative")
        by_integrator[row["integrator"]].append(_ZERO_FLOOR if error == 0 else error)

    if not data:
        raise ValueError("integrator data is empty")
    if any(len(values) != 1 for values in dofs.values()):
        raise ValueError("each model must have exactly one dof")
    if any(not values for by_integrator in data.values() for values in by_integrator.values()):
        raise ValueError("each model needs rows for every supported integrator")

    envs = sorted(data, key=lambda env: (next(iter(dofs[env])), env))
    labels = [f"{env.replace('dm_', '')}\n({next(iter(dofs[env]))} dof)" for env in envs]
    x = np.arange(len(envs))

    lo = min(min(v) for d in data.values() for v in d.values())
    fig, ax = plt.subplots(figsize=(1.15 * len(envs) + 1.5, 5.2))
    ax.axhspan(lo * 0.5, 1e-5, color="#55a868", alpha=0.06, zorder=0)
    for j, ig in enumerate(INTEGRATORS):
        boxes = [data[e][ig] for e in envs]
        bp = ax.boxplot(boxes, positions=x + (j - 1) * 0.26, widths=0.22, patch_artist=True,
                        manage_ticks=False, whis=(0, 100))
        _style(bp, INTEGRATOR[ig])

    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, len(envs) - 0.4)
    ax.set_ylabel("transition-FD VJP vs whole-rollout FD relative error")
    ax.grid(True, axis="y")
    ax.text(0.012, 0.04, "agreement region (< 1e-5); zeros shown at 1e-16", transform=ax.transAxes,
            va="bottom", ha="left", fontsize=9, color="#2f7d4f", fontweight="bold")
    ax.legend(handles=[Patch(facecolor=INTEGRATOR[ig], alpha=0.65, label=ig) for ig in INTEGRATORS],
              title="integrator", loc="upper left", ncol=3)
    titled(fig, ax,
           "Gradient agreement across every supported integrator",
           "transition-FD VJP vs whole-rollout FD relative error over the dt sweep · "
           f"reference eps = {', '.join(f'{value:g}' for value in sorted(reference_eps))} · "
           f"transition eps = {', '.join(f'{value:g}' for value in sorted(transition_eps))} · "
           "whiskers = full range · RK4 is outside MuJoCo transition-FD support")
    fig.tight_layout()
    save(fig, args.out)


if __name__ == "__main__":
    main()
