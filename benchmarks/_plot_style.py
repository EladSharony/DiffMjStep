"""Shared plotting style so every generated figure looks like one family.

Import `apply_style()` and the palette/helpers; call `apply_style()` once at the top of a
plot's main(). One source of truth for colors, fonts, spines, grid, saving, and the `×`/SI
tick formatters.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, FuncFormatter

# --- semantic palette (one source of truth) -----------------------------------------------
FORWARD = "#e8743b"    # forward pass (mujoco_rollout)
BACKWARD = "#3aa757"   # backward pass (cpp_vjp)
BASELINE = "#8a8a8a"   # reference paths (always dashed)
INTEGRATOR = {"Euler": "#4c72b0", "implicit": "#c44e52", "implicitfast": "#55a868"}
# categorical colors for per-model lines in a single-panel figure (up to 9 dm_control envs)
MODEL_PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
                 "#937860", "#da8bc3", "#64b5cd", "#ccb974"]

# tick formatters
MULTIPLIER = FuncFormatter(lambda v, _: f"{v:g}×")
ENGINEERING = EngFormatter(places=2)


def engineering(value: float) -> str:
    """Format a value with a compact SI prefix for annotations."""
    return ENGINEERING(value).replace(" ", "")


def read_manifest(path: str | Path) -> dict:
    """Read an adjacent manifest after verifying that it names and hashes the CSV."""
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if manifest.get("data_file") != path.name:
        raise ValueError(f"manifest data_file does not name {path.name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest.get("data_sha256") != digest:
        raise ValueError(f"manifest checksum does not match {path.name}")
    return manifest


def read_rows(path: str | Path) -> list[dict[str, str]]:
    """Read benchmark rows after verifying their adjacent manifest."""
    path = Path(path)
    read_manifest(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_unique(rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    """Reject duplicate logical result rows."""
    seen = set()
    for row in rows:
        identity = tuple(row[field] for field in fields)
        if identity in seen:
            raise ValueError(f"duplicate row identity for {', '.join(fields)}: {identity}")
        seen.add(identity)


def finite_float(value: str, label: str) -> float:
    """Parse a finite numeric artifact value."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be numeric") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def positive_float(value: str, label: str) -> float:
    """Parse a finite, strictly positive artifact value."""
    number = finite_float(value, label)
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.8,
        "figure.titlesize": 12.5,
        "figure.titleweight": "bold",
    })


def titled(fig, ax, main: str, sub: str) -> None:
    """Bold one-line title + smaller gray metadata subtitle (keeps titles from overflowing)."""
    ax.set_title(sub, fontsize=9.5, color="#666666", fontweight="normal", pad=10)
    fig.suptitle(main, fontsize=13.5, fontweight="bold", y=1.0)


def save(fig, path: str) -> None:
    """Create the destination directory, save consistently, and report the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
