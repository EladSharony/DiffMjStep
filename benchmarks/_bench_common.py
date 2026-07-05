"""Shared helpers for DiffMjStep benchmarks.

Keep this small: a model registry, deterministic input generation, separated
forward/backward timing, and CSV/table output based on actual result fields.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import mujoco

if __package__:
    from ._manifest import _csv_path, _validate_experiment, _write_csv_and_manifest
else:
    from _manifest import _csv_path, _validate_experiment, _write_csv_and_manifest

# Make the repo root importable when these scripts are run from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from diffmjstep.state import (  # noqa: E402
    infer_state_dims,
    set_data_control,
    set_data_state,
)


# Benchmark models are the dm_control suite (the standard, citable env set). There are no
# repo-local model XMLs; dm_control domains usable today (nq == nv after the Euler override).
# Sourced live from the installed dm_control package as XML + assets and loaded through the
# raw mujoco bindings -- dm_control's own Physics wrapper is version-skewed against mujoco
# 3.9, but get_model_and_assets() + from_xml_string is not. humanoid/fish are free-base
# (nq != nv) and stay blocked until tangent-state support lands.
_DM_CONTROL_DOMAIN = {
    "dm_cartpole": "cartpole", "dm_acrobot": "acrobot", "dm_cheetah": "cheetah",
    "dm_walker": "walker", "dm_hopper": "hopper", "dm_finger": "finger",
    "dm_reacher": "reacher", "dm_pendulum": "pendulum", "dm_pointmass": "point_mass",
}

MODEL_NAMES = list(_DM_CONTROL_DOMAIN)

CSV_COLUMNS = [
    "backend",
    "model",
    "batch",
    "horizon",
    "dtype",
    "nthread",
    "grad_rel_error",
    "grad_cosine",
]


def build_model(name: str) -> mujoco.MjModel:
    try:
        return _build_dm_control(_DM_CONTROL_DOMAIN[name])
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}") from None


def _build_dm_control(domain: str) -> mujoco.MjModel:
    import importlib

    try:
        mod = importlib.import_module(f"dm_control.suite.{domain}")
    except ImportError as exc:
        raise ImportError("dm_control models need `pip install dm_control`") from exc
    xml, assets = mod.get_model_and_assets()
    model = mujoco.MjModel.from_xml_string(xml, assets)
    # mjd_transitionFD rejects RK4 (a MuJoCo limit); dm_control cartpole/acrobot default to it.
    # Normalize to Euler -- this changes the physics from canonical dm_control, same caveat as
    # the repo-local Gym assets, so don't compare these timings to published dm_control numbers.
    if int(model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_RK4):
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    return model


def make_inputs(
    model: mujoco.MjModel,
    batch: int,
    horizon: int,
    dtype: str,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic small initial state and control sequence for a model."""
    rng = np.random.default_rng(seed)
    nx = int(model.nq + model.nv + model.na)
    x0 = np.zeros((batch, nx), dtype=np.float64)
    x0[:, : model.nq] = 0.05 * rng.standard_normal((batch, int(model.nq)))
    U = 0.1 * rng.standard_normal((batch, horizon, int(model.nu)))
    torch_dtype = getattr(torch, dtype)
    return (
        torch.tensor(x0, dtype=torch_dtype),
        torch.tensor(U, dtype=torch_dtype),
    )


def python_rollout(
    model: mujoco.MjModel, x0: torch.Tensor, U: torch.Tensor
) -> torch.Tensor:
    """Small Python-loop baseline for forward benchmarks only."""
    dims = infer_state_dims(model)
    x0_np = x0.detach().numpy()
    controls = U.detach().numpy()
    states = np.empty((x0.shape[0], U.shape[1] + 1, dims.nx_qpos))
    states[:, 0] = x0_np
    for batch in range(x0.shape[0]):
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        set_data_state(model, data, x0_np[batch])
        mujoco.mj_forward(model, data)
        for step in range(U.shape[1]):
            set_data_control(model, data, controls[batch, step])
            mujoco.mj_step(model, data)
            states[batch, step + 1] = np.concatenate(
                (data.qpos, data.qvel, data.act)
            )
    return torch.as_tensor(states, dtype=x0.dtype)


def samples_forward(thunk: Callable[[], Any], *, repeat: int, warmup: int) -> list[float]:
    """All per-run forward times in milliseconds (for distribution plots)."""
    for _ in range(warmup):
        value = thunk()
        del value
    out = []
    for _ in range(repeat):
        start = time.perf_counter()
        value = thunk()
        elapsed_ms = 1e3 * (time.perf_counter() - start)
        del value
        out.append(elapsed_ms)
    return out


def samples_backward(build_loss: Callable[[], torch.Tensor], *, repeat: int, warmup: int) -> list[float]:
    """All per-run backward times in milliseconds.

    `build_loss` rebuilds the graph and returns a scalar loss each call; only the
    `.backward()` call is timed, so the forward pass is excluded.
    """

    def one() -> float:
        loss = build_loss()
        start = time.perf_counter()
        loss.backward()
        return 1e3 * (time.perf_counter() - start)

    for _ in range(warmup):
        one()
    return [one() for _ in range(repeat)]


def write_csv(
    path: str | Path, rows: list[dict[str, Any]], **manifest: Any
) -> None:
    path = _csv_path(path)
    _validate_experiment(manifest)
    if not rows:
        raise ValueError("cannot write an empty benchmark result")
    columns = _columns(rows)
    if not columns:
        raise ValueError("cannot write a benchmark result without columns")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_csv_and_manifest(path, output.getvalue().encode("utf-8"), **manifest)


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    present = set().union(*(row.keys() for row in rows))
    preferred = [column for column in CSV_COLUMNS if column in present]
    return preferred + sorted(present - set(preferred))


def build_argparser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--backend", nargs="+", default=["python_ref"])
    parser.add_argument("--model", nargs="+", default=["dm_acrobot"], choices=MODEL_NAMES)
    parser.add_argument("--batch", nargs="+", type=int, default=[1, 16, 256])
    parser.add_argument("--horizon", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default=None, help="optional CSV output path")
    return parser
