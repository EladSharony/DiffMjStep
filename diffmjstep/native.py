"""Lazy JIT build + thin wrapper for the native rollout VJP backend.

The C++ extension is compiled on first use via torch.utils.cpp_extension.load and
cached under TORCH_EXTENSIONS_DIR; subsequent imports are instant. It links the
MuJoCo shared library that ships with the `mujoco` wheel, so no system MuJoCo install
is required. Building needs a C++17 compiler.
"""
from __future__ import annotations

import functools
import os
import shlex
import sys
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import torch


def _ensure_ninja_on_path() -> None:
    # torch's JIT build shells out to the `ninja` binary; pip installs it next to the
    # interpreter, which is not on PATH unless the venv is activated. Add it so the
    # extension builds whether or not the environment was sourced.
    bin_dir = os.path.dirname(sys.executable)
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir and bin_dir not in parts:
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def _find_mujoco_library(mujoco_dir: Path, version: str) -> Path:
    if sys.platform == "linux":
        name = f"libmujoco.so.{version}"
    elif sys.platform == "darwin":
        name = f"libmujoco.{version}.dylib"
    else:
        raise RuntimeError(
            f"native cpp_vjp does not support {sys.platform!r}; "
            "supported targets are Linux and macOS"
        )
    library = mujoco_dir / name
    if not library.is_file():
        raise RuntimeError(f"could not find MuJoCo shared library {library}")
    return library


@functools.lru_cache(maxsize=1)
def load_extension():
    import mujoco
    from torch.utils.cpp_extension import load

    _ensure_ninja_on_path()
    mujoco_dir = Path(mujoco.__file__).resolve().parent
    source_resource = files("diffmjstep") / "cpp" / "rollout_vjp.cpp"
    library = _find_mujoco_library(mujoco_dir, version=mujoco.__version__)
    include = mujoco_dir / "include"
    if not include.is_dir():
        raise RuntimeError(f"MuJoCo headers are missing: {include}")

    pthread = ["-pthread"] if sys.platform == "linux" else []
    with as_file(source_resource) as source:
        if not source.is_file():
            raise RuntimeError(f"packaged native source is missing: {source}")
        return load(
            name="diffmjstep_cpp",
            sources=[str(source)],
            extra_include_paths=[str(include)],
            extra_cflags=["-O3", "-std=c++17", *pthread],
            extra_ldflags=[
                shlex.quote(str(library)),
                shlex.quote(f"-Wl,-rpath,{mujoco_dir}"),
                *pthread,
            ],
            verbose=False,
        )


def rollout_backward_vjp(
    model: Any,
    full_snapshots: Any,
    U: Any,
    grad_X: Any | None,
    grad_Y: Any | None = None,
    *,
    return_all: bool,
    needs_control_grad: bool,
    eps: float,
    centered: bool,
    nthread: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Native reverse-time VJP.

    full_snapshots: [B, T+1, nfull], U: [B, T, nu]. grad_X is optional
    [B, T+1, nx] or [B, nx], and grad_Y is optional [B, T, nsensordata]. Returns
    (grad_x0 [B, nx], grad_U [B, T, nu] or None).
    """
    import mujoco

    ext = load_extension()
    full_snapshots = (
        torch.as_tensor(full_snapshots).detach().to(torch.float64).cpu().contiguous()
    )
    U = torch.as_tensor(U).detach().to(torch.float64).cpu().contiguous()
    grad_X = (
        torch.empty(0, dtype=torch.float64, device="cpu")
        if grad_X is None
        else torch.as_tensor(grad_X).detach().to(torch.float64).cpu().contiguous()
    )
    grad_Y = (
        torch.empty(0, dtype=torch.float64, device="cpu")
        if grad_Y is None
        else torch.as_tensor(grad_Y).detach().to(torch.float64).cpu().contiguous()
    )
    has_work = (
        full_snapshots.ndim == 3
        and U.ndim == 3
        and full_snapshots.shape[0] > 0
        and U.shape[1] > 0
    )
    worker_count = (
        min(max(1, int(nthread)), int(full_snapshots.shape[0]))
        if has_work
        else 0
    )
    scratch_data = [mujoco.MjData(model) for _ in range(worker_count)]
    data_addresses = torch.tensor(
        [data._address for data in scratch_data], dtype=torch.int64, device="cpu"
    )
    grad_x0, grad_U = ext.rollout_backward_vjp(
        full_snapshots,
        U,
        grad_X,
        grad_Y,
        data_addresses,
        int(model._address),
        int(nthread),
        bool(return_all),
        bool(needs_control_grad),
        float(eps),
        bool(centered),
    )
    return grad_x0, grad_U if needs_control_grad else None
