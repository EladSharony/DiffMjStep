from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

import mujoco


FULLPHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS


def validate_tensor(name: str, value: object) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{name} must be a float32 or float64 torch.Tensor, "
            f"got {type(value).__name__}"
        )
    if value.layout != torch.strided:
        raise TypeError(f"{name} must have strided layout, got {value.layout}")
    if value.dtype not in {torch.float32, torch.float64}:
        raise TypeError(f"{name} must be float32 or float64, got {value.dtype}")


def validate_devices(state: torch.Tensor, control: torch.Tensor) -> None:
    if state.device != control.device:
        raise ValueError(
            "state and control tensors must be on the same device, got "
            f"{state.device} and {control.device}"
        )
    if state.device.type != "cpu":
        raise ValueError(f"state and control must be on CPU, got {state.device}")


def validate_eps(eps: float) -> float:
    eps = float(eps)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be finite and positive, got {eps!r}")
    return eps


def validate_fullphysics(model: mujoco.MjModel) -> int:
    actual = int(mujoco.mj_stateSize(model, FULLPHYSICS))
    expected = 1 + int(model.nq) + int(model.nv) + int(model.na)
    if actual != expected:
        raise NotImplementedError(
            "plugin/history FULLPHYSICS state is unsupported: "
            f"expected {expected} values (time+qpos+qvel+act), got {actual}"
        )
    return actual


def snapshot_model(model: mujoco.MjModel) -> mujoco.MjModel:
    private = copy.copy(model)
    validate_fullphysics(private)
    private.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
    return private


@dataclass(frozen=True)
class StateDims:
    nq: int
    nv: int
    na: int
    nu: int
    nx_qpos: int
    nx_tangent: int


def infer_state_dims(model: Any) -> StateDims:
    """Return the state dimensions DiffMjStep uses for the given MuJoCo model."""
    nq = int(model.nq)
    nv = int(model.nv)
    na = int(model.na)
    nu = int(model.nu)
    return StateDims(
        nq=nq,
        nv=nv,
        na=na,
        nu=nu,
        nx_qpos=nq + nv + na,
        nx_tangent=2 * nv + na,
    )


def _shape(value: Any) -> tuple[int, ...]:
    if not hasattr(value, "shape"):
        raise TypeError("expected an array-like object with a shape")
    return tuple(int(dim) for dim in value.shape)


def check_rollout_state_supported(model: Any) -> None:
    dims = infer_state_dims(model)
    if dims.nq != dims.nv:
        raise NotImplementedError(
            "rollout state requires nq == nv; free-joint and quaternion models "
            "are supported by mj_linearize only"
        )


def validate_state(model: Any, x: Any, *, require_square: bool = True) -> None:
    """Validate a state tensor/array shape; raise ValueError if it is wrong."""
    shape = _shape(x)
    if len(shape) not in (1, 2):
        raise ValueError(f"state must have shape [nx] or [B, nx], got {shape}")

    if require_square:
        check_rollout_state_supported(model)
    nx = infer_state_dims(model).nx_qpos
    if shape[-1] != nx:
        raise ValueError(f"state last dimension must be {nx}, got {shape[-1]}")


def validate_control(model: Any, u: Any, allow_sequence: bool = True) -> None:
    """Validate a control tensor/array shape; raise ValueError if it is wrong."""
    shape = _shape(u)
    allowed = (1, 2, 3) if allow_sequence else (1, 2)
    if len(shape) not in allowed:
        expected = "[nu], [B, nu], or [B, T, nu]" if allow_sequence else "[nu] or [B, nu]"
        raise ValueError(f"control must have shape {expected}, got {shape}")

    nu = int(model.nu)
    if shape[-1] != nu:
        raise ValueError(f"control last dimension must be {nu}, got {shape[-1]}")


def as_numpy(value: Any) -> np.ndarray:
    """Convert a CPU tensor/array-like object to a detached NumPy array."""
    if hasattr(value, "detach"):
        if getattr(value, "device", None) is not None and value.device.type != "cpu":
            raise ValueError("MuJoCo CPU backend received a non-CPU tensor")
        return value.detach().cpu().numpy()
    return np.asarray(value)


def set_data_state(model: Any, data: Any, x: np.ndarray) -> None:
    """Set qpos/qvel/act in mjData from a full [nq+nv+na] DiffMjStep state vector.

    Slicing qpos/qvel/act is valid for any model, including free joints. Rollout's
    nq == nv restriction is enforced by validate_state at the public boundary.
    """
    if x.ndim != 1:
        raise ValueError(f"state must have shape [nx], got {_shape(x)}")
    validate_state(model, x, require_square=False)
    x = np.asarray(x, dtype=np.float64)
    dims = infer_state_dims(model)

    data.qpos[:] = x[: dims.nq]
    data.qvel[:] = x[dims.nq : dims.nq + dims.nv]
    if dims.na:
        data.act[:] = x[dims.nq + dims.nv : dims.nq + dims.nv + dims.na]


def set_data_control(model: Any, data: Any, u: np.ndarray) -> None:
    u = np.asarray(u, dtype=np.float64)
    if u.shape[-1] != int(model.nu):
        raise ValueError(f"control last dimension must be {model.nu}, got {u.shape[-1]}")
    data.ctrl[:] = u
