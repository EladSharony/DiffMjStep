from __future__ import annotations

from typing import Any

import numpy as np

import mujoco
from .state import (
    FULLPHYSICS,
    as_numpy,
    infer_state_dims,
    set_data_control,
    set_data_state,
    snapshot_model,
    validate_control,
    validate_eps,
    validate_state,
    validate_tensor,
)


def _enum_value(value: Any) -> int:
    return int(value.value) if hasattr(value, "value") else int(value)


def check_transition_fd_integrator(model: Any) -> None:
    if _enum_value(model.opt.integrator) == _enum_value(mujoco.mjtIntegrator.mjINT_RK4):
        raise ValueError("mjd_transitionFD does not support RK4 integrator; use Euler or implicit integration")


def _as_real_float64(name: str, value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        validate_tensor(name, value)
        array = as_numpy(value)
    else:
        array = np.asarray(value)
        if isinstance(value, np.ndarray) and not np.issubdtype(array.dtype, np.floating):
            raise TypeError(f"{name} must be a real floating array, got {array.dtype}")
        if not isinstance(value, np.ndarray) and array.dtype.kind not in {"i", "u", "f"}:
            raise TypeError(f"{name} must contain real numeric values, got {array.dtype}")
    return np.asarray(array, dtype=np.float64)


def _as_batched_state(model: Any, x: Any) -> np.ndarray:
    x_np = _as_real_float64("state", x)
    validate_state(model, x_np, require_square=False)
    return x_np[None, :] if x_np.ndim == 1 else x_np


def _as_batched_control(model: Any, u: Any) -> np.ndarray:
    u_np = _as_real_float64("control", u)
    validate_control(model, u_np, allow_sequence=False)
    return u_np[None, :] if u_np.ndim == 1 else u_np


def _return_like_input(reference: Any, value: np.ndarray):
    if hasattr(reference, "detach"):
        import torch

        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    return value


def _transition_fd_at_snapshot(
    model: Any,
    data: Any,
    full_snapshot: np.ndarray,
    u: np.ndarray,
    *,
    sensors: bool = False,
    needs_control_grad: bool = True,
    eps: float,
    centered: bool,
):
    nx = 2 * int(model.nv) + int(model.na)
    nu = int(model.nu)
    ns = int(model.nsensordata)
    A = np.empty((nx, nx), dtype=np.float64)
    B = np.empty((nx, nu), dtype=np.float64) if needs_control_grad else None
    C = np.empty((ns, nx), dtype=np.float64) if sensors else None
    D = (
        np.empty((ns, nu), dtype=np.float64)
        if sensors and needs_control_grad
        else None
    )

    mujoco.mj_resetData(model, data)
    mujoco.mj_setState(model, data, full_snapshot, FULLPHYSICS)
    data.ctrl[:] = u
    mujoco.mj_forward(model, data)
    forward_difference_D = centered and D is not None
    mujoco.mjd_transitionFD(
        model,
        data,
        eps,
        int(centered),
        A,
        B,
        C,
        None if forward_difference_D else D,
    )
    if forward_difference_D:
        # MuJoCo's centered-D ordering reverses interior control derivatives; a
        # D-only forward pass also preserves its one-sided control-limit handling.
        mujoco.mjd_transitionFD(model, data, eps, 0, None, None, None, D)
    return A, B, C, D


def _linearize(
    model: Any,
    x: Any,
    u: Any,
    *,
    sensors: bool = False,
    eps: float = 1e-8,
    centered: bool = True,
):
    dims = infer_state_dims(model)
    x_np = _as_batched_state(model, x)
    nx = dims.nx_tangent

    u_np = _as_batched_control(model, u)
    if x_np.shape[0] != u_np.shape[0]:
        raise ValueError(f"state/control batch sizes must match, got {x_np.shape[0]} and {u_np.shape[0]}")
    batch_size = x_np.shape[0]
    A = np.empty((batch_size, nx, nx), dtype=np.float64)
    B = np.empty((batch_size, nx, dims.nu), dtype=np.float64)
    C = np.empty((batch_size, int(model.nsensordata), nx), dtype=np.float64) if sensors else None
    D = np.empty((batch_size, int(model.nsensordata), dims.nu), dtype=np.float64) if sensors else None

    data = mujoco.MjData(model)
    centered_flag = 1 if centered else 0
    for batch in range(batch_size):
        mujoco.mj_resetData(model, data)
        set_data_state(model, data, x_np[batch])
        set_data_control(model, data, u_np[batch])
        mujoco.mj_forward(model, data)
        forward_difference_D = sensors and centered
        mujoco.mjd_transitionFD(
            model,
            data,
            eps,
            centered_flag,
            A[batch],
            B[batch],
            C[batch] if sensors else None,
            None if forward_difference_D else D[batch] if sensors else None,
        )
        if forward_difference_D:
            # MuJoCo's centered-D ordering workaround: request D alone with forward
            # differences, which also keeps one-sided control-limit behavior intact.
            mujoco.mjd_transitionFD(
                model, data, eps, 0, None, None, None, D[batch]
            )

    A_out = _return_like_input(x, A)
    B_out = _return_like_input(x, B)
    if sensors:
        return A_out, B_out, _return_like_input(x, C), _return_like_input(x, D)
    return A_out, B_out


def mj_linearize(
    model: Any,
    x: Any,
    u: Any,
    *,
    sensors: bool = False,
    eps: float = 1e-8,
    centered: bool = True,
):
    """Compute MuJoCo transition Jacobians A/B, and optionally sensor Jacobians C/D.

    Returned tensors/arrays are batched: A has shape [B, nx, nx], B has shape [B, nx, nu].

    For free-joint / quaternion models (nq != nv), the input state is the full
    [nq + nv + na] vector and A/B are returned in MuJoCo's tangent space (nx = 2*nv + na) --
    the standard linearization for trajopt/iLQR. The qpos-square form is only defined when
    nq == nv.
    """
    private_model = snapshot_model(model)
    eps = validate_eps(eps)
    check_transition_fd_integrator(private_model)
    return _linearize(
        private_model,
        x,
        u,
        sensors=sensors,
        eps=eps,
        centered=centered,
    )
