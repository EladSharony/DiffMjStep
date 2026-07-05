from __future__ import annotations

import operator
import os
import warnings
from typing import Any

import numpy as np
import torch
from torch.autograd.function import once_differentiable

import mujoco
from mujoco import rollout
from . import native
from .linearize import (
    _transition_fd_at_snapshot,
    check_transition_fd_integrator,
)
from .state import (
    infer_state_dims,
    snapshot_model,
    validate_control,
    validate_devices,
    validate_eps,
    validate_fullphysics,
    validate_state,
    validate_tensor,
)


_SUPPORTED_BACKENDS = {"auto", "mujoco_rollout", "cpp_vjp"}
_AUTO_NATIVE_DISABLED = False
_AUTO_NATIVE_LOADING_PID: int | None = None


def _check_backend(backend: str) -> None:
    if backend not in _SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(_SUPPORTED_BACKENDS))
        raise NotImplementedError(f"backend {backend!r} is unknown; choose from {supported}")


def _auto_native_ready() -> bool:
    global _AUTO_NATIVE_DISABLED, _AUTO_NATIVE_LOADING_PID
    pid = os.getpid()
    if _AUTO_NATIVE_DISABLED:
        return False
    if _AUTO_NATIVE_LOADING_PID == pid:
        return False
    # No Python call between the checks and assignment: the GIL makes this the single
    # loading claim without a lock that could be inherited held by fork.
    _AUTO_NATIVE_LOADING_PID = pid
    try:
        native.load_extension()
    except Exception as exc:
        _AUTO_NATIVE_DISABLED = True
        warnings.warn(
            "DiffMjStep could not load the native cpp_vjp backend; "
            "backend='auto' will keep MuJoCo's C rollout forward and use the "
            "Python VJP. Install Ninja and a C++17 compiler. "
            f"Native load failed with {type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    else:
        return True
    finally:
        if _AUTO_NATIVE_LOADING_PID == pid:
            _AUTO_NATIVE_LOADING_PID = None

def _python_callbacks() -> tuple[Any | None, ...]:
    return (
        mujoco.get_mjcb_act_bias(),
        mujoco.get_mjcb_act_dyn(),
        mujoco.get_mjcb_act_gain(),
        mujoco.get_mjcb_contactfilter(),
        mujoco.get_mjcb_control(),
        mujoco.get_mjcb_passive(),
        mujoco.get_mjcb_sensor(),
        mujoco.get_mjcb_time(),
    )


def _require_same_python_callbacks(expected: tuple[Any | None, ...]) -> None:
    if any(
        current is not captured
        for current, captured in zip(_python_callbacks(), expected, strict=True)
    ):
        raise RuntimeError(
            "MuJoCo callback registration must remain unchanged from forward through "
            "backward; register callbacks before mj_rollout"
        )


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _public_output_from_snapshots(
    snapshots: np.ndarray,
    nx: int,
    return_all: bool,
) -> np.ndarray:
    public_states = (
        snapshots[..., 1 : 1 + nx]
        if return_all
        else snapshots[:, -1, 1 : 1 + nx]
    )
    return np.array(
        public_states,
        dtype=np.float64,
        copy=True,
        order="C",
    )


_AUTO_NTHREAD_CAP = 8  # past this, per-call mjData allocation outweighs added parallelism


def _auto_nthread(batch_size: int, nthread: int | None) -> int:
    if batch_size == 0:
        return 1
    if nthread is not None:
        return min(max(1, nthread), batch_size)
    if batch_size < 8:
        return 1
    return min(_AUTO_NTHREAD_CAP, os.cpu_count() or 1, batch_size)


def _rollout_mujoco(
    model: Any,
    x0: np.ndarray,
    U: np.ndarray,
    *,
    return_all: bool,
    sensors: bool,
    nthread: int | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Fast forward rollout via mujoco.rollout (C-threaded, GIL released)."""
    dims = infer_state_dims(model)
    nfull = validate_fullphysics(model)
    batch_size, horizon, _ = U.shape
    init = np.zeros((batch_size, nfull), dtype=np.float64)
    init[:, 1:] = x0  # leading element is simulation time = 0

    snapshots = np.empty((batch_size, horizon + 1, nfull), dtype=np.float64)
    snapshots[:, 0, :] = init
    if batch_size == 0 or horizon == 0:
        X = _public_output_from_snapshots(
            snapshots, dims.nx_qpos, return_all
        )
        Y = (
            np.empty(
                (batch_size, horizon, int(model.nsensordata)),
                dtype=np.float64,
            )
            if sensors
            else None
        )
        return X, Y, snapshots

    nthread = _auto_nthread(batch_size, nthread)
    if nthread > 1:
        data_arg: mujoco.MjData | list[mujoco.MjData] = [
            mujoco.MjData(model) for _ in range(nthread)
        ]
    else:
        data_arg = mujoco.MjData(model)

    states, sensordata = rollout.rollout(model, data_arg, init, U)
    snapshots[:, 1:, :] = states
    X = _public_output_from_snapshots(snapshots, dims.nx_qpos, return_all)
    Y = sensordata if sensors else None
    return X, Y, snapshots


def _rollout_vjp_numpy(
    model: Any,
    U: np.ndarray,
    full_snapshots: np.ndarray,
    grad_X: np.ndarray | None,
    grad_Y: np.ndarray | None,
    *,
    return_all: bool,
    needs_control_grad: bool,
    eps: float,
    centered: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    batch_size, horizon, _ = U.shape
    nx = infer_state_dims(model).nx_qpos
    grad_x0 = np.zeros((batch_size, nx), dtype=np.float64)
    grad_U = np.zeros_like(U) if needs_control_grad else None
    if batch_size == 0 or (grad_X is None and grad_Y is None):
        return grad_x0, grad_U
    if horizon == 0:
        if grad_X is not None:
            grad_x0[:] = grad_X[:, 0] if return_all else grad_X
        return grad_x0, grad_U

    work_data = mujoco.MjData(model)
    for batch in range(batch_size):
        lambda_x = (
            np.zeros(nx, dtype=np.float64)
            if return_all or grad_X is None
            else grad_X[batch]
        )
        for step in reversed(range(horizon)):
            if return_all and grad_X is not None:
                lambda_x += grad_X[batch, step + 1]
            A, B, C, D = _transition_fd_at_snapshot(
                model,
                work_data,
                full_snapshots[batch, step],
                U[batch, step],
                sensors=grad_Y is not None,
                needs_control_grad=needs_control_grad,
                eps=eps,
                centered=centered,
            )
            gy = None if grad_Y is None else grad_Y[batch, step]
            if needs_control_grad:
                grad_U[batch, step] = lambda_x @ B
                if gy is not None:
                    grad_U[batch, step] += gy @ D
            lambda_x = lambda_x @ A
            if gy is not None:
                lambda_x += gy @ C
        grad_x0[batch] = lambda_x
        if return_all and grad_X is not None:
            grad_x0[batch] += grad_X[batch, 0]
    return grad_x0, grad_U


class _MjRolloutFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x0: torch.Tensor,
        U: torch.Tensor,
        model: Any,
        X_np: np.ndarray,
        Y_np: np.ndarray | None,
        full_snapshots: np.ndarray,
        return_all: bool,
        eps: float,
        centered: bool,
        nthread: int | None,
        backward_impl: str,
        callbacks: tuple[Any | None, ...] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        ctx.model = model
        ctx.full_snapshots = full_snapshots
        ctx.return_all = return_all
        ctx.eps = eps
        ctx.centered = centered
        ctx.nthread = nthread
        ctx.backward_impl = backward_impl
        ctx.callbacks = callbacks
        ctx.save_for_backward(U.detach().cpu())

        X_t = torch.as_tensor(X_np, dtype=x0.dtype, device=x0.device)
        Y_t = (
            torch.as_tensor(Y_np, dtype=x0.dtype, device=x0.device)
            if Y_np is not None
            else x0.new_empty(0)
        )
        return X_t, Y_t

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_X: torch.Tensor | None, grad_Y: torch.Tensor | None):
        (U_cpu,) = ctx.saved_tensors
        model = ctx.model
        return_all = ctx.return_all

        U_np = U_cpu.numpy().astype(np.float64, copy=False)
        full_snapshots = ctx.full_snapshots
        callbacks = ctx.callbacks

        device = U_cpu.device

        if grad_Y is not None and not bool(torch.any(grad_Y)):
            grad_Y = None

        if grad_X is None and grad_Y is None:
            grad_x0_t = U_cpu.new_zeros(
                (full_snapshots.shape[0], infer_state_dims(model).nx_qpos)
            )
            grad_U_t = (
                torch.zeros_like(U_cpu)
                if ctx.needs_input_grad[1]
                else None
            )
            return (
                grad_x0_t,
                grad_U_t,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        if callbacks is None:
            raise RuntimeError("internal error: differentiable rollout lost callback state")
        _require_same_python_callbacks(callbacks)

        if any(callback is not None for callback in callbacks):
            use_native = False
        elif ctx.backward_impl == "cpp":
            native.load_extension()
            use_native = True
        elif ctx.backward_impl == "auto":
            use_native = _auto_native_ready()
        else:
            use_native = False

        if use_native:
            nthr = _auto_nthread(int(full_snapshots.shape[0]), ctx.nthread)
            grad_x0_t, grad_U_t = native.rollout_backward_vjp(
                model,
                full_snapshots,
                U_np,
                grad_X,
                grad_Y,
                return_all=return_all,
                needs_control_grad=ctx.needs_input_grad[1],
                eps=ctx.eps,
                centered=ctx.centered,
                nthread=nthr,
            )
            grad_x0_t = grad_x0_t.to(dtype=U_cpu.dtype, device=device)
            if grad_U_t is not None:
                grad_U_t = grad_U_t.to(dtype=U_cpu.dtype, device=device)
        else:
            grad_X_np = (
                grad_X.detach().cpu().numpy().astype(np.float64, copy=False)
                if grad_X is not None
                else None
            )
            grad_Y_np = (
                grad_Y.detach().cpu().numpy().astype(np.float64, copy=False)
                if grad_Y is not None
                else None
            )
            grad_x0, grad_U = _rollout_vjp_numpy(
                model,
                U_np,
                full_snapshots,
                grad_X_np,
                grad_Y_np,
                return_all=return_all,
                needs_control_grad=ctx.needs_input_grad[1],
                eps=ctx.eps,
                centered=ctx.centered,
            )

            grad_x0_t = torch.as_tensor(
                grad_x0, dtype=U_cpu.dtype, device=device
            )
            grad_U_t = (
                torch.as_tensor(grad_U, dtype=U_cpu.dtype, device=device)
                if grad_U is not None
                else None
            )

        _require_same_python_callbacks(callbacks)

        return (
            grad_x0_t,
            grad_U_t,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def mj_rollout(
    model: Any,
    x0: torch.Tensor,
    U: torch.Tensor,
    *,
    backend: str = "auto",
    return_all: bool = True,
    return_sensors: bool = False,
    eps: float = 1e-8,
    centered: bool = True,
    nthread: int | None = None,
):
    """Roll out MuJoCo dynamics from PyTorch tensors.

    backend="auto" uses the C-threaded mujoco.rollout forward and native threaded
    backward when it can be loaded, with a Python VJP fallback. backend="mujoco_rollout"
    uses the fast forward with the Python VJP; backend="cpp_vjp" requires native backward.

    With return_sensors=True the call returns (X, Y), where Y is the [B, T, nsensordata]
    sensor trajectory, and the loss may depend on X, Y, or both.
    """
    validate_tensor("state", x0)
    validate_tensor("control", U)
    validate_devices(x0, U)
    if x0.dtype != U.dtype:
        raise TypeError(
            "state and control tensors must have the same dtype, got "
            f"{x0.dtype} and {U.dtype}"
        )
    eps = validate_eps(eps)
    _check_backend(backend)
    validate_state(model, x0)
    validate_control(model, U, allow_sequence=True)
    if x0.ndim != 2:
        raise ValueError("state must include a batch axis with shape [B, nx]")
    if U.ndim != 3:
        raise ValueError("control must include batch and time axes with shape [B, T, nu]")
    if x0.shape[0] != U.shape[0]:
        raise ValueError(
            f"state/control batch sizes must match, got {x0.shape[0]} and {U.shape[0]}"
        )

    needs_graph = torch.is_grad_enabled() and (
        x0.requires_grad or U.requires_grad
    )
    needs_derivatives = needs_graph
    warmstart_disabled = int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
    if needs_derivatives or not (int(model.opt.disableflags) & warmstart_disabled):
        # Copy first, then validate the private model so a concurrent caller mutation
        # cannot invalidate the model between validation and snapshotting.
        runtime_model = snapshot_model(model)
    else:
        validate_fullphysics(model)
        runtime_model = model

    if needs_derivatives or backend == "cpp_vjp":
        check_transition_fd_integrator(runtime_model)

    backward_impl = (
        "auto"
        if backend == "auto"
        else "cpp"
        if backend == "cpp_vjp"
        else "python"
    )
    U_contiguous = U.contiguous()
    x0_np = np.asarray(_tensor_to_numpy(x0), dtype=np.float64)
    U_np = np.asarray(_tensor_to_numpy(U_contiguous), dtype=np.float64)

    callbacks = _python_callbacks() if needs_derivatives else None
    X_np, Y_np, full_snapshots = _rollout_mujoco(
        runtime_model,
        x0_np,
        U_np,
        sensors=return_sensors,
        return_all=return_all,
        nthread=nthread,
    )
    if callbacks is not None:
        _require_same_python_callbacks(callbacks)
    X_out, Y_out = _MjRolloutFunction.apply(
        x0,
        U_contiguous,
        runtime_model,
        X_np,
        Y_np,
        full_snapshots,
        return_all,
        eps,
        centered,
        nthread,
        backward_impl,
        callbacks,
    )
    return (X_out, Y_out) if return_sensors else X_out


def mj_step(
    model: Any,
    x0: torch.Tensor,
    u: torch.Tensor,
    *,
    nstep: int = 1,
    backend: str = "auto",
    return_sensors: bool = False,
    eps: float = 1e-8,
    centered: bool = True,
    nthread: int | None = None,
):
    """Apply one action for nstep MuJoCo steps and return the final state.

    With return_sensors=True returns (xT, Y), where Y is the [B, nstep, nsensordata]
    sensor trajectory over the repeated steps.
    """
    validate_tensor("state", x0)
    validate_tensor("control", u)
    eps = validate_eps(eps)

    try:
        nstep = operator.index(nstep)
    except TypeError as exc:
        raise TypeError(
            f"nstep must be an integer, got {type(nstep).__name__}"
        ) from exc
    if nstep < 1:
        raise ValueError(f"nstep must be >= 1, got {nstep}")
    validate_state(model, x0)
    validate_control(model, u, allow_sequence=False)
    if x0.ndim != 2 or u.ndim != 2:
        raise ValueError("state and control must include batch axes [B, nx] and [B, nu]")
    if x0.shape[0] != u.shape[0]:
        raise ValueError(
            f"state/control batch sizes must match, got {x0.shape[0]} and {u.shape[0]}"
        )

    U = u[:, None, :].expand(-1, nstep, -1).contiguous()

    return mj_rollout(
        model,
        x0,
        U,
        backend=backend,
        return_all=False,
        return_sensors=return_sensors,
        eps=eps,
        centered=centered,
        nthread=nthread,
    )


__all__ = ["mj_rollout", "mj_step"]
