"""Validation and error-path contract: bad shapes, unsupported options, flag combos."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import mujoco

from diffmjstep import mj_linearize, mj_rollout, mj_step


def test_rk4_rejected_by_linearize(pendulum):
    pendulum.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    with pytest.raises(ValueError, match="RK4"):
        mj_linearize(pendulum, np.array([0.1, 0.0]), np.array([0.0]))


@pytest.mark.parametrize(
    ("x", "u", "name"),
    [
        ([np.timedelta64(0, "s"), np.timedelta64(0, "s")], [0], "state"),
        ([0, 0], [np.timedelta64(0, "s")], "control"),
    ],
)
def test_linearize_rejects_timedelta_literal_sequences(pendulum, x, u, name):
    with pytest.raises(TypeError, match=rf"{name}.*real numeric"):
        mj_linearize(pendulum, x, u)


def test_linearize_accepts_integer_literal_sequences(pendulum):
    A, B = mj_linearize(pendulum, [0, 0], [0])
    assert A.shape == (1, 2, 2)
    assert B.shape == (1, 2, 1)


def test_rk4_rejected_by_cpp_vjp_at_call(pendulum):
    pendulum.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    x0 = torch.zeros(1, 2, dtype=torch.float64)
    U = torch.zeros(1, 1, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="RK4"):
        mj_rollout(pendulum, x0, U, backend="cpp_vjp")


def test_rk4_differentiable_call_is_rejected_at_call(pendulum):
    pendulum.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    x0 = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    U = torch.zeros(1, 1, 1, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="RK4"):
        mj_rollout(pendulum, x0, U)


@pytest.mark.parametrize("bad_eps", [0.0, -1e-8, float("nan"), float("inf")])
def test_rollout_rejects_invalid_eps(pendulum, bad_eps):
    x = torch.zeros(1, 2, dtype=torch.float64)
    u = torch.zeros(1, 1, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="eps must be finite and positive"):
        mj_rollout(pendulum, x, u, eps=bad_eps)


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.complex64])
def test_rollout_rejects_non_real_floating_inputs(pendulum, dtype):
    x = torch.zeros(1, 2, dtype=dtype)
    u = torch.zeros(1, 1, 1, dtype=dtype)
    with pytest.raises(TypeError, match=rf"state.*float32 or float64.*{dtype}"):
        mj_rollout(pendulum, x, u)


def test_rollout_rejects_mixed_dtypes(pendulum):
    x = torch.zeros(1, 2, dtype=torch.float32)
    u = torch.zeros(1, 1, 1, dtype=torch.float64)
    with pytest.raises(TypeError, match="same dtype"):
        mj_rollout(pendulum, x, u)


def test_rollout_rejects_non_cpu_device_without_hardware(pendulum):
    x = torch.empty(1, 2, dtype=torch.float64, device="meta")
    u = torch.empty(1, 1, 1, dtype=torch.float64, device="meta")
    with pytest.raises(ValueError, match="must be on CPU, got meta"):
        mj_rollout(pendulum, x, u)


def test_rollout_rejects_mixed_devices(pendulum):
    x = torch.zeros(1, 2, dtype=torch.float64)
    u = torch.empty(1, 1, 1, dtype=torch.float64, device="meta")
    with pytest.raises(ValueError, match="same device.*cpu.*meta"):
        mj_rollout(pendulum, x, u)


def test_rollout_rejects_history_state(history_model):
    x = torch.zeros(
        1,
        history_model.nq + history_model.nv + history_model.na,
        dtype=torch.float64,
    )
    u = torch.zeros(1, 1, history_model.nu, dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="plugin/history"):
        mj_rollout(history_model, x, u)


def test_backend_error_lists_supported_values(pendulum):
    x = torch.zeros(1, 2, dtype=torch.float64)
    u = torch.zeros(1, 1, 1, dtype=torch.float64)
    with pytest.raises(
        NotImplementedError,
        match="unknown.*cpp_vjp.*mujoco_rollout",
    ):
        mj_rollout(pendulum, x, u, backend="unknown")


def test_nstep_must_be_positive(pendulum):
    x0 = torch.zeros(1, 2, dtype=torch.float64)
    u = torch.zeros(1, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="nstep"):
        mj_step(pendulum, x0, u, nstep=0)


@pytest.mark.parametrize("bad_nstep", [1.5, "2"])
def test_nstep_requires_integer_protocol(pendulum, bad_nstep):
    x0 = torch.zeros(1, 2, dtype=torch.float64)
    u = torch.zeros(1, 1, dtype=torch.float64)

    with pytest.raises(TypeError, match=r"nstep.*integer"):
        mj_step(pendulum, x0, u, nstep=bad_nstep)


@pytest.mark.parametrize("bad_name", ["state", "control"])
def test_step_rejects_non_tensor_inputs(pendulum, bad_name):
    x0 = [0.0, 0.0] if bad_name == "state" else torch.zeros(2, dtype=torch.float64)
    u = [0.0] if bad_name == "control" else torch.zeros(1, dtype=torch.float64)
    with pytest.raises(TypeError, match=rf"{bad_name}.*float32 or float64.*list"):
        mj_step(pendulum, x0, u)


def test_step_rejects_invalid_eps_before_control_expansion(pendulum):
    x0 = torch.zeros(2, dtype=torch.float64)
    u = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="eps must be finite and positive"):
        mj_step(pendulum, x0, u, nstep=1.5, eps=0.0)


@pytest.mark.parametrize("entrypoint", ["rollout", "step"])
@pytest.mark.parametrize("bad_name", ["state", "control"])
def test_dynamics_rejects_non_strided_tensor_layout(pendulum, entrypoint, bad_name):
    x0 = torch.zeros(1, 2, dtype=torch.float64)
    u_shape = (1, 1, 1) if entrypoint == "rollout" else (1, 1)
    u = torch.zeros(u_shape, dtype=torch.float64)
    if bad_name == "state":
        x0 = x0.to_sparse()
    else:
        u = u.to_sparse()

    with pytest.raises(TypeError, match=rf"{bad_name}.*strided layout.*sparse_coo"):
        if entrypoint == "rollout":
            mj_rollout(pendulum, x0, u)
        else:
            mj_step(pendulum, x0, u)


def test_bad_state_dim_raises(pendulum):
    x0 = torch.zeros(1, 3, dtype=torch.float64)  # nx should be 2
    u = torch.zeros(1, 1, dtype=torch.float64)
    with pytest.raises(ValueError):
        mj_step(pendulum, x0, u)


def test_bad_control_dim_raises(pendulum):
    x0 = torch.zeros(1, 2, dtype=torch.float64)
    u = torch.zeros(1, 2, dtype=torch.float64)  # nu should be 1
    with pytest.raises(ValueError):
        mj_step(pendulum, x0, u)


def test_mismatched_batch_raises(cartpole):
    x0 = torch.zeros(3, 4, dtype=torch.float64)
    U = torch.zeros(2, 5, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match="batch"):
        mj_rollout(cartpole, x0, U)


def test_autograd_rejects_free_joint(free_joint):
    d = mujoco.MjData(free_joint)
    mujoco.mj_resetData(free_joint, d)
    x_full = torch.tensor(np.concatenate([d.qpos, d.qvel]), dtype=torch.float64)
    U = torch.zeros(1, free_joint.nu, dtype=torch.float64)
    with pytest.raises(NotImplementedError, match=r"rollout.*nq == nv"):
        mj_rollout(free_joint, x_full, U)


@pytest.mark.parametrize(
    ("function", "args", "removed_argument"),
    [
        (mj_rollout, (None, None, None), "state" "_mode"),
        (mj_rollout, (None, None, None), "allow_device" "_transfer"),
        (mj_rollout, (None, None, None), "save" "_jacobians"),
        (mj_rollout, (None, None, None), "return" "_jacobians"),
        (mj_rollout, (None, None, None), "keep" "dim"),
        (mj_step, (None, None, None), "state" "_mode"),
        (mj_step, (None, None, None), "allow_device" "_transfer"),
        (mj_step, (None, None, None), "keep" "dim"),
        (mj_linearize, (None, None, None), "state" "_mode"),
    ],
)
def test_removed_arguments_raise_normal_type_error(function, args, removed_argument):
    with pytest.raises(TypeError, match=rf"unexpected keyword argument.*{removed_argument}"):
        function(*args, **{removed_argument: False})
