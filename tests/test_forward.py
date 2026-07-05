"""Forward-pass correctness: outputs must match an independent MuJoCo rollout."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from diffmjstep import mj_rollout, mj_step
from conftest import oracle_rollout

BACKENDS = ["mujoco_rollout", "cpp_vjp"]


@pytest.fixture
def inputs(cartpole):
    rng = np.random.default_rng(0)
    x0 = torch.tensor(0.1 * rng.standard_normal((4, 4)), dtype=torch.float64)
    U = torch.tensor(0.2 * rng.standard_normal((4, 6, 1)), dtype=torch.float64)
    return x0, U


@pytest.mark.parametrize("backend", BACKENDS)
def test_rollout_matches_oracle(cartpole, inputs, backend):
    x0, U = inputs
    X = mj_rollout(cartpole, x0, U, backend=backend, return_all=True).detach().numpy()
    assert X.shape == (4, 7, 4)
    for b in range(x0.shape[0]):
        ref = oracle_rollout(cartpole, x0[b].numpy(), U[b].numpy())
        np.testing.assert_allclose(X[b], ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_return_all_false_is_terminal_state(cartpole, inputs, backend):
    x0, U = inputs
    XT = mj_rollout(cartpole, x0, U, backend=backend, return_all=False).detach().numpy()
    assert XT.shape == (4, 4)
    for b in range(x0.shape[0]):
        ref = oracle_rollout(cartpole, x0[b].numpy(), U[b].numpy())[-1]
        np.testing.assert_allclose(XT[b], ref, rtol=0, atol=1e-12)


def test_mj_step_repeats_control_nstep(pendulum):
    x0 = torch.tensor([[0.3, 0.0]], dtype=torch.float64)
    u = torch.tensor([[0.5]], dtype=torch.float64)
    nstep = 5
    xT = mj_step(pendulum, x0, u, nstep=nstep).detach().numpy()
    # Oracle: apply the same control for nstep steps.
    U = np.repeat(u.numpy(), nstep, axis=0)  # [nstep, nu]
    ref = oracle_rollout(pendulum, x0[0].numpy(), U)[-1]
    np.testing.assert_allclose(xT[0], ref, rtol=0, atol=1e-12)


def test_mj_step_accepts_integer_like_nstep(pendulum):
    x0 = torch.tensor([[0.3, 0.0]], dtype=torch.float64)
    u = torch.tensor([[0.5]], dtype=torch.float64)
    xT = mj_step(pendulum, x0, u, nstep=np.int64(2)).detach().numpy()
    ref = oracle_rollout(pendulum, x0[0].numpy(), np.repeat(u.numpy(), 2, axis=0))[-1]
    np.testing.assert_allclose(xT[0], ref, rtol=0, atol=1e-12)


def test_mj_step_default_nstep_is_one(pendulum):
    x0 = torch.tensor([[0.3, 0.0]], dtype=torch.float64)
    u = torch.tensor([[0.5]], dtype=torch.float64)
    one = mj_step(pendulum, x0, u).detach().numpy()
    ref = oracle_rollout(pendulum, x0[0].numpy(), u.numpy())[-1]
    np.testing.assert_allclose(one[0], ref, rtol=0, atol=1e-12)


def test_mj_step_sensor_trajectory_shape(pendulum_sensors):
    x0 = torch.tensor([[0.3, 0.0]], dtype=torch.float64)
    u = torch.tensor([[0.0]], dtype=torch.float64)
    xT, Y = mj_step(pendulum_sensors, x0, u, nstep=4, return_sensors=True)
    assert xT.shape == (1, 2)
    assert Y.shape == (1, 4, pendulum_sensors.nsensordata)


# --- shape / batching contract -----------------------------------------------------

@pytest.mark.parametrize(
    ("entrypoint", "state", "control"),
    [
        (mj_rollout, torch.zeros(2), torch.zeros(1, 1, 1)),
        (mj_rollout, torch.zeros(1, 2), torch.zeros(1, 1)),
        (mj_step, torch.zeros(2), torch.zeros(1, 1)),
        (mj_step, torch.zeros(1, 2), torch.zeros(1)),
    ],
)
def test_public_dynamics_require_explicit_batch_axes(
    pendulum, entrypoint, state, control
):
    with pytest.raises(ValueError, match="batch"):
        entrypoint(pendulum, state.to(torch.float64), control.to(torch.float64))


def test_horizon_zero_returns_initial_state(pendulum):
    x0 = torch.tensor([[0.2, 0.1]], dtype=torch.float64)
    U = torch.zeros(1, 0, 1, dtype=torch.float64)
    X = mj_rollout(pendulum, x0, U, return_all=True)
    assert X.shape == (1, 1, 2)
    np.testing.assert_allclose(X[0, 0].detach().numpy(), x0[0].numpy(), atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_output_dtype_follows_input(pendulum, dtype):
    x0 = torch.tensor([[0.1, 0.0]], dtype=dtype)
    u = torch.tensor([[0.2]], dtype=dtype)
    out = mj_step(pendulum, x0, u)
    assert out.dtype == dtype
