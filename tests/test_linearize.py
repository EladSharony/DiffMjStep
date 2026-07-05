"""Standalone transition-linearization contracts."""

import mujoco
import numpy as np
import torch

from conftest import oracle_jacobian_fd
from diffmjstep import mj_linearize


def test_linearize_matches_independent_finite_difference(cartpole):
    state = np.array([0.0, 0.1, 0.0, 0.0])
    control = np.array([0.3])
    actual_A, actual_B = mj_linearize(cartpole, state, control)
    expected_A, expected_B = oracle_jacobian_fd(cartpole, state, control)
    np.testing.assert_allclose(actual_A[0], expected_A, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(actual_B[0], expected_B, rtol=1e-4, atol=1e-5)


def test_linearize_preserves_batch_and_input_family(cartpole):
    state = torch.zeros(3, 4, dtype=torch.float64)
    control = torch.zeros(3, 1, dtype=torch.float64)
    A, B = mj_linearize(cartpole, state, control)
    assert isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor)
    assert A.shape == (3, 4, 4)
    assert B.shape == (3, 4, 1)

    A_np, B_np = mj_linearize(cartpole, state[0].numpy(), control[0].numpy())
    assert isinstance(A_np, np.ndarray) and isinstance(B_np, np.ndarray)


def test_sensor_linearization_returns_C_and_D(actuator_force_sensor):
    state = np.array([0.2, -0.1])
    control = np.array([0.25])
    A, B, C, D = mj_linearize(
        actuator_force_sensor, state, control, sensors=True, eps=1e-6
    )
    assert A.shape == (1, 2, 2)
    assert B.shape == (1, 2, 1)
    assert C.shape == (1, actuator_force_sensor.nsensordata, 2)
    assert D.shape == (1, actuator_force_sensor.nsensordata, 1)


def test_free_joint_uses_tangent_space(free_joint):
    data = mujoco.MjData(free_joint)
    mujoco.mj_resetData(free_joint, data)
    state = np.concatenate((data.qpos, data.qvel))
    A, B = mj_linearize(free_joint, state, np.zeros(free_joint.nu))
    tangent = 2 * free_joint.nv + free_joint.na
    assert A.shape == (1, tangent, tangent)
    assert B.shape == (1, tangent, free_joint.nu)


def test_centered_and_forward_schemes_agree(cartpole):
    state = np.array([0.0, 0.1, 0.0, 0.0])
    control = np.array([0.3])
    centered, _ = mj_linearize(cartpole, state, control, centered=True)
    forward, _ = mj_linearize(cartpole, state, control, centered=False)
    np.testing.assert_allclose(centered, forward, rtol=1e-5, atol=1e-6)
