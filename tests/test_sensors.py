"""Sensor trajectory values and public gradient contracts."""

import numpy as np
import pytest
import torch

from conftest import oracle_sensors
from diffmjstep import mj_rollout


@pytest.mark.parametrize("backend", ["mujoco_rollout", "cpp_vjp"])
def test_sensor_outputs_share_state_forward(backend, pendulum_sensors):
    state = torch.tensor([[0.2, 0.0]], dtype=torch.float64)
    controls = torch.tensor([[[0.1], [-0.1], [0.0]]], dtype=torch.float64)
    plain = mj_rollout(pendulum_sensors, state, controls, backend=backend)
    with_sensors, sensors = mj_rollout(
        pendulum_sensors,
        state,
        controls,
        backend=backend,
        return_sensors=True,
    )
    torch.testing.assert_close(with_sensors, plain, rtol=0, atol=0)
    assert sensors.shape == (1, 3, pendulum_sensors.nsensordata)


def test_sensor_values_and_pre_step_timing_match_oracle(pendulum_sensors):
    state = torch.tensor([[0.37, 0.0], [-0.2, 0.1]], dtype=torch.float64)
    controls = torch.zeros(2, 4, 1, dtype=torch.float64)
    _, sensors = mj_rollout(
        pendulum_sensors, state, controls, return_sensors=True
    )
    for batch in range(2):
        expected = oracle_sensors(
            pendulum_sensors, state[batch].numpy(), controls[batch].numpy()
        )
        np.testing.assert_allclose(sensors[batch].numpy(), expected, atol=1e-12)
    assert sensors[0, 0, 0].item() == pytest.approx(state[0, 0].item())


def test_mixed_state_and_sensor_loss_passes_gradcheck(pendulum_sensors):
    state = torch.tensor([[0.2, 0.0]], dtype=torch.float64, requires_grad=True)
    controls = torch.tensor(
        [[[0.1], [-0.1]]], dtype=torch.float64, requires_grad=True
    )

    def loss(x, u):
        states, sensors = mj_rollout(
            pendulum_sensors,
            x,
            u,
            backend="mujoco_rollout",
            return_sensors=True,
        )
        return states.square().sum() + sensors.square().sum()

    assert torch.autograd.gradcheck(
        loss, (state, controls), eps=1e-6, atol=1e-4, rtol=1e-3
    )


def test_native_sensor_gradients_match_python_vjp(actuator_force_sensor):
    state_value = torch.tensor([[0.2, -0.1]], dtype=torch.float64)
    control_value = torch.tensor([[[0.25], [-0.4]]], dtype=torch.float64)

    def gradients(backend):
        state = state_value.clone().requires_grad_()
        controls = control_value.clone().requires_grad_()
        states, sensors = mj_rollout(
            actuator_force_sensor,
            state,
            controls,
            backend=backend,
            return_sensors=True,
        )
        (states.square().sum() + sensors.square().sum()).backward()
        return state.grad, controls.grad

    expected = gradients("mujoco_rollout")
    actual = gradients("cpp_vjp")
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)


def test_ignored_sensor_output_does_not_change_state_gradients(pendulum_sensors):
    state_value = torch.tensor([[0.2, 0.0]], dtype=torch.float64)
    control_value = torch.tensor([[[0.1], [-0.1]]], dtype=torch.float64)

    def gradients(return_sensors):
        state = state_value.clone().requires_grad_()
        controls = control_value.clone().requires_grad_()
        output = mj_rollout(
            pendulum_sensors,
            state,
            controls,
            return_sensors=return_sensors,
        )
        states = output[0] if return_sensors else output
        states.square().sum().backward()
        return state.grad, controls.grad

    for got, want in zip(gradients(True), gradients(False), strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
