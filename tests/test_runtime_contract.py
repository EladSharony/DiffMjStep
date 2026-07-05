"""Externally visible replay and model-isolation contracts."""

import copy

import mujoco
import pytest
import torch

from diffmjstep import mj_rollout


def _weighted_gradients(model, state_value, control_value, mutate=False):
    state = state_value.clone().requires_grad_()
    controls = control_value.clone().requires_grad_()
    output = mj_rollout(model, state, controls)
    weights = torch.arange(
        1, output.numel() + 1, dtype=output.dtype
    ).reshape_as(output)
    loss = (output * weights).sum()
    if mutate:
        output.detach().fill_(12345)
    loss.backward()
    return state.grad, controls.grad


def test_output_mutation_does_not_corrupt_saved_replay(cartpole):
    state = torch.tensor([[0.1, -0.2, 0.05, 0.15]], dtype=torch.float64)
    controls = torch.tensor(
        [[[0.3], [-0.2], [0.1], [0.25]]], dtype=torch.float64
    )
    expected = _weighted_gradients(cartpole, state, controls)
    actual = _weighted_gradients(cartpole, state, controls, mutate=True)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_callback_registration_must_not_change_before_backward(pendulum):
    state = torch.tensor([[0.3, -0.1]], dtype=torch.float64, requires_grad=True)
    controls = torch.tensor([[[0.2]]], dtype=torch.float64, requires_grad=True)

    def replacement(_model, _data):
        pass

    previous = mujoco.get_mjcb_passive()
    try:
        mujoco.set_mjcb_passive(None)
        output = mj_rollout(pendulum, state, controls, backend="mujoco_rollout")
        mujoco.set_mjcb_passive(replacement)
        with pytest.raises(RuntimeError, match="registration must remain unchanged"):
            output.sum().backward()
    finally:
        mujoco.set_mjcb_passive(previous)


def test_time_dependent_callback_passes_whole_rollout_gradcheck(pendulum):
    def time_dependent_force(_model, data):
        data.qfrc_passive[0] += (0.5 + data.time) * data.qpos[0]

    previous = mujoco.get_mjcb_passive()
    try:
        mujoco.set_mjcb_passive(time_dependent_force)
        state = torch.tensor(
            [[0.3, -0.1]], dtype=torch.float64, requires_grad=True
        )
        controls = torch.tensor(
            [[[0.2], [-0.1], [0.3]]], dtype=torch.float64, requires_grad=True
        )
        assert torch.autograd.gradcheck(
            lambda x, u: mj_rollout(
                pendulum, x, u, backend="mujoco_rollout", return_all=False
            ),
            (state, controls),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )
    finally:
        mujoco.set_mjcb_passive(previous)


def test_backward_is_isolated_from_caller_model_mutation(cartpole):
    state_value = torch.tensor(
        [[0.1, -0.2, 0.05, 0.15]], dtype=torch.float64
    )
    control_value = torch.tensor(
        [[[0.3], [-0.2], [0.1]]], dtype=torch.float64
    )
    expected = _weighted_gradients(copy.copy(cartpole), state_value, control_value)

    state = state_value.clone().requires_grad_()
    controls = control_value.clone().requires_grad_()
    output = mj_rollout(cartpole, state, controls)
    original_timestep = cartpole.opt.timestep
    original_damping = cartpole.dof_damping.copy()
    try:
        cartpole.opt.timestep *= 7
        cartpole.dof_damping[:] = 50
        weights = torch.arange(
            1, output.numel() + 1, dtype=output.dtype
        ).reshape_as(output)
        (output * weights).sum().backward()
    finally:
        cartpole.opt.timestep = original_timestep
        cartpole.dof_damping[:] = original_damping

    for got, want in zip((state.grad, controls.grad), expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-7, atol=1e-9)


def test_warmstart_hidden_state_does_not_change_derivative_mode(early_stop_contact):
    model = early_stop_contact
    model.opt.iterations = 1
    model.opt.tolerance = 0
    original_flags = int(model.opt.disableflags)
    state = torch.tensor(
        [[0.01561909, 0.10938033, 0.08923619, 0.42154183, -0.39839765, 0.35403894]],
        dtype=torch.float64,
    )
    controls = 0.2 * torch.randn(
        (1, 8, int(model.nu)), generator=torch.Generator().manual_seed(17),
        dtype=torch.float64,
    )
    with torch.no_grad():
        inference = mj_rollout(model, state, controls)
    differentiable = mj_rollout(model, state.clone().requires_grad_(), controls)
    torch.testing.assert_close(inference, differentiable, rtol=0, atol=0)
    assert int(model.opt.disableflags) == original_flags


def test_early_stopped_contact_passes_whole_rollout_gradcheck(early_stop_contact):
    state = torch.tensor(
        [[0.01561909, 0.10938033, 0.08923619, 0.42154183, -0.39839765, 0.35403894]],
        dtype=torch.float64,
        requires_grad=True,
    )
    controls = torch.tensor(
        [[
            [-0.06235758, 0.16901246, -0.10644170],
            [0.07543371, 0.00420950, -0.23734728],
            [0.15588126, 0.19376655, -0.01020909],
        ]],
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        lambda x, u: mj_rollout(
            early_stop_contact,
            x,
            u,
            backend="mujoco_rollout",
            return_all=False,
        ),
        (state, controls),
        eps=1e-6,
        atol=5e-5,
        rtol=1e-5,
    )
