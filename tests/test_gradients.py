"""Public autograd checks against whole-rollout finite differences."""

import numpy as np
import torch

from diffmjstep import mj_rollout, mj_step


def test_gradcheck_rollout(cartpole):
    x0 = torch.tensor(
        [[0.0, 0.1, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    controls = torch.tensor(
        [[[0.5], [-0.3], [0.2]]], dtype=torch.float64, requires_grad=True
    )

    def rollout(state, actions):
        return mj_rollout(
            cartpole, state, actions, backend="mujoco_rollout", return_all=True
        )

    assert torch.autograd.gradcheck(
        rollout, (x0, controls), eps=1e-6, atol=1e-4, rtol=1e-3
    )


def test_gradcheck_terminal_and_state_only(cartpole):
    x0 = torch.tensor(
        [[0.0, 0.1, 0.0, 0.0]], dtype=torch.float64, requires_grad=True
    )
    controls = torch.tensor([[[0.4], [-0.2]]], dtype=torch.float64)

    def terminal(state):
        return mj_rollout(cartpole, state, controls, return_all=False)

    assert torch.autograd.gradcheck(
        terminal, (x0,), eps=1e-6, atol=1e-4, rtol=1e-3
    )


def test_gradcheck_mj_step_multistep(pendulum):
    x0 = torch.tensor([[0.2, 0.0]], dtype=torch.float64, requires_grad=True)
    control = torch.tensor([[0.3]], dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda state, action: mj_step(pendulum, state, action, nstep=4),
        (x0, control),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_native_gradients_match_python_vjp(cartpole):
    rng = np.random.default_rng(2)
    state_value = torch.tensor(
        0.05 * rng.standard_normal((2, 4)), dtype=torch.float64
    )
    control_value = torch.tensor(
        0.2 * rng.standard_normal((2, 3, 1)), dtype=torch.float64
    )

    def gradients(backend):
        state = state_value.clone().requires_grad_()
        controls = control_value.clone().requires_grad_()
        mj_rollout(cartpole, state, controls, backend=backend).square().sum().backward()
        return state.grad, controls.grad

    expected = gradients("mujoco_rollout")
    actual = gradients("cpp_vjp")
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)
