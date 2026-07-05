"""Backend selection through public behavior."""

import warnings

import mujoco
import pytest
import torch

import diffmjstep.core as core
from diffmjstep import mj_rollout


@pytest.fixture(autouse=True)
def reset_auto_state():
    core._AUTO_NATIVE_DISABLED = False
    core._AUTO_NATIVE_LOADING_PID = None
    getattr(core.native.load_extension, "cache_clear", lambda: None)()
    yield
    core._AUTO_NATIVE_DISABLED = False
    core._AUTO_NATIVE_LOADING_PID = None
    getattr(core.native.load_extension, "cache_clear", lambda: None)()


def _gradients(model, backend):
    state = torch.tensor([[0.2, -0.1]], dtype=torch.float64, requires_grad=True)
    controls = torch.tensor(
        [[[0.3], [-0.2]]], dtype=torch.float64, requires_grad=True
    )
    mj_rollout(model, state, controls, backend=backend).square().sum().backward()
    return state.grad, controls.grad


def test_auto_matches_required_native_backend(pendulum):
    expected = _gradients(pendulum, "cpp_vjp")
    actual = _gradients(pendulum, "auto")
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_auto_warns_once_and_falls_back_to_python_vjp(monkeypatch, pendulum):
    def unavailable():
        raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(core.native, "load_extension", unavailable)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = _gradients(pendulum, "auto")
        second = _gradients(pendulum, "auto")

    assert len(caught) == 1
    assert "compiler unavailable" in str(caught[0].message)
    expected = _gradients(pendulum, "mujoco_rollout")
    for got, want in zip(first + second, expected + expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_active_python_callback_uses_python_vjp(monkeypatch, pendulum):
    monkeypatch.setattr(core, "_auto_native_ready", lambda: True)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native VJP must not replay Python callbacks")

    monkeypatch.setattr(core.native, "rollout_backward_vjp", forbidden)

    def passive(_model, data):
        data.qfrc_passive[0] += 0.1 * data.qpos[0]

    previous = mujoco.get_mjcb_passive()
    try:
        mujoco.set_mjcb_passive(passive)
        state_grad, control_grad = _gradients(pendulum, "auto")
    finally:
        mujoco.set_mjcb_passive(previous)

    assert torch.isfinite(state_grad).all()
    assert torch.isfinite(control_grad).all()


def test_explicit_cpp_failure_is_not_hidden(monkeypatch, pendulum):
    def unavailable():
        raise RuntimeError("native build failed")

    monkeypatch.setattr(core.native, "load_extension", unavailable)
    state = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    controls = torch.zeros(1, 1, 1, dtype=torch.float64, requires_grad=True)
    with pytest.raises(RuntimeError, match="native build failed"):
        mj_rollout(pendulum, state, controls, backend="cpp_vjp").sum().backward()


def test_unknown_backend_lists_supported_choices(pendulum):
    state = torch.zeros(1, 2, dtype=torch.float64)
    controls = torch.zeros(1, 1, 1, dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="auto.*cpp_vjp.*mujoco_rollout"):
        mj_rollout(pendulum, state, controls, backend="unknown")
