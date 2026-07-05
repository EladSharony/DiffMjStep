"""Native loader and public zero-work safety."""

import sys

import pytest
import torch

from diffmjstep import mj_rollout
from diffmjstep import native


pytestmark = pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="native backend is unsupported"
)


@pytest.mark.parametrize(
    ("platform", "name"),
    [
        ("linux", "libmujoco.so.3.9.0"),
        ("darwin", "libmujoco.3.9.0.dylib"),
    ],
)
def test_native_library_lookup_is_exact(monkeypatch, tmp_path, platform, name):
    monkeypatch.setattr(native.sys, "platform", platform)
    expected = tmp_path / name
    expected.touch()
    assert native._find_mujoco_library(tmp_path, "3.9.0") == expected


def test_native_library_lookup_rejects_unsupported_platform(monkeypatch, tmp_path):
    monkeypatch.setattr(native.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Linux and macOS"):
        native._find_mujoco_library(tmp_path, "3.9.0")


def test_packaged_extension_builds_and_loads():
    native.load_extension.cache_clear()
    assert native.load_extension() is not None


@pytest.mark.parametrize("return_all", [True, False])
def test_native_zero_horizon_has_identity_gradient(pendulum, return_all):
    state = torch.tensor(
        [[0.2, -0.1], [-0.3, 0.4]], dtype=torch.float64, requires_grad=True
    )
    controls = torch.empty((2, 0, 1), dtype=torch.float64, requires_grad=True)
    output = mj_rollout(
        pendulum,
        state,
        controls,
        backend="cpp_vjp",
        return_all=return_all,
    )
    weights = torch.arange(
        1, output.numel() + 1, dtype=torch.float64
    ).reshape_as(output)
    (output * weights).sum().backward()
    expected = weights[:, 0] if return_all else weights
    torch.testing.assert_close(state.grad, expected, rtol=0, atol=0)
    assert controls.grad is not None and controls.grad.shape == controls.shape


def test_native_empty_batch_is_safe(pendulum_sensors):
    state = torch.empty((0, 2), dtype=torch.float64, requires_grad=True)
    controls = torch.empty((0, 3, 1), dtype=torch.float64, requires_grad=True)
    states, sensors = mj_rollout(
        pendulum_sensors,
        state,
        controls,
        backend="cpp_vjp",
        return_sensors=True,
    )
    assert states.shape == (0, 4, 2)
    assert sensors.shape == (0, 3, pendulum_sensors.nsensordata)
    (states.sum() + sensors.sum()).backward()
    assert state.grad is not None and state.grad.shape == state.shape
    assert controls.grad is not None and controls.grad.shape == controls.shape
