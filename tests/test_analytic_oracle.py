from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from diffmjstep import mj_linearize, mj_rollout

XML = """
<mujoco model="linear_mass">
  <option gravity="0 0 0" integrator="Euler"/>
  <worldbody>
    <body>
      <joint name="x" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.01" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor joint="x" gear="1"/></actuator>
</mujoco>
"""


@pytest.mark.parametrize(
    ("dt", "horizon", "seed"),
    [(0.001, 1, 0), (0.01, 5, 1), (0.05, 16, 2)],
)
def test_transition_and_rollout_gradients_match_closed_form(dt, horizon, seed):
    model = mujoco.MjModel.from_xml_string(XML)
    model.opt.timestep = dt
    A_exact = np.array([[1.0, dt], [0.0, 1.0]])
    B_exact = np.array([[dt * dt], [dt]])
    A, B = mj_linearize(model, np.array([0.2, -0.1]), np.array([0.3]))
    assert A.shape == (1, 2, 2)
    assert B.shape == (1, 2, 1)
    np.testing.assert_allclose(A[0], A_exact, rtol=0, atol=1e-8)
    np.testing.assert_allclose(B[0], B_exact, rtol=0, atol=1e-8)

    rng = np.random.default_rng(seed)
    x0_value = rng.normal(scale=0.1, size=(1, 2))
    controls = rng.normal(scale=0.1, size=(1, horizon, 1))
    x0 = torch.tensor(x0_value, dtype=torch.float64, requires_grad=True)
    U = torch.tensor(controls, dtype=torch.float64, requires_grad=True)
    mj_rollout(model, x0, U, backend="mujoco_rollout").square().sum().backward()

    states = [x0_value[0]]
    for control in controls[0]:
        states.append(A_exact @ states[-1] + B_exact[:, 0] * control[0])
    lam = 2.0 * states[-1]
    grad_u = np.empty((horizon, 1))
    for step in range(horizon - 1, -1, -1):
        grad_u[step] = lam @ B_exact
        lam = 2.0 * states[step] + lam @ A_exact
    exact = np.concatenate([lam, grad_u.ravel()])
    actual = np.concatenate([x0.grad.numpy().ravel(), U.grad.numpy().ravel()])
    relative_error = np.linalg.norm(actual - exact) / np.linalg.norm(exact)
    assert relative_error < 1e-8
