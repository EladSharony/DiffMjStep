"""Shared fixtures, model builders, and independent oracles for the DiffMjStep suite.

Everything here is built from inline MuJoCo XML so the tests need no external assets.
The "oracle" helpers re-derive forward rollouts and transition Jacobians directly from
the raw `mujoco` bindings, independent of `diffmjstep`'s own code paths, so a test that
compares against them is an independent check, not a snapshot of this implementation.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import mujoco

torch.manual_seed(0)


# --- inline models -----------------------------------------------------------------

PENDULUM_XML = """
<mujoco model="pendulum">
  <option timestep="0.01" integrator="Euler"/>
  <worldbody>
    <body name="pole">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.05" density="1"/>
    </body>
  </worldbody>
  <actuator><motor name="torque" joint="hinge" gear="1"/></actuator>
</mujoco>
"""

# nq == nv == 2, nu == 1 — the smallest non-trivial coupled system.
CARTPOLE_XML = """
<mujoco model="cartpole">
  <option timestep="0.01" integrator="Euler"/>
  <worldbody>
    <body name="cart">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.2 0.2 0.2" mass="1"/>
      <body name="pole">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.04" density="100"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="slide" joint="slide" gear="10"/></actuator>
</mujoco>
"""

# Pendulum with sensors: jointpos(1), jointvel(1), framepos(3) -> nsensordata == 5.
PENDULUM_SENSORS_XML = """
<mujoco model="pendulum_sensors">
  <option timestep="0.01" integrator="Euler"/>
  <worldbody>
    <body name="pole">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.05" density="1"/>
      <site name="tip" pos="0 0 -1"/>
    </body>
  </worldbody>
  <actuator><motor name="torque" joint="hinge" gear="1"/></actuator>
  <sensor>
    <jointpos joint="hinge"/>
    <jointvel joint="hinge"/>
    <framepos objtype="site" objname="tip"/>
  </sensor>
</mujoco>
"""

# A control-linear sensor with an exactly known derivative dy/du = 3.  Keeping the
# control range finite also exercises MuJoCo's one-sided finite difference at the limit.
ACTUATOR_FORCE_SENSOR_XML = """
<mujoco model="actuator_force_sensor">
  <option timestep="0.01" integrator="Euler"/>
  <worldbody>
    <body>
      <joint name="hinge"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <general name="actuator" joint="hinge" gainprm="3"
             ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
  <sensor><actuatorfrc actuator="actuator"/></sensor>
</mujoco>
"""

# Free base (nq=8 != nv=7): a free body carrying a hinge-actuated child.
FREE_JOINT_XML = """
<mujoco model="freebase">
  <option timestep="0.01" integrator="Euler"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
      <body name="arm" pos="0.1 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="m" joint="hinge" gear="1"/></actuator>
</mujoco>
"""

HISTORY_XML = """
<mujoco model="history">
  <option timestep="0.01" integrator="Euler"/>
  <worldbody>
    <body><joint name="j"/><geom type="sphere" size=".1"/></body>
  </worldbody>
  <actuator><motor joint="j" nsample="2"/></actuator>
</mujoco>
"""

EARLY_STOP_CONTACT_XML = """
<mujoco model="early_stop_contact">
  <option timestep="0.01" integrator="Euler" solver="Newton"
          iterations="2" tolerance="1e-6" gravity="0 0 -9.81"/>
  <worldbody>
    <geom type="plane" size="2 2 .1"/>
    <body>
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="z" type="slide" axis="0 0 1"/>
      <joint name="r" type="hinge" axis="0 1 0"/>
      <geom type="box" size=".3 .1 .1" mass="1" solref="0.03 1"
            friction="1 0.005 0.0001"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="x"/><motor joint="z"/><motor joint="r"/>
  </actuator>
</mujoco>
"""


def _model(xml: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml)


@pytest.fixture
def pendulum():
    return _model(PENDULUM_XML)


@pytest.fixture
def cartpole():
    return _model(CARTPOLE_XML)


@pytest.fixture
def pendulum_sensors():
    return _model(PENDULUM_SENSORS_XML)


@pytest.fixture
def actuator_force_sensor():
    return _model(ACTUATOR_FORCE_SENSOR_XML)


@pytest.fixture
def free_joint():
    return _model(FREE_JOINT_XML)


@pytest.fixture
def history_model():
    return _model(HISTORY_XML)


@pytest.fixture
def early_stop_contact():
    return mujoco.MjModel.from_xml_string(EARLY_STOP_CONTACT_XML)


# --- independent oracles -----------------------------------------------------------

def oracle_rollout(model, x0, U):
    """Plain MuJoCo rollout, re-implemented from the raw bindings (nq == nv only).

    x0: [nx], U: [T, nu]. Returns the full state trajectory X[T+1, nx].
    """
    nq, nv, na = int(model.nq), int(model.nv), int(model.na)
    x0 = np.asarray(x0, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    horizon = U.shape[0]
    nx = nq + nv + na

    def read(d):
        parts = [np.array(d.qpos), np.array(d.qvel)]
        if na:
            parts.append(np.array(d.act))
        return np.concatenate(parts)

    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    d.qpos[:] = x0[:nq]
    d.qvel[:] = x0[nq:nq + nv]
    if na:
        d.act[:] = x0[nq + nv:]
    mujoco.mj_forward(model, d)

    X = np.empty((horizon + 1, nx), dtype=np.float64)
    X[0] = x0
    for t in range(horizon):
        d.ctrl[:] = U[t]
        mujoco.mj_step(model, d)
        X[t + 1] = read(d)
    return X


def oracle_sensors(model, x0, U):
    """Re-implemented rollout that also records sensordata after each step."""
    nq, nv, na = int(model.nq), int(model.nv), int(model.na)
    x0 = np.asarray(x0, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    horizon = U.shape[0]
    Y = np.empty((horizon, int(model.nsensordata)), dtype=np.float64)

    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    d.qpos[:] = x0[:nq]
    d.qvel[:] = x0[nq:nq + nv]
    if na:
        d.act[:] = x0[nq + nv:]
    mujoco.mj_forward(model, d)
    for t in range(horizon):
        d.ctrl[:] = U[t]
        mujoco.mj_step(model, d)
        Y[t] = np.array(d.sensordata)
    return Y


def oracle_jacobian_fd(model, x, u, eps=1e-6):
    """Central finite-difference A = dx_next/dx, B = dx_next/du for one step (nq == nv).

    Independent of `mjd_transitionFD`: perturbs the DiffMjStep state directly and rolls
    one MuJoCo step each way. Only valid where nq == nv.
    """
    x = np.asarray(x, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    nx = x.shape[0]
    nu = u.shape[0]

    def step(xv, uv):
        return oracle_rollout(model, xv, uv[None, :])[1]

    A = np.empty((nx, nx))
    for i in range(nx):
        dp = x.copy(); dp[i] += eps
        dm = x.copy(); dm[i] -= eps
        A[:, i] = (step(dp, u) - step(dm, u)) / (2 * eps)
    B = np.empty((nx, nu))
    for j in range(nu):
        up = u.copy(); up[j] += eps
        um = u.copy(); um[j] -= eps
        B[:, j] = (step(x, up) - step(x, um)) / (2 * eps)
    return A, B
