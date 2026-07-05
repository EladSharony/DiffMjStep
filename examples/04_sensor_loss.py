"""Differentiate a loss on the sensor trajectory: drive the pendulum tip to a target.

The framepos sensor gives the tip's world position; we optimize controls so the tip
reaches a target, using gradients that flow through MuJoCo's sensor Jacobians (C/D).
"""
import torch
import mujoco

from diffmjstep import mj_rollout

SENSORS_XML = """
<mujoco model="pendulum_sensors">
  <option timestep="0.01" integrator="Euler"/>
  <worldbody>
    <body name="pole" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.05" density="1"/>
      <site name="tip" pos="0 0 -1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque" joint="hinge" gear="1"/>
  </actuator>
  <sensor>
    <jointpos joint="hinge"/>
    <jointvel joint="hinge"/>
    <framepos objtype="site" objname="tip"/>
  </sensor>
</mujoco>
"""


def main() -> None:
    model = mujoco.MjModel.from_xml_string(SENSORS_XML)

    # sensordata layout: [jointpos(1), jointvel(1), framepos(3)]; tip xyz is the last 3.
    horizon = 12
    x0 = torch.tensor([[0.1, 0.0]], dtype=torch.float64)
    # reachable point on the radius-1 tip circle: (sin 0.6, 0, -cos 0.6)
    target_xyz = torch.tensor([0.5646, 0.0, -0.8253], dtype=torch.float64)
    U = torch.zeros(1, horizon, model.nu, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([U], lr=0.05)

    for _ in range(40):
        opt.zero_grad()
        _, Y = mj_rollout(model, x0, U, return_sensors=True, return_all=True)
        tip = Y[:, -1, -3:]
        loss = (tip - target_xyz).square().sum() + 1e-3 * U.square().sum()
        loss.backward()
        opt.step()

    _, Y = mj_rollout(model, x0, U, return_sensors=True, return_all=True)
    print("last pre-step tip xyz:", Y[0, -1, -3:].detach().numpy())
    print("target tip xyz:", target_xyz.numpy())


if __name__ == "__main__":
    main()
