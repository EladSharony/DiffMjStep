import torch
import mujoco

from diffmjstep import mj_step


PENDULUM_XML = """
<mujoco model="pendulum">
  <option timestep="0.01" integrator="Euler"/>
  <worldbody>
    <body name="pole" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -1" size="0.05" density="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque" joint="hinge" gear="1"/>
  </actuator>
</mujoco>
"""


def main() -> None:
    model = mujoco.MjModel.from_xml_string(PENDULUM_XML)

    x0 = torch.tensor([[0.1, 0.0]], dtype=torch.float64, requires_grad=True)
    u = torch.tensor([[0.2]], dtype=torch.float64, requires_grad=True)

    x1 = mj_step(model, x0, u, nstep=1)
    loss = x1.square().sum()
    loss.backward()

    print("x1:", x1.detach().numpy())
    print("d loss / d x0:", x0.grad.numpy())
    print("d loss / d u:", u.grad.numpy())


if __name__ == "__main__":
    main()
