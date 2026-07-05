import torch
import mujoco

from diffmjstep import mj_rollout


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
    torch.manual_seed(0)

    model = mujoco.MjModel.from_xml_string(PENDULUM_XML)

    horizon = 12
    x0 = torch.tensor([[0.1, 0.0]], dtype=torch.float64)
    target = torch.tensor([[0.5, 0.0]], dtype=torch.float64)
    U = torch.zeros(1, horizon, model.nu, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([U], lr=0.05)

    for _ in range(25):
        optimizer.zero_grad()
        X = mj_rollout(model, x0, U, return_all=True)
        terminal_loss = (X[:, -1] - target).square().sum()
        control_loss = 1e-3 * U.square().sum()
        loss = terminal_loss + control_loss
        loss.backward()
        optimizer.step()

    X = mj_rollout(model, x0, U, return_all=True)
    print("final state:", X[:, -1].detach().numpy())
    print("target:", target.numpy())
    print("optimized controls:", U.detach().numpy())


if __name__ == "__main__":
    main()
