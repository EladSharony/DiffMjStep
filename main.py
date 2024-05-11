import torch
import mujoco as mj
from autograd_mujoco import MjStep

import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def jacobian(f: callable, x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Computes the Jacobian of a function f at x using centered finite differences.
    :param f: function
    :param x: input
    :param eps: epsilon
    :return: Jacobian of f at x
    """
    x = x.view(1, -1)
    n = x.numel()
    e = torch.eye(n, dtype=torch.float64, device=x.device)
    J = [(f(x + eps * e[i]) - f(x - eps * e[i])) / (2 * eps) for i in range(n)]
    return torch.stack(J).reshape(n, -1).T


def call_jacobian(state: torch.Tensor, ctrl: torch.Tensor, n_steps: int, mj_model: mj.MjModel, mj_data: mj.MjData) -> \
        tuple[torch.Tensor, torch.Tensor]:
    """
    Uses the jacobian function to compute the dydx and dydu.
    """
    state.requires_grad = False
    ctrl.requires_grad = False
    n_batch = state.shape[0]
    naive_dydx, naive_dydu = zip(
        *[(jacobian(lambda s: MjStep.apply(s, ctrl[[i]], n_steps, mj_model, mj_data)[0], state[[i]], 1e-4),
           jacobian(lambda a: MjStep.apply(state[[i]], a, n_steps, mj_model, mj_data)[0], ctrl[[i]], 1e-4))
          for i in range(n_batch)])
    naive_dydx, naive_dydu = torch.stack(naive_dydx), torch.stack(naive_dydu)
    return naive_dydx, naive_dydu


if __name__ == '__main__':
    # path to the xml file
    xml_path = 'assets/half_cheetah.xml'

    # Create an instance of the MjModel and MjData
    mj_model = mj.MjModel.from_xml_path(filename=xml_path)
    mj_data = mj.MjData(mj_model)

    n_steps = 4  # Number of steps to unroll the dynamics

    torch.manual_seed(0)

    batch_sizes = [2 ** i for i in range(0, 12 + 1)]

    devices = [torch.device('cpu'), torch.device('cuda')]

    df = pd.DataFrame(columns=['device', 'method', 'batch_size', 'execution_time'])

    # Define the number of runs
    n_runs = 5
    for run in range(n_runs):
        for device in devices:
            for n_batch in batch_sizes:
                # Define the state and control
                state = torch.rand(n_batch, mj_model.nq + mj_model.nv + mj_model.na, device=device, requires_grad=True)
                ctrl = torch.rand(n_batch, mj_model.nu, device=device, requires_grad=True)

                # Measure the execution time of MjStep.apply
                start_time = time.time()
                next_state, dydx, dydu = MjStep.apply(state, ctrl, n_steps, mj_model, mj_data)
                end_time = time.time()
                new_row = pd.DataFrame({'run': [run], 'device': [device], 'method': ['MjStep'], 'batch_size': [n_batch],
                                        'execution_time': [end_time - start_time]})
                df = pd.concat([df, new_row], ignore_index=True)

                # Measure the execution time of naive FD
                start_time = time.time()
                naive_dydx, naive_dydu = call_jacobian(state, ctrl, n_steps, mj_model, mj_data)
                end_time = time.time()
                new_row = pd.DataFrame(
                    {'run': [run], 'device': [device], 'method': ['Naive FD'], 'batch_size': [n_batch],
                     'execution_time': [end_time - start_time]})
                df = pd.concat([df, new_row], ignore_index=True)

    # Plot the execution times with error bars
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.2)
    sns.set_palette('deep')

    ax = sns.lineplot(data=df, x='batch_size', y='execution_time', hue='method', style='device', errorbar='sd')

    sns.move_legend(ax, "upper center", ncol=2)
    ax.set_xticks(batch_sizes, [f'{batch_size}' for batch_size in batch_sizes])
    ax.set_xlim([batch_sizes[0], batch_sizes[-1]])
    ax.set_xscale('log', base=2)
    ax.set_title(f'Execution Time of MjStep and Naive FD (Over {n_runs} Runs)')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Execution Time (s)')
    ax.grid(True)

    # Save the plot to svg
    plt.savefig('execution_time.svg', format='svg')
    plt.savefig('execution_time.png', format='png')
