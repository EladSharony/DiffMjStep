"""Compare MuJoCo transition-FD VJPs with whole-rollout central finite differences.

The reference perturbs each entry of x0 and U, re-runs the forward rollout, and
central-differences a scalar loss. It is O(#params) forward rollouts, so keep models
and horizons tiny.

Reports the worst relative error and can persist every paired component.
The transition VJP uses a MuJoCo finite-difference epsilon fixed at 1e-8;
``--reference-eps`` only controls the independent whole-rollout central-difference reference.

Example:
    python benchmarks/compare_finite_difference.py --model dm_pendulum dm_acrobot --horizon 3
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from _bench_common import MODEL_NAMES, build_model, make_inputs, write_csv
from diffmjstep import mj_rollout

TRANSITION_EPS = 1e-8


def _loss(model, x0, U) -> torch.Tensor:
    X = mj_rollout(
        model,
        x0,
        U,
        backend="mujoco_rollout",
        return_all=True,
        eps=TRANSITION_EPS,
    )
    return X.square().sum()


def _transition_fd_grad(model, x0, U) -> np.ndarray:
    xb = x0.detach().clone().requires_grad_(True)
    Ub = U.detach().clone().requires_grad_(True)
    _loss(model, xb, Ub).backward()
    return np.concatenate([xb.grad.reshape(-1).numpy(), Ub.grad.reshape(-1).numpy()])


def _fd_grad(model, x0, U, reference_eps: float) -> np.ndarray:
    flat = torch.cat([x0.reshape(-1), U.reshape(-1)]).clone()
    nx = x0.numel()
    shape_x, shape_U = x0.shape, U.shape

    def loss_at(vec: torch.Tensor) -> float:
        x = vec[:nx].reshape(shape_x)
        u = vec[nx:].reshape(shape_U)
        with torch.no_grad():
            return float(_loss(model, x, u))

    grad = np.zeros(flat.numel())
    for i in range(flat.numel()):
        plus = flat.clone()
        plus[i] += reference_eps
        minus = flat.clone()
        minus[i] -= reference_eps
        grad[i] = (loss_at(plus) - loss_at(minus)) / (2 * reference_eps)
    return grad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", nargs="+", default=["dm_pendulum", "dm_acrobot", "dm_reacher"], choices=MODEL_NAMES)
    parser.add_argument("--horizon", nargs="+", type=int, default=[1, 3, 8])
    parser.add_argument(
        "--reference-eps",
        nargs="+",
        type=float,
        default=[1e-5, 1e-6, 1e-7],
        help="whole-rollout central-difference epsilon",
    )
    parser.add_argument("--seed", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    component_rows = []
    configurations = 0
    worst_relative_error = 0.0
    for model_name in args.model:
        model = build_model(model_name)
        for seed in args.seed:
            for horizon in args.horizon:
                x0, U = make_inputs(
                    model,
                    batch=1,
                    horizon=horizon,
                    dtype="float64",
                    seed=seed,
                )
                g_transition_fd = _transition_fd_grad(model, x0, U)
                for reference_eps in args.reference_eps:
                    g_rollout_fd = _fd_grad(model, x0, U, reference_eps)
                    component_rows.extend(
                        {
                            "model": model_name,
                            "seed": seed,
                            "horizon": horizon,
                            "reference_eps": reference_eps,
                            "transition_eps": TRANSITION_EPS,
                            "component": component,
                            "transition_fd_vjp": transition_fd,
                            "whole_rollout_fd": rollout_fd,
                        }
                        for component, (transition_fd, rollout_fd) in enumerate(
                            zip(g_transition_fd, g_rollout_fd, strict=True)
                        )
                    )
                    relative_error = np.linalg.norm(
                        g_transition_fd - g_rollout_fd
                    ) / (np.linalg.norm(g_rollout_fd) + 1e-30)
                    worst_relative_error = max(
                        worst_relative_error, float(relative_error)
                    )
                    configurations += 1

    print(
        f"measured {configurations} configurations; "
        f"worst relative error {worst_relative_error:.3g}"
    )
    manifest = {
        "seed": args.seed,
        "repeats": 1,
        "warmups": 0,
        "dtype": "float64",
        "nthread": 1,
        "reference_eps": args.reference_eps,
        "transition_eps": TRANSITION_EPS,
        "horizons": args.horizon,
        "models": args.model,
    }
    if args.out:
        write_csv(args.out, component_rows, **manifest)
        print(f"wrote {len(component_rows)} components to {args.out}")


if __name__ == "__main__":
    main()
