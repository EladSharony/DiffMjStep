"""Integrator ablation: gradient agreement and forward stability per integrator.

DiffMjStep is NOT limited to Euler. `mjd_transitionFD` supports Euler, implicit, and
implicitfast; only RK4 is rejected (a MuJoCo limitation, raised by MuJoCo itself). This
script sweeps integrator x timestep on one model and reports, for each cell:

  - grad_rel_error / grad_cosine : MuJoCo transition-FD VJP vs a whole-rollout central-FD
    reference. The fixed transition epsilon (1e-8) and configured reference epsilon are
    recorded separately. These metrics measure numerical agreement for the selected model
    and step.
  - fwd_rel_error : final-state difference from RK4 at the same dt (RK4 runs forward but
    cannot be used with `mjd_transitionFD`). MuJoCo's "Euler" already integrates joint
    damping implicitly, so the difference from implicit/implicitfast is model-dependent.
  - max_abs_state : max |state| over the rollout, a blow-up detector for stability.

RK4 is included only as a forward comparison; it has no gradient row.

Example:
    python benchmarks/ablate_integrator.py --model dm_cartpole --dt 0.005 0.01 0.02 0.05
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import mujoco

from _bench_common import MODEL_NAMES, build_model, make_inputs, write_csv
from compare_finite_difference import TRANSITION_EPS, _fd_grad, _transition_fd_grad
from diffmjstep import mj_rollout

# Mutate in-memory options instead of maintaining one XML asset per configuration.
INTEGRATORS = {
    "Euler": mujoco.mjtIntegrator.mjINT_EULER,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
}
DIFFERENTIABLE = tuple(INTEGRATORS)  # RK4 excluded: mjd_transitionFD rejects it.

def _forward(model, x0, U) -> np.ndarray:
    with torch.no_grad():
        return mj_rollout(model, x0, U, backend="mujoco_rollout", return_all=True).numpy()


def _row(model_name, integ, dt, horizon, x0, U, reference_eps):
    model = build_model(model_name)
    model.opt.timestep = dt

    # RK4 reference trajectory at the same dt (forward only; RK4 isn't differentiable here).
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    ref = _forward(model, x0, U)

    model.opt.integrator = INTEGRATORS[integ]
    traj = _forward(model, x0, U)
    fwd_rel = float(np.linalg.norm(traj - ref) / (np.linalg.norm(ref) + 1e-30))
    max_abs = float(np.abs(traj).max())

    g_transition_fd = _transition_fd_grad(model, x0, U)
    g_rollout_fd = _fd_grad(model, x0, U, reference_eps)
    rel = float(np.linalg.norm(g_transition_fd - g_rollout_fd)
                / (np.linalg.norm(g_rollout_fd) + 1e-30))
    cos = float(g_transition_fd @ g_rollout_fd
                / ((np.linalg.norm(g_transition_fd) * np.linalg.norm(g_rollout_fd)) + 1e-30))
    return {
        "model": model_name,
        "dof": int(model.nv),
        "integrator": integ,
        "dt": dt,
        "horizon": horizon,
        "reference_eps": reference_eps,
        "transition_eps": TRANSITION_EPS,
        "grad_rel_error": rel,
        "grad_cosine": cos,
        "fwd_rel_error": fwd_rel,
        "max_abs_state": max_abs,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="dm_acrobot", choices=MODEL_NAMES)
    p.add_argument("--dt", nargs="+", type=float, default=[0.005, 0.01, 0.02, 0.05])
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument(
        "--reference-eps",
        type=float,
        default=1e-6,
        help="whole-rollout central-difference epsilon",
    )
    p.add_argument("--out", default=None)
    args = p.parse_args()

    base = build_model(args.model)
    x0, U = make_inputs(base, batch=1, horizon=args.horizon, dtype="float64")

    rows = [_row(args.model, integ, dt, args.horizon, x0, U, args.reference_eps)
            for dt in args.dt for integ in DIFFERENTIABLE]

    print(f"measured {len(rows)} rows")

    if args.out:
        write_csv(
            args.out,
            rows,
            seed=0,
            repeats=1,
            warmups=0,
            dtype="float64",
            nthread=1,
            reference_eps=args.reference_eps,
            transition_eps=TRANSITION_EPS,
            timesteps=args.dt,
            horizon=args.horizon,
            integrators=list(DIFFERENTIABLE),
        )
        print(f"\nwrote {len(rows)} rows to {args.out}")

    # Self-check: require close numerical agreement at the finest requested timestep.
    fine = min(args.dt)
    for r in rows:
        if r["dt"] == fine:
            assert r["grad_rel_error"] < 1e-4, (
                f"{r['integrator']} gradient disagreement at dt={fine}: "
                f"{r['grad_rel_error']:.2e}"
            )


if __name__ == "__main__":
    main()
