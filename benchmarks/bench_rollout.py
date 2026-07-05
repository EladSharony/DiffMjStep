"""Forward-rollout timing for DiffMjStep backends.

Example:
    python benchmarks/bench_rollout.py \\
        --backend python_ref mujoco_rollout \\
        --model dm_acrobot --batch 1 16 256 --horizon 8 32 --repeat 20 --warmup 5
"""
from __future__ import annotations

import mujoco
import torch

from _bench_common import (
    build_argparser,
    build_model,
    make_inputs,
    python_rollout,
    samples_forward,
    write_csv,
)
from diffmjstep import mj_rollout
from diffmjstep.core import _auto_nthread

FORWARD_BACKENDS = {"python_ref", "mujoco_rollout"}


def main() -> None:
    parser = build_argparser(__doc__)
    parser.add_argument("--nthread", type=int, default=None, help="threads for mujoco_rollout (default auto)")
    args = parser.parse_args()
    rows = []

    for model_name in args.model:
        model = build_model(model_name)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        for backend in args.backend:
            if backend not in FORWARD_BACKENDS:
                raise SystemExit(f"bench_rollout supports {sorted(FORWARD_BACKENDS)}, got {backend!r}")
            for batch in args.batch:
                for horizon in args.horizon:
                    x0, U = make_inputs(model, batch, horizon, args.dtype)
                    is_fast = backend == "mujoco_rollout"
                    nthread = _auto_nthread(batch, args.nthread) if is_fast else 1

                    def forward() -> torch.Tensor:
                        with torch.no_grad():
                            if backend == "python_ref":
                                return python_rollout(model, x0, U)
                            return mj_rollout(
                                model, x0, U, backend=backend, return_all=True,
                                nthread=args.nthread if is_fast else None,
                            )

                    samples = samples_forward(
                        forward, repeat=args.repeat, warmup=args.warmup
                    )
                    for sample, elapsed_ms in enumerate(samples):
                        rows.append(
                            {
                                "backend": backend,
                                "model": model_name,
                                "batch": batch,
                                "horizon": horizon,
                                "dtype": args.dtype,
                                "dof": int(model.nv),
                                "nthread": nthread,
                                "sample": sample,
                                "elapsed_ms": elapsed_ms,
                            }
                        )

    print(f"measured {len(rows)} rows")
    if args.out:
        write_csv(
            args.out,
            rows,
            seed=0,
            repeats=args.repeat,
            warmups=args.warmup,
            dtype=args.dtype,
            nthread="auto" if args.nthread is None else args.nthread,
            effective_nthreads=sorted({row["nthread"] for row in rows}),
        )
        print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
