"""Backward (reverse-time VJP) timing for DiffMjStep backends.

Only the `.backward()` call is timed; the forward pass that rebuilds the graph is
excluded. This isolates the transition-Jacobian sweep that an optimized native
backend would target.

Example:
    python benchmarks/bench_backward.py \\
        --backend python_vjp cpp_vjp --model dm_cheetah \\
        --batch 1 16 256 --horizon 1 8 32 --repeat 20 --warmup 5
"""
from __future__ import annotations


from _bench_common import (
    build_argparser,
    build_model,
    make_inputs,
    samples_backward,
    write_csv,
)
from diffmjstep import mj_rollout
from diffmjstep.core import _auto_nthread

BACKWARD_BACKENDS = {"python_vjp", "cpp_vjp"}


def main() -> None:
    parser = build_argparser(__doc__)
    parser.set_defaults(backend=["python_vjp"])
    parser.add_argument("--nthread", type=int, default=None, help="threads for cpp_vjp (default auto)")
    args = parser.parse_args()
    rows = []

    for model_name in args.model:
        model = build_model(model_name)
        for backend in args.backend:
            if backend not in BACKWARD_BACKENDS:
                raise SystemExit(f"bench_backward supports {sorted(BACKWARD_BACKENDS)}, got {backend!r}")
            for batch in args.batch:
                for horizon in args.horizon:
                    x0, U = make_inputs(model, batch, horizon, args.dtype)
                    is_cpp = backend == "cpp_vjp"
                    runtime_backend = "cpp_vjp" if is_cpp else "mujoco_rollout"
                    nthread = _auto_nthread(batch, args.nthread) if is_cpp else 1

                    def build_loss():
                        xb = x0.detach().clone().requires_grad_(True)
                        Ub = U.detach().clone().requires_grad_(True)
                        X = mj_rollout(
                            model, xb, Ub, backend=runtime_backend, return_all=True,
                            nthread=args.nthread if is_cpp else None,
                        )
                        return X.square().sum()

                    samples = samples_backward(
                        build_loss, repeat=args.repeat, warmup=args.warmup
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
