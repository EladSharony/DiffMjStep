# Benchmarks

Benchmark outputs are machine-specific and intentionally untracked. Each runner writes a CSV
and adjacent JSON manifest containing its command, revision, environment, and checksum. Run
experiments on an idle target machine, then render figures from those files.

## Environment

```bash
uv python install 3.12
uv sync --frozen --python 3.12 --extra dev
```

## CPU measurements

```bash
mkdir -p benchmarks/results

uv run --frozen python benchmarks/bench_rollout.py \
  --backend python_ref mujoco_rollout \
  --model dm_pendulum dm_acrobot dm_cartpole dm_finger dm_hopper dm_walker dm_cheetah dm_reacher dm_pointmass \
  --batch 1 16 64 256 1024 4096 --horizon 32 --dtype float64 \
  --repeat 15 --warmup 3 --out benchmarks/results/cpu_forward_samples.csv

uv run --frozen python benchmarks/bench_backward.py \
  --backend python_vjp cpp_vjp \
  --model dm_pendulum dm_acrobot dm_cartpole dm_finger dm_hopper dm_walker dm_cheetah dm_reacher dm_pointmass \
  --batch 1 16 64 256 1024 4096 --horizon 32 --dtype float64 \
  --repeat 8 --warmup 2 --out benchmarks/results/cpu_backward_samples.csv

uv run --frozen python benchmarks/compare_finite_difference.py \
  --model dm_pendulum dm_acrobot dm_reacher dm_pointmass \
  --horizon 1 3 8 --reference-eps 1e-5 1e-6 1e-7 --seed 0 1 2 \
  --out benchmarks/results/gradient_components.csv

for model in dm_pendulum dm_acrobot dm_cartpole dm_finger dm_hopper dm_walker dm_cheetah dm_reacher dm_pointmass; do
  uv run --frozen python benchmarks/ablate_integrator.py \
    --model "$model" --dt 0.005 0.01 0.02 0.05 0.1 --horizon 5 \
    --reference-eps 1e-6 \
    --out "benchmarks/results/integrator_ablation_${model}.csv"
done
```

The CPU runners load dm_control XML and assets through raw MuJoCo bindings. Domains using RK4
are normalized to Euler because `mjd_transitionFD` supports Euler, implicit, and implicitfast,
not RK4. The forward runner disables solver warm-start once so both compared paths execute
the same zero-copy dynamics. `bench_backward.py` excludes graph construction and times only
`.backward()`.

## Figures

```bash
uv run --frozen python benchmarks/plot_dm_control.py
uv run --frozen python benchmarks/plot_efficiency.py
uv run --frozen python benchmarks/plot_accuracy.py
uv run --frozen python benchmarks/plot_integrator_ablation.py
```

Plotters only read supplied CSV/manifest pairs; they never run simulations. Keep generated
results outside Git or publish them with the paper's immutable artifact release.
