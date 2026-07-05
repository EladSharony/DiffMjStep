# Tests

The suite covers the public batched API, independent numerical references, native
Linux/macOS execution, runtime isolation, sensors, and generated research artifacts.

```bash
uv sync --frozen --python 3.12 --extra dev
uv run --frozen pytest -q
```

Use `float64` for gradient checks. The primary derivative checks use closed-form dynamics,
whole-rollout finite differences, or `torch.autograd.gradcheck`; cross-backend agreement is
only a regression check.

The supported exports are `mj_step`, `mj_rollout`, and `mj_linearize`. Step and rollout
inputs always include an explicit batch axis. Runtime backends are `auto`,
`mujoco_rollout`, and `cpp_vjp`.

Sensor timing is `Y[t] = g(x_t, u_t)`, so the final sensor value is the final pre-step
measurement. Contact-rich finite differences test a useful local estimate, not smoothness
through contact discontinuities.

Benchmark tests verify CSV/manifest integrity and small end-to-end generators.
