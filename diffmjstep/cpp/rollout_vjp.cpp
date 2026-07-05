// Native reverse-time VJP through a MuJoCo rollout, threaded over the batch.
//
// The forward (and therefore the per-step trajectory) is computed in Python via
// mujoco.rollout; this kernel only does the expensive backward: at each (batch, step)
// it recomputes the transition Jacobians A_t, B_t with mjd_transitionFD and accumulates
// the adjoint. The FD sweep is GIL-bound from Python, so here we release the GIL and
// run one mjData + one set of scratch buffers per worker thread.
//
// Layout (all float64, contiguous):
//   full_snapshots : [B, T+1, nfull] full physics state including x0 at t = 0
//   U              : [B, T, nu]
//   grad_X         : optional [B, T+1, nx] when return_all else [B, nx]
//   grad_Y         : optional [B, T, nsensordata], or an empty sentinel
// returns grad_x0 [B, nx], grad_U [B, T, nu] or an empty sentinel.
//
// Correctness mirrors the Python rollout VJPs; qpos state-mode with nq == nv, so the
// nx = 2*nv+na transition Jacobians act directly on the packed state.

#include <mujoco/mujoco.h>
#include <torch/extension.h>

#include <algorithm>
#include <thread>
#include <vector>

namespace {
void process_chunk(const mjModel* m, mjData* d, int b_begin, int b_end,
                   const double* full_snapshots, const double* U, const double* grad_X,
                   const double* grad_Y, double* grad_x0, double* grad_U,
                   int T, int nx, int nfull, int nu, int ns,
                   bool return_all, bool needs_control_grad, bool use_state,
                   bool use_sensors,
                   double eps, mjtByte centered) {
  std::vector<mjtNum> A(static_cast<size_t>(nx) * nx);
  std::vector<mjtNum> B;
  if (needs_control_grad) {
    B.resize(static_cast<size_t>(nx) * nu);
  }
  std::vector<mjtNum> C;
  if (use_sensors) {
    C.resize(static_cast<size_t>(ns) * nx);
  }
  std::vector<mjtNum> D;
  if (use_sensors && needs_control_grad) {
    D.resize(static_cast<size_t>(ns) * nu);
  }
  std::vector<mjtNum> lam(nx), lam_new(nx);

  for (int b = b_begin; b < b_end; ++b) {
    if (return_all || !use_state) {
      std::fill(lam.begin(), lam.end(), 0.0);
    } else {
      const double* g = grad_X + static_cast<size_t>(b) * nx;
      for (int i = 0; i < nx; ++i) lam[i] = g[i];
    }

    for (int t = T - 1; t >= 0; --t) {
      if (return_all && use_state) {
        const double* g = grad_X + (static_cast<size_t>(b) * (T + 1) + (t + 1)) * nx;
        for (int i = 0; i < nx; ++i) lam[i] += g[i];
      }

      const double* full_snapshot =
          full_snapshots + (static_cast<size_t>(b) * (T + 1) + t) * nfull;
      mj_resetData(m, d);
      mj_setState(m, d, full_snapshot, mjSTATE_FULLPHYSICS);
      const double* u = U + (static_cast<size_t>(b) * T + t) * nu;
      for (int i = 0; i < nu; ++i) d->ctrl[i] = u[i];

      mj_forward(m, d);
      const bool forward_difference_D =
          centered && use_sensors && needs_control_grad;
      mjd_transitionFD(m, d, eps, centered, A.data(),
                       needs_control_grad ? B.data() : nullptr,
                       use_sensors ? C.data() : nullptr,
                       use_sensors && needs_control_grad && !forward_difference_D
                           ? D.data()
                           : nullptr);
      if (forward_difference_D) {
        // MuJoCo's centered-D ordering workaround: a D-only forward pass also
        // preserves one-sided control-limit handling.
        mjd_transitionFD(m, d, eps, 0, nullptr, nullptr, nullptr, D.data());
      }

      const double* gy = use_sensors
          ? grad_Y + (static_cast<size_t>(b) * T + t) * ns
          : nullptr;

      if (needs_control_grad) {
        // grad_U[b, t, k] = sum_i lam[i] * B[i, k] + sum_s gy[s] * D[s, k]
        double* gu = grad_U + (static_cast<size_t>(b) * T + t) * nu;
        for (int k = 0; k < nu; ++k) {
          double s = 0.0;
          for (int i = 0; i < nx; ++i) {
            s += lam[i] * B[static_cast<size_t>(i) * nu + k];
          }
          if (use_sensors) {
            for (int sensor = 0; sensor < ns; ++sensor) {
              s += gy[sensor] * D[static_cast<size_t>(sensor) * nu + k];
            }
          }
          gu[k] = s;
        }
      }
      // lam <- lam @ A + gy @ C
      for (int j = 0; j < nx; ++j) {
        double s = 0.0;
        for (int i = 0; i < nx; ++i) s += lam[i] * A[static_cast<size_t>(i) * nx + j];
        if (use_sensors) {
          for (int sensor = 0; sensor < ns; ++sensor) {
            s += gy[sensor] * C[static_cast<size_t>(sensor) * nx + j];
          }
        }
        lam_new[j] = s;
      }
      std::swap(lam, lam_new);
    }

    double* gx = grad_x0 + static_cast<size_t>(b) * nx;
    if (return_all && use_state) {
      const double* g0 = grad_X + static_cast<size_t>(b) * (T + 1) * nx;
      for (int i = 0; i < nx; ++i) gx[i] = lam[i] + g0[i];
    } else {
      for (int i = 0; i < nx; ++i) gx[i] = lam[i];
    }
  }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> rollout_backward_vjp(
    torch::Tensor full_snapshots, torch::Tensor U, torch::Tensor grad_X,
    torch::Tensor grad_Y,
    torch::Tensor data_addresses, int64_t model_addr, int64_t nthread,
    bool return_all, bool needs_control_grad, double eps, bool centered) {
  TORCH_CHECK(full_snapshots.dtype() == torch::kFloat64,
              "full_snapshots must be float64");
  TORCH_CHECK(U.dtype() == torch::kFloat64, "U must be float64");
  TORCH_CHECK(grad_X.dtype() == torch::kFloat64, "grad_X must be float64");
  TORCH_CHECK(grad_Y.dtype() == torch::kFloat64, "grad_Y must be float64");
  TORCH_CHECK(data_addresses.dtype() == torch::kInt64,
              "data_addresses must be int64");
  TORCH_CHECK(full_snapshots.device().is_cpu() && U.device().is_cpu() &&
                  grad_X.device().is_cpu() && grad_Y.device().is_cpu() &&
                  data_addresses.device().is_cpu(),
              "full_snapshots, U, grad_X, grad_Y, and data_addresses "
              "must be CPU tensors");
  TORCH_CHECK(full_snapshots.dim() == 3,
              "full_snapshots must have shape [B, T+1, nfull]");
  TORCH_CHECK(U.dim() == 3, "U must have shape [B, T, model.nu]");
  TORCH_CHECK(data_addresses.dim() == 1,
              "data_addresses must have shape [nthread]");
  const bool use_state = !(grad_X.dim() == 1 && grad_X.size(0) == 0);
  const bool use_sensors = !(grad_Y.dim() == 1 && grad_Y.size(0) == 0);
  TORCH_CHECK(!use_state || grad_X.dim() == (return_all ? 3 : 2),
              return_all ? "grad_X must have shape [B, T+1, nx]"
                         : "grad_X must have shape [B, nx]");
  TORCH_CHECK(!use_sensors || grad_Y.dim() == 3,
              "grad_Y must have shape [B, T, nsensordata]");
  full_snapshots = full_snapshots.contiguous();
  U = U.contiguous();
  grad_X = grad_X.contiguous();
  grad_Y = grad_Y.contiguous();
  data_addresses = data_addresses.contiguous();

  const mjModel* m = reinterpret_cast<const mjModel*>(model_addr);
  const int B = static_cast<int>(full_snapshots.size(0));
  const int Tp1 = static_cast<int>(full_snapshots.size(1));
  const int snapshot_width = static_cast<int>(full_snapshots.size(2));
  const int T = static_cast<int>(U.size(1));
  const int nq = m->nq, nv = m->nv, na = m->na, nu = m->nu;
  const int nx = 2 * nv + na;
  const int ns = m->nsensordata;
  const int nfull = mj_stateSize(m, mjSTATE_FULLPHYSICS);

  TORCH_CHECK(nq == nv, "cpp_vjp supports qpos state-mode with nq == nv only");
  TORCH_CHECK(m->opt.integrator != mjINT_RK4,
              "mjd_transitionFD does not support RK4 integrator; use Euler or implicit integration");
  TORCH_CHECK(Tp1 == T + 1, "full_snapshots must have T+1 steps along dim 1");
  TORCH_CHECK(snapshot_width == nfull,
              "FULLPHYSICS snapshot width mismatch: expected ", nfull,
              ", got ", snapshot_width);
  TORCH_CHECK(static_cast<int>(U.size(0)) == B &&
                  static_cast<int>(U.size(2)) == nu,
              "U must have shape [B, T, model.nu]");
  if (use_state) {
    if (return_all) {
      TORCH_CHECK(static_cast<int>(grad_X.size(0)) == B &&
                      static_cast<int>(grad_X.size(1)) == Tp1 &&
                      static_cast<int>(grad_X.size(2)) == nx,
                  "grad_X must have shape [B, T+1, nx]");
    } else {
      TORCH_CHECK(static_cast<int>(grad_X.size(0)) == B &&
                      static_cast<int>(grad_X.size(1)) == nx,
                  "grad_X must have shape [B, nx]");
    }
  }
  if (use_sensors) {
    TORCH_CHECK(static_cast<int>(grad_Y.size(0)) == B &&
                    static_cast<int>(grad_Y.size(1)) == T &&
                    static_cast<int>(grad_Y.size(2)) == ns,
                "grad_Y must have shape [B, T, nsensordata]");
  }

  auto opts = torch::TensorOptions().dtype(torch::kFloat64);
  torch::Tensor grad_x0 = torch::zeros({B, nx}, opts);
  torch::Tensor grad_U = needs_control_grad
      ? torch::zeros({B, T, nu}, opts)
      : torch::empty({0}, opts);
  if (B == 0 || T == 0) {
    TORCH_CHECK(data_addresses.size(0) == 0,
                "data_addresses must be empty when B or T is zero");
    if (T == 0 && use_state) {
      grad_x0.copy_(return_all ? grad_X.select(1, 0) : grad_X);
    }
    return {grad_x0, grad_U};
  }

  int nthr = std::max<int64_t>(1, nthread);
  nthr = std::min(nthr, B);
  TORCH_CHECK(static_cast<int>(data_addresses.size(0)) == nthr,
              "data_addresses must have shape [nthread]");

  const double* full_snapshots_p = full_snapshots.data_ptr<double>();
  const double* U_p = U.data_ptr<double>();
  const double* grad_X_p = use_state ? grad_X.data_ptr<double>() : nullptr;
  const double* grad_Y_p = use_sensors ? grad_Y.data_ptr<double>() : nullptr;
  double* gx0_p = grad_x0.data_ptr<double>();
  double* gU_p = needs_control_grad ? grad_U.data_ptr<double>() : nullptr;

  const mjtByte cflag = centered ? 1 : 0;

  // Python owns and registers these mjData instances for callback compatibility.
  // The synchronous extension call keeps their wrappers alive until all workers join.
  const int64_t* data_addresses_p = data_addresses.data_ptr<int64_t>();
  std::vector<mjData*> datas;
  datas.reserve(nthr);
  for (int i = 0; i < nthr; ++i) {
    mjData* data = reinterpret_cast<mjData*>(data_addresses_p[i]);
    TORCH_CHECK(data != nullptr, "data address must not be null");
    datas.push_back(data);
  }

  {
    pybind11::gil_scoped_release release;
    if (nthr <= 1) {
      process_chunk(m, datas[0], 0, B, full_snapshots_p, U_p, grad_X_p,
                    grad_Y_p, gx0_p, gU_p, T, nx, nfull, nu, ns, return_all,
                    needs_control_grad, use_state, use_sensors, eps, cflag);
    } else {
      std::vector<std::thread> threads;
      for (int ti = 0; ti < nthr; ++ti) {
        const int b0 = ti * B / nthr;
        const int b1 = (ti + 1) * B / nthr;
        threads.emplace_back(process_chunk, m, datas[ti], b0, b1,
                             full_snapshots_p, U_p, grad_X_p, grad_Y_p,
                             gx0_p, gU_p, T, nx, nfull, nu, ns, return_all,
                             needs_control_grad, use_state, use_sensors, eps, cflag);
      }
      for (auto& th : threads) th.join();
    }
  }
  return {grad_x0, grad_U};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
  mod.def("rollout_backward_vjp", &rollout_backward_vjp,
          "Threaded reverse-time VJP through a MuJoCo rollout (CPU, float64).");
}
