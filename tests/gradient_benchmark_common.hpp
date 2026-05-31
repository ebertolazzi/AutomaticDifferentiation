#pragma once

#include <Eigen/Core>

#include <TinyAD/Scalar.hh>
#include <cppad/cg.hpp>
#include <cppad/cppad.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef AUTODIFF_HAS_ENZYME
#include <enzyme/utils>
#endif

namespace autodiff_benchmark {

constexpr std::size_t kDimension = 128;
constexpr double kFiniteDiffStep = 1e-7;
constexpr double kFiniteDiffTolerance = 5e-5;
constexpr double kCrossTolerance = 1e-10;

using DenseVector = std::vector<double>;
using CGScalar = CppAD::cg::CG<double>;
using CppADTape = CppAD::ADFun<double>;
using CppADCGTape = CppAD::ADFun<CGScalar>;

struct GradientResult {
  double value = 0.0;
  DenseVector gradient;
};

struct ErrorMetrics {
  double max_abs = 0.0;
  double rms = 0.0;
};

template <class T, class = void>
struct has_tinyad_dynamic_mode : std::false_type {};

template <class T>
struct has_tinyad_dynamic_mode<T, std::void_t<decltype(T::dynamic_mode_)>> : std::true_type {};

template <class Scalar>
Scalar scalar_constant_like(const Scalar& like, const double value) {
  if constexpr (has_tinyad_dynamic_mode<Scalar>::value) {
    if constexpr (Scalar::dynamic_mode_) {
      return Scalar::make_passive(value, like.grad.size());
    } else {
      return Scalar(value);
    }
  } else {
    return Scalar(value);
  }
}

template <class T>
T ad_sin(const T& value) {
  using std::sin;
  return sin(value);
}

template <class T>
T ad_cos(const T& value) {
  using std::cos;
  return cos(value);
}

template <class T>
T ad_exp(const T& value) {
  using std::exp;
  return exp(value);
}

template <class T>
T ad_log(const T& value) {
  using std::log;
  return log(value);
}

template <class Scalar, class Getter>
Scalar coupled_objective(Getter&& get_value) {
  const auto& seed = get_value(0);
  const auto c = [&](const double value) { return scalar_constant_like(seed, value); };

  Scalar total = c(0.0);
  const Scalar c001 = c(0.01);
  const Scalar c002 = c(0.02);
  const Scalar c003 = c(0.03);
  const Scalar c005 = c(0.05);
  const Scalar c010 = c(0.10);
  const Scalar c015 = c(0.15);
  const Scalar c020 = c(0.20);
  const Scalar c025 = c(0.25);
  const Scalar c035 = c(0.35);
  const Scalar c050 = c(0.50);
  const Scalar c100 = c(1.0);
  const Scalar inv_n = c(1.0 / static_cast<double>(kDimension));

  for (std::size_t i = 0; i < kDimension; ++i) {
    const auto& x0 = get_value(i);
    const auto& x1 = get_value((i + 1) % kDimension);
    const auto& x2 = get_value((i + 3) % kDimension);
    const auto& x3 = get_value((i + 7) % kDimension);
    const auto& x4 = get_value((i + 11) % kDimension);
    const auto& x5 = get_value((i + 19) % kDimension);

    const Scalar affine = x0 + c010 * x1 - c005 * x2;
    const Scalar trig = ad_exp(c005 * x0) * ad_sin(x0 + c025 * x1) +
                        c035 * ad_cos(x2 - c015 * x3);
    const Scalar log_term = ad_log(c100 + x0 * x0 + c020 * x4 * x4);
    const Scalar mix = affine * affine * log_term;
    const Scalar cross = c002 * ad_sin(x0 * x3) * ad_exp(c003 * x5);
    const Scalar left = x0 - c050 * x4;
    const Scalar right = x1 + c020 * x5;
    const Scalar quartic = c001 * left * left * right * right;

    total += trig + mix + cross + quartic;
  }

  return total * inv_n;
}

inline DenseVector make_input() {
  DenseVector x(kDimension);
  for (std::size_t i = 0; i < kDimension; ++i) {
    const double angle = 0.13 * static_cast<double>(i + 1);
    x[i] = 0.25 * std::sin(angle) + 0.10 * std::cos(0.37 * angle);
  }
  return x;
}

inline double passive_objective(const DenseVector& x) {
  return coupled_objective<double>([&](std::size_t i) -> const double& { return x[i]; });
}

inline ErrorMetrics compute_error(const DenseVector& lhs, const DenseVector& rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::runtime_error("Mismatched gradient sizes");
  }

  double max_abs = 0.0;
  double sum_sq = 0.0;

  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const double diff = lhs[i] - rhs[i];
    max_abs = std::max(max_abs, std::abs(diff));
    sum_sq += diff * diff;
  }

  return {max_abs, std::sqrt(sum_sq / static_cast<double>(lhs.size()))};
}

inline DenseVector finite_difference_gradient(const DenseVector& x, const double step) {
  DenseVector grad(kDimension, 0.0);
  DenseVector x_work = x;

  for (std::size_t i = 0; i < kDimension; ++i) {
    x_work[i] += step;
    const double fp = passive_objective(x_work);
    x_work[i] -= 2.0 * step;
    const double fm = passive_objective(x_work);
    x_work[i] += step;
    grad[i] = (fp - fm) / (2.0 * step);
  }

  return grad;
}

inline CppADTape build_cppad_tape() {
  using ADScalar = CppAD::AD<double>;

  std::vector<ADScalar> ax(kDimension);
  std::fill(ax.begin(), ax.end(), ADScalar(0.0));
  CppAD::Independent(ax);

  std::vector<ADScalar> ay(1);
  ay[0] = coupled_objective<ADScalar>([&](std::size_t i) -> const ADScalar& { return ax[i]; });

  return CppADTape(ax, ay);
}

inline CppADCGTape build_cppadcg_tape() {
  using ADScalar = CppAD::AD<CGScalar>;

  std::vector<ADScalar> ax(kDimension);
  std::fill(ax.begin(), ax.end(), ADScalar(CGScalar(0.0)));
  CppAD::Independent(ax);

  std::vector<ADScalar> ay(1);
  ay[0] = coupled_objective<ADScalar>([&](std::size_t i) -> const ADScalar& { return ax[i]; });

  return CppADCGTape(ax, ay);
}

inline GradientResult eval_cppad_forward(CppADTape& fun, const DenseVector& x) {
  GradientResult out;
  out.gradient.assign(kDimension, 0.0);

  const auto y0 = fun.Forward(0, x);
  out.value = y0[0];

  DenseVector direction(kDimension, 0.0);
  for (std::size_t j = 0; j < kDimension; ++j) {
    std::fill(direction.begin(), direction.end(), 0.0);
    direction[j] = 1.0;
    const auto dy = fun.Forward(1, direction);
    out.gradient[j] = dy[0];
  }

  return out;
}

inline GradientResult eval_cppad_reverse(CppADTape& fun, const DenseVector& x) {
  GradientResult out;
  const auto y0 = fun.Forward(0, x);
  out.value = y0[0];

  const DenseVector weights(1, 1.0);
  out.gradient = fun.Reverse(1, weights);

  return out;
}

inline GradientResult eval_tinyad_forward(const DenseVector& x) {
  using ADScalar = TinyAD::Scalar<Eigen::Dynamic, double, false>;

  std::vector<ADScalar> ax(kDimension);
  for (std::size_t i = 0; i < kDimension; ++i) {
    ax[i] = ADScalar::make_active(x[i], static_cast<int>(i), static_cast<int>(kDimension));
  }

  const ADScalar y = coupled_objective<ADScalar>([&](std::size_t i) -> const ADScalar& { return ax[i]; });

  GradientResult out;
  out.value = y.val;
  out.gradient.resize(kDimension);
  for (std::size_t i = 0; i < kDimension; ++i) {
    out.gradient[i] = y.grad(static_cast<Eigen::Index>(i));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Analytic gradient and Hessian of coupled_objective
// ---------------------------------------------------------------------------

inline GradientResult eval_analytic_gradient(const DenseVector& x) {
  const std::size_t N = kDimension;
  const double inv_n = 1.0 / static_cast<double>(N);

  GradientResult out;
  out.value = passive_objective(x);
  out.gradient.assign(N, 0.0);

  for (std::size_t i = 0; i < N; ++i) {
    const std::size_t idx0 = i;
    const std::size_t idx1 = (i +  1) % N;
    const std::size_t idx2 = (i +  3) % N;
    const std::size_t idx3 = (i +  7) % N;
    const std::size_t idx4 = (i + 11) % N;
    const std::size_t idx5 = (i + 19) % N;

    const double x0 = x[idx0];
    const double x1 = x[idx1];
    const double x2 = x[idx2];
    const double x3 = x[idx3];
    const double x4 = x[idx4];
    const double x5 = x[idx5];

    const double e05x0       = std::exp(0.05 * x0);
    const double arg_s       = x0 + 0.25 * x1;
    const double sin_s       = std::sin(arg_s);
    const double cos_s       = std::cos(arg_s);
    const double arg_c       = x2 - 0.15 * x3;
    const double cos_c       = std::cos(arg_c);
    const double sin_c       = std::sin(arg_c);

    const double affine      = x0 + 0.10 * x1 - 0.05 * x2;
    const double log_denom   = 1.0 + x0 * x0 + 0.20 * x4 * x4;
    const double log_t       = std::log(log_denom);

    const double left        = x0 - 0.50 * x4;
    const double right       = x1 + 0.20 * x5;

    const double sin_x0x3   = std::sin(x0 * x3);
    const double cos_x0x3   = std::cos(x0 * x3);
    const double exp_03x5   = std::exp(0.03 * x5);

    const double dtrig_dx0  = 0.05 * e05x0 * sin_s + e05x0 * cos_s;
    const double dtrig_dx1  = e05x0 * cos_s * 0.25;
    const double dtrig_dx2  = -0.35 * sin_c;
    const double dtrig_dx3  =  0.35 * sin_c * 0.15;

    const double daffine_dx0 =  1.0;
    const double daffine_dx1 =  0.10;
    const double daffine_dx2 = -0.05;
    const double dlog_dx0   = 2.0 * x0 / log_denom;
    const double dlog_dx4   = 0.40 * x4 / log_denom;

    const double dmix_dx0   = 2.0 * affine * daffine_dx0 * log_t + affine * affine * dlog_dx0;
    const double dmix_dx1   = 2.0 * affine * daffine_dx1 * log_t;
    const double dmix_dx2   = 2.0 * affine * daffine_dx2 * log_t;
    const double dmix_dx4   = affine * affine * dlog_dx4;

    const double dcross_dx0 = 0.02 * cos_x0x3 * x3 * exp_03x5;
    const double dcross_dx3 = 0.02 * cos_x0x3 * x0 * exp_03x5;
    const double dcross_dx5 = 0.02 * sin_x0x3 * exp_03x5 * 0.03;

    const double dquartic_dx0 = 0.01 * 2.0 * left * right * right;
    const double dquartic_dx1 = 0.01 * left * left * 2.0 * right;
    const double dquartic_dx4 = 0.01 * 2.0 * left * (-0.50) * right * right;
    const double dquartic_dx5 = 0.01 * left * left * 2.0 * right * 0.20;

    out.gradient[idx0] += inv_n * (dtrig_dx0 + dmix_dx0 + dcross_dx0 + dquartic_dx0);
    out.gradient[idx1] += inv_n * (dtrig_dx1 + dmix_dx1 + dquartic_dx1);
    out.gradient[idx2] += inv_n * (dtrig_dx2 + dmix_dx2);
    out.gradient[idx3] += inv_n * (dtrig_dx3 + dcross_dx3);
    out.gradient[idx4] += inv_n * (dmix_dx4 + dquartic_dx4);
    out.gradient[idx5] += inv_n * (dcross_dx5 + dquartic_dx5);
  }

  return out;
}

inline std::vector<double> eval_analytic_hessian_matrix(const DenseVector& x) {
  const std::size_t N = kDimension;
  const double inv_n = 1.0 / static_cast<double>(N);
  std::vector<double> H(N * N, 0.0);

  const auto addH = [&](std::size_t r, std::size_t c, double v) {
    if (r == c) {
      H[r * N + c] += v;
    } else {
      H[r * N + c] += v;
      H[c * N + r] += v;
    }
  };

  for (std::size_t i = 0; i < N; ++i) {
    const std::size_t idx0 = i;
    const std::size_t idx1 = (i +  1) % N;
    const std::size_t idx2 = (i +  3) % N;
    const std::size_t idx3 = (i +  7) % N;
    const std::size_t idx4 = (i + 11) % N;
    const std::size_t idx5 = (i + 19) % N;

    const double x0 = x[idx0];
    const double x1 = x[idx1];
    const double x2 = x[idx2];
    const double x3 = x[idx3];
    const double x4 = x[idx4];
    const double x5 = x[idx5];

    const double e05x0     = std::exp(0.05 * x0);
    const double arg_s     = x0 + 0.25 * x1;
    const double sin_s     = std::sin(arg_s);
    const double cos_s     = std::cos(arg_s);
    const double arg_c     = x2 - 0.15 * x3;
    const double sin_c     = std::sin(arg_c);
    const double cos_c     = std::cos(arg_c);

    const double affine    = x0 + 0.10 * x1 - 0.05 * x2;
    const double log_denom = 1.0 + x0 * x0 + 0.20 * x4 * x4;
    const double log_t     = std::log(log_denom);

    const double left      = x0 - 0.50 * x4;
    const double right     = x1 + 0.20 * x5;

    const double sin_x0x3 = std::sin(x0 * x3);
    const double cos_x0x3 = std::cos(x0 * x3);
    const double exp_03x5 = std::exp(0.03 * x5);

    const double dlog_dx0 = 2.0 * x0 / log_denom;
    const double dlog_dx4 = 0.40 * x4 / log_denom;

    // trig second derivatives
    addH(idx0, idx0, inv_n * e05x0 * ((0.0025 - 1.0) * sin_s + 0.10 * cos_s));
    addH(idx0, idx1, inv_n * 0.25 * e05x0 * (0.05 * cos_s - sin_s));
    addH(idx1, idx1, inv_n * (-0.0625) * e05x0 * sin_s);
    addH(idx2, idx2, inv_n * 0.35 * (-cos_c));
    addH(idx2, idx3, inv_n * 0.35 * 0.15 * cos_c);
    addH(idx3, idx3, inv_n * (-0.35 * 0.0225 * cos_c));

    // mix second derivatives
    const double d2log_dx0x0 = (2.0 * log_denom - 4.0 * x0 * x0) / (log_denom * log_denom);
    const double d2log_dx4x4 = 0.40 * (log_denom - 0.40 * x4 * x4) / (log_denom * log_denom);
    const double d2log_dx0x4 = -2.0 * x0 * dlog_dx4 / log_denom;

    struct Role { std::size_t idx; double dA; double dL; };
    const Role roles[6] = {
      {idx0,  1.00, dlog_dx0},
      {idx1,  0.10, 0.0},
      {idx2, -0.05, 0.0},
      {idx3,  0.0,  0.0},
      {idx4,  0.0,  dlog_dx4},
      {idx5,  0.0,  0.0},
    };
    auto d2log = [&](int ra, int rb) -> double {
      if (ra == 0 && rb == 0) return d2log_dx0x0;
      if (ra == 4 && rb == 4) return d2log_dx4x4;
      if ((ra == 0 && rb == 4) || (ra == 4 && rb == 0)) return d2log_dx0x4;
      return 0.0;
    };
    for (int ra = 0; ra < 6; ++ra) {
      for (int rb = ra; rb < 6; ++rb) {
        double v = 2.0 * roles[ra].dA * roles[rb].dA * log_t
                 + 2.0 * affine * roles[ra].dA * roles[rb].dL
                 + 2.0 * affine * roles[rb].dA * roles[ra].dL
                 + affine * affine * d2log(ra, rb);
        if (v != 0.0) addH(roles[ra].idx, roles[rb].idx, inv_n * v);
      }
    }

    // cross second derivatives
    addH(idx0, idx0, inv_n * 0.02 * (-sin_x0x3) * x3 * x3 * exp_03x5);
    addH(idx0, idx3, inv_n * 0.02 * (cos_x0x3 - sin_x0x3 * x0 * x3) * exp_03x5);
    addH(idx0, idx5, inv_n * 0.02 * cos_x0x3 * x3 * exp_03x5 * 0.03);
    addH(idx3, idx3, inv_n * 0.02 * (-sin_x0x3) * x0 * x0 * exp_03x5);
    addH(idx3, idx5, inv_n * 0.02 * cos_x0x3 * x0 * exp_03x5 * 0.03);
    addH(idx5, idx5, inv_n * 0.02 * sin_x0x3 * exp_03x5 * 0.0009);

    // quartic second derivatives
    addH(idx0, idx0, inv_n * 0.02 * right * right);
    addH(idx0, idx1, inv_n * 0.04 * left * right);
    addH(idx0, idx4, inv_n * (-0.01) * right * right);
    addH(idx0, idx5, inv_n * 0.02 * left * 2.0 * right * 0.20);
    addH(idx1, idx1, inv_n * 0.02 * left * left);
    addH(idx1, idx4, inv_n * (-0.02) * left * right);
    addH(idx1, idx5, inv_n * 0.004 * left * left);
    addH(idx4, idx4, inv_n * 0.005 * right * right);
    addH(idx4, idx5, inv_n * (-0.004) * left * right);
    addH(idx5, idx5, inv_n * 0.0008 * left * left);
  }
  return H;
}

struct HessianResultAnalytic {
  double value = 0.0;
  DenseVector hessian;
  DenseVector gradient;
};

inline HessianResultAnalytic eval_analytic_full(const DenseVector& x) {
  HessianResultAnalytic out;
  auto gr = eval_analytic_gradient(x);
  out.value    = gr.value;
  out.gradient = std::move(gr.gradient);
  out.hessian  = eval_analytic_hessian_matrix(x);
  return out;
}

// ---------------------------------------------------------------------------
// Enzyme AD
// ---------------------------------------------------------------------------
#ifdef AUTODIFF_HAS_ENZYME

extern int __enzyme_autodiff(...);
extern double __enzyme_fwddiff(...);

#ifndef AUTODIFF_ENZYME_IMPL_DEFINED
#define AUTODIFF_ENZYME_IMPL_DEFINED

inline double enzyme_objective_raw(const double* x, std::size_t n) noexcept {
  (void)n;
  return coupled_objective<double>([&](std::size_t i) -> const double& { return x[i]; });
}

#endif

inline GradientResult eval_enzyme_reverse(const DenseVector& x) {
  const std::size_t n = x.size();
  GradientResult out;
  out.value = passive_objective(x);
  out.gradient.assign(n, 0.0);
  __enzyme_autodiff(
      reinterpret_cast<void*>(enzyme_objective_raw),
      enzyme_dup,   x.data(), out.gradient.data(),
      enzyme_const, n);
  return out;
}

inline GradientResult eval_enzyme_forward(const DenseVector& x) {
  const std::size_t n = x.size();
  GradientResult out;
  out.gradient.resize(n);
  out.value = enzyme_objective_raw(x.data(), n);
  for (std::size_t j = 0; j < n; ++j) {
    DenseVector ej(n, 0.0);
    ej[j] = 1.0;
    out.gradient[j] = __enzyme_fwddiff(
        reinterpret_cast<void*>(enzyme_objective_raw),
        enzyme_dup,   x.data(), ej.data(),
        enzyme_const, n);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Gradient helper
// ---------------------------------------------------------------------------

inline void enzyme_gradient_raw(
  const double* x,
  double* grad,
  std::size_t n
) noexcept {

  std::fill_n(grad, n, 0.0);

  __enzyme_autodiff(
    reinterpret_cast<void*>(enzyme_objective_raw),
    enzyme_dup,
      const_cast<double*>(x),
      grad,
    enzyme_const,
      n
  );
}

inline double enzyme_gradient_component_raw(
  const double* x,
  std::size_t n,
  std::size_t component
) noexcept {

  std::array<double, kDimension> grad{};

  enzyme_gradient_raw(
    x,
    grad.data(),
    n
  );

  return grad[component];
}

// ---------------------------------------------------------------------------
// Hessian (Forward-over-Reverse)
// ---------------------------------------------------------------------------

inline DenseVector eval_enzyme_hessian(
  const DenseVector& x
) {

  const std::size_t n = x.size();

  DenseVector H(n * n, 0.0);

  DenseVector direction(n, 0.0);

  for (std::size_t j = 0; j < n; ++j) {

    std::fill(
      direction.begin(),
      direction.end(),
      0.0
    );

    direction[j] = 1.0;

    for (std::size_t i = 0; i < n; ++i) {

      H[i * n + j] =
        __enzyme_fwddiff(
          reinterpret_cast<void*>(
            enzyme_gradient_component_raw
          ),

          enzyme_dup,
            const_cast<double*>(x.data()),
            direction.data(),

          enzyme_const,
            n,

          enzyme_const,
            i
        );
    }
  }

  return H;
}

// ---------------------------------------------------------------------------
// Full result with Hessian (reverse-over-reverse, kept for compatibility)
// ---------------------------------------------------------------------------
struct EnzymeHessianResult {
  double value = 0.0;
  DenseVector gradient;
  DenseVector hessian;
};

inline EnzymeHessianResult eval_enzyme_full(const DenseVector& x) {
  EnzymeHessianResult out;
  auto gr = eval_enzyme_reverse(x);
  out.value    = gr.value;
  out.gradient = std::move(gr.gradient);
  out.hessian  = eval_enzyme_hessian(x);
  return out;
}

#endif  // AUTODIFF_HAS_ENZYME

}  // namespace autodiff_benchmark
