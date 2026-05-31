#pragma once

#include <Eigen/Core>

#include <TinyAD/Scalar.hh>
#include <cppad/cg.hpp>
#include <cppad/cppad.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

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

}  // namespace autodiff_benchmark
