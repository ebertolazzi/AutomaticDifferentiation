#include "gradient_benchmark_common.hpp"

#include <cppad/cg/model/dynamic_lib/linux/linux_dynamiclib.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using autodiff_benchmark::DenseVector;
using autodiff_benchmark::ErrorMetrics;

constexpr double kFiniteDiffHessianStep = 1e-5;
constexpr double kFiniteDiffHessianTolerance = 2e-4;

volatile double g_sink_hessian = 0.0;

struct HessianResult {
  double value = 0.0;
  DenseVector hessian;
};

struct HessianMethodSummary {
  std::string name;
  HessianResult result;
  ErrorMetrics ref_error;
  double finite_diff_max = 0.0;
  double average_ns = 0.0;
  double speedup_vs_analytic = 0.0;
};

std::string generated_library_path() {
#ifdef AUTODIFF_CODEGEN_LIBRARY_PATH
  return AUTODIFF_CODEGEN_LIBRARY_PATH;
#else
  return {};
#endif
}

std::string model_name() {
#ifdef AUTODIFF_CODEGEN_MODEL_NAME
  return AUTODIFF_CODEGEN_MODEL_NAME;
#else
  return "autodiff_hessian_model";
#endif
}

int parse_iterations(int argc, char** argv) {
  int iterations = 3;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--iters" && i + 1 < argc) {
      iterations = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [--iters N]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  return iterations;
}

std::size_t flat_index(const std::size_t row, const std::size_t col) {
  return row * autodiff_benchmark::kDimension + col;
}

ErrorMetrics compute_dense_error(const DenseVector& lhs, const DenseVector& rhs) {
  return autodiff_benchmark::compute_error(lhs, rhs);
}

std::vector<std::pair<std::size_t, std::size_t>> hessian_probe_entries() {
  return {
    {0, 0}, {1, 1}, {17, 17}, {64, 64}, {95, 95}, {127, 127},
    {0, 1}, {3, 7}, {11, 19}, {32, 35}, {64, 75}, {100, 111}
  };
}

double finite_difference_hessian_entry(const DenseVector& x, const std::size_t i, const std::size_t j, const double h) {
  DenseVector xp = x;
  DenseVector xm = x;

  if (i == j) {
    xp[i] += h;
    xm[i] -= h;
    const double fp = autodiff_benchmark::passive_objective(xp);
    const double f0 = autodiff_benchmark::passive_objective(x);
    const double fm = autodiff_benchmark::passive_objective(xm);
    return (fp - 2.0 * f0 + fm) / (h * h);
  }

  DenseVector xpp = x;
  DenseVector xpm = x;
  DenseVector xmp = x;
  DenseVector xmm = x;
  xpp[i] += h; xpp[j] += h;
  xpm[i] += h; xpm[j] -= h;
  xmp[i] -= h; xmp[j] += h;
  xmm[i] -= h; xmm[j] -= h;

  const double fpp = autodiff_benchmark::passive_objective(xpp);
  const double fpm = autodiff_benchmark::passive_objective(xpm);
  const double fmp = autodiff_benchmark::passive_objective(xmp);
  const double fmm = autodiff_benchmark::passive_objective(xmm);
  return (fpp - fpm - fmp + fmm) / (4.0 * h * h);
}

double max_probe_error(const DenseVector& x, const DenseVector& hessian) {
  double max_error = 0.0;
  for (const auto& [i, j] : hessian_probe_entries()) {
    const double fd = finite_difference_hessian_entry(x, i, j, kFiniteDiffHessianStep);
    const double ref = hessian[flat_index(i, j)];
    max_error = std::max(max_error, std::abs(fd - ref));
  }
  return max_error;
}

HessianResult eval_cppad_hessian(autodiff_benchmark::CppADTape& fun, const DenseVector& x) {
  HessianResult out;
  out.value = fun.Forward(0, x)[0];
  out.hessian = fun.Hessian(x, 0);
  return out;
}

HessianResult eval_tinyad_hessian(const DenseVector& x) {
  using ADScalar = TinyAD::Scalar<Eigen::Dynamic, double, true>;

  std::vector<ADScalar> ax(autodiff_benchmark::kDimension);
  for (std::size_t i = 0; i < autodiff_benchmark::kDimension; ++i) {
    ax[i] = ADScalar::make_active(x[i], static_cast<int>(i), static_cast<int>(autodiff_benchmark::kDimension));
  }

  const ADScalar y = autodiff_benchmark::coupled_objective<ADScalar>([&](std::size_t i) -> const ADScalar& { return ax[i]; });

  HessianResult out;
  out.value = y.val;
  out.hessian.resize(autodiff_benchmark::kDimension * autodiff_benchmark::kDimension);
  for (std::size_t r = 0; r < autodiff_benchmark::kDimension; ++r) {
    for (std::size_t c = 0; c < autodiff_benchmark::kDimension; ++c) {
      out.hessian[flat_index(r, c)] = y.Hess(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
    }
  }
  return out;
}

HessianResult eval_cppadcg_hessian(CppAD::cg::GenericModel<double>& model, const DenseVector& x) {
  HessianResult out;
  out.value = model.ForwardZero(x)[0];
  out.hessian = model.Hessian(x, 0);
  return out;
}

HessianResult eval_analytic_hessian_result(const DenseVector& x) {
  HessianResult out;
  auto full = autodiff_benchmark::eval_analytic_full(x);
  out.value   = full.value;
  out.hessian = std::move(full.hessian);
  return out;
}

template <class Fn>
double benchmark_average_ns(const int iterations, Fn&& fn) {
  if (iterations <= 0) {
    throw std::runtime_error("iterations must be positive");
  }

  auto warmup = fn();
  g_sink_hessian += warmup.value + warmup.hessian.front();

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    const auto result = fn();
    g_sink_hessian += result.value + result.hessian[static_cast<std::size_t>(i) % result.hessian.size()];
  }
  const auto stop = std::chrono::steady_clock::now();

  return std::chrono::duration<double, std::nano>(stop - start).count() / static_cast<double>(iterations);
}

void print_summary_header() {
  std::cout << std::left << std::setw(24) << "Method"
            << std::right << std::setw(16) << "max|H-H_ref|"
            << std::setw(16) << "rms|H-H_ref|"
            << std::setw(16) << "max|H-H_fd|"
            << std::setw(16) << "avg ns"
            << std::setw(12) << "speedup"
            << '\n';
  std::cout << std::string(100, '-') << '\n';
}

void print_summary_row(const HessianMethodSummary& summary) {
  std::cout << std::left << std::setw(24) << summary.name
            << std::right << std::setw(16) << std::scientific << std::setprecision(3) << summary.ref_error.max_abs
            << std::setw(16) << summary.ref_error.rms
            << std::setw(16) << summary.finite_diff_max
            << std::setw(16) << std::fixed << std::setprecision(2) << summary.average_ns
            << std::setw(12) << std::setprecision(4) << summary.speedup_vs_analytic
            << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int iterations = parse_iterations(argc, argv);
    const DenseVector x = autodiff_benchmark::make_input();
    const double reference_value = autodiff_benchmark::passive_objective(x);

    auto cppad_fun = autodiff_benchmark::build_cppad_tape();

    const std::string library_path = generated_library_path();
    if (library_path.empty()) {
      throw std::runtime_error("Generated CppADCodegen Hessian library path is empty");
    }

    CppAD::cg::LinuxDynamicLib<double> dynamic_library(library_path);
    std::unique_ptr<CppAD::cg::GenericModel<double>> generated_model = dynamic_library.model(model_name());
    if (!generated_model) {
      throw std::runtime_error("Failed to load generated CppADCodegen Hessian model");
    }

    const HessianResult cppad_hessian = eval_cppad_hessian(cppad_fun, x);
    const HessianResult tinyad_hessian = eval_tinyad_hessian(x);
    const HessianResult cppadcg_hessian = eval_cppadcg_hessian(*generated_model, x);
    const HessianResult analytic_hessian = eval_analytic_hessian_result(x);

#ifdef AUTODIFF_HAS_ENZYME
    // Enzyme forward-over-reverse Hessian
    const auto enzyme_full = autodiff_benchmark::eval_enzyme_full(x);
    HessianResult enzyme_hessian;
    enzyme_hessian.value   = enzyme_full.value;
    enzyme_hessian.hessian = enzyme_full.hessian;
#endif

    std::vector<HessianMethodSummary> summaries = {
      {
        "Analytic",
        analytic_hessian,
        compute_dense_error(analytic_hessian.hessian, cppad_hessian.hessian),
        max_probe_error(x, analytic_hessian.hessian),
        benchmark_average_ns(iterations, [&] { return eval_analytic_hessian_result(x); })
      },
      {
        "TinyAD Hessian",
        tinyad_hessian,
        compute_dense_error(tinyad_hessian.hessian, cppad_hessian.hessian),
        max_probe_error(x, tinyad_hessian.hessian),
        benchmark_average_ns(iterations, [&] { return eval_tinyad_hessian(x); })
      },
      {
        "CppAD Hessian",
        cppad_hessian,
        ErrorMetrics{0.0, 0.0},
        max_probe_error(x, cppad_hessian.hessian),
        benchmark_average_ns(iterations, [&] { return eval_cppad_hessian(cppad_fun, x); })
      },
      {
        "CppADCodegen Hessian",
        cppadcg_hessian,
        compute_dense_error(cppadcg_hessian.hessian, cppad_hessian.hessian),
        max_probe_error(x, cppadcg_hessian.hessian),
        benchmark_average_ns(iterations, [&] { return eval_cppadcg_hessian(*generated_model, x); })
      }
#ifdef AUTODIFF_HAS_ENZYME
      ,
      {
        "Enzyme Hessian",
        enzyme_hessian,
        compute_dense_error(enzyme_hessian.hessian, cppad_hessian.hessian),
        max_probe_error(x, enzyme_hessian.hessian),
        benchmark_average_ns(iterations, [&] {
          auto ef = autodiff_benchmark::eval_enzyme_full(x);
          HessianResult r;
          r.value   = ef.value;
          r.hessian = std::move(ef.hessian);
          return r;
        })
      }
#endif
    };

    const double analytic_average_ns = summaries[0].average_ns;
    for (auto& summary : summaries) {
      summary.speedup_vs_analytic = analytic_average_ns / summary.average_ns;
    }

    std::cout << "Hessian benchmark on a coupled objective with " << autodiff_benchmark::kDimension << " variables\n";
    std::cout << "Finite-difference step : " << std::scientific << kFiniteDiffHessianStep << '\n';
    std::cout << "Benchmark iterations   : " << std::fixed << iterations << '\n';
    std::cout << "Reference value        : " << std::scientific << reference_value << '\n';
    std::cout << "Generated library      : " << library_path << "\n\n";

    print_summary_header();
    for (const auto& summary : summaries) {
      print_summary_row(summary);
    }

    std::cout << "\nValue checks:\n";
    for (const auto& summary : summaries) {
      const double value_error = std::abs(summary.result.value - reference_value);
      std::cout << "  " << std::left << std::setw(22) << summary.name
                << " value error = " << std::scientific << value_error << '\n';
    }

    std::cout << "\n";
    std::cout << "Analytic note: hand-derived closed-form Hessian of the benchmark objective.\n";
    std::cout << "TinyAD note: Hessian is computed in forward mode through TinyAD::Scalar with second-order derivatives.\n";
    std::cout << "CppADCodegen note: Hessian timings measure the generated dense Hessian library built from cppadcg_hessian_runtime/.\n";
#ifdef AUTODIFF_HAS_ENZYME
    std::cout << "Enzyme note: Hessian is computed via reverse-over-forward, one reverse sweep per gradient component.\n";
#else
    std::cout << "Enzyme status: disabled at build time; configure tests with -DAUTODIFF_ENABLE_ENZYME=ON to include Enzyme timings.\n";
#endif

    bool ok = true;
    std::ostringstream errors;

    for (const auto& summary : summaries) {
      const double value_error = std::abs(summary.result.value - reference_value);
      if (value_error > 1e-11) {
        ok = false;
        errors << summary.name << ": value mismatch " << value_error << '\n';
      }
      if (summary.finite_diff_max > kFiniteDiffHessianTolerance) {
        ok = false;
        errors << summary.name << ": finite-difference mismatch " << summary.finite_diff_max << '\n';
      }
    }

    if (summaries[1].ref_error.max_abs > autodiff_benchmark::kCrossTolerance) {
      ok = false;
      errors << "TinyAD Hessian vs CppAD mismatch " << summaries[1].ref_error.max_abs << '\n';
    }
    if (summaries[3].ref_error.max_abs > autodiff_benchmark::kCrossTolerance) {
      ok = false;
      errors << "CppADCodegen Hessian vs CppAD mismatch " << summaries[3].ref_error.max_abs << '\n';
    }
    if (summaries[0].ref_error.max_abs > autodiff_benchmark::kCrossTolerance) {
      ok = false;
      errors << "Analytic Hessian vs CppAD mismatch " << summaries[0].ref_error.max_abs << '\n';
    }

#ifdef AUTODIFF_HAS_ENZYME
    // Enzyme is the last entry when enabled (index 4)
    if (summaries.size() > 4 && summaries[4].ref_error.max_abs > autodiff_benchmark::kCrossTolerance) {
      ok = false;
      errors << "Enzyme Hessian vs CppAD mismatch " << summaries[4].ref_error.max_abs << '\n';
    }
#endif

    if (!ok) {
      std::cerr << "\nVerification failed:\n" << errors.str();
      return 1;
    }

    return g_sink_hessian < 0.0 ? 2 : 0;
  } catch (const std::exception& err) {
    std::cerr << "Error: " << err.what() << '\n';
    return 1;
  }
}
