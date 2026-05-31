#include "gradient_benchmark_common.hpp"

#include <cppad/cg/model/dynamic_lib/linux/linux_dynamiclib.hpp>

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using autodiff_benchmark::DenseVector;
using autodiff_benchmark::ErrorMetrics;
using autodiff_benchmark::GradientResult;

volatile double g_sink = 0.0;

struct MethodSummary {
  std::string name;
  GradientResult result;
  ErrorMetrics fd_error;
  ErrorMetrics pair_error;
  double average_ns = 0.0;
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
  return "autodiff_codegen_model";
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

GradientResult eval_cppadcg_generated_forward(CppAD::cg::GenericModel<double>& model, const DenseVector& x) {
  GradientResult out;
  out.gradient.assign(autodiff_benchmark::kDimension, 0.0);

  DenseVector tx(2 * autodiff_benchmark::kDimension, 0.0);
  for (std::size_t j = 0; j < autodiff_benchmark::kDimension; ++j) {
    tx[2 * j] = x[j];
  }
  out.value = model.ForwardZero(x)[0];

  for (std::size_t j = 0; j < autodiff_benchmark::kDimension; ++j) {
    tx[2 * j + 1] = 1.0;
    const auto dy = model.ForwardOne(tx);
    out.gradient[j] = dy[0];
    tx[2 * j + 1] = 0.0;
  }

  return out;
}

GradientResult eval_cppadcg_generated_reverse(CppAD::cg::GenericModel<double>& model, const DenseVector& x) {
  GradientResult out;
  out.value = model.ForwardZero(x)[0];
  DenseVector ty(1, out.value);
  DenseVector weights(1, 1.0);
  out.gradient = model.ReverseOne(x, ty, weights);
  return out;
}

template <class Fn>
double benchmark_average_ns(const int iterations, Fn&& fn) {
  if (iterations <= 0) {
    throw std::runtime_error("iterations must be positive");
  }

  auto warmup = fn();
  g_sink += warmup.value + warmup.gradient.front();

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    const auto result = fn();
    g_sink += result.value + result.gradient[static_cast<std::size_t>(i) % result.gradient.size()];
  }
  const auto stop = std::chrono::steady_clock::now();

  const auto elapsed = std::chrono::duration<double, std::nano>(stop - start).count();
  return elapsed / static_cast<double>(iterations);
}

void print_summary_header() {
  std::cout << std::left << std::setw(24) << "Method"
            << std::right << std::setw(16) << "max|g-g_fd|"
            << std::setw(16) << "rms|g-g_fd|"
            << std::setw(16) << "max|g-g_ref|"
            << std::setw(16) << "avg ns"
            << '\n';
  std::cout << std::string(88, '-') << '\n';
}

void print_summary_row(const MethodSummary& summary) {
  std::cout << std::left << std::setw(24) << summary.name
            << std::right << std::setw(16) << std::scientific << std::setprecision(3) << summary.fd_error.max_abs
            << std::setw(16) << summary.fd_error.rms
            << std::setw(16) << summary.pair_error.max_abs
            << std::setw(16) << std::fixed << std::setprecision(1) << summary.average_ns
            << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const int iterations = parse_iterations(argc, argv);
    const DenseVector x = autodiff_benchmark::make_input();
    const double reference_value = autodiff_benchmark::passive_objective(x);
    const DenseVector finite_diff = autodiff_benchmark::finite_difference_gradient(x, autodiff_benchmark::kFiniteDiffStep);

    auto cppad_fun = autodiff_benchmark::build_cppad_tape();

    const std::string library_path = generated_library_path();
    if (library_path.empty()) {
      throw std::runtime_error("Generated CppADCodegen library path is empty");
    }

    CppAD::cg::LinuxDynamicLib<double> dynamic_library(library_path);
    std::unique_ptr<CppAD::cg::GenericModel<double>> generated_model = dynamic_library.model(model_name());
    if (!generated_model) {
      throw std::runtime_error("Failed to load generated CppADCodegen model");
    }

    const GradientResult cppad_reverse = autodiff_benchmark::eval_cppad_reverse(cppad_fun, x);
    const GradientResult cppad_forward = autodiff_benchmark::eval_cppad_forward(cppad_fun, x);
    const GradientResult cppadcg_reverse = eval_cppadcg_generated_reverse(*generated_model, x);
    const GradientResult cppadcg_forward = eval_cppadcg_generated_forward(*generated_model, x);
    const GradientResult tinyad_forward = autodiff_benchmark::eval_tinyad_forward(x);

    const auto cppad_forward_pair = autodiff_benchmark::compute_error(cppad_forward.gradient, cppad_reverse.gradient);
    const auto cppadcg_reverse_pair = autodiff_benchmark::compute_error(cppadcg_reverse.gradient, cppad_reverse.gradient);
    const auto cppadcg_forward_pair = autodiff_benchmark::compute_error(cppadcg_forward.gradient, cppad_reverse.gradient);
    const auto tinyad_forward_pair = autodiff_benchmark::compute_error(tinyad_forward.gradient, cppad_reverse.gradient);

    std::vector<MethodSummary> summaries = {
      {
        "TinyAD forward",
        tinyad_forward,
        autodiff_benchmark::compute_error(tinyad_forward.gradient, finite_diff),
        tinyad_forward_pair,
        benchmark_average_ns(iterations, [&] { return autodiff_benchmark::eval_tinyad_forward(x); })
      },
      {
        "CppAD forward",
        cppad_forward,
        autodiff_benchmark::compute_error(cppad_forward.gradient, finite_diff),
        cppad_forward_pair,
        benchmark_average_ns(iterations, [&] { return autodiff_benchmark::eval_cppad_forward(cppad_fun, x); })
      },
      {
        "CppAD reverse",
        cppad_reverse,
        autodiff_benchmark::compute_error(cppad_reverse.gradient, finite_diff),
        ErrorMetrics{0.0, 0.0},
        benchmark_average_ns(iterations, [&] { return autodiff_benchmark::eval_cppad_reverse(cppad_fun, x); })
      },
      {
        "CppADCodegen forward",
        cppadcg_forward,
        autodiff_benchmark::compute_error(cppadcg_forward.gradient, finite_diff),
        cppadcg_forward_pair,
        benchmark_average_ns(iterations, [&] { return eval_cppadcg_generated_forward(*generated_model, x); })
      },
      {
        "CppADCodegen reverse",
        cppadcg_reverse,
        autodiff_benchmark::compute_error(cppadcg_reverse.gradient, finite_diff),
        cppadcg_reverse_pair,
        benchmark_average_ns(iterations, [&] { return eval_cppadcg_generated_reverse(*generated_model, x); })
      }
    };

    std::cout << "Gradient benchmark on a coupled objective with " << autodiff_benchmark::kDimension << " variables\n";
    std::cout << "Finite-difference step : " << std::scientific << autodiff_benchmark::kFiniteDiffStep << '\n';
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
    std::cout << "TinyAD note: only forward mode is available in this benchmark; reverse mode is not exposed by TinyAD.\n";
    std::cout << "CppADCodegen note: timings measure the generated shared library built from cppadcg_benchmark_runtime/.\n";

    bool ok = true;
    std::ostringstream errors;

    for (const auto& summary : summaries) {
      const double value_error = std::abs(summary.result.value - reference_value);
      if (value_error > 1e-11) {
        ok = false;
        errors << summary.name << ": value mismatch " << value_error << '\n';
      }
      if (summary.fd_error.max_abs > autodiff_benchmark::kFiniteDiffTolerance) {
        ok = false;
        errors << summary.name << ": finite-difference mismatch " << summary.fd_error.max_abs << '\n';
      }
    }

    if (cppad_forward_pair.max_abs > autodiff_benchmark::kCrossTolerance) {
      ok = false;
      errors << "CppAD forward vs reverse mismatch " << cppad_forward_pair.max_abs << '\n';
    }
    if (cppadcg_forward_pair.max_abs > autodiff_benchmark::kCrossTolerance) {
      ok = false;
      errors << "CppADCodegen forward vs CppAD reverse mismatch " << cppadcg_forward_pair.max_abs << '\n';
    }
    if (cppadcg_reverse_pair.max_abs > autodiff_benchmark::kCrossTolerance) {
      ok = false;
      errors << "CppADCodegen reverse vs CppAD reverse mismatch " << cppadcg_reverse_pair.max_abs << '\n';
    }
    if (tinyad_forward_pair.max_abs > autodiff_benchmark::kFiniteDiffTolerance) {
      ok = false;
      errors << "TinyAD forward vs CppAD reverse mismatch " << tinyad_forward_pair.max_abs << '\n';
    }

    if (!ok) {
      std::cerr << "\nVerification failed:\n" << errors.str();
      return 1;
    }

    return g_sink < 0.0 ? 2 : 0;
  } catch (const std::exception& err) {
    std::cerr << "Error: " << err.what() << '\n';
    return 1;
  }
}
