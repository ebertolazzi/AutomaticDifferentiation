#include "dual.hh"
#include "test_sparse_common.hh"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @file test1.cc
 * @brief Verbose numerical tests for the `AD::Dual` class.
 */

namespace {

  //! Convenient alias for double-precision tests.
  using DUAL = AD::Dual<double>;

  //! Nested dual number used to evaluate second derivatives.
  using DUAL2 = AD::Dual<DUAL>;

  //! Dense sample point used by the large sparse test case.
  using SparsePoint = TestSparse::VariableArray<double>;

  //! Dense residual vector used by the large sparse vector map.
  using SparseResidual = TestSparse::ResidualArray<double>;

  //! Dense Jacobian matrix used by the large sparse vector map.
  using SparseJacobian = std::array<std::array<double, TestSparse::kLargeVariableCount>, TestSparse::kLargeResidualCount>;

  //! Dense Hessian matrix used by the large sparse scalar objective.
  using SparseHessian = std::array<std::array<double, TestSparse::kLargeVariableCount>, TestSparse::kLargeVariableCount>;

  //! Dense sample point used by the huge sparse vector-map test case.
  using HugeSparsePoint = TestSparse::HugeVariableArray<double>;

  //! Dense Jacobian matrix used by the huge sparse vector map.
  using HugeSparseJacobian = std::array<std::array<double, TestSparse::kHugeSparseVariableCount>, TestSparse::kHugeSparseResidualCount>;

  //! Dense sample point used by the dense ten-variable test case.
  using DensePoint = TestSparse::DenseVariableArray<double>;

  //! Dense Jacobian matrix used by the dense ten-variable vector map.
  using DenseJacobian = std::array<std::array<double, TestSparse::kDenseVariableCount>, TestSparse::kDenseResidualCount>;

  //! Dense Hessian matrix used by the dense ten-variable scalar objective.
  using DenseHessian = std::array<std::array<double, TestSparse::kDenseVariableCount>, TestSparse::kDenseVariableCount>;

  /**
   * @brief Minimal context used to collect, print, and validate test checks.
   */
  struct TestContext {
    int total  = 0;
    int failed = 0;

    /**
     * @brief Prints a section header for a logical test group.
     *
     * @param title section title.
     */
    void section( std::string const & title ) const {
      std::cout << "\n== " << title << " ==\n";
    }

    /**
     * @brief Checks whether two scalars are close and prints a verbose line.
     *
     * @param label short description of the check.
     * @param actual computed value.
     * @param expected expected value.
     * @param tolerance absolute tolerance.
     */
    void expect_close(
      std::string const & label,
      double              actual,
      double              expected,
      double              tolerance = 1e-12
    ) {
      ++total;
      bool const ok = std::abs(actual - expected) <= tolerance;
      if ( !ok ) ++failed;

      std::ostream & out = ok ? std::cout : std::cerr;
      out
        << "Test " << total
        << ": " << label
        << " | expected " << expected
        << " | actual " << actual
        << " | tolerance " << tolerance
        << " | " << (ok ? "OK" : "FAIL")
        << '\n';
    }

    /**
     * @brief Checks both the primal value and derivative of a dual number.
     *
     * @param label short description of the check.
     * @param actual computed dual result.
     * @param expected_value expected primal value.
     * @param expected_dual expected derivative.
     * @param tolerance absolute tolerance.
     */
    void expect_dual(
      std::string const & label,
      DUAL const        & actual,
      double              expected_value,
      double              expected_dual,
      double              tolerance = 1e-12
    ) {
      expect_close(label + " [value]", actual.value(), expected_value, tolerance);
      expect_close(label + " [dual]",  actual.dual(),  expected_dual,  tolerance);
    }

    /**
     * @brief Finishes the run and throws if at least one check failed.
     */
    void finalize() const {
      std::cout << "\nSummary: executed " << total << " checks";
      if ( failed == 0 ) {
        std::cout << ", all passed.\n";
        return;
      }
      std::cout << ", failures: " << failed << ".\n";
      throw std::runtime_error("dual tests failed");
    }
  };

  /**
   * @brief Builds an independent variable with unit derivative.
   *
   * @param x variable value.
   * @return DUAL corresponding dual variable.
   */
  DUAL variable( double x ) {
    return DUAL::variable(x);
  }

  /**
   * @brief Checks a unary function against expected value and derivative.
   *
   * @tparam DualFunction callable dual overload.
   * @tparam ScalarFunction callable scalar function.
   * @tparam DerivativeFunction callable scalar derivative.
   * @param ctx test context.
   * @param label test name.
   * @param x evaluation point.
   * @param dual_function function evaluated on dual numbers.
   * @param scalar_function scalar reference function.
   * @param derivative expected scalar derivative.
   * @param tolerance absolute tolerance.
   */
  template <typename DualFunction, typename ScalarFunction, typename DerivativeFunction>
  void check_unary(
    TestContext         & ctx,
    std::string const   & label,
    double                x,
    DualFunction const  & dual_function,
    ScalarFunction const& scalar_function,
    DerivativeFunction const & derivative,
    double                tolerance = 1e-12
  ) {
    DUAL const result = dual_function(variable(x));
    ctx.expect_dual(label, result, scalar_function(x), derivative(x), tolerance);
  }

  /**
   * @brief Checks a binary function with sensitivities on both arguments.
   *
   * @tparam DualFunction callable dual overload.
   * @tparam ScalarFunction callable scalar function.
   * @tparam DfdxFunction callable partial derivative with respect to `x`.
   * @tparam DfdyFunction callable partial derivative with respect to `y`.
   * @param ctx test context.
   * @param label test name.
   * @param x first argument value.
   * @param dx derivative carried by the first argument.
   * @param y second argument value.
   * @param dy derivative carried by the second argument.
   * @param dual_function function evaluated on dual numbers.
   * @param scalar_function scalar reference function.
   * @param dfdx scalar partial derivative with respect to `x`.
   * @param dfdy scalar partial derivative with respect to `y`.
   * @param tolerance absolute tolerance.
   */
  template <
    typename DualFunction,
    typename ScalarFunction,
    typename DfdxFunction,
    typename DfdyFunction
  >
  void check_binary(
    TestContext           & ctx,
    std::string const     & label,
    double                  x,
    double                  dx,
    double                  y,
    double                  dy,
    DualFunction const    & dual_function,
    ScalarFunction const  & scalar_function,
    DfdxFunction const    & dfdx,
    DfdyFunction const    & dfdy,
    double                  tolerance = 1e-12
  ) {
    DUAL const lhs(x, dx);
    DUAL const rhs(y, dy);
    DUAL const result = dual_function(lhs, rhs);
    double const expected_dual = dfdx(x, y) * dx + dfdy(x, y) * dy;
    ctx.expect_dual(label, result, scalar_function(x, y), expected_dual, tolerance);
  }

  /**
   * @brief Runs tests for constructors, assignments, and basic operators.
   *
   * @param ctx test context.
   */
  void test_basic_operations( TestContext & ctx ) {
    ctx.section("Basic operations");

    DUAL a;
    ctx.expect_dual("default constructor", a, 0.0, 0.0);

    DUAL b(2.5, -1.5);
    ctx.expect_dual("value constructor", b, 2.5, -1.5);

    DUAL c = DUAL::variable(4.0);
    ctx.expect_dual("variable constructor", c, 4.0, 1.0);

    c.set(-3.0, 7.0);
    ctx.expect_dual("set", c, -3.0, 7.0);

    c = 5.0;
    ctx.expect_dual("scalar assignment", c, 5.0, 0.0);

    DUAL const x(2.0, 3.0);
    DUAL const y(-1.5, 4.0);

    ctx.expect_dual("unary plus", +x, 2.0, 3.0);
    ctx.expect_dual("unary minus", -x, -2.0, -3.0);

    ctx.expect_dual("dual sum", x + y, 0.5, 7.0);
    ctx.expect_dual("scalar left sum", 1.25 + x, 3.25, 3.0);
    ctx.expect_dual("scalar right sum", x + 1.25, 3.25, 3.0);

    ctx.expect_dual("dual difference", x - y, 3.5, -1.0);
    ctx.expect_dual("scalar left difference", 1.25 - x, -0.75, -3.0);
    ctx.expect_dual("scalar right difference", x - 1.25, 0.75, 3.0);

    ctx.expect_dual("dual product", x * y, -3.0, 3.5);
    ctx.expect_dual("scalar left product", -2.0 * x, -4.0, -6.0);
    ctx.expect_dual("scalar right product", x * -2.0, -4.0, -6.0);

    ctx.expect_dual("dual quotient", x / y, -4.0 / 3.0, -50.0 / 9.0, 1e-11);
    ctx.expect_dual("scalar right quotient", x / 2.0, 1.0, 1.5);
    ctx.expect_dual("scalar left quotient", 6.0 / x, 3.0, -4.5);

    DUAL accum(1.0, 2.0);
    accum += DUAL(3.0, -4.0);
    ctx.expect_dual("operator +=", accum, 4.0, -2.0);
    accum += 2.0;
    ctx.expect_dual("operator += scalar", accum, 6.0, -2.0);
    accum -= DUAL(1.0, 3.0);
    ctx.expect_dual("operator -=", accum, 5.0, -5.0);
    accum -= 4.0;
    ctx.expect_dual("operator -= scalar", accum, 1.0, -5.0);
    accum *= DUAL(-2.0, 1.0);
    ctx.expect_dual("operator *=", accum, -2.0, 11.0);
    accum *= -0.5;
    ctx.expect_dual("operator *= scalar", accum, 1.0, -5.5);
    accum /= DUAL(4.0, -2.0);
    ctx.expect_dual("operator /=", accum, 0.25, -1.25);
    accum /= 0.5;
    ctx.expect_dual("operator /= scalar", accum, 0.5, -2.5);

    ctx.expect_close("operator == with scalar", (DUAL(3.0, 0.0) == 3.0) ? 1.0 : 0.0, 1.0);
    ctx.expect_close("operator != with dual", (DUAL(3.0, 1.0) != DUAL(3.0, 0.0)) ? 1.0 : 0.0, 1.0);

    std::ostringstream out;
    out << DUAL(-1.5, 2.25);
    ctx.expect_close("stream output is not empty", out.str().empty() ? 0.0 : 1.0, 1.0);
  }

  /**
   * @brief Runs tests for trigonometric and inverse trigonometric functions.
   *
   * @param ctx test context.
   */
  void test_trigonometric_functions( TestContext & ctx ) {
    ctx.section("Trigonometric functions");

    check_unary(ctx, "sin", 0.3, AD::sin<double>, []( double x ) { return std::sin(x); }, []( double x ) {
      return std::cos(x);
    });
    check_unary(ctx, "cos", 0.3, AD::cos<double>, []( double x ) { return std::cos(x); }, []( double x ) {
      return -std::sin(x);
    });
    check_unary(ctx, "tan", 0.3, AD::tan<double>, []( double x ) { return std::tan(x); }, []( double x ) {
      double const c = std::cos(x);
      return 1.0 / (c * c);
    });
    check_unary(ctx, "asin", 0.2, AD::asin<double>, []( double x ) { return std::asin(x); }, []( double x ) {
      return 1.0 / std::sqrt(1.0 - x * x);
    });
    check_unary(ctx, "acos", 0.2, AD::acos<double>, []( double x ) { return std::acos(x); }, []( double x ) {
      return -1.0 / std::sqrt(1.0 - x * x);
    });
    check_unary(ctx, "atan", 0.7, AD::atan<double>, []( double x ) { return std::atan(x); }, []( double x ) {
      return 1.0 / (1.0 + x * x);
    });

    check_binary(
      ctx, "atan2", 0.7, 1.5, 1.2, -0.5,
      []( DUAL const & y, DUAL const & x ) { return AD::atan2(y, x); },
      []( double y, double x ) { return std::atan2(y, x); },
      []( double y, double x ) { return x / (x * x + y * y); },
      []( double y, double x ) { return -y / (x * x + y * y); }
    );
  }

  /**
   * @brief Runs tests for hyperbolic and inverse hyperbolic functions.
   *
   * @param ctx test context.
   */
  void test_hyperbolic_functions( TestContext & ctx ) {
    ctx.section("Hyperbolic functions");

    check_unary(ctx, "sinh", 0.4, AD::sinh<double>, []( double x ) { return std::sinh(x); }, []( double x ) {
      return std::cosh(x);
    });
    check_unary(ctx, "cosh", 0.4, AD::cosh<double>, []( double x ) { return std::cosh(x); }, []( double x ) {
      return std::sinh(x);
    });
    check_unary(ctx, "tanh", 0.4, AD::tanh<double>, []( double x ) { return std::tanh(x); }, []( double x ) {
      double const c = std::cosh(x);
      return 1.0 / (c * c);
    });
    check_unary(ctx, "asinh", -0.6, AD::asinh<double>, []( double x ) { return std::asinh(x); }, []( double x ) {
      return 1.0 / std::sqrt(x * x + 1.0);
    });
    check_unary(ctx, "acosh", 1.7, AD::acosh<double>, []( double x ) { return std::acosh(x); }, []( double x ) {
      return 1.0 / (std::sqrt(x - 1.0) * std::sqrt(x + 1.0));
    });
    check_unary(ctx, "atanh", 0.25, AD::atanh<double>, []( double x ) { return std::atanh(x); }, []( double x ) {
      return 1.0 / (1.0 - x * x);
    });
  }

  /**
   * @brief Runs tests for exponential and logarithmic functions.
   *
   * @param ctx test context.
   */
  void test_exponential_and_logarithmic_functions( TestContext & ctx ) {
    ctx.section("Exponential and logarithmic functions");

    check_unary(ctx, "exp", 0.45, AD::exp<double>, []( double x ) { return std::exp(x); }, []( double x ) {
      return std::exp(x);
    });
    check_unary(ctx, "exp2", 0.45, AD::exp2<double>, []( double x ) { return std::exp2(x); }, []( double x ) {
      return std::exp2(x) * std::log(2.0);
    });
    check_unary(ctx, "expm1", 0.45, AD::expm1<double>, []( double x ) { return std::expm1(x); }, []( double x ) {
      return std::exp(x);
    });
    check_unary(ctx, "log", 1.8, AD::log<double>, []( double x ) { return std::log(x); }, []( double x ) {
      return 1.0 / x;
    });
    check_unary(ctx, "log2", 1.8, AD::log2<double>, []( double x ) { return std::log2(x); }, []( double x ) {
      return 1.0 / (x * std::log(2.0));
    });
    check_unary(ctx, "log10", 1.8, AD::log10<double>, []( double x ) { return std::log10(x); }, []( double x ) {
      return 1.0 / (x * std::log(10.0));
    });
    check_unary(ctx, "log1p", 0.35, AD::log1p<double>, []( double x ) { return std::log1p(x); }, []( double x ) {
      return 1.0 / (1.0 + x);
    });
  }

  /**
   * @brief Runs tests for powers, roots, and related functions.
   *
   * @param ctx test context.
   */
  void test_power_and_root_functions( TestContext & ctx ) {
    ctx.section("Power and root functions");

    check_unary(ctx, "sqrt", 2.25, AD::sqrt<double>, []( double x ) { return std::sqrt(x); }, []( double x ) {
      return 1.0 / (2.0 * std::sqrt(x));
    });
    check_unary(ctx, "cbrt", 1.7, AD::cbrt<double>, []( double x ) { return std::cbrt(x); }, []( double x ) {
      double const r = std::cbrt(x);
      return 1.0 / (3.0 * r * r);
    });

    check_binary(
      ctx, "pow dual-dual", 1.8, 0.7, 1.2, -0.4,
      []( DUAL const & x, DUAL const & y ) { return AD::pow(x, y); },
      []( double x, double y ) { return std::pow(x, y); },
      []( double x, double y ) { return y * std::pow(x, y - 1.0); },
      []( double x, double y ) { return std::pow(x, y) * std::log(x); },
      2e-12
    );

    DUAL const a(1.8, 0.7);
    DUAL const b = AD::pow(a, 2.5);
    ctx.expect_dual(
      "pow dual-scalar",
      b,
      std::pow(1.8, 2.5),
      2.5 * std::pow(1.8, 1.5) * 0.7,
      2e-12
    );

    DUAL const c = AD::pow(2.5, DUAL(0.4, -1.2));
    ctx.expect_dual(
      "pow scalar-dual",
      c,
      std::pow(2.5, 0.4),
      std::pow(2.5, 0.4) * std::log(2.5) * -1.2,
      2e-12
    );

    DUAL const h = AD::hypot(DUAL(3.0, 1.5), DUAL(4.0, -2.0));
    ctx.expect_dual("hypot", h, 5.0, (3.0 * 1.5 + 4.0 * -2.0) / 5.0);

    DUAL const e = AD::fma(DUAL(2.0, 3.0), DUAL(-1.0, 4.0), DUAL(5.0, -2.0));
    ctx.expect_dual("fma", e, 3.0, 3.0);
  }

  /**
   * @brief Runs tests for auxiliary and special functions.
   *
   * @param ctx test context.
   */
  void test_auxiliary_functions( TestContext & ctx ) {
    ctx.section("Auxiliary and special functions");

    check_unary(ctx, "abs negative", -2.4, AD::abs<double>, []( double x ) {
      return std::abs(x);
    }, []( double ) {
      return -1.0;
    });

    check_unary(ctx, "floor", 3.8, AD::floor<double>, []( double x ) {
      return std::floor(x);
    }, []( double ) {
      return 0.0;
    });
    check_unary(ctx, "ceil", 3.2, AD::ceil<double>, []( double x ) {
      return std::ceil(x);
    }, []( double ) {
      return 0.0;
    });
    check_unary(ctx, "trunc", -3.8, AD::trunc<double>, []( double x ) {
      return std::trunc(x);
    }, []( double ) {
      return 0.0;
    });
    check_unary(ctx, "round", 3.2, AD::round<double>, []( double x ) {
      return std::round(x);
    }, []( double ) {
      return 0.0;
    });

    check_unary(ctx, "erf", 0.4, AD::erf<double>, []( double x ) {
      return std::erf(x);
    }, []( double x ) {
      return 2.0 / std::sqrt(std::acos(-1.0)) * std::exp(-x * x);
    }, 2e-12);

    check_unary(ctx, "erfc", 0.4, AD::erfc<double>, []( double x ) {
      return std::erfc(x);
    }, []( double x ) {
      return -2.0 / std::sqrt(std::acos(-1.0)) * std::exp(-x * x);
    }, 2e-12);

    DUAL const scaled = AD::ldexp(DUAL(0.75, -2.0), 3);
    ctx.expect_dual("ldexp", scaled, std::ldexp(0.75, 3), std::ldexp(-2.0, 3));

    DUAL const scalbn_value = AD::scalbn(DUAL(0.75, -2.0), 2);
    ctx.expect_dual("scalbn", scalbn_value, std::scalbn(0.75, 2), std::scalbn(-2.0, 2));

    DUAL const scalbln_value = AD::scalbln(DUAL(0.75, -2.0), 2L);
    ctx.expect_dual(
      "scalbln",
      scalbln_value,
      std::scalbln(0.75, 2L),
      std::scalbln(-2.0, 2L)
    );
  }

  /**
   * @brief Runs a non-trivial composed-expression test.
   *
   * @param ctx test context.
   */
  void test_composed_expression( TestContext & ctx ) {
    ctx.section("Composed expression");

    auto const function = []( DUAL const & x ) {
      using AD::cos;
      using AD::exp;
      using AD::log1p;
      using AD::sin;
      return (x * x + 2.0 * x + 1.0) * exp(sin(x)) / log1p(x) + cos(x);
    };

    double const x = 0.4;
    DUAL const result = function(variable(x));

    double const value =
      ((x * x + 2.0 * x + 1.0) * std::exp(std::sin(x)) / std::log1p(x)) +
      std::cos(x);

    double const deriv =
      ((2.0 * x + 2.0) * std::exp(std::sin(x)) +
       (x * x + 2.0 * x + 1.0) * std::exp(std::sin(x)) * std::cos(x))
      / std::log1p(x)
      - (x * x + 2.0 * x + 1.0) * std::exp(std::sin(x))
      / (std::log1p(x) * std::log1p(x) * (1.0 + x))
      - std::sin(x);

    ctx.expect_dual("composed expression", result, value, deriv, 5e-12);
  }

  /**
   * @brief Evaluates the large sparse scalar objective gradient with repeated dual sweeps.
   *
   * @param x evaluation point.
   * @return SparsePoint dense gradient vector.
   */
  SparsePoint evaluate_sparse_gradient_dual( SparsePoint const & x ) {
    SparsePoint gradient{};
    for ( int col = 0; col < TestSparse::kLargeVariableCount; ++col ) {
      TestSparse::VariableArray<DUAL> active{};
      for ( int i = 0; i < TestSparse::kLargeVariableCount; ++i ) {
        active[i] = DUAL(x[i], i == col ? 1.0 : 0.0);
      }
      gradient[col] = TestSparse::sparse_scalar_objective(active).dual();
    }
    return gradient;
  }

  /**
   * @brief Evaluates the large sparse vector-map Jacobian with repeated dual sweeps.
   *
   * @param x evaluation point.
   * @return SparseJacobian dense Jacobian matrix.
   */
  SparseJacobian evaluate_sparse_jacobian_dual( SparsePoint const & x ) {
    SparseJacobian jacobian{};
    for ( int col = 0; col < TestSparse::kLargeVariableCount; ++col ) {
      TestSparse::VariableArray<DUAL> active{};
      for ( int i = 0; i < TestSparse::kLargeVariableCount; ++i ) {
        active[i] = DUAL(x[i], i == col ? 1.0 : 0.0);
      }
      auto const residuals = TestSparse::sparse_vector_map(active);
      for ( int row = 0; row < TestSparse::kLargeResidualCount; ++row ) {
        jacobian[row][col] = residuals[row].dual();
      }
    }
    return jacobian;
  }

  /**
   * @brief Evaluates the large sparse scalar objective Hessian with nested dual numbers.
   *
   * @param x evaluation point.
   * @return SparseHessian dense Hessian matrix.
   */
  SparseHessian evaluate_sparse_hessian_dual( SparsePoint const & x ) {
    SparseHessian hessian{};
    for ( int row = 0; row < TestSparse::kLargeVariableCount; ++row ) {
      for ( int col = 0; col < TestSparse::kLargeVariableCount; ++col ) {
        TestSparse::VariableArray<DUAL2> active{};
        for ( int i = 0; i < TestSparse::kLargeVariableCount; ++i ) {
          active[i] = DUAL2(
            DUAL(x[i], i == row ? 1.0 : 0.0),
            DUAL(i == col ? 1.0 : 0.0, 0.0)
          );
        }
        hessian[row][col] = TestSparse::sparse_scalar_objective(active).dual().dual();
      }
    }
    return hessian;
  }

  /**
   * @brief Evaluates the scalar sparse objective without derivatives.
   *
   * @param x evaluation point.
   * @return double scalar objective value.
   */
  double sparse_scalar_objective_passive( SparsePoint const & x ) {
    return TestSparse::sparse_scalar_objective(x);
  }

  /**
   * @brief Evaluates one residual of the sparse vector map without derivatives.
   *
   * @param row residual index.
   * @param x evaluation point.
   * @return double residual value.
   */
  double sparse_residual_passive( int row, SparsePoint const & x ) {
    return TestSparse::sparse_residual_row(row, x);
  }

  /**
   * @brief Computes a centered finite-difference gradient.
   *
   * @param x evaluation point.
   * @param h finite-difference step.
   * @return SparsePoint finite-difference gradient.
   */
  SparsePoint finite_difference_sparse_gradient( SparsePoint const & x, double h = 1e-6 ) {
    SparsePoint gradient{};
    for ( int col = 0; col < TestSparse::kLargeVariableCount; ++col ) {
      SparsePoint xp = x;
      SparsePoint xm = x;
      xp[col] += h;
      xm[col] -= h;
      gradient[col] = (sparse_scalar_objective_passive(xp) - sparse_scalar_objective_passive(xm)) / (2.0 * h);
    }
    return gradient;
  }

  /**
   * @brief Computes a centered finite-difference Jacobian.
   *
   * @param x evaluation point.
   * @param h finite-difference step.
   * @return SparseJacobian finite-difference Jacobian.
   */
  SparseJacobian finite_difference_sparse_jacobian( SparsePoint const & x, double h = 1e-6 ) {
    SparseJacobian jacobian{};
    for ( int col = 0; col < TestSparse::kLargeVariableCount; ++col ) {
      SparsePoint xp = x;
      SparsePoint xm = x;
      xp[col] += h;
      xm[col] -= h;
      for ( int row = 0; row < TestSparse::kLargeResidualCount; ++row ) {
        jacobian[row][col] =
          (sparse_residual_passive(row, xp) - sparse_residual_passive(row, xm)) / (2.0 * h);
      }
    }
    return jacobian;
  }

  /**
   * @brief Computes a centered finite-difference Hessian.
   *
   * @param x evaluation point.
   * @param h finite-difference step.
   * @return SparseHessian finite-difference Hessian.
   */
  SparseHessian finite_difference_sparse_hessian( SparsePoint const & x, double h = 1e-5 ) {
    SparseHessian hessian{};
    for ( int row = 0; row < TestSparse::kLargeVariableCount; ++row ) {
      for ( int col = 0; col < TestSparse::kLargeVariableCount; ++col ) {
        SparsePoint xpp = x;
        SparsePoint xpm = x;
        SparsePoint xmp = x;
        SparsePoint xmm = x;
        xpp[row] += h; xpp[col] += h;
        xpm[row] += h; xpm[col] -= h;
        xmp[row] -= h; xmp[col] += h;
        xmm[row] -= h; xmm[col] -= h;
        hessian[row][col] =
          (sparse_scalar_objective_passive(xpp) - sparse_scalar_objective_passive(xpm) -
           sparse_scalar_objective_passive(xmp) + sparse_scalar_objective_passive(xmm)) /
          (4.0 * h * h);
      }
    }
    return hessian;
  }

  /**
   * @brief Counts structural nonzeros in a dense Jacobian matrix.
   *
   * @param jacobian dense Jacobian matrix.
   * @param tolerance structural zero tolerance.
   * @return int nonzero count.
   */
  int count_jacobian_nonzeros( SparseJacobian const & jacobian, double tolerance = 1e-12 ) {
    int count = 0;
    for ( auto const & row : jacobian ) {
      for ( double const value : row ) {
        if ( std::abs(value) > tolerance ) ++count;
      }
    }
    return count;
  }

  /**
   * @brief Counts structural nonzeros in a dense Hessian matrix.
   *
   * @param hessian dense Hessian matrix.
   * @param tolerance structural zero tolerance.
   * @return int nonzero count.
   */
  int count_hessian_nonzeros( SparseHessian const & hessian, double tolerance = 1e-12 ) {
    int count = 0;
    for ( auto const & row : hessian ) {
      for ( double const value : row ) {
        if ( std::abs(value) > tolerance ) ++count;
      }
    }
    return count;
  }

  /**
   * @brief Computes the maximum absolute difference between two dense gradients.
   *
   * @param lhs first gradient.
   * @param rhs second gradient.
   * @return double maximum absolute difference.
   */
  double max_gradient_difference( SparsePoint const & lhs, SparsePoint const & rhs ) {
    double error = 0.0;
    for ( int i = 0; i < TestSparse::kLargeVariableCount; ++i ) {
      error = std::max(error, std::abs(lhs[i] - rhs[i]));
    }
    return error;
  }

  /**
   * @brief Computes the maximum absolute difference between two dense Jacobians.
   *
   * @param lhs first Jacobian.
   * @param rhs second Jacobian.
   * @return double maximum absolute difference.
   */
  double max_jacobian_difference( SparseJacobian const & lhs, SparseJacobian const & rhs ) {
    double error = 0.0;
    for ( int row = 0; row < TestSparse::kLargeResidualCount; ++row ) {
      for ( int col = 0; col < TestSparse::kLargeVariableCount; ++col ) {
        error = std::max(error, std::abs(lhs[row][col] - rhs[row][col]));
      }
    }
    return error;
  }

  /**
   * @brief Computes the maximum absolute difference between two dense Hessians.
   *
   * @param lhs first Hessian.
   * @param rhs second Hessian.
   * @return double maximum absolute difference.
   */
  double max_hessian_difference( SparseHessian const & lhs, SparseHessian const & rhs ) {
    double error = 0.0;
    for ( int row = 0; row < TestSparse::kLargeVariableCount; ++row ) {
      for ( int col = 0; col < TestSparse::kLargeVariableCount; ++col ) {
        error = std::max(error, std::abs(lhs[row][col] - rhs[row][col]));
      }
    }
    return error;
  }

  /**
   * @brief Runs a large sparse gradient, Jacobian, and Hessian test.
   *
   * @param ctx test context.
   */
  void test_large_sparse_derivatives( TestContext & ctx ) {
    ctx.section("Large sparse derivatives");

    SparsePoint const point = TestSparse::build_sparse_sample_point(7, 0.11);
    SparsePoint const gradient_dual = evaluate_sparse_gradient_dual(point);
    SparsePoint const gradient_fd   = finite_difference_sparse_gradient(point);
    SparseJacobian const jacobian_dual = evaluate_sparse_jacobian_dual(point);
    SparseJacobian const jacobian_fd   = finite_difference_sparse_jacobian(point);
    SparseHessian const hessian_dual = evaluate_sparse_hessian_dual(point);
    SparseHessian const hessian_fd   = finite_difference_sparse_hessian(point);

    ctx.expect_close(
      "large sparse gradient max error",
      max_gradient_difference(gradient_dual, gradient_fd),
      0.0,
      2e-6
    );
    ctx.expect_close(
      "large sparse Jacobian max error",
      max_jacobian_difference(jacobian_dual, jacobian_fd),
      0.0,
      3e-6
    );
    ctx.expect_close(
      "large sparse Hessian max error",
      max_hessian_difference(hessian_dual, hessian_fd),
      0.0,
      2e-4
    );
    ctx.expect_close(
      "large sparse Jacobian structural nonzeros",
      double(count_jacobian_nonzeros(jacobian_dual)),
      double(TestSparse::expected_sparse_jacobian_nonzeros()),
      0.0
    );
    ctx.expect_close(
      "large sparse Hessian structural nonzeros",
      double(count_hessian_nonzeros(hessian_dual)),
      double(TestSparse::expected_sparse_hessian_nonzeros()),
      0.0
    );
    ctx.expect_close("large sparse Jacobian zero away from stencil", jacobian_dual[0][5], 0.0, 1e-12);
    ctx.expect_close("large sparse Hessian zero away from band", hessian_dual[0][7], 0.0, 1e-12);
  }

  /**
   * @brief Evaluates the dense ten-variable gradient with repeated dual sweeps.
   *
   * @param x evaluation point.
   * @return DensePoint dense gradient vector.
   */
  DensePoint evaluate_dense_gradient_dual( DensePoint const & x ) {
    DensePoint gradient{};
    for ( int col = 0; col < TestSparse::kDenseVariableCount; ++col ) {
      TestSparse::DenseVariableArray<DUAL> active{};
      for ( int i = 0; i < TestSparse::kDenseVariableCount; ++i ) {
        active[i] = DUAL(x[i], i == col ? 1.0 : 0.0);
      }
      gradient[col] = TestSparse::dense_scalar_objective(active).dual();
    }
    return gradient;
  }

  /**
   * @brief Evaluates the dense ten-variable Jacobian with repeated dual sweeps.
   *
   * @param x evaluation point.
   * @return DenseJacobian dense Jacobian matrix.
   */
  DenseJacobian evaluate_dense_jacobian_dual( DensePoint const & x ) {
    DenseJacobian jacobian{};
    for ( int col = 0; col < TestSparse::kDenseVariableCount; ++col ) {
      TestSparse::DenseVariableArray<DUAL> active{};
      for ( int i = 0; i < TestSparse::kDenseVariableCount; ++i ) {
        active[i] = DUAL(x[i], i == col ? 1.0 : 0.0);
      }
      auto const residuals = TestSparse::dense_vector_map(active);
      for ( int row = 0; row < TestSparse::kDenseResidualCount; ++row ) {
        jacobian[row][col] = residuals[row].dual();
      }
    }
    return jacobian;
  }

  /**
   * @brief Evaluates the dense ten-variable Hessian with nested dual numbers.
   *
   * @param x evaluation point.
   * @return DenseHessian dense Hessian matrix.
   */
  DenseHessian evaluate_dense_hessian_dual( DensePoint const & x ) {
    DenseHessian hessian{};
    for ( int row = 0; row < TestSparse::kDenseVariableCount; ++row ) {
      for ( int col = 0; col < TestSparse::kDenseVariableCount; ++col ) {
        TestSparse::DenseVariableArray<DUAL2> active{};
        for ( int i = 0; i < TestSparse::kDenseVariableCount; ++i ) {
          active[i] = DUAL2(
            DUAL(x[i], i == row ? 1.0 : 0.0),
            DUAL(i == col ? 1.0 : 0.0, 0.0)
          );
        }
        hessian[row][col] = TestSparse::dense_scalar_objective(active).dual().dual();
      }
    }
    return hessian;
  }

  /**
   * @brief Evaluates the huge sparse Jacobian with repeated dual sweeps.
   *
   * @param x evaluation point.
   * @return HugeSparseJacobian dense Jacobian matrix.
   */
  HugeSparseJacobian evaluate_huge_sparse_jacobian_dual( HugeSparsePoint const & x ) {
    HugeSparseJacobian jacobian{};
    for ( int col = 0; col < TestSparse::kHugeSparseVariableCount; ++col ) {
      TestSparse::HugeVariableArray<DUAL> active{};
      for ( int i = 0; i < TestSparse::kHugeSparseVariableCount; ++i ) {
        active[i] = DUAL(x[i], i == col ? 1.0 : 0.0);
      }
      auto const residuals = TestSparse::huge_sparse_vector_map(active);
      for ( int row = 0; row < TestSparse::kHugeSparseResidualCount; ++row ) {
        jacobian[row][col] = residuals[row].dual();
      }
    }
    return jacobian;
  }

  /**
   * @brief Evaluates the dense scalar objective without derivatives.
   *
   * @param x evaluation point.
   * @return double objective value.
   */
  double dense_scalar_objective_passive( DensePoint const & x ) {
    return TestSparse::dense_scalar_objective(x);
  }

  /**
   * @brief Evaluates one dense residual without derivatives.
   *
   * @param row residual index.
   * @param x evaluation point.
   * @return double residual value.
   */
  double dense_residual_passive( int row, DensePoint const & x ) {
    return TestSparse::dense_residual_row(row, x);
  }

  /**
   * @brief Evaluates one huge sparse residual without derivatives.
   *
   * @param row residual index.
   * @param x evaluation point.
   * @return double residual value.
   */
  double huge_sparse_residual_passive( int row, HugeSparsePoint const & x ) {
    return TestSparse::huge_sparse_residual_row(row, x);
  }

  /**
   * @brief Approximates the dense ten-variable gradient with centered differences.
   *
   * @param x evaluation point.
   * @param h finite-difference step.
   * @return DensePoint finite-difference gradient.
   */
  DensePoint finite_difference_dense_gradient( DensePoint const & x, double h = 1e-6 ) {
    DensePoint gradient{};
    for ( int col = 0; col < TestSparse::kDenseVariableCount; ++col ) {
      DensePoint xp = x;
      DensePoint xm = x;
      xp[col] += h;
      xm[col] -= h;
      gradient[col] =
        (dense_scalar_objective_passive(xp) - dense_scalar_objective_passive(xm)) / (2.0 * h);
    }
    return gradient;
  }

  /**
   * @brief Approximates the dense ten-variable Jacobian with centered differences.
   *
   * @param x evaluation point.
   * @param h finite-difference step.
   * @return DenseJacobian finite-difference Jacobian.
   */
  DenseJacobian finite_difference_dense_jacobian( DensePoint const & x, double h = 1e-6 ) {
    DenseJacobian jacobian{};
    for ( int col = 0; col < TestSparse::kDenseVariableCount; ++col ) {
      DensePoint xp = x;
      DensePoint xm = x;
      xp[col] += h;
      xm[col] -= h;
      for ( int row = 0; row < TestSparse::kDenseResidualCount; ++row ) {
        jacobian[row][col] =
          (dense_residual_passive(row, xp) - dense_residual_passive(row, xm)) / (2.0 * h);
      }
    }
    return jacobian;
  }

  /**
   * @brief Approximates the dense ten-variable Hessian with second differences.
   *
   * @param x evaluation point.
   * @param h finite-difference step.
   * @return DenseHessian finite-difference Hessian.
   */
  DenseHessian finite_difference_dense_hessian( DensePoint const & x, double h = 1e-5 ) {
    DenseHessian hessian{};
    for ( int row = 0; row < TestSparse::kDenseVariableCount; ++row ) {
      for ( int col = 0; col < TestSparse::kDenseVariableCount; ++col ) {
        DensePoint xpp = x;
        DensePoint xpm = x;
        DensePoint xmp = x;
        DensePoint xmm = x;
        xpp[row] += h; xpp[col] += h;
        xpm[row] += h; xpm[col] -= h;
        xmp[row] -= h; xmp[col] += h;
        xmm[row] -= h; xmm[col] -= h;
        hessian[row][col] =
          (
            dense_scalar_objective_passive(xpp) -
            dense_scalar_objective_passive(xpm) -
            dense_scalar_objective_passive(xmp) +
            dense_scalar_objective_passive(xmm)
          ) / (4.0 * h * h);
      }
    }
    return hessian;
  }

  /**
   * @brief Approximates the huge sparse Jacobian with centered differences.
   *
   * @param x evaluation point.
   * @param h finite-difference step.
   * @return HugeSparseJacobian finite-difference Jacobian.
   */
  HugeSparseJacobian finite_difference_huge_sparse_jacobian( HugeSparsePoint const & x, double h = 1e-6 ) {
    HugeSparseJacobian jacobian{};
    for ( int col = 0; col < TestSparse::kHugeSparseVariableCount; ++col ) {
      HugeSparsePoint xp = x;
      HugeSparsePoint xm = x;
      xp[col] += h;
      xm[col] -= h;
      for ( int row = 0; row < TestSparse::kHugeSparseResidualCount; ++row ) {
        jacobian[row][col] =
          (huge_sparse_residual_passive(row, xp) - huge_sparse_residual_passive(row, xm)) / (2.0 * h);
      }
    }
    return jacobian;
  }

  /**
   * @brief Counts structural nonzeros in the dense Jacobian matrix.
   *
   * @param jacobian matrix under inspection.
   * @param tolerance structural zero tolerance.
   * @return int nonzero count.
   */
  int count_dense_jacobian_nonzeros( DenseJacobian const & jacobian, double tolerance = 1e-12 ) {
    int count = 0;
    for ( auto const & row : jacobian ) {
      for ( double value : row ) if ( std::abs(value) > tolerance ) ++count;
    }
    return count;
  }

  /**
   * @brief Counts structural nonzeros in the dense Hessian matrix.
   *
   * @param hessian matrix under inspection.
   * @param tolerance structural zero tolerance.
   * @return int nonzero count.
   */
  int count_dense_hessian_nonzeros( DenseHessian const & hessian, double tolerance = 1e-12 ) {
    int count = 0;
    for ( auto const & row : hessian ) {
      for ( double value : row ) if ( std::abs(value) > tolerance ) ++count;
    }
    return count;
  }

  /**
   * @brief Counts structural nonzeros in the huge sparse Jacobian matrix.
   *
   * @param jacobian matrix under inspection.
   * @param tolerance structural zero tolerance.
   * @return int nonzero count.
   */
  int count_huge_sparse_jacobian_nonzeros( HugeSparseJacobian const & jacobian, double tolerance = 1e-12 ) {
    int count = 0;
    for ( auto const & row : jacobian ) {
      for ( double value : row ) if ( std::abs(value) > tolerance ) ++count;
    }
    return count;
  }

  /**
   * @brief Computes the maximum absolute difference between two dense gradients.
   *
   * @param lhs first gradient.
   * @param rhs second gradient.
   * @return double maximum absolute difference.
   */
  double max_dense_gradient_difference( DensePoint const & lhs, DensePoint const & rhs ) {
    double error = 0.0;
    for ( int i = 0; i < TestSparse::kDenseVariableCount; ++i ) {
      error = std::max(error, std::abs(lhs[i] - rhs[i]));
    }
    return error;
  }

  /**
   * @brief Computes the maximum absolute difference between two dense Jacobians.
   *
   * @param lhs first Jacobian.
   * @param rhs second Jacobian.
   * @return double maximum absolute difference.
   */
  double max_dense_jacobian_difference( DenseJacobian const & lhs, DenseJacobian const & rhs ) {
    double error = 0.0;
    for ( int row = 0; row < TestSparse::kDenseResidualCount; ++row ) {
      for ( int col = 0; col < TestSparse::kDenseVariableCount; ++col ) {
        error = std::max(error, std::abs(lhs[row][col] - rhs[row][col]));
      }
    }
    return error;
  }

  /**
   * @brief Computes the maximum absolute difference between two dense Hessians.
   *
   * @param lhs first Hessian.
   * @param rhs second Hessian.
   * @return double maximum absolute difference.
   */
  double max_dense_hessian_difference( DenseHessian const & lhs, DenseHessian const & rhs ) {
    double error = 0.0;
    for ( int row = 0; row < TestSparse::kDenseVariableCount; ++row ) {
      for ( int col = 0; col < TestSparse::kDenseVariableCount; ++col ) {
        error = std::max(error, std::abs(lhs[row][col] - rhs[row][col]));
      }
    }
    return error;
  }

  /**
   * @brief Computes the maximum absolute difference between two huge sparse Jacobians.
   *
   * @param lhs first Jacobian.
   * @param rhs second Jacobian.
   * @return double maximum absolute difference.
   */
  double max_huge_sparse_jacobian_difference( HugeSparseJacobian const & lhs, HugeSparseJacobian const & rhs ) {
    double error = 0.0;
    for ( int row = 0; row < TestSparse::kHugeSparseResidualCount; ++row ) {
      for ( int col = 0; col < TestSparse::kHugeSparseVariableCount; ++col ) {
        error = std::max(error, std::abs(lhs[row][col] - rhs[row][col]));
      }
    }
    return error;
  }

  /**
   * @brief Runs a dense ten-variable gradient, Jacobian, and Hessian test.
   *
   * @param ctx test context.
   */
  void test_dense10_derivatives( TestContext & ctx ) {
    ctx.section("Dense 10-variable derivatives");

    DensePoint const point = TestSparse::build_dense_sample_point(4, 0.13);
    DensePoint const gradient_dual = evaluate_dense_gradient_dual(point);
    DensePoint const gradient_fd   = finite_difference_dense_gradient(point);
    DenseJacobian const jacobian_dual = evaluate_dense_jacobian_dual(point);
    DenseJacobian const jacobian_fd   = finite_difference_dense_jacobian(point);
    DenseHessian const hessian_dual = evaluate_dense_hessian_dual(point);
    DenseHessian const hessian_fd   = finite_difference_dense_hessian(point);

    ctx.expect_close("dense10 gradient max error", max_dense_gradient_difference(gradient_dual, gradient_fd), 0.0, 3e-6);
    ctx.expect_close("dense10 Jacobian max error", max_dense_jacobian_difference(jacobian_dual, jacobian_fd), 0.0, 4e-6);
    ctx.expect_close("dense10 Hessian max error", max_dense_hessian_difference(hessian_dual, hessian_fd), 0.0, 3e-4);
    ctx.expect_close(
      "dense10 Jacobian structural nonzeros",
      double(count_dense_jacobian_nonzeros(jacobian_dual)),
      double(TestSparse::expected_dense_jacobian_nonzeros()),
      0.0
    );
    ctx.expect_close(
      "dense10 Hessian structural nonzeros",
      double(count_dense_hessian_nonzeros(hessian_dual)),
      double(TestSparse::expected_dense_hessian_nonzeros()),
      0.0
    );
  }

  /**
   * @brief Runs a huge sparse vector-map Jacobian test.
   *
   * @param ctx test context.
   */
  void test_huge_sparse_vector_map( TestContext & ctx ) {
    ctx.section("Huge sparse vector-map Jacobian");

    HugeSparsePoint const point = TestSparse::build_huge_sparse_sample_point(5, 0.10);
    HugeSparseJacobian const jacobian_dual = evaluate_huge_sparse_jacobian_dual(point);
    HugeSparseJacobian const jacobian_fd   = finite_difference_huge_sparse_jacobian(point);

    ctx.expect_close(
      "huge sparse Jacobian max error",
      max_huge_sparse_jacobian_difference(jacobian_dual, jacobian_fd),
      0.0,
      5e-6
    );
    ctx.expect_close(
      "huge sparse Jacobian structural nonzeros",
      double(count_huge_sparse_jacobian_nonzeros(jacobian_dual)),
      double(TestSparse::expected_huge_sparse_jacobian_nonzeros()),
      0.0
    );
    ctx.expect_close("huge sparse Jacobian zero away from stencil", jacobian_dual[0][9], 0.0, 1e-12);
  }

}

/**
 * @brief Entry point of the test program.
 *
 * @return int exit code `0` on success.
 */
int main() {
  try {
    TestContext ctx;
    test_basic_operations(ctx);
    test_trigonometric_functions(ctx);
    test_hyperbolic_functions(ctx);
    test_exponential_and_logarithmic_functions(ctx);
    test_power_and_root_functions(ctx);
    test_auxiliary_functions(ctx);
    test_composed_expression(ctx);
    test_large_sparse_derivatives(ctx);
    test_dense10_derivatives(ctx);
    test_huge_sparse_vector_map(ctx);
    ctx.finalize();
  } catch ( std::exception const & exc ) {
    std::cerr << exc.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
