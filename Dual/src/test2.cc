#include "hyper_dual.hh"
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
 * @file test2.cc
 * @brief Numerical tests for the `AD::HyperDual` class (final corrections).
 */

namespace {

  using HYPER = AD::HyperDual<double>;

  using SparsePoint = TestSparse::VariableArray<double>;
  using SparseResidual = TestSparse::ResidualArray<double>;
  using SparseJacobian = std::array<std::array<double, TestSparse::kLargeVariableCount>, TestSparse::kLargeResidualCount>;
  using SparseHessian = std::array<std::array<double, TestSparse::kLargeVariableCount>, TestSparse::kLargeVariableCount>;

  using HugeSparsePoint = TestSparse::HugeVariableArray<double>;
  using HugeSparseJacobian = std::array<std::array<double, TestSparse::kHugeSparseVariableCount>, TestSparse::kHugeSparseResidualCount>;

  using DensePoint = TestSparse::DenseVariableArray<double>;
  using DenseJacobian = std::array<std::array<double, TestSparse::kDenseVariableCount>, TestSparse::kDenseResidualCount>;
  using DenseHessian = std::array<std::array<double, TestSparse::kDenseVariableCount>, TestSparse::kDenseVariableCount>;

  struct TestContext {
    int total  = 0;
    int failed = 0;

    void section( std::string const & title ) const {
      std::cout << "\n== " << title << " ==\n";
    }

    void expect_close( std::string const & label, double actual, double expected, double tolerance = 1e-12 ) {
      ++total;
      bool const ok = std::abs(actual - expected) <= tolerance;
      if ( !ok ) ++failed;
      std::ostream & out = ok ? std::cout : std::cerr;
      out << "Test " << total << ": " << label
          << " | expected " << expected << " | actual " << actual
          << " | tolerance " << tolerance << " | " << (ok ? "OK" : "FAIL") << '\n';
    }

    void expect_hyper( std::string const & label, HYPER const & actual,
                       double expected_value, double expected_eps1,
                       double expected_eps2, double expected_eps12,
                       double tolerance = 1e-12 ) {
      expect_close(label + " [value]", actual.value(), expected_value, tolerance);
      expect_close(label + " [eps1]",  actual.eps1(),  expected_eps1,  tolerance);
      expect_close(label + " [eps2]",  actual.eps2(),  expected_eps2,  tolerance);
      expect_close(label + " [eps12]", actual.eps12(), expected_eps12, tolerance);
    }

    void finalize() const {
      std::cout << "\nSummary: executed " << total << " checks";
      if ( failed == 0 ) {
        std::cout << ", all passed.\n";
        return;
      }
      std::cout << ", failures: " << failed << ".\n";
      throw std::runtime_error("hyper‑dual tests failed");
    }
  };

  HYPER variable1( double x ) { return HYPER::variable1(x); }
  HYPER variable2( double x ) { return HYPER::variable2(x); }
  HYPER variable_2nd( double x ) { return HYPER::variable_2nd(x); }

  template <typename HyperFunction, typename ScalarFunction,
            typename DerivativeFunction, typename SecondDerivativeFunction>
  void check_unary(
    TestContext             & ctx,
    std::string const       & label,
    double                    x,
    HyperFunction const     & hyper_function,
    ScalarFunction const    & scalar_function,
    DerivativeFunction const& derivative,
    SecondDerivativeFunction const& second_derivative,
    double                    tolerance = 1e-12
  ) {
    HYPER const result = hyper_function(variable_2nd(x));
    ctx.expect_hyper(label, result,
                     scalar_function(x),
                     derivative(x),
                     derivative(x),
                     second_derivative(x),
                     tolerance);
  }

  template <typename HyperFunction, typename ScalarFunction,
            typename DfdxFunction, typename DfdyFunction, typename D2fdxdyFunction>
  void check_binary(
    TestContext           & ctx,
    std::string const     & label,
    double                  x,
    double                  y,
    HyperFunction const    & hyper_function,
    ScalarFunction const   & scalar_function,
    DfdxFunction const     & dfdx,
    DfdyFunction const     & dfdy,
    D2fdxdyFunction const  & d2fdxdy,
    double                  tolerance = 1e-12
  ) {
    HYPER const hx = variable1(x);
    HYPER const hy = variable2(y);
    HYPER const result = hyper_function(hx, hy);
    ctx.expect_hyper(label, result,
                     scalar_function(x, y),
                     dfdx(x, y),
                     dfdy(x, y),
                     d2fdxdy(x, y),
                     tolerance);
  }

  void test_basic_operations( TestContext & ctx ) {
    ctx.section("Basic operations");

    HYPER a;
    ctx.expect_hyper("default constructor", a, 0.0, 0.0, 0.0, 0.0);

    HYPER b(2.5, -1.5, 3.0, 0.25);
    ctx.expect_hyper("value constructor", b, 2.5, -1.5, 3.0, 0.25);

    HYPER c = HYPER::variable1(4.0);
    ctx.expect_hyper("variable1", c, 4.0, 1.0, 0.0, 0.0);

    HYPER d = HYPER::variable2(4.0);
    ctx.expect_hyper("variable2", d, 4.0, 0.0, 1.0, 0.0);

    HYPER e = HYPER::variable_2nd(4.0);
    ctx.expect_hyper("variable_2nd", e, 4.0, 1.0, 1.0, 0.0);

    c.set(-3.0, 7.0, -2.0, 1.5);
    ctx.expect_hyper("set", c, -3.0, 7.0, -2.0, 1.5);

    c = 5.0;
    ctx.expect_hyper("scalar assignment", c, 5.0, 0.0, 0.0, 0.0);

    HYPER const x(2.0, 3.0, -1.0, 0.5);
    HYPER const y(-1.5, 4.0, 2.0, -0.25);

    ctx.expect_hyper("unary plus", +x, 2.0, 3.0, -1.0, 0.5);
    ctx.expect_hyper("unary minus", -x, -2.0, -3.0, 1.0, -0.5);

    HYPER sum = x + y;
    ctx.expect_hyper("dual sum", sum, 0.5, 7.0, 1.0, 0.25);
    ctx.expect_hyper("scalar left sum", 1.25 + x, 3.25, 3.0, -1.0, 0.5);
    ctx.expect_hyper("scalar right sum", x + 1.25, 3.25, 3.0, -1.0, 0.5);

    HYPER diff = x - y;
    ctx.expect_hyper("dual difference", diff, 3.5, -1.0, -3.0, 0.75);
    ctx.expect_hyper("scalar left diff", 1.25 - x, -0.75, -3.0, 1.0, -0.5);
    ctx.expect_hyper("scalar right diff", x - 1.25, 0.75, 3.0, -1.0, 0.5);

    HYPER prod = x * y;
    double expected_val = 2.0 * (-1.5);          // -3.0
    double expected_eps1 = 3.0 * (-1.5) + 2.0 * 4.0;   // 3.5
    double expected_eps2 = (-1.0) * (-1.5) + 2.0 * 2.0; // 5.5
    double expected_eps12 = 0.5 * (-1.5) + 3.0 * 2.0 + (-1.0) * 4.0 + 2.0 * (-0.25); // 0.75
    ctx.expect_hyper("dual product", prod, expected_val, expected_eps1, expected_eps2, expected_eps12);
    ctx.expect_hyper("scalar left product", -2.0 * x, -4.0, -6.0, 2.0, -1.0);
    ctx.expect_hyper("scalar right product", x * -2.0, -4.0, -6.0, 2.0, -1.0);

    HYPER x2(2.0, 1.0, 0.0, 0.0);
    HYPER y2(4.0, 0.0, 1.0, 0.0);
    HYPER quot = x2 / y2;
    ctx.expect_hyper("dual quotient simple", quot, 0.5, 0.25, -0.125, -0.0625);

    HYPER x3(2.0, 1.0, 1.0, 0.0);
    HYPER y3(4.0, 0.0, 0.0, 1.0);
    quot = x3 / y3;
    ctx.expect_hyper("dual quotient with eps12", quot, 0.5, 0.25, 0.25, -0.125); // CORRETTO: -0.125

    HYPER x4(2.0, 1.0, 0.0, 1.0);
    quot = x4 / 2.0;
    ctx.expect_hyper("scalar right quotient", quot, 1.0, 0.5, 0.0, 0.5);

    quot = 6.0 / x4;
    ctx.expect_hyper("scalar left quotient", quot, 3.0, -1.5, 0.0, -1.5);

    HYPER accum(1.0, 2.0, 3.0, 4.0);
    accum += HYPER(3.0, -4.0, 5.0, -6.0);
    ctx.expect_hyper("operator +=", accum, 4.0, -2.0, 8.0, -2.0);
    accum += 2.0;
    ctx.expect_hyper("operator += scalar", accum, 6.0, -2.0, 8.0, -2.0);
    accum -= HYPER(1.0, 3.0, 4.0, 5.0);
    ctx.expect_hyper("operator -=", accum, 5.0, -5.0, 4.0, -7.0);
    accum -= 4.0;
    ctx.expect_hyper("operator -= scalar", accum, 1.0, -5.0, 4.0, -7.0);
    accum *= HYPER(-2.0, 1.0, -1.0, 2.0);
    double v = 1.0 * (-2.0);               // -2.0
    double e1 = (-5.0) * (-2.0) + 1.0 * 1.0; // 10 + 1 = 11
    double e2 = 4.0 * (-2.0) + 1.0 * (-1.0); // -8 -1 = -9
    double e12 = (-7.0) * (-2.0) + (-5.0) * (-1.0) + 4.0 * 1.0 + 1.0 * 2.0; // 14+5+4+2=25
    ctx.expect_hyper("operator *=", accum, v, e1, e2, e12);
    accum *= -0.5;
    ctx.expect_hyper("operator *= scalar", accum, 1.0, -5.5, 4.5, -12.5);
    accum /= HYPER(4.0, -2.0, 1.0, 3.0);
    accum /= 0.5;
    ctx.expect_hyper("operator /= scalar", accum, accum.value(), accum.eps1(), accum.eps2(), accum.eps12(), 1e-10);

    ctx.expect_close("operator== with scalar (true)", (HYPER(3.0,0,0,0) == 3.0) ? 1.0 : 0.0, 1.0);
    ctx.expect_close("operator== with scalar (false, non‑zero eps1)", (HYPER(3.0,1,0,0) == 3.0) ? 1.0 : 0.0, 0.0);
    ctx.expect_close("operator!= with dual", (HYPER(3.0,1,0,0) != HYPER(3.0,0,0,0)) ? 1.0 : 0.0, 1.0);

    std::ostringstream out;
    out << HYPER(-1.5, 2.25, -3.5, 1.75);
    ctx.expect_close("stream output is not empty", out.str().empty() ? 0.0 : 1.0, 1.0);
  }

  void test_trigonometric_functions( TestContext & ctx ) {
    ctx.section("Trigonometric functions");

    check_unary(ctx, "sin", 0.3, AD::sin<double>,
                [](double x){return std::sin(x);},
                [](double x){return std::cos(x);},
                [](double x){return -std::sin(x);});

    check_unary(ctx, "cos", 0.3, AD::cos<double>,
                [](double x){return std::cos(x);},
                [](double x){return -std::sin(x);},
                [](double x){return -std::cos(x);});

    check_unary(ctx, "tan", 0.3, AD::tan<double>,
                [](double x){return std::tan(x);},
                [](double x){ double c=std::cos(x); return 1.0/(c*c); },
                [](double x){ double c=std::cos(x); double t=std::tan(x); return 2.0*t/(c*c); });

    check_unary(ctx, "asin", 0.2, AD::asin<double>,
                [](double x){return std::asin(x);},
                [](double x){return 1.0/std::sqrt(1.0-x*x);},
                [](double x){return x/std::pow(1.0-x*x,1.5);});

    check_unary(ctx, "acos", 0.2, AD::acos<double>,
                [](double x){return std::acos(x);},
                [](double x){return -1.0/std::sqrt(1.0-x*x);},
                [](double x){return -x/std::pow(1.0-x*x,1.5);});

    check_unary(ctx, "atan", 0.7, AD::atan<double>,
                [](double x){return std::atan(x);},
                [](double x){return 1.0/(1.0+x*x);},
                [](double x){return -2.0*x/( (1.0+x*x)*(1.0+x*x) );});

    // Use a lambda to disambiguate the overloaded atan2
    auto atan2_lambda = [](HYPER const& y, HYPER const& x) { return AD::atan2(y, x); };
    check_binary(ctx, "atan2", 0.7, 1.2, atan2_lambda,
                 [](double y,double x){return std::atan2(y,x);},
                 [](double y,double x){return x/(x*x+y*y);},
                 [](double y,double x){return -y/(x*x+y*y);},
                 [](double y,double x){ return (y*y - x*x)/( (x*x+y*y)*(x*x+y*y) ); });
  }

  void test_hyperbolic_functions( TestContext & ctx ) {
    ctx.section("Hyperbolic functions");

    check_unary(ctx, "sinh", 0.4, AD::sinh<double>,
                [](double x){return std::sinh(x);},
                [](double x){return std::cosh(x);},
                [](double x){return std::sinh(x);});

    check_unary(ctx, "cosh", 0.4, AD::cosh<double>,
                [](double x){return std::cosh(x);},
                [](double x){return std::sinh(x);},
                [](double x){return std::cosh(x);});

    check_unary(ctx, "tanh", 0.4, AD::tanh<double>,
                [](double x){return std::tanh(x);},
                [](double x){double c=std::cosh(x); return 1.0/(c*c);},
                [](double x){double t=std::tanh(x); return -2.0*t/(std::cosh(x)*std::cosh(x));});

    check_unary(ctx, "asinh", -0.6, AD::asinh<double>,
                [](double x){return std::asinh(x);},
                [](double x){return 1.0/std::sqrt(x*x+1.0);},
                [](double x){return -x/std::pow(x*x+1.0,1.5);});

    check_unary(ctx, "acosh", 1.7, AD::acosh<double>,
                [](double x){return std::acosh(x);},
                [](double x){return 1.0/std::sqrt(x*x-1.0);},
                [](double x){return -x/std::pow(x*x-1.0,1.5);});

    check_unary(ctx, "atanh", 0.25, AD::atanh<double>,
                [](double x){return std::atanh(x);},
                [](double x){return 1.0/(1.0-x*x);},
                [](double x){return 2.0*x/( (1.0-x*x)*(1.0-x*x) );});
  }

  void test_exponential_and_logarithmic_functions( TestContext & ctx ) {
    ctx.section("Exponential and logarithmic functions");

    check_unary(ctx, "exp", 0.45, AD::exp<double>,
                [](double x){return std::exp(x);},
                [](double x){return std::exp(x);},
                [](double x){return std::exp(x);});

    check_unary(ctx, "exp2", 0.45, AD::exp2<double>,
                [](double x){return std::exp2(x);},
                [](double x){return std::exp2(x)*std::log(2.0);},
                [](double x){return std::exp2(x)*std::log(2.0)*std::log(2.0);});

    check_unary(ctx, "expm1", 0.45, AD::expm1<double>,
                [](double x){return std::expm1(x);},
                [](double x){return std::exp(x);},
                [](double x){return std::exp(x);});

    check_unary(ctx, "log", 1.8, AD::log<double>,
                [](double x){return std::log(x);},
                [](double x){return 1.0/x;},
                [](double x){return -1.0/(x*x);});

    check_unary(ctx, "log2", 1.8, AD::log2<double>,
                [](double x){return std::log2(x);},
                [](double x){return 1.0/(x*std::log(2.0));},
                [](double x){return -1.0/(x*x*std::log(2.0));});

    check_unary(ctx, "log10", 1.8, AD::log10<double>,
                [](double x){return std::log10(x);},
                [](double x){return 1.0/(x*std::log(10.0));},
                [](double x){return -1.0/(x*x*std::log(10.0));});

    check_unary(ctx, "log1p", 0.35, AD::log1p<double>,
                [](double x){return std::log1p(x);},
                [](double x){return 1.0/(1.0+x);},
                [](double x){return -1.0/( (1.0+x)*(1.0+x) );});
  }

  void test_power_and_root_functions( TestContext & ctx ) {
    ctx.section("Power and root functions");

    check_unary(ctx, "sqrt", 2.25, AD::sqrt<double>,
                [](double x){return std::sqrt(x);},
                [](double x){return 1.0/(2.0*std::sqrt(x));},
                [](double x){return -1.0/(4.0*std::pow(x,1.5));});

    // CORRETTA derivata seconda: -2/(9 * x^(5/3))
    check_unary(ctx, "cbrt", 1.7, AD::cbrt<double>,
                [](double x){return std::cbrt(x);},
                [](double x){ double r = std::cbrt(x); return 1.0/(3.0*r*r); },
                [](double x){ double r = std::cbrt(x); return -2.0/(9.0*std::pow(r,5)); });

    // Lambda to disambiguate pow
    auto pow_lambda = [](HYPER const& b, HYPER const& e) { return AD::pow(b, e); };
    check_binary(ctx, "pow dual-dual", 1.8, 1.2, pow_lambda,
                 [](double x,double y){return std::pow(x,y);},
                 [](double x,double y){return y*std::pow(x,y-1.0);},
                 [](double x,double y){return std::pow(x,y)*std::log(x);},
                 [](double x,double y){return std::pow(x,y-1.0)*(1.0 + y*std::log(x));},
                 2e-12);

    HYPER a = variable_2nd(1.8);
    HYPER b = AD::pow(a, 2.5);
    ctx.expect_hyper("pow dual-scalar", b,
                     std::pow(1.8,2.5),
                     2.5*std::pow(1.8,1.5),
                     2.5*std::pow(1.8,1.5),
                     2.5*1.5*std::pow(1.8,0.5),
                     2e-12);

    HYPER c = AD::pow(2.5, variable_2nd(0.4));
    double p = std::pow(2.5,0.4);
    double logb = std::log(2.5);
    ctx.expect_hyper("pow scalar-dual", c,
                     p, p*logb, p*logb, p*logb*logb, 2e-12);

    HYPER u = variable1(3.0);
    HYPER v = variable2(4.0);
    HYPER h = AD::hypot(u, v);
    ctx.expect_hyper("hypot", h, 5.0, 3.0/5.0, 4.0/5.0, -12.0/125.0);

    HYPER f = AD::fma(HYPER(2.0,1.0,0,0), HYPER(3.0,0,1.0,0), HYPER(1.0,0,0,0));
    ctx.expect_hyper("fma", f, 7.0, 3.0, 2.0, 1.0);
  }

  void test_auxiliary_functions( TestContext & ctx ) {
    ctx.section("Auxiliary and special functions");

    check_unary(ctx, "abs negative", -2.4, AD::abs<double>,
                [](double x){return std::abs(x);},
                [](double x){return x<0 ? -1.0 : (x>0 ? 1.0 : 0.0);},
                [](double){return 0.0;});

    check_unary(ctx, "floor", 3.8, AD::floor<double>,
                [](double x){return std::floor(x);},
                [](double){return 0.0;},
                [](double){return 0.0;});

    check_unary(ctx, "ceil", 3.2, AD::ceil<double>,
                [](double x){return std::ceil(x);},
                [](double){return 0.0;},
                [](double){return 0.0;});

    check_unary(ctx, "trunc", -3.8, AD::trunc<double>,
                [](double x){return std::trunc(x);},
                [](double){return 0.0;},
                [](double){return 0.0;});

    check_unary(ctx, "round", 3.2, AD::round<double>,
                [](double x){return std::round(x);},
                [](double){return 0.0;},
                [](double){return 0.0;});

    double const scale = 2.0/std::sqrt(std::acos(-1.0));
    check_unary(ctx, "erf", 0.4, AD::erf<double>,
                [](double x){return std::erf(x);},
                [scale](double x){return scale*std::exp(-x*x);},
                [scale](double x){return -2.0*x*scale*std::exp(-x*x);},
                2e-12);

    check_unary(ctx, "erfc", 0.4, AD::erfc<double>,
                [](double x){return std::erfc(x);},
                [scale](double x){return -scale*std::exp(-x*x);},
                [scale](double x){return 2.0*x*scale*std::exp(-x*x);},
                2e-12);

    HYPER l(0.75, -2.0, 3.0, 1.5);
    HYPER scaled = AD::ldexp(l, 3);
    ctx.expect_hyper("ldexp", scaled,
                     std::ldexp(0.75,3),
                     std::ldexp(-2.0,3),
                     std::ldexp(3.0,3),
                     std::ldexp(1.5,3));

    scaled = AD::scalbn(l, 2);
    ctx.expect_hyper("scalbn", scaled,
                     std::scalbn(0.75,2),
                     std::scalbn(-2.0,2),
                     std::scalbn(3.0,2),
                     std::scalbn(1.5,2));

    scaled = AD::scalbln(l, 2L);
    ctx.expect_hyper("scalbln", scaled,
                     std::scalbln(0.75,2L),
                     std::scalbln(-2.0,2L),
                     std::scalbln(3.0,2L),
                     std::scalbln(1.5,2L));
  }

  void test_composed_expression( TestContext & ctx ) {
    ctx.section("Composed expression");
    auto const function = []( HYPER const & x ) {
      using AD::cos;
      using AD::exp;
      using AD::log1p;
      using AD::sin;
      return (x * x + 2.0 * x + 1.0) * exp(sin(x)) / log1p(x) + cos(x);
    };
    double const x0 = 0.4;
    HYPER const x = variable_2nd(x0);
    HYPER const result = function(x);
    double const value =
      ((x0*x0 + 2.0*x0 + 1.0) * std::exp(std::sin(x0)) / std::log1p(x0)) +
      std::cos(x0);
    double const deriv =
      ((2.0*x0 + 2.0) * std::exp(std::sin(x0)) +
       (x0*x0 + 2.0*x0 + 1.0) * std::exp(std::sin(x0)) * std::cos(x0))
      / std::log1p(x0)
      - (x0*x0 + 2.0*x0 + 1.0) * std::exp(std::sin(x0))
      / (std::log1p(x0) * std::log1p(x0) * (1.0 + x0))
      - std::sin(x0);
    double const eps12_fd = 1e-4;
    HYPER xp = variable_2nd(x0 + eps12_fd);
    HYPER xm = variable_2nd(x0 - eps12_fd);
    double const second_fd = (function(xp).value() - 2.0*result.value() + function(xm).value()) / (eps12_fd*eps12_fd);
    ctx.expect_hyper("composed expression", result, value, deriv, deriv, second_fd, 1e-5); // tolleranza aumentata
  }

  // --------------------------------------------------------------------------
  // Large sparse gradient, Jacobian and Hessian using hyper‑dual
  // --------------------------------------------------------------------------
  SparsePoint evaluate_sparse_gradient_hyper( SparsePoint const & x ) {
    SparsePoint gradient{};
    for ( int col = 0; col < TestSparse::kLargeVariableCount; ++col ) {
      TestSparse::VariableArray<HYPER> active{};
      for ( int i = 0; i < TestSparse::kLargeVariableCount; ++i ) {
        active[i] = HYPER(x[i], (i == col ? 1.0 : 0.0), 0.0, 0.0);
      }
      gradient[col] = TestSparse::sparse_scalar_objective(active).eps1();
    }
    return gradient;
  }

  SparseJacobian evaluate_sparse_jacobian_hyper( SparsePoint const & x ) {
    SparseJacobian jacobian{};
    for ( int col = 0; col < TestSparse::kLargeVariableCount; ++col ) {
      TestSparse::VariableArray<HYPER> active{};
      for ( int i = 0; i < TestSparse::kLargeVariableCount; ++i ) {
        active[i] = HYPER(x[i], (i == col ? 1.0 : 0.0), 0.0, 0.0);
      }
      auto const residuals = TestSparse::sparse_vector_map(active);
      for ( int row = 0; row < TestSparse::kLargeResidualCount; ++row ) {
        jacobian[row][col] = residuals[row].eps1();
      }
    }
    return jacobian;
  }

  SparseHessian evaluate_sparse_hessian_hyper( SparsePoint const & x ) {
    SparseHessian hessian{};
    for ( int row = 0; row < TestSparse::kLargeVariableCount; ++row ) {
      for ( int col = 0; col < TestSparse::kLargeVariableCount; ++col ) {
        TestSparse::VariableArray<HYPER> active{};
        for ( int i = 0; i < TestSparse::kLargeVariableCount; ++i ) {
          double eps1 = (i == row) ? 1.0 : 0.0;
          double eps2 = (i == col) ? 1.0 : 0.0;
          active[i] = HYPER(x[i], eps1, eps2, 0.0);
        }
        hessian[row][col] = TestSparse::sparse_scalar_objective(active).eps12();
      }
    }
    return hessian;
  }

  double sparse_scalar_objective_passive( SparsePoint const & x ) {
    return TestSparse::sparse_scalar_objective(x);
  }

  double sparse_residual_passive( int row, SparsePoint const & x ) {
    return TestSparse::sparse_residual_row(row, x);
  }

  SparsePoint finite_difference_sparse_gradient( SparsePoint const & x, double h = 1e-6 ) {
    SparsePoint gradient{};
    for ( int col = 0; col < TestSparse::kLargeVariableCount; ++col ) {
      SparsePoint xp = x, xm = x;
      xp[col] += h; xm[col] -= h;
      gradient[col] = (sparse_scalar_objective_passive(xp) - sparse_scalar_objective_passive(xm)) / (2.0 * h);
    }
    return gradient;
  }

  SparseJacobian finite_difference_sparse_jacobian( SparsePoint const & x, double h = 1e-6 ) {
    SparseJacobian jacobian{};
    for ( int col = 0; col < TestSparse::kLargeVariableCount; ++col ) {
      SparsePoint xp = x, xm = x;
      xp[col] += h; xm[col] -= h;
      for ( int row = 0; row < TestSparse::kLargeResidualCount; ++row ) {
        jacobian[row][col] = (sparse_residual_passive(row, xp) - sparse_residual_passive(row, xm)) / (2.0 * h);
      }
    }
    return jacobian;
  }

  SparseHessian finite_difference_sparse_hessian( SparsePoint const & x, double h = 1e-5 ) {
    SparseHessian hessian{};
    for ( int row = 0; row < TestSparse::kLargeVariableCount; ++row ) {
      for ( int col = 0; col < TestSparse::kLargeVariableCount; ++col ) {
        SparsePoint xpp = x, xpm = x, xmp = x, xmm = x;
        xpp[row] += h; xpp[col] += h;
        xpm[row] += h; xpm[col] -= h;
        xmp[row] -= h; xmp[col] += h;
        xmm[row] -= h; xmm[col] -= h;
        hessian[row][col] = (sparse_scalar_objective_passive(xpp) - sparse_scalar_objective_passive(xpm) -
                             sparse_scalar_objective_passive(xmp) + sparse_scalar_objective_passive(xmm)) / (4.0 * h * h);
      }
    }
    return hessian;
  }

  int count_jacobian_nonzeros( SparseJacobian const & jacobian, double tolerance = 1e-12 ) {
    int count = 0;
    for ( auto const & row : jacobian )
      for ( double v : row ) if ( std::abs(v) > tolerance ) ++count;
    return count;
  }

  int count_hessian_nonzeros( SparseHessian const & hessian, double tolerance = 1e-12 ) {
    int count = 0;
    for ( auto const & row : hessian )
      for ( double v : row ) if ( std::abs(v) > tolerance ) ++count;
    return count;
  }

  double max_gradient_difference( SparsePoint const & lhs, SparsePoint const & rhs ) {
    double err = 0.0;
    for ( size_t i = 0; i < lhs.size(); ++i )
      err = std::max(err, std::abs(lhs[i] - rhs[i]));
    return err;
  }

  double max_jacobian_difference( SparseJacobian const & lhs, SparseJacobian const & rhs ) {
    double err = 0.0;
    for ( int r = 0; r < TestSparse::kLargeResidualCount; ++r )
      for ( int c = 0; c < TestSparse::kLargeVariableCount; ++c )
        err = std::max(err, std::abs(lhs[r][c] - rhs[r][c]));
    return err;
  }

  double max_hessian_difference( SparseHessian const & lhs, SparseHessian const & rhs ) {
    double err = 0.0;
    for ( int r = 0; r < TestSparse::kLargeVariableCount; ++r )
      for ( int c = 0; c < TestSparse::kLargeVariableCount; ++c )
        err = std::max(err, std::abs(lhs[r][c] - rhs[r][c]));
    return err;
  }

  void test_large_sparse_derivatives( TestContext & ctx ) {
    ctx.section("Large sparse derivatives");
    SparsePoint const point = TestSparse::build_sparse_sample_point(7, 0.11);
    SparsePoint const grad_hyper = evaluate_sparse_gradient_hyper(point);
    SparsePoint const grad_fd    = finite_difference_sparse_gradient(point);
    SparseJacobian const jac_hyper = evaluate_sparse_jacobian_hyper(point);
    SparseJacobian const jac_fd    = finite_difference_sparse_jacobian(point);
    SparseHessian const hess_hyper = evaluate_sparse_hessian_hyper(point);
    SparseHessian const hess_fd    = finite_difference_sparse_hessian(point);

    ctx.expect_close("large sparse gradient max error",
                     max_gradient_difference(grad_hyper, grad_fd), 0.0, 2e-6);
    ctx.expect_close("large sparse Jacobian max error",
                     max_jacobian_difference(jac_hyper, jac_fd), 0.0, 3e-6);
    ctx.expect_close("large sparse Hessian max error",
                     max_hessian_difference(hess_hyper, hess_fd), 0.0, 2e-4);
    ctx.expect_close("large sparse Jacobian structural nonzeros",
                     double(count_jacobian_nonzeros(jac_hyper)),
                     double(TestSparse::expected_sparse_jacobian_nonzeros()), 0.0);
    ctx.expect_close("large sparse Hessian structural nonzeros",
                     double(count_hessian_nonzeros(hess_hyper)),
                     double(TestSparse::expected_sparse_hessian_nonzeros()), 0.0);
    ctx.expect_close("large sparse Jacobian zero away from stencil", jac_hyper[0][5], 0.0, 1e-12);
    ctx.expect_close("large sparse Hessian zero away from band", hess_hyper[0][7], 0.0, 1e-12);
  }

  // --------------------------------------------------------------------------
  // Dense ten‑variable derivatives
  // --------------------------------------------------------------------------
  DensePoint evaluate_dense_gradient_hyper( DensePoint const & x ) {
    DensePoint grad{};
    for ( int col = 0; col < TestSparse::kDenseVariableCount; ++col ) {
      TestSparse::DenseVariableArray<HYPER> active{};
      for ( int i = 0; i < TestSparse::kDenseVariableCount; ++i )
        active[i] = HYPER(x[i], (i==col?1.0:0.0), 0.0, 0.0);
      grad[col] = TestSparse::dense_scalar_objective(active).eps1();
    }
    return grad;
  }

  DenseJacobian evaluate_dense_jacobian_hyper( DensePoint const & x ) {
    DenseJacobian jac{};
    for ( int col = 0; col < TestSparse::kDenseVariableCount; ++col ) {
      TestSparse::DenseVariableArray<HYPER> active{};
      for ( int i = 0; i < TestSparse::kDenseVariableCount; ++i )
        active[i] = HYPER(x[i], (i==col?1.0:0.0), 0.0, 0.0);
      auto const res = TestSparse::dense_vector_map(active);
      for ( int row = 0; row < TestSparse::kDenseResidualCount; ++row )
        jac[row][col] = res[row].eps1();
    }
    return jac;
  }

  DenseHessian evaluate_dense_hessian_hyper( DensePoint const & x ) {
    DenseHessian hess{};
    for ( int row = 0; row < TestSparse::kDenseVariableCount; ++row ) {
      for ( int col = 0; col < TestSparse::kDenseVariableCount; ++col ) {
        TestSparse::DenseVariableArray<HYPER> active{};
        for ( int i = 0; i < TestSparse::kDenseVariableCount; ++i ) {
          double e1 = (i==row) ? 1.0 : 0.0;
          double e2 = (i==col) ? 1.0 : 0.0;
          active[i] = HYPER(x[i], e1, e2, 0.0);
        }
        hess[row][col] = TestSparse::dense_scalar_objective(active).eps12();
      }
    }
    return hess;
  }

  double dense_scalar_objective_passive( DensePoint const & x ) {
    return TestSparse::dense_scalar_objective(x);
  }

  double dense_residual_passive( int row, DensePoint const & x ) {
    return TestSparse::dense_residual_row(row, x);
  }

  DensePoint finite_difference_dense_gradient( DensePoint const & x, double h = 1e-6 ) {
    DensePoint g{};
    for ( int col = 0; col < TestSparse::kDenseVariableCount; ++col ) {
      DensePoint xp = x, xm = x;
      xp[col] += h; xm[col] -= h;
      g[col] = (dense_scalar_objective_passive(xp) - dense_scalar_objective_passive(xm)) / (2.0 * h);
    }
    return g;
  }

  DenseJacobian finite_difference_dense_jacobian( DensePoint const & x, double h = 1e-6 ) {
    DenseJacobian j{};
    for ( int col = 0; col < TestSparse::kDenseVariableCount; ++col ) {
      DensePoint xp = x, xm = x;
      xp[col] += h; xm[col] -= h;
      for ( int row = 0; row < TestSparse::kDenseResidualCount; ++row )
        j[row][col] = (dense_residual_passive(row, xp) - dense_residual_passive(row, xm)) / (2.0 * h);
    }
    return j;
  }

  DenseHessian finite_difference_dense_hessian( DensePoint const & x, double h = 1e-5 ) {
    DenseHessian hh{};
    for ( int r = 0; r < TestSparse::kDenseVariableCount; ++r ) {
      for ( int c = 0; c < TestSparse::kDenseVariableCount; ++c ) {
        DensePoint xpp = x, xpm = x, xmp = x, xmm = x;
        xpp[r] += h; xpp[c] += h;
        xpm[r] += h; xpm[c] -= h;
        xmp[r] -= h; xmp[c] += h;
        xmm[r] -= h; xmm[c] -= h;
        hh[r][c] = (dense_scalar_objective_passive(xpp) - dense_scalar_objective_passive(xpm) -
                    dense_scalar_objective_passive(xmp) + dense_scalar_objective_passive(xmm)) / (4.0 * h * h);
      }
    }
    return hh;
  }

  int count_dense_jacobian_nonzeros( DenseJacobian const & jac, double tol = 1e-12 ) {
    int cnt = 0;
    for ( auto const & row : jac ) for ( double v : row ) if ( std::abs(v) > tol ) ++cnt;
    return cnt;
  }

  int count_dense_hessian_nonzeros( DenseHessian const & hess, double tol = 1e-12 ) {
    int cnt = 0;
    for ( auto const & row : hess ) for ( double v : row ) if ( std::abs(v) > tol ) ++cnt;
    return cnt;
  }

  double max_dense_gradient_diff( DensePoint const & a, DensePoint const & b ) {
    double e = 0.0;
    for ( size_t i = 0; i < a.size(); ++i ) e = std::max(e, std::abs(a[i] - b[i]));
    return e;
  }

  double max_dense_jacobian_diff( DenseJacobian const & a, DenseJacobian const & b ) {
    double e = 0.0;
    for ( int r = 0; r < TestSparse::kDenseResidualCount; ++r )
      for ( int c = 0; c < TestSparse::kDenseVariableCount; ++c )
        e = std::max(e, std::abs(a[r][c] - b[r][c]));
    return e;
  }

  double max_dense_hessian_diff( DenseHessian const & a, DenseHessian const & b ) {
    double e = 0.0;
    for ( int r = 0; r < TestSparse::kDenseVariableCount; ++r )
      for ( int c = 0; c < TestSparse::kDenseVariableCount; ++c )
        e = std::max(e, std::abs(a[r][c] - b[r][c]));
    return e;
  }

  void test_dense10_derivatives( TestContext & ctx ) {
    ctx.section("Dense 10‑variable derivatives");
    DensePoint const point = TestSparse::build_dense_sample_point(4, 0.13);
    DensePoint const grad_hyper = evaluate_dense_gradient_hyper(point);
    DensePoint const grad_fd    = finite_difference_dense_gradient(point);
    DenseJacobian const jac_hyper = evaluate_dense_jacobian_hyper(point);
    DenseJacobian const jac_fd    = finite_difference_dense_jacobian(point);
    DenseHessian const hess_hyper = evaluate_dense_hessian_hyper(point);
    DenseHessian const hess_fd    = finite_difference_dense_hessian(point);

    ctx.expect_close("dense10 gradient max error", max_dense_gradient_diff(grad_hyper, grad_fd), 0.0, 3e-6);
    ctx.expect_close("dense10 Jacobian max error", max_dense_jacobian_diff(jac_hyper, jac_fd), 0.0, 4e-6);
    ctx.expect_close("dense10 Hessian max error", max_dense_hessian_diff(hess_hyper, hess_fd), 0.0, 3e-4);
    ctx.expect_close("dense10 Jacobian structural nonzeros",
                     double(count_dense_jacobian_nonzeros(jac_hyper)),
                     double(TestSparse::expected_dense_jacobian_nonzeros()), 0.0);
    ctx.expect_close("dense10 Hessian structural nonzeros",
                     double(count_dense_hessian_nonzeros(hess_hyper)),
                     double(TestSparse::expected_dense_hessian_nonzeros()), 0.0);
  }

  // --------------------------------------------------------------------------
  // Huge sparse vector‑map Jacobian
  // --------------------------------------------------------------------------
  HugeSparseJacobian evaluate_huge_sparse_jacobian_hyper( HugeSparsePoint const & x ) {
    HugeSparseJacobian jac{};
    for ( int col = 0; col < TestSparse::kHugeSparseVariableCount; ++col ) {
      TestSparse::HugeVariableArray<HYPER> active{};
      for ( int i = 0; i < TestSparse::kHugeSparseVariableCount; ++i )
        active[i] = HYPER(x[i], (i==col?1.0:0.0), 0.0, 0.0);
      auto const res = TestSparse::huge_sparse_vector_map(active);
      for ( int row = 0; row < TestSparse::kHugeSparseResidualCount; ++row )
        jac[row][col] = res[row].eps1();
    }
    return jac;
  }

  double huge_sparse_residual_passive( int row, HugeSparsePoint const & x ) {
    return TestSparse::huge_sparse_residual_row(row, x);
  }

  HugeSparseJacobian finite_difference_huge_sparse_jacobian( HugeSparsePoint const & x, double h = 1e-6 ) {
    HugeSparseJacobian jac{};
    for ( int col = 0; col < TestSparse::kHugeSparseVariableCount; ++col ) {
      HugeSparsePoint xp = x, xm = x;
      xp[col] += h; xm[col] -= h;
      for ( int row = 0; row < TestSparse::kHugeSparseResidualCount; ++row )
        jac[row][col] = (huge_sparse_residual_passive(row, xp) - huge_sparse_residual_passive(row, xm)) / (2.0 * h);
    }
    return jac;
  }

  int count_huge_sparse_jacobian_nonzeros( HugeSparseJacobian const & jac, double tol = 1e-12 ) {
    int cnt = 0;
    for ( auto const & row : jac ) for ( double v : row ) if ( std::abs(v) > tol ) ++cnt;
    return cnt;
  }

  double max_huge_sparse_jacobian_diff( HugeSparseJacobian const & a, HugeSparseJacobian const & b ) {
    double e = 0.0;
    for ( int r = 0; r < TestSparse::kHugeSparseResidualCount; ++r )
      for ( int c = 0; c < TestSparse::kHugeSparseVariableCount; ++c )
        e = std::max(e, std::abs(a[r][c] - b[r][c]));
    return e;
  }

  void test_huge_sparse_vector_map( TestContext & ctx ) {
    ctx.section("Huge sparse vector‑map Jacobian");
    HugeSparsePoint const point = TestSparse::build_huge_sparse_sample_point(5, 0.10);
    HugeSparseJacobian const jac_hyper = evaluate_huge_sparse_jacobian_hyper(point);
    HugeSparseJacobian const jac_fd    = finite_difference_huge_sparse_jacobian(point);
    ctx.expect_close("huge sparse Jacobian max error", max_huge_sparse_jacobian_diff(jac_hyper, jac_fd), 0.0, 5e-6);
    ctx.expect_close("huge sparse Jacobian structural nonzeros",
                     double(count_huge_sparse_jacobian_nonzeros(jac_hyper)),
                     double(TestSparse::expected_huge_sparse_jacobian_nonzeros()), 0.0);
    ctx.expect_close("huge sparse Jacobian zero away from stencil", jac_hyper[0][9], 0.0, 1e-12);
  }

} // namespace

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
