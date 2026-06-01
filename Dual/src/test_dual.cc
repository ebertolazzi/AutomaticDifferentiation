// test_dual.cc
// Comprehensive test suite for the Dual number library (dual.hh)
// Compile: c++ -std=c++17 -o test_dual test_dual.cc

#include "dual.hh"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <string>

using namespace AD;

const double eps = 1e-12;

bool approx_equal(double a, double b, double tol = eps) {
    return std::abs(a - b) <= tol;
}

#define TEST(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAIL: " << msg << std::endl; \
            return 1; \
        } \
    } while (0)

// Test constructors, assignment, basic access
int test_basics() {
    std::cout << "Testing basics...\n";

    // Default constructor
    Dual<double> d0;
    TEST(d0.value() == 0.0 && d0.dual() == 0.0, "default constructor");

    // Value constructor
    Dual<double> d1(3.5);
    TEST(d1.value() == 3.5 && d1.dual() == 0.0, "value constructor");

    // Full constructor
    Dual<double> d2(2.0, 1.5);
    TEST(d2.value() == 2.0 && d2.dual() == 1.5, "full constructor");

    // Copy constructor
    Dual<double> d3(d2);
    TEST(d3.value() == 2.0 && d3.dual() == 1.5, "copy constructor");

    // Move constructor
    Dual<double> d4(std::move(d3));
    TEST(d4.value() == 2.0 && d4.dual() == 1.5, "move constructor");

    // Assignment from scalar
    d0 = 4.2;
    TEST(d0.value() == 4.2 && d0.dual() == 0.0, "scalar assignment");

    // variable factory
    auto x = Dual<double>::variable(5.0);
    TEST(x.value() == 5.0 && x.dual() == 1.0, "variable()");

    // set method
    x.set(7.0, 2.0);
    TEST(x.value() == 7.0 && x.dual() == 2.0, "set()");

    std::cout << "Basics tests passed.\n";
    return 0;
}

// Test arithmetic operators
int test_arithmetic() {
    std::cout << "Testing arithmetic...\n";

    Dual<double> a(2.0, 1.0);
    Dual<double> b(3.0, 0.5);

    // Addition
    auto sum = a + b;
    TEST(approx_equal(sum.value(), 5.0), "a+b value");
    TEST(approx_equal(sum.dual(), 1.5), "a+b dual");

    auto sum_scalar1 = a + 5.0;
    TEST(approx_equal(sum_scalar1.value(), 7.0) && approx_equal(sum_scalar1.dual(), 1.0), "a+5");
    auto sum_scalar2 = 5.0 + a;
    TEST(approx_equal(sum_scalar2.value(), 7.0) && approx_equal(sum_scalar2.dual(), 1.0), "5+a");

    // Subtraction
    auto diff = a - b;
    TEST(approx_equal(diff.value(), -1.0), "a-b value");
    TEST(approx_equal(diff.dual(), 0.5), "a-b dual");

    auto diff_scalar1 = a - 1.0;
    TEST(approx_equal(diff_scalar1.value(), 1.0) && approx_equal(diff_scalar1.dual(), 1.0), "a-1");
    auto diff_scalar2 = 5.0 - a;
    TEST(approx_equal(diff_scalar2.value(), 3.0) && approx_equal(diff_scalar2.dual(), -1.0), "5-a");

    // Multiplication
    auto prod = a * b;
    TEST(approx_equal(prod.value(), 6.0), "a*b value");
    TEST(approx_equal(prod.dual(), 1.0*3.0 + 2.0*0.5), "a*b dual");

    auto prod_scalar = a * 2.0;
    TEST(approx_equal(prod_scalar.value(), 4.0) && approx_equal(prod_scalar.dual(), 2.0), "a*2");
    auto prod_scalar2 = 2.0 * a;
    TEST(approx_equal(prod_scalar2.value(), 4.0) && approx_equal(prod_scalar2.dual(), 2.0), "2*a");

    // Division
    auto quot = a / b;
    TEST(approx_equal(quot.value(), 2.0/3.0), "a/b value");
    double expected_dual = (1.0*3.0 - 2.0*0.5) / (3.0*3.0);
    TEST(approx_equal(quot.dual(), expected_dual), "a/b dual");

    auto quot_scalar1 = a / 2.0;
    TEST(approx_equal(quot_scalar1.value(), 1.0) && approx_equal(quot_scalar1.dual(), 0.5), "a/2");
    auto quot_scalar2 = 6.0 / a;
    TEST(approx_equal(quot_scalar2.value(), 3.0), "6/a value");
    TEST(approx_equal(quot_scalar2.dual(), -6.0*1.0/(2.0*2.0)), "6/a dual");

    // Compound assignments
    Dual<double> c = a;
    c += b;
    TEST(approx_equal(c.value(), 5.0) && approx_equal(c.dual(), 1.5), "c+=b");
    c = a; c -= b;
    TEST(approx_equal(c.value(), -1.0) && approx_equal(c.dual(), 0.5), "c-=b");
    c = a; c *= b;
    TEST(approx_equal(c.value(), 6.0) && approx_equal(c.dual(), 1.0*3.0+2.0*0.5), "c*=b");
    c = a; c /= b;
    TEST(approx_equal(c.value(), 2.0/3.0) && approx_equal(c.dual(), expected_dual), "c/=b");

    // Unary plus/minus
    auto pos = +a;
    TEST(pos.value() == a.value() && pos.dual() == a.dual(), "unary +");
    auto neg = -a;
    TEST(neg.value() == -a.value() && neg.dual() == -a.dual(), "unary -");

    std::cout << "Arithmetic tests passed.\n";
    return 0;
}

// Test comparisons
int test_comparisons() {
    std::cout << "Testing comparisons...\n";

    Dual<double> a(2.0, 1.0);
    Dual<double> b(2.0, 0.0);
    Dual<double> c(3.0, 0.0);

    TEST(a == a, "equality self");
    TEST(!(a == b), "different dual part -> not equal");
    TEST(b == 2.0, "dual zero and value match -> equal to scalar");
    TEST(2.0 == b, "scalar equality symmetric");
    TEST(!(a == 2.0), "non-zero dual -> not equal to scalar");
    TEST(a != b, "inequality true");
    TEST(b != c, "inequality different value");

    std::cout << "Comparison tests passed.\n";
    return 0;
}

// Test mathematical functions
int test_functions() {
    std::cout << "Testing mathematical functions...\n";

    double x0 = 0.5;
    auto x = Dual<double>::variable(x0);  // value = x0, dual = 1

    // sin
    auto s = sin(x);
    TEST(approx_equal(s.value(), std::sin(x0)), "sin value");
    TEST(approx_equal(s.dual(), std::cos(x0)), "sin derivative");

    // cos
    auto c = cos(x);
    TEST(approx_equal(c.value(), std::cos(x0)), "cos value");
    TEST(approx_equal(c.dual(), -std::sin(x0)), "cos derivative");

    // tan
    auto t = tan(x);
    TEST(approx_equal(t.value(), std::tan(x0)), "tan value");
    double sec2 = 1.0 / (std::cos(x0)*std::cos(x0));
    TEST(approx_equal(t.dual(), sec2), "tan derivative");

    // asin
    auto as = asin(x);
    TEST(approx_equal(as.value(), std::asin(x0)), "asin value");
    double asin_d = 1.0 / std::sqrt(1.0 - x0*x0);
    TEST(approx_equal(as.dual(), asin_d), "asin derivative");

    // acos
    auto ac = acos(x);
    TEST(approx_equal(ac.value(), std::acos(x0)), "acos value");
    TEST(approx_equal(ac.dual(), -asin_d), "acos derivative");

    // atan
    auto at = atan(x);
    TEST(approx_equal(at.value(), std::atan(x0)), "atan value");
    TEST(approx_equal(at.dual(), 1.0/(1.0+x0*x0)), "atan derivative");

    // atan2 with both dual
    auto y = Dual<double>::variable(0.5);  // value = 0.5, dual = 1
    auto x2 = Dual<double>::variable(0.5);
    auto a2 = atan2(y, x2);
    TEST(approx_equal(a2.value(), std::atan2(0.5, 0.5)), "atan2 value");
    // derivative: (x*dy - y*dx)/(x^2+y^2) with dx=dy=1 -> (0.5*1 - 0.5*1)/(0.5^2+0.5^2)=0
    TEST(approx_equal(a2.dual(), 0.0), "atan2 derivative");
    // scalar versions
    auto a2s = atan2(y, 1.0);
    TEST(approx_equal(a2s.dual(), 1.0/(1.0+0.5*0.5)), "atan2(y,scalar) derivative");
    auto a2s2 = atan2(1.0, x2);
    TEST(approx_equal(a2s2.dual(), -1.0/(1.0+0.5*0.5)), "atan2(scalar,x) derivative");

    // sinh
    auto sh = sinh(x);
    TEST(approx_equal(sh.value(), std::sinh(x0)), "sinh value");
    TEST(approx_equal(sh.dual(), std::cosh(x0)), "sinh derivative");

    // cosh
    auto ch = cosh(x);
    TEST(approx_equal(ch.value(), std::cosh(x0)), "cosh value");
    TEST(approx_equal(ch.dual(), std::sinh(x0)), "cosh derivative");

    // tanh
    auto th = tanh(x);
    TEST(approx_equal(th.value(), std::tanh(x0)), "tanh value");
    double sech2 = 1.0 / (std::cosh(x0)*std::cosh(x0));
    TEST(approx_equal(th.dual(), sech2), "tanh derivative");

    // asinh
    auto ash = asinh(x);
    TEST(approx_equal(ash.value(), std::asinh(x0)), "asinh value");
    TEST(approx_equal(ash.dual(), 1.0/std::sqrt(1.0+x0*x0)), "asinh derivative");

    // acosh for x>1
    double x1 = 1.5;
    auto x_acosh = Dual<double>::variable(x1);
    auto ach = acosh(x_acosh);
    TEST(approx_equal(ach.value(), std::acosh(x1)), "acosh value");
    TEST(approx_equal(ach.dual(), 1.0/std::sqrt(x1*x1-1.0)), "acosh derivative");

    // atanh for |x|<1
    double xa = 0.3;
    auto x_atanh = Dual<double>::variable(xa);
    auto ath = atanh(x_atanh);
    TEST(approx_equal(ath.value(), std::atanh(xa)), "atanh value");
    TEST(approx_equal(ath.dual(), 1.0/(1.0-xa*xa)), "atanh derivative");

    // exp
    auto e = exp(x);
    double exp_val = std::exp(x0);
    TEST(approx_equal(e.value(), exp_val), "exp value");
    TEST(approx_equal(e.dual(), exp_val), "exp derivative");

    // exp2
    auto e2 = exp2(x);
    double exp2_val = std::exp2(x0);
    TEST(approx_equal(e2.value(), exp2_val), "exp2 value");
    TEST(approx_equal(e2.dual(), exp2_val * std::log(2.0)), "exp2 derivative");

    // expm1
    auto em1 = expm1(x);
    TEST(approx_equal(em1.value(), std::expm1(x0)), "expm1 value");
    TEST(approx_equal(em1.dual(), exp_val), "expm1 derivative");

    // log
    auto l = log(x);
    TEST(approx_equal(l.value(), std::log(x0)), "log value");
    TEST(approx_equal(l.dual(), 1.0/x0), "log derivative");

    // log2
    auto l2 = log2(x);
    TEST(approx_equal(l2.value(), std::log2(x0)), "log2 value");
    TEST(approx_equal(l2.dual(), 1.0/(x0*std::log(2.0))), "log2 derivative");

    // log10
    auto l10 = log10(x);
    TEST(approx_equal(l10.value(), std::log10(x0)), "log10 value");
    TEST(approx_equal(l10.dual(), 1.0/(x0*std::log(10.0))), "log10 derivative");

    // log1p
    auto lp = log1p(x);
    TEST(approx_equal(lp.value(), std::log1p(x0)), "log1p value");
    TEST(approx_equal(lp.dual(), 1.0/(1.0+x0)), "log1p derivative");

    // sqrt
    auto sq = sqrt(x);
    double sqrt_val = std::sqrt(x0);
    TEST(approx_equal(sq.value(), sqrt_val), "sqrt value");
    TEST(approx_equal(sq.dual(), 1.0/(2.0*sqrt_val)), "sqrt derivative");

    // cbrt
    auto cb = cbrt(x);
    double cbrt_val = std::cbrt(x0);
    TEST(approx_equal(cb.value(), cbrt_val), "cbrt value");
    TEST(approx_equal(cb.dual(), 1.0/(3.0*cbrt_val*cbrt_val)), "cbrt derivative");

    // pow with dual base and scalar exponent
    double expo = 2.5;
    auto p1 = pow(x, expo);
    TEST(approx_equal(p1.value(), std::pow(x0, expo)), "pow(base,scalar) value");
    TEST(approx_equal(p1.dual(), expo * std::pow(x0, expo-1.0)), "pow(base,scalar) derivative");

    // pow with scalar base and dual exponent
    double base_s = 2.0;
    auto p2 = pow(base_s, x);
    TEST(approx_equal(p2.value(), std::pow(base_s, x0)), "pow(scalar,exponent) value");
    TEST(approx_equal(p2.dual(), std::pow(base_s, x0) * std::log(base_s)), "pow(scalar,exponent) derivative");

    // pow with both dual (use x^x as test)
    auto p3 = pow(x, x);
    double expected = std::pow(x0, x0);
    double d = expected * (1.0 + std::log(x0));
    TEST(approx_equal(p3.value(), expected, 1e-10), "pow(both) value");
    TEST(approx_equal(p3.dual(), d, 1e-10), "pow(both) derivative");

    // hypot
    auto u = Dual<double>::variable(3.0);
    auto v = Dual<double>::variable(4.0);
    auto h = hypot(u, v);
    TEST(approx_equal(h.value(), 5.0), "hypot value");
    TEST(approx_equal(h.dual(), (3.0*1.0 + 4.0*1.0)/5.0), "hypot derivative");
    auto h_scalar1 = hypot(u, 4.0);
    TEST(approx_equal(h_scalar1.dual(), 3.0/5.0), "hypot(dual,scalar) derivative");
    auto h_scalar2 = hypot(3.0, v);
    TEST(approx_equal(h_scalar2.dual(), 4.0/5.0), "hypot(scalar,dual) derivative");

    // floor, ceil, trunc, round (derivatives zero)
    Dual<double> xf(1.7, 2.0);
    auto fl = floor(xf);
    TEST(fl.value() == 1.0 && fl.dual() == 0.0, "floor");
    auto cl = ceil(xf);
    TEST(cl.value() == 2.0 && cl.dual() == 0.0, "ceil");
    auto tr = trunc(xf);
    TEST(tr.value() == 1.0 && tr.dual() == 0.0, "trunc");
    auto rnd = round(xf);
    TEST(rnd.value() == 2.0 && rnd.dual() == 0.0, "round");

    // ldexp, scalbn, scalbln
    Dual<double> xl(1.5, 0.5);
    auto ld = ldexp(xl, 3);
    TEST(approx_equal(ld.value(), 12.0) && approx_equal(ld.dual(), 4.0), "ldexp");
    auto sc = scalbn(xl, -1);
    TEST(approx_equal(sc.value(), 0.75) && approx_equal(sc.dual(), 0.25), "scalbn");
    auto scl = scalbln(xl, 2L);
    TEST(approx_equal(scl.value(), 6.0) && approx_equal(scl.dual(), 2.0), "scalbln");

    // fma - define a and b locally
    Dual<double> afma(2.0, 1.0);
    Dual<double> bfma(3.0, 0.5);
    auto fma_res = fma(afma, bfma, Dual<double>(1.0)); // afma*bfma+1
    TEST(approx_equal(fma_res.value(), 7.0), "fma value");
    TEST(approx_equal(fma_res.dual(), 1.0*3.0 + 2.0*0.5), "fma dual");

    // erf, erfc
    auto erf_h = erf(x);
    double erf_val = std::erf(x0);
    double erf_d1 = 2.0/std::sqrt(M_PI) * std::exp(-x0*x0);
    TEST(approx_equal(erf_h.value(), erf_val), "erf value");
    TEST(approx_equal(erf_h.dual(), erf_d1), "erf derivative");

    auto erfc_h = erfc(x);
    TEST(approx_equal(erfc_h.value(), std::erfc(x0)), "erfc value");
    TEST(approx_equal(erfc_h.dual(), -erf_d1), "erfc derivative");

    // abs
    Dual<double> xneg(-2.0, 1.0);
    auto abs_h = abs(xneg);
    TEST(approx_equal(abs_h.value(), 2.0), "abs value");
    TEST(approx_equal(abs_h.dual(), -1.0), "abs derivative (sign)");
    Dual<double> xzero(0.0, 1.0);
    auto abs_z = abs(xzero);
    TEST(approx_equal(abs_z.value(), 0.0) && abs_z.dual() == 0.0, "abs(0) derivative set to 0");

    std::cout << "Function tests passed.\n";
    return 0;
}

// Test stream output
int test_stream() {
    std::cout << "\nTesting stream output...\n";
    Dual<double> d(2.5, -1.75);
    std::ostringstream oss;
    oss << d;
    std::string s = oss.str();
    // Expected: "2.5 - 1.75 * eps"
    TEST(s.find("2.5") != std::string::npos, "stream contains value");
    TEST(s.find("1.75") != std::string::npos, "stream contains absolute dual");
    TEST(s.find("eps") != std::string::npos, "stream contains eps");
    std::cout << "Stream output: " << s << std::endl;
    return 0;
}

int main() {
    std::cout << std::setprecision(12);
    if (test_basics() != 0) return 1;
    if (test_arithmetic() != 0) return 1;
    if (test_comparisons() != 0) return 1;
    if (test_functions() != 0) return 1;
    if (test_stream() != 0) return 1;
    std::cout << "\nAll Dual library tests passed successfully.\n";
    return 0;
}
