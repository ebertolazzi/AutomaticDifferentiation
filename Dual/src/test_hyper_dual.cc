// test_hyper_dual.cc
// Comprehensive test suite for the HyperDual class.

#include "hyper_dual.hh"
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

int test_arithmetic() {
    std::cout << "Testing arithmetic operations...\n";

    HyperDual<double> a(2.0, 1.0, 0.5, 0.25);
    HyperDual<double> b(3.0, 0.5, 1.0, 0.75);

    // Addition
    auto sum = a + b;
    TEST(approx_equal(sum.value(), 5.0), "a+b value");
    TEST(approx_equal(sum.eps1(), 1.5), "a+b eps1");
    TEST(approx_equal(sum.eps2(), 1.5), "a+b eps2");
    TEST(approx_equal(sum.eps12(), 1.0), "a+b eps12");

    // Subtraction
    auto diff = a - b;
    TEST(approx_equal(diff.value(), -1.0), "a-b value");
    TEST(approx_equal(diff.eps1(), 0.5), "a-b eps1");
    TEST(approx_equal(diff.eps2(), -0.5), "a-b eps2");
    TEST(approx_equal(diff.eps12(), -0.5), "a-b eps12");

    // Multiplication
    auto prod = a * b;
    TEST(approx_equal(prod.value(), 6.0), "a*b value");
    TEST(approx_equal(prod.eps1(), 2.0*0.5 + 1.0*3.0), "a*b eps1");
    TEST(approx_equal(prod.eps2(), 2.0*1.0 + 0.5*3.0), "a*b eps2");
    double expected_eps12 = 0.25*3.0 + 1.0*1.0 + 0.5*0.5 + 2.0*0.75;
    TEST(approx_equal(prod.eps12(), expected_eps12), "a*b eps12");

    // Division
    auto quot = a / b;
    double inv_b_val = 1.0/3.0;
    double inv_b_eps1 = -0.5/9.0;
    double inv_b_eps2 = -1.0/9.0;
    double inv_b_eps12 = (2.0*0.5*1.0/27.0) - (0.75/9.0);
    double expected_val = 2.0 * inv_b_val;
    double expected_eps1 = 1.0*inv_b_val + 2.0*inv_b_eps1;
    double expected_eps2 = 0.5*inv_b_val + 2.0*inv_b_eps2;
    double expected_eps12_val = 0.25*inv_b_val + 1.0*inv_b_eps2 + 0.5*inv_b_eps1 + 2.0*inv_b_eps12;
    TEST(approx_equal(quot.value(), expected_val), "a/b value");
    TEST(approx_equal(quot.eps1(), expected_eps1), "a/b eps1");
    TEST(approx_equal(quot.eps2(), expected_eps2), "a/b eps2");
    TEST(approx_equal(quot.eps12(), expected_eps12_val), "a/b eps12");

    // Scalar operations
    auto scalar_add = a + 5.0;
    TEST(approx_equal(scalar_add.value(), 7.0), "a+5 value");
    TEST(approx_equal(scalar_add.eps1(), 1.0), "a+5 eps1");
    auto scalar_sub = 5.0 - a;
    TEST(approx_equal(scalar_sub.value(), 3.0), "5-a value");
    TEST(approx_equal(scalar_sub.eps1(), -1.0), "5-a eps1");
    auto scalar_mul = a * 2.0;
    TEST(approx_equal(scalar_mul.value(), 4.0), "a*2 value");
    TEST(approx_equal(scalar_mul.eps1(), 2.0), "a*2 eps1");
    auto scalar_div = a / 2.0;
    TEST(approx_equal(scalar_div.value(), 1.0), "a/2 value");
    TEST(approx_equal(scalar_div.eps1(), 0.5), "a/2 eps1");

    // Compound assignments
    HyperDual<double> c = a;
    c += b;
    TEST(approx_equal(c.value(), 5.0), "c+=b value");
    c = a;
    c -= b;
    TEST(approx_equal(c.value(), -1.0), "c-=b value");
    c = a;
    c *= b;
    TEST(approx_equal(c.value(), 6.0), "c*=b value");
    c = a;
    c /= b;
    TEST(approx_equal(c.value(), expected_val), "c/=b value");

    // Unary minus
    auto neg = -a;
    TEST(approx_equal(neg.value(), -2.0), "-a value");
    TEST(approx_equal(neg.eps1(), -1.0), "-a eps1");

    // Comparison
    TEST(a == a, "equality self");
    TEST(!(a == b), "inequality a vs b");
    TEST(a != b, "inequality not");

    // Equality with scalar: only true if dual parts are zero
    HyperDual<double> a_real(2.0, 0.0, 0.0, 0.0);
    TEST(a_real == 2.0, "equality with scalar (dual parts zero)");
    TEST(2.0 == a_real, "scalar equality (dual parts zero)");
    TEST(!(a == 2.0), "non-zero dual parts prevent equality");
    TEST(!(a == 3.0), "different primal value");

    std::cout << "Arithmetic tests passed.\n";
    return 0;
}

int test_functions() {
    std::cout << "\nTesting mathematical functions...\n";

    double x0 = 0.5;
    HyperDual<double> x = HyperDual<double>::variable_2nd(x0);

    auto s = sin(x);
    TEST(approx_equal(s.value(), std::sin(x0)), "sin value");
    TEST(approx_equal(s.eps1(), std::cos(x0)), "sin derivative");
    TEST(approx_equal(s.eps12(), -std::sin(x0)), "sin second derivative");

    auto c = cos(x);
    TEST(approx_equal(c.value(), std::cos(x0)), "cos value");
    TEST(approx_equal(c.eps1(), -std::sin(x0)), "cos derivative");
    TEST(approx_equal(c.eps12(), -std::cos(x0)), "cos second derivative");

    auto t = tan(x);
    double sec2 = 1.0 / (std::cos(x0)*std::cos(x0));
    double tan2 = std::tan(x0);
    TEST(approx_equal(t.value(), std::tan(x0)), "tan value");
    TEST(approx_equal(t.eps1(), sec2), "tan derivative");
    TEST(approx_equal(t.eps12(), 2.0 * sec2 * tan2), "tan second derivative");

    auto e = exp(x);
    double exp_val = std::exp(x0);
    TEST(approx_equal(e.value(), exp_val), "exp value");
    TEST(approx_equal(e.eps1(), exp_val), "exp derivative");
    TEST(approx_equal(e.eps12(), exp_val), "exp second derivative");

    auto l = log(x);
    TEST(approx_equal(l.value(), std::log(x0)), "log value");
    TEST(approx_equal(l.eps1(), 1.0/x0), "log derivative");
    TEST(approx_equal(l.eps12(), -1.0/(x0*x0)), "log second derivative");

    auto sq = sqrt(x);
    double sqrt_val = std::sqrt(x0);
    TEST(approx_equal(sq.value(), sqrt_val), "sqrt value");
    TEST(approx_equal(sq.eps1(), 0.5/sqrt_val), "sqrt derivative");
    TEST(approx_equal(sq.eps12(), -0.25/(x0*sqrt_val)), "sqrt second derivative");

    auto erf_h = erf(x);
    double erf_val = std::erf(x0);
    double erf_d1 = 2.0/std::sqrt(M_PI) * std::exp(-x0*x0);
    double erf_d2 = -2.0*x0 * erf_d1;
    TEST(approx_equal(erf_h.value(), erf_val), "erf value");
    TEST(approx_equal(erf_h.eps1(), erf_d1), "erf derivative");
    TEST(approx_equal(erf_h.eps12(), erf_d2), "erf second derivative");

    std::cout << "Function tests passed.\n";
    return 0;
}

int test_two_arg_functions() {
    std::cout << "\nTesting two-argument functions...\n";

    HyperDual<double> u = HyperDual<double>::variable1(3.0);
    HyperDual<double> v = HyperDual<double>::variable2(4.0);
    auto h = hypot(u, v);
    TEST(approx_equal(h.value(), 5.0), "hypot value");
    TEST(approx_equal(h.eps1(), 3.0/5.0), "hypot derivative w.r.t u");
    TEST(approx_equal(h.eps2(), 4.0/5.0), "hypot derivative w.r.t v");
    TEST(approx_equal(h.eps12(), -3.0*4.0/(125.0)), "hypot mixed second");

    double x0 = 0.5;
    HyperDual<double> y = HyperDual<double>::variable1(x0*x0);
    HyperDual<double> x2 = HyperDual<double>::variable2(x0);
    auto a2 = atan2(y, x2);
    double denom = x0*x0 + x0*x0*x0*x0;
    double d1y = x0 / denom;
    double d1x = -x0*x0 / denom;
    double d2xy = (x0*x0*x0*x0 - x0*x0) / (denom*denom);
    TEST(approx_equal(a2.value(), std::atan(x0)), "atan2 value");
    TEST(approx_equal(a2.eps1(), d1y), "atan2 derivative w.r.t y");
    TEST(approx_equal(a2.eps2(), d1x), "atan2 derivative w.r.t x");
    TEST(approx_equal(a2.eps12(), d2xy), "atan2 mixed second derivative");

    std::cout << "Two-argument function tests passed.\n";
    return 0;
}

int test_stream() {
    std::cout << "\nTesting stream output...\n";
    HyperDual<double> h(1.5, 0.25, -0.5, 0.75);
    std::ostringstream oss;
    oss << h;
    std::string s = oss.str();
    TEST(s.find("1.5") != std::string::npos, "stream contains value");
    TEST(s.find("ε₁") != std::string::npos, "stream contains ε₁");
    TEST(s.find("ε₂") != std::string::npos, "stream contains ε₂");
    TEST(s.find("ε₁ε₂") != std::string::npos, "stream contains ε₁ε₂");
    std::cout << "Stream output: " << s << std::endl;
    return 0;
}

int main() {
    std::cout << std::setprecision(12);
    if (test_arithmetic() != 0) return 1;
    if (test_functions() != 0) return 1;
    if (test_two_arg_functions() != 0) return 1;
    if (test_stream() != 0) return 1;
    std::cout << "\nAll HyperDual tests passed successfully.\n";
    return 0;
}

