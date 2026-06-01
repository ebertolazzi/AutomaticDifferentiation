#pragma once

#ifndef HYPER_DUAL_HH
#define HYPER_DUAL_HH

#include <cmath>
#include <ostream>

/**
 * @file HyperDual.hh
 * @brief Hyper-dual number implementation for forward-mode automatic differentiation,
 *        enabling computation of first and second derivatives simultaneously.
 */

namespace AD {

namespace detail {

    /**
     * @brief Returns the square of a value.
     */
    template <typename T>
    constexpr inline T square(T const &x) noexcept(noexcept(x * x)) {
        return x * x;
    }

    /**
     * @brief Returns the constant factor in the derivative of `erf`.
     */
    template <typename T>
    inline T erf_scale() {
        using std::acos;
        using std::sqrt;
        return T(2) / sqrt(acos(T(-1)));
    }

    /**
     * @brief Returns the cube of a value.
     */
    template <typename T>
    constexpr inline T cube(T const &x) noexcept(noexcept(x * x * x)) {
        return x * x * x;
    }

} // namespace detail

/**
 * @brief Hyper-dual number of the form `a + b*ε₁ + c*ε₂ + d*ε₁ε₂`.
 *
 * Hyper-dual numbers extend dual numbers with two nilpotent units ε₁ and ε₂
 * such that ε₁² = ε₂² = 0 and ε₁ε₂ = ε₂ε₁ ≠ 0. This allows computing both
 * first derivatives (coefficients of ε₁ and ε₂) and the second derivative
 * (coefficient of ε₁ε₂) of a function in a single evaluation.
 *
 * @tparam T underlying scalar type.
 */
template <typename T>
class HyperDual {
public:
    /// Underlying scalar type.
    using value_type = T;

private:
    T _val  = T(0);   ///< Primal value.
    T _eps1 = T(0);   ///< Coefficient of ε₁.
    T _eps2 = T(0);   ///< Coefficient of ε₂.
    T _eps12 = T(0);  ///< Coefficient of ε₁ε₂.

public:
    /// Default constructor (zero).
    constexpr HyperDual() noexcept = default;

    /// Copy constructor.
    constexpr HyperDual(HyperDual const &) noexcept = default;

    /// Move constructor.
    constexpr HyperDual(HyperDual &&) noexcept = default;

    /**
     * @brief Construct a hyper-dual number from its components.
     * @param value primal value.
     * @param eps1 coefficient of ε₁.
     * @param eps2 coefficient of ε₂.
     * @param eps12 coefficient of ε₁ε₂.
     */
    constexpr explicit HyperDual(T value, T eps1 = T(0), T eps2 = T(0), T eps12 = T(0)) noexcept
        : _val(value), _eps1(eps1), _eps2(eps2), _eps12(eps12) {}

    /// Copy assignment.
    constexpr HyperDual &operator=(HyperDual const &) noexcept = default;

    /// Move assignment.
    constexpr HyperDual &operator=(HyperDual &&) noexcept = default;

    /**
     * @brief Create an independent variable with respect to the first direction (ε₁).
     * @param value variable value.
     * @return HyperDual with derivative 1 w.r.t. ε₁.
     */
    [[nodiscard]] static constexpr HyperDual variable1(T value) noexcept {
        return HyperDual(value, T(1), T(0), T(0));
    }

    /**
     * @brief Create an independent variable with respect to the second direction (ε₂).
     * @param value variable value.
     * @return HyperDual with derivative 1 w.r.t. ε₂.
     */
    [[nodiscard]] static constexpr HyperDual variable2(T value) noexcept {
        return HyperDual(value, T(0), T(1), T(0));
    }

    /**
     * @brief Create a variable for univariate second derivative computation.
     * Both ε₁ and ε₂ are set to 1; the ε₁ε₂ coefficient will contain f''(x).
     * @param value variable value.
     * @return HyperDual with eps1=1, eps2=1, eps12=0.
     */
    [[nodiscard]] static constexpr HyperDual variable_2nd(T value) noexcept {
        return HyperDual(value, T(1), T(1), T(0));
    }

    /// Returns the primal value.
    [[nodiscard]] constexpr T value() const noexcept { return _val; }

    /// Returns the coefficient of ε₁ (first derivative w.r.t. first direction).
    [[nodiscard]] constexpr T eps1() const noexcept { return _eps1; }

    /// Returns the coefficient of ε₂ (first derivative w.r.t. second direction).
    [[nodiscard]] constexpr T eps2() const noexcept { return _eps2; }

    /// Returns the coefficient of ε₁ε₂ (mixed second derivative / second derivative).
    [[nodiscard]] constexpr T eps12() const noexcept { return _eps12; }

    /**
     * @brief Set all components at once.
     */
    constexpr void set(T value, T eps1, T eps2, T eps12) noexcept {
        _val = value;
        _eps1 = eps1;
        _eps2 = eps2;
        _eps12 = eps12;
    }

    /**
     * @brief Assign a scalar; dual parts are reset to zero.
     */
    constexpr HyperDual &operator=(T value) noexcept {
        _val = value;
        _eps1 = _eps2 = _eps12 = T(0);
        return *this;
    }

    // Compound assignment operators -----------------------------------------

    /// Add a hyper-dual number.
    constexpr HyperDual &operator+=(HyperDual const &rhs) noexcept {
        _val += rhs._val;
        _eps1 += rhs._eps1;
        _eps2 += rhs._eps2;
        _eps12 += rhs._eps12;
        return *this;
    }

    /// Add a scalar.
    constexpr HyperDual &operator+=(T rhs) noexcept {
        _val += rhs;
        return *this;
    }

    /// Subtract a hyper-dual number.
    constexpr HyperDual &operator-=(HyperDual const &rhs) noexcept {
        _val -= rhs._val;
        _eps1 -= rhs._eps1;
        _eps2 -= rhs._eps2;
        _eps12 -= rhs._eps12;
        return *this;
    }

    /// Subtract a scalar.
    constexpr HyperDual &operator-=(T rhs) noexcept {
        _val -= rhs;
        return *this;
    }

    /// Multiply by a hyper-dual number.
    constexpr HyperDual &operator*=(HyperDual const &rhs) noexcept {
        T const val  = _val;
        T const e1   = _eps1;
        T const e2   = _eps2;
        T const e12  = _eps12;
        _val   = val * rhs._val;
        _eps1  = e1 * rhs._val + val * rhs._eps1;
        _eps2  = e2 * rhs._val + val * rhs._eps2;
        _eps12 = e12 * rhs._val + e1 * rhs._eps2 + e2 * rhs._eps1 + val * rhs._eps12;
        return *this;
    }

    /// Multiply by a scalar.
    constexpr HyperDual &operator*=(T rhs) noexcept {
        _val *= rhs;
        _eps1 *= rhs;
        _eps2 *= rhs;
        _eps12 *= rhs;
        return *this;
    }

    /// Divide by a hyper-dual number.
    constexpr HyperDual &operator/=(HyperDual const &rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }

    /// Divide by a scalar.
    constexpr HyperDual &operator/=(T rhs) noexcept {
        _val /= rhs;
        _eps1 /= rhs;
        _eps2 /= rhs;
        _eps12 /= rhs;
        return *this;
    }

    /// Unary plus.
    [[nodiscard]] constexpr HyperDual operator+() const noexcept {
        return *this;
    }

    /// Unary minus.
    [[nodiscard]] constexpr HyperDual operator-() const noexcept {
        return HyperDual(-_val, -_eps1, -_eps2, -_eps12);
    }
};

// Binary operators ---------------------------------------------------------

/// Addition of two hyper-dual numbers.
template <typename T>
[[nodiscard]] constexpr inline HyperDual<T>
operator+(HyperDual<T> const &lhs, HyperDual<T> const &rhs) noexcept {
    return HyperDual<T>(lhs.value() + rhs.value(),
                        lhs.eps1() + rhs.eps1(),
                        lhs.eps2() + rhs.eps2(),
                        lhs.eps12() + rhs.eps12());
}

/// Addition of scalar and hyper-dual number.
template <typename T>
[[nodiscard]] constexpr inline HyperDual<T>
operator+(T lhs, HyperDual<T> const &rhs) noexcept {
    return HyperDual<T>(lhs + rhs.value(), rhs.eps1(), rhs.eps2(), rhs.eps12());
}

/// Addition of hyper-dual number and scalar.
template <typename T>
[[nodiscard]] constexpr inline HyperDual<T>
operator+(HyperDual<T> const &lhs, T rhs) noexcept {
    return HyperDual<T>(lhs.value() + rhs, lhs.eps1(), lhs.eps2(), lhs.eps12());
}

/// Subtraction of two hyper-dual numbers.
template <typename T>
[[nodiscard]] constexpr inline HyperDual<T>
operator-(HyperDual<T> const &lhs, HyperDual<T> const &rhs) noexcept {
    return HyperDual<T>(lhs.value() - rhs.value(),
                        lhs.eps1() - rhs.eps1(),
                        lhs.eps2() - rhs.eps2(),
                        lhs.eps12() - rhs.eps12());
}

/// Subtraction of hyper-dual number from scalar.
template <typename T>
[[nodiscard]] constexpr inline HyperDual<T>
operator-(T lhs, HyperDual<T> const &rhs) noexcept {
    return HyperDual<T>(lhs - rhs.value(), -rhs.eps1(), -rhs.eps2(), -rhs.eps12());
}

/// Subtraction of scalar from hyper-dual number.
template <typename T>
[[nodiscard]] constexpr inline HyperDual<T>
operator-(HyperDual<T> const &lhs, T rhs) noexcept {
    return HyperDual<T>(lhs.value() - rhs, lhs.eps1(), lhs.eps2(), lhs.eps12());
}

/// Multiplication of two hyper-dual numbers.
template <typename T>
[[nodiscard]] constexpr inline HyperDual<T>
operator*(HyperDual<T> const &lhs, HyperDual<T> const &rhs) noexcept {
    return HyperDual<T>(
        lhs.value() * rhs.value(),
        lhs.eps1() * rhs.value() + lhs.value() * rhs.eps1(),
        lhs.eps2() * rhs.value() + lhs.value() * rhs.eps2(),
        lhs.eps12() * rhs.value() + lhs.eps1() * rhs.eps2() +
            lhs.eps2() * rhs.eps1() + lhs.value() * rhs.eps12()
    );
}

/// Multiplication of hyper-dual number by scalar.
template <typename T>
[[nodiscard]] constexpr inline HyperDual<T>
operator*(HyperDual<T> const &lhs, T rhs) noexcept {
    return HyperDual<T>(lhs.value() * rhs, lhs.eps1() * rhs, lhs.eps2() * rhs, lhs.eps12() * rhs);
}

/// Multiplication of scalar by hyper-dual number.
template <typename T>
[[nodiscard]] constexpr inline HyperDual<T>
operator*(T lhs, HyperDual<T> const &rhs) noexcept {
    return HyperDual<T>(lhs * rhs.value(), lhs * rhs.eps1(), lhs * rhs.eps2(), lhs * rhs.eps12());
}

/// Division of two hyper-dual numbers.
template <typename T>
[[nodiscard]] inline HyperDual<T> operator/(HyperDual<T> const &lhs, HyperDual<T> const &rhs) {
    // Compute inverse of rhs
    T const denom = rhs.value() * rhs.value();
    T const denom3 = denom * rhs.value(); // (value)^3
    T const inv_val = T(1) / rhs.value();
    T const inv_eps1 = -rhs.eps1() / denom;
    T const inv_eps2 = -rhs.eps2() / denom;
    T const inv_eps12 = (T(2) * rhs.eps1() * rhs.eps2() / denom3) - (rhs.eps12() / denom);
    HyperDual<T> const inv(inv_val, inv_eps1, inv_eps2, inv_eps12);
    return lhs * inv;
}

/// Division of hyper-dual number by scalar.
template <typename T>
[[nodiscard]] constexpr inline HyperDual<T>
operator/(HyperDual<T> const &lhs, T rhs) noexcept {
    return HyperDual<T>(lhs.value() / rhs, lhs.eps1() / rhs, lhs.eps2() / rhs, lhs.eps12() / rhs);
}

/// Division of scalar by hyper-dual number.
template <typename T>
[[nodiscard]] inline HyperDual<T> operator/(T lhs, HyperDual<T> const &rhs) {
    return HyperDual<T>(lhs) / rhs;
}

// Comparison operators -----------------------------------------------------

/// Exact equality of two hyper-dual numbers.
template <typename T>
constexpr inline bool operator==(HyperDual<T> const &lhs, HyperDual<T> const &rhs) noexcept {
    return lhs.value() == rhs.value() && lhs.eps1() == rhs.eps1() &&
           lhs.eps2() == rhs.eps2() && lhs.eps12() == rhs.eps12();
}

/// Equality with scalar (dual parts must be zero).
template <typename T>
constexpr inline bool operator==(HyperDual<T> const &lhs, T rhs) noexcept {
    return lhs.value() == rhs && lhs.eps1() == T(0) && lhs.eps2() == T(0) && lhs.eps12() == T(0);
}

/// Equality with scalar (symmetry).
template <typename T>
constexpr inline bool operator==(T lhs, HyperDual<T> const &rhs) noexcept {
    return rhs == lhs;
}

/// Inequality.
template <typename T>
constexpr inline bool operator!=(HyperDual<T> const &lhs, HyperDual<T> const &rhs) noexcept {
    return !(lhs == rhs);
}

template <typename T>
constexpr inline bool operator!=(HyperDual<T> const &lhs, T rhs) noexcept {
    return !(lhs == rhs);
}

template <typename T>
constexpr inline bool operator!=(T lhs, HyperDual<T> const &rhs) noexcept {
    return !(lhs == rhs);
}

// Stream output ------------------------------------------------------------

/**
 * @brief Print hyper-dual number in the form "a + b*ε₁ + c*ε₂ + d*ε₁ε₂".
 */
template <typename CharT, typename Traits, typename T>
inline std::basic_ostream<CharT, Traits> &
operator<<(std::basic_ostream<CharT, Traits> &stream, HyperDual<T> const &h) {
    using std::abs;
    stream << h.value();
    if (h.eps1() != T(0)) {
        stream << (h.eps1() >= T(0) ? " + " : " - ") << abs(h.eps1()) << "*ε₁";
    }
    if (h.eps2() != T(0)) {
        stream << (h.eps2() >= T(0) ? " + " : " - ") << abs(h.eps2()) << "*ε₂";
    }
    if (h.eps12() != T(0)) {
        stream << (h.eps12() >= T(0) ? " + " : " - ") << abs(h.eps12()) << "*ε₁ε₂";
    }
    return stream;
}

// Elementary functions -----------------------------------------------------

/**
 * @brief Absolute value.
 * @note Derivative is sign(x) (0 at x=0).
 */
template <typename T>
inline HyperDual<T> abs(HyperDual<T> const &x) {
    using std::abs;
    using std::copysign;
    T const val = abs(x.value());
    T const d1 = (x.value() == T(0)) ? T(0) : copysign(T(1), x.value());
    T const d2 = T(0); // second derivative of abs is zero except at 0 where it's undefined
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Sine.
 */
template <typename T>
inline HyperDual<T> sin(HyperDual<T> const &x) {
    using std::cos;
    using std::sin;
    T const val = sin(x.value());
    T const d1  = cos(x.value());
    T const d2  = -val;
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Cosine.
 */
template <typename T>
inline HyperDual<T> cos(HyperDual<T> const &x) {
    using std::cos;
    using std::sin;
    T const val = cos(x.value());
    T const d1  = -sin(x.value());
    T const d2  = -val;
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Tangent.
 */
template <typename T>
inline HyperDual<T> tan(HyperDual<T> const &x) {
    using std::cos;
    using std::tan;
    T const c = cos(x.value());
    T const val = tan(x.value());
    T const d1  = T(1) / detail::square(c);
    T const d2  = T(2) * val / detail::square(c); // derivative of sec^2(x) = 2 sec^2(x) tan(x)
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Arcsine.
 */
template <typename T>
inline HyperDual<T> asin(HyperDual<T> const &x) {
    using std::asin;
    using std::sqrt;
    T const val = asin(x.value());
    T const d1  = T(1) / sqrt(T(1) - detail::square(x.value()));
    T const d2  = x.value() * detail::cube(d1); // derivative of d1 = x / (1-x^2)^{3/2}
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Arccosine.
 */
template <typename T>
inline HyperDual<T> acos(HyperDual<T> const &x) {
    using std::acos;
    using std::sqrt;
    T const val = acos(x.value());
    T const d1  = -T(1) / sqrt(T(1) - detail::square(x.value()));
    T const d2  = -x.value() * detail::cube(-d1); // derivative of -1/√(1-x^2)
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Arctangent.
 */
template <typename T>
inline HyperDual<T> atan(HyperDual<T> const &x) {
    using std::atan;
    T const val = atan(x.value());
    T const d1  = T(1) / (T(1) + detail::square(x.value()));
    T const d2  = -T(2) * x.value() * detail::square(d1);
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Two-argument arctangent.
 */
template <typename T>
inline HyperDual<T> atan2(HyperDual<T> const &y, HyperDual<T> const &x) {
    using std::atan2;
    T const val = atan2(y.value(), x.value());
    T const denom = detail::square(x.value()) + detail::square(y.value());
    T const d1x = -y.value() / denom;
    T const d1y =  x.value() / denom;
    // Second derivatives (only mixed term needed for ε₁ε₂ coefficient)
    T const d2xx =  T(2) * x.value() * y.value() / detail::square(denom);
    T const d2yy = -T(2) * x.value() * y.value() / detail::square(denom);
    T const d2xy = (detail::square(y.value()) - detail::square(x.value())) / detail::square(denom);
    // For hyper-dual numbers, the ε₁ε₂ coefficient is:
    // f'(x)·x_12 + f'(y)·y_12 + ½ f''(x,x)·x_1·x_2 + ½ f''(y,y)·y_1·y_2 + f''(x,y)·(x_1·y_2 + x_2·y_1)/? Actually careful:
    // The second-order term from Taylor: ½ (f_xx (dx)^2 + 2 f_xy dx dy + f_yy (dy)^2). With dx = x_eps1 ε1 + x_eps2 ε2 + x_eps12 ε1ε2? But we only retain ε1ε2 terms: contributions from dx²: 2 x_eps1 x_eps2 ε1ε2, dy²: 2 y_eps1 y_eps2 ε1ε2, dx dy: (x_eps1 y_eps2 + x_eps2 y_eps1) ε1ε2. So total ε1ε2 from second order = f_xx * x_eps1 x_eps2 + f_yy * y_eps1 y_eps2 + f_xy * (x_eps1 y_eps2 + x_eps2 y_eps1). The formula for hyper-dual then: new_eps12 = f_x * x_eps12 + f_y * y_eps12 + f_xx * x_eps1 x_eps2 + f_yy * y_eps1 y_eps2 + f_xy * (x_eps1 y_eps2 + x_eps2 y_eps1).
    T const eps12 = d1x * x.eps12() + d1y * y.eps12()
                    + d2xx * x.eps1() * x.eps2()
                    + d2yy * y.eps1() * y.eps2()
                    + d2xy * (x.eps1() * y.eps2() + x.eps2() * y.eps1());
    return HyperDual<T>(val,
                        d1x * x.eps1() + d1y * y.eps1(),
                        d1x * x.eps2() + d1y * y.eps2(),
                        eps12);
}

// Scalar versions of atan2
template <typename T>
inline HyperDual<T> atan2(HyperDual<T> const &y, T x) {
    return atan2(y, HyperDual<T>(x));
}

template <typename T>
inline HyperDual<T> atan2(T y, HyperDual<T> const &x) {
    return atan2(HyperDual<T>(y), x);
}

/**
 * @brief Hyperbolic sine.
 */
template <typename T>
inline HyperDual<T> sinh(HyperDual<T> const &x) {
    using std::cosh;
    using std::sinh;
    T const val = sinh(x.value());
    T const d1  = cosh(x.value());
    T const d2  = val;
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Hyperbolic cosine.
 */
template <typename T>
inline HyperDual<T> cosh(HyperDual<T> const &x) {
    using std::cosh;
    using std::sinh;
    T const val = cosh(x.value());
    T const d1  = sinh(x.value());
    T const d2  = val;
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Hyperbolic tangent.
 */
template <typename T>
inline HyperDual<T> tanh(HyperDual<T> const &x) {
    using std::cosh;
    using std::tanh;
    T const c = cosh(x.value());
    T const val = tanh(x.value());
    T const d1  = T(1) / detail::square(c);
    T const d2  = -T(2) * val * d1;
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Inverse hyperbolic sine.
 */
template <typename T>
inline HyperDual<T> asinh(HyperDual<T> const &x) {
    using std::asinh;
    using std::sqrt;
    T const val = asinh(x.value());
    T const d1  = T(1) / sqrt(T(1) + detail::square(x.value()));
    T const d2  = -x.value() * detail::cube(d1);
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Inverse hyperbolic cosine.
 */
template <typename T>
inline HyperDual<T> acosh(HyperDual<T> const &x) {
    using std::acosh;
    using std::sqrt;
    T const val = acosh(x.value());
    T const d1  = T(1) / sqrt(detail::square(x.value()) - T(1));
    T const d2  = -x.value() * detail::cube(d1);
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Inverse hyperbolic tangent.
 */
template <typename T>
inline HyperDual<T> atanh(HyperDual<T> const &x) {
    using std::atanh;
    T const val = atanh(x.value());
    T const d1  = T(1) / (T(1) - detail::square(x.value()));
    T const d2  = T(2) * x.value() * detail::square(d1);
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Exponential function.
 */
template <typename T>
inline HyperDual<T> exp(HyperDual<T> const &x) {
    using std::exp;
    T const val = exp(x.value());
    T const d1  = val;
    T const d2  = val;
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Base-2 exponential.
 */
template <typename T>
inline HyperDual<T> exp2(HyperDual<T> const &x) {
    using std::exp2;
    using std::log;
    T const val = exp2(x.value());
    T const d1  = val * log(T(2));
    T const d2  = d1 * log(T(2));
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief exp(x) - 1.
 */
template <typename T>
inline HyperDual<T> expm1(HyperDual<T> const &x) {
    using std::exp;
    using std::expm1;
    T const val = expm1(x.value());
    T const d1  = exp(x.value());
    T const d2  = d1;
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Natural logarithm.
 */
template <typename T>
inline HyperDual<T> log(HyperDual<T> const &x) {
    using std::log;
    T const val = log(x.value());
    T const d1  = T(1) / x.value();
    T const d2  = -detail::square(d1);
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Base-2 logarithm.
 */
template <typename T>
inline HyperDual<T> log2(HyperDual<T> const &x) {
    using std::log;
    using std::log2;
    T const val = log2(x.value());
    T const d1  = T(1) / (x.value() * log(T(2)));
    T const d2  = -T(1) / (detail::square(x.value()) * log(T(2)));
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Base-10 logarithm.
 */
template <typename T>
inline HyperDual<T> log10(HyperDual<T> const &x) {
    using std::log;
    using std::log10;
    T const val = log10(x.value());
    T const d1  = T(1) / (x.value() * log(T(10)));
    T const d2  = -T(1) / (detail::square(x.value()) * log(T(10)));
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief log(1 + x).
 */
template <typename T>
inline HyperDual<T> log1p(HyperDual<T> const &x) {
    using std::log1p;
    T const val = log1p(x.value());
    T const d1  = T(1) / (T(1) + x.value());
    T const d2  = -detail::square(d1);
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Square root.
 */
template <typename T>
inline HyperDual<T> sqrt(HyperDual<T> const &x) {
    using std::sqrt;
    T const val = sqrt(x.value());
    T const d1  = T(1) / (T(2) * val);
    T const d2  = -d1 / (T(2) * x.value());
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Cubic root.
 */
template <typename T>
inline HyperDual<T> cbrt(HyperDual<T> const &x) {
    using std::cbrt;
    T const val = cbrt(x.value());
    T const d1  = T(1) / (T(3) * val * val);
    T const d2  = -T(2) * d1 / (T(3) * x.value());
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Power function: base^exponent (both hyper-dual).
 * Implemented as exp(exponent * log(base)).
 */
template <typename T>
inline HyperDual<T> pow(HyperDual<T> const &base, HyperDual<T> const &exponent) {
    return exp(exponent * log(base));
}

/**
 * @brief Power function: base^exponent (hyper-dual base, scalar exponent).
 */
template <typename T>
inline HyperDual<T> pow(HyperDual<T> const &base, T exponent) {
    using std::pow;
    T const val = pow(base.value(), exponent);
    T const d1  = exponent * pow(base.value(), exponent - T(1));
    T const d2  = exponent * (exponent - T(1)) * pow(base.value(), exponent - T(2));
    return HyperDual<T>(val,
                        d1 * base.eps1(),
                        d1 * base.eps2(),
                        d1 * base.eps12() + d2 * base.eps1() * base.eps2());
}

/**
 * @brief Power function: base^exponent (scalar base, hyper-dual exponent).
 */
template <typename T>
inline HyperDual<T> pow(T base, HyperDual<T> const &exponent) {
    using std::log;
    using std::pow;
    T const val = pow(base, exponent.value());
    T const d1  = val * log(base);
    T const d2  = d1 * log(base);
    return HyperDual<T>(val,
                        d1 * exponent.eps1(),
                        d1 * exponent.eps2(),
                        d1 * exponent.eps12() + d2 * exponent.eps1() * exponent.eps2());
}

/**
 * @brief Euclidean distance: hypot(x, y).
 */
template <typename T>
inline HyperDual<T> hypot(HyperDual<T> const &x, HyperDual<T> const &y) {
    using std::hypot;
    T const val = hypot(x.value(), y.value());
    T const d1x = x.value() / val;
    T const d1y = y.value() / val;
    // Second derivatives
    T const d2xx = (y.value() * y.value()) / (val * val * val);
    T const d2yy = (x.value() * x.value()) / (val * val * val);
    T const d2xy = -x.value() * y.value() / (val * val * val);
    T const eps12 = d1x * x.eps12() + d1y * y.eps12()
                    + d2xx * x.eps1() * x.eps2()
                    + d2yy * y.eps1() * y.eps2()
                    + d2xy * (x.eps1() * y.eps2() + x.eps2() * y.eps1());
    return HyperDual<T>(val,
                        d1x * x.eps1() + d1y * y.eps1(),
                        d1x * x.eps2() + d1y * y.eps2(),
                        eps12);
}

template <typename T>
inline HyperDual<T> hypot(HyperDual<T> const &x, T y) {
    return hypot(x, HyperDual<T>(y));
}

template <typename T>
inline HyperDual<T> hypot(T x, HyperDual<T> const &y) {
    return hypot(HyperDual<T>(x), y);
}

/**
 * @brief Floor function (derivative zero wherever defined).
 */
template <typename T>
inline HyperDual<T> floor(HyperDual<T> const &x) {
    using std::floor;
    return HyperDual<T>(floor(x.value()), T(0), T(0), T(0));
}

/**
 * @brief Ceiling function (derivative zero wherever defined).
 */
template <typename T>
inline HyperDual<T> ceil(HyperDual<T> const &x) {
    using std::ceil;
    return HyperDual<T>(ceil(x.value()), T(0), T(0), T(0));
}

/**
 * @brief Truncation function (derivative zero wherever defined).
 */
template <typename T>
inline HyperDual<T> trunc(HyperDual<T> const &x) {
    using std::trunc;
    return HyperDual<T>(trunc(x.value()), T(0), T(0), T(0));
}

/**
 * @brief Round function (derivative zero wherever defined).
 */
template <typename T>
inline HyperDual<T> round(HyperDual<T> const &x) {
    using std::round;
    return HyperDual<T>(round(x.value()), T(0), T(0), T(0));
}

/**
 * @brief Multiply by 2^exponent.
 */
template <typename T>
inline HyperDual<T> ldexp(HyperDual<T> const &x, int exponent) {
    using std::ldexp;
    return HyperDual<T>(ldexp(x.value(), exponent),
                        ldexp(x.eps1(), exponent),
                        ldexp(x.eps2(), exponent),
                        ldexp(x.eps12(), exponent));
}

/**
 * @brief Multiply by FLT_RADIX^exponent.
 */
template <typename T>
inline HyperDual<T> scalbn(HyperDual<T> const &x, int exponent) {
    using std::scalbn;
    return HyperDual<T>(scalbn(x.value(), exponent),
                        scalbn(x.eps1(), exponent),
                        scalbn(x.eps2(), exponent),
                        scalbn(x.eps12(), exponent));
}

/**
 * @brief Multiply by FLT_RADIX^exponent (long exponent).
 */
template <typename T>
inline HyperDual<T> scalbln(HyperDual<T> const &x, long exponent) {
    using std::scalbln;
    return HyperDual<T>(scalbln(x.value(), exponent),
                        scalbln(x.eps1(), exponent),
                        scalbln(x.eps2(), exponent),
                        scalbln(x.eps12(), exponent));
}

/**
 * @brief Fused multiply-add: x*y + z.
 */
template <typename T>
inline HyperDual<T> fma(HyperDual<T> const &x, HyperDual<T> const &y, HyperDual<T> const &z) {
    return x * y + z;
}

/**
 * @brief Error function.
 */
template <typename T>
inline HyperDual<T> erf(HyperDual<T> const &x) {
    using std::erf;
    using std::exp;
    T const val = erf(x.value());
    T const d1  = detail::erf_scale<T>() * exp(-detail::square(x.value()));
    T const d2  = -T(2) * x.value() * d1;
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

/**
 * @brief Complementary error function.
 */
template <typename T>
inline HyperDual<T> erfc(HyperDual<T> const &x) {
    using std::erfc;
    using std::exp;
    T const val = erfc(x.value());
    T const d1  = -detail::erf_scale<T>() * exp(-detail::square(x.value()));
    T const d2  = T(2) * x.value() * detail::erf_scale<T>() * exp(-detail::square(x.value()));
    return HyperDual<T>(val,
                        d1 * x.eps1(),
                        d1 * x.eps2(),
                        d1 * x.eps12() + d2 * x.eps1() * x.eps2());
}

} // namespace AD

#endif // HYPERDUAL_HH
