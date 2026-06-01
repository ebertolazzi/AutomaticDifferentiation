#pragma once

#ifndef DUAL_HH
#define DUAL_HH

#include <cmath>
#include <ostream>

/**
 * @file dual.hh
 * @brief Dual-number implementation for forward-mode automatic differentiation.
 */

namespace AD {

  namespace detail {

    /**
     * @brief Returns the square of a value.
     *
     * @tparam T value type.
     * @param x value to square.
     * @return T square of @p x.
     */
    template <typename T>
    constexpr inline T square( T const & x ) noexcept(noexcept(x * x)) {
      return x * x;
    }

    /**
     * @brief Returns the constant factor in the derivative of `erf`.
     *
     * @tparam T value type.
     * @return T value of `2 / sqrt(pi)`.
     */
    template <typename T>
    inline T erf_scale() {
      using std::acos;
      using std::sqrt;
      return T(2) / sqrt(acos(T(-1)));
    }

  }

  /**
   * @brief Dual number of the form `a + b eps`.
   *
   * A dual number stores both the function value (`a`) and the first
   * derivative with respect to an independent variable (`b`).
   *
   * @tparam T scalar coefficient type.
   */
  template <typename T>
  class Dual {
  public:

    //! Underlying scalar type.
    using value_type = T;

  private:

    T _value = T(0);
    T _dual  = T(0);

  public:

    /**
     * @brief Builds the zero dual number.
     */
    constexpr Dual() noexcept = default;

    /**
     * @brief Builds a dual number by copy.
     */
    constexpr Dual( Dual const & ) noexcept = default;

    /**
     * @brief Builds a dual number by move.
     */
    constexpr Dual( Dual && ) noexcept = default;

    /**
     * @brief Builds a dual number from primal and dual parts.
     *
     * @param value primal value.
     * @param dual dual coefficient.
     */
    constexpr explicit Dual( T value, T dual = T(0) ) noexcept
    : _value(value)
    , _dual(dual)
    {}

    /**
     * @brief Assigns a dual number by copy.
     *
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator = ( Dual const & ) noexcept = default;

    /**
     * @brief Assigns a dual number by move.
     *
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator = ( Dual && ) noexcept = default;

    /**
     * @brief Builds an independent variable.
     *
     * @param value variable value.
     * @return Dual dual number with unit initial derivative.
     */
    [[nodiscard]] static constexpr Dual variable( T value ) noexcept {
      return Dual(value, T(1));
    }

    /**
     * @brief Returns the primal value.
     *
     * @return T primal value.
     */
    [[nodiscard]] constexpr T value() const noexcept { return _value; }

    /**
     * @brief Returns the dual component.
     *
     * @return T derivative coefficient.
     */
    [[nodiscard]] constexpr T dual() const noexcept { return _dual; }

    /**
     * @brief Sets both components of the dual number.
     *
     * @param value new primal value.
     * @param dual new dual coefficient.
     */
    constexpr void set( T value, T dual ) noexcept {
      _value = value;
      _dual  = dual;
    }

    /**
     * @brief Assigns a scalar value.
     *
     * The dual part is reset to zero.
     *
     * @param value new primal value.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator = ( T value ) noexcept {
      _value = value;
      _dual  = T(0);
      return *this;
    }

    /**
     * @brief Adds a dual number.
     *
     * @param rhs addend.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator += ( Dual const & rhs ) noexcept {
      _value += rhs._value;
      _dual  += rhs._dual;
      return *this;
    }

    /**
     * @brief Adds a scalar.
     *
     * @param rhs scalar addend.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator += ( T rhs ) noexcept {
      _value += rhs;
      return *this;
    }

    /**
     * @brief Subtracts a dual number.
     *
     * @param rhs subtrahend.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator -= ( Dual const & rhs ) noexcept {
      _value -= rhs._value;
      _dual  -= rhs._dual;
      return *this;
    }

    /**
     * @brief Subtracts a scalar.
     *
     * @param rhs scalar subtrahend.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator -= ( T rhs ) noexcept {
      _value -= rhs;
      return *this;
    }

    /**
     * @brief Multiplies by a dual number.
     *
     * @param rhs factor.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator *= ( Dual const & rhs ) noexcept {
      T const value = _value;
      T const dual  = _dual;
      _value = value * rhs._value;
      _dual  = dual * rhs._value + value * rhs._dual;
      return *this;
    }

    /**
     * @brief Multiplies by a scalar.
     *
     * @param rhs scalar factor.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator *= ( T rhs ) noexcept {
      _value *= rhs;
      _dual  *= rhs;
      return *this;
    }

    /**
     * @brief Divides by a dual number.
     *
     * @param rhs divisor.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator /= ( Dual const & rhs ) noexcept {
      T const value = _value;
      T const dual  = _dual;
      T const denom = rhs._value * rhs._value;
      _value = value / rhs._value;
      _dual  = (dual * rhs._value - value * rhs._dual) / denom;
      return *this;
    }

    /**
     * @brief Divides by a scalar.
     *
     * @param rhs scalar divisor.
     * @return Dual& reference to the updated object.
     */
    constexpr Dual & operator /= ( T rhs ) noexcept {
      _value /= rhs;
      _dual  /= rhs;
      return *this;
    }

    /**
     * @brief Applies unary plus.
     *
     * @return Dual unchanged copy of the dual number.
     */
    [[nodiscard]] constexpr Dual operator + () const noexcept {
      return *this;
    }

    /**
     * @brief Applies unary minus.
     *
     * @return Dual negated dual number.
     */
    [[nodiscard]] constexpr Dual operator - () const noexcept {
      return Dual(-_value, -_dual);
    }

  };

  /**
   * @brief Compares two dual numbers for exact equality.
   *
   * @tparam T scalar type.
   * @param lhs first operand.
   * @param rhs second operand.
   * @return bool `true` if both components match exactly.
   */
  template <typename T>
  constexpr inline bool operator == ( Dual<T> const & lhs, Dual<T> const & rhs ) noexcept {
    return lhs.value() == rhs.value() && lhs.dual() == rhs.dual();
  }

  /**
   * @brief Compares a dual number with a scalar.
   *
   * @tparam T scalar type.
   * @param lhs dual operand.
   * @param rhs scalar operand.
   * @return bool `true` if the primal values match and the dual part is zero.
   */
  template <typename T>
  constexpr inline bool operator == ( Dual<T> const & lhs, T rhs ) noexcept {
    return lhs.value() == rhs && lhs.dual() == T(0);
  }

  /**
   * @brief Compares a scalar with a dual number.
   *
   * @tparam T scalar type.
   * @param lhs scalar operand.
   * @param rhs dual operand.
   * @return bool `true` if the primal values match and the dual part is zero.
   */
  template <typename T>
  constexpr inline bool operator == ( T lhs, Dual<T> const & rhs ) noexcept {
    return rhs == lhs;
  }

  /**
   * @brief Compares two dual numbers for inequality.
   *
   * @tparam T scalar type.
   * @param lhs first operand.
   * @param rhs second operand.
   * @return bool negation of `operator==`.
   */
  template <typename T>
  constexpr inline bool operator != ( Dual<T> const & lhs, Dual<T> const & rhs ) noexcept {
    return !(lhs == rhs);
  }

  /**
   * @brief Adds two dual numbers.
   *
   * @tparam T scalar type.
   * @param lhs first addend.
   * @param rhs second addend.
   * @return Dual<T> sum of the two dual numbers.
   */
  template <typename T>
  [[nodiscard]] constexpr inline Dual<T> operator + ( Dual<T> const & lhs, Dual<T> const & rhs ) noexcept {
    return Dual<T>(lhs.value() + rhs.value(), lhs.dual() + rhs.dual());
  }

  /**
   * @brief Adds a scalar and a dual number.
   *
   * @tparam T scalar type.
   * @param lhs scalar addend.
   * @param rhs dual addend.
   * @return Dual<T> sum result.
   */
  template <typename T>
  [[nodiscard]] constexpr inline Dual<T> operator + ( T lhs, Dual<T> const & rhs ) noexcept {
    return Dual<T>(lhs + rhs.value(), rhs.dual());
  }

  /**
   * @brief Adds a dual number and a scalar.
   *
   * @tparam T scalar type.
   * @param lhs dual addend.
   * @param rhs scalar addend.
   * @return Dual<T> sum result.
   */
  template <typename T>
  [[nodiscard]] constexpr inline Dual<T> operator + ( Dual<T> const & lhs, T rhs ) noexcept {
    return Dual<T>(lhs.value() + rhs, lhs.dual());
  }

  /**
   * @brief Subtracts two dual numbers.
   *
   * @tparam T scalar type.
   * @param lhs minuend.
   * @param rhs subtrahend.
   * @return Dual<T> subtraction result.
   */
  template <typename T>
  [[nodiscard]] constexpr inline Dual<T> operator - ( Dual<T> const & lhs, Dual<T> const & rhs ) noexcept {
    return Dual<T>(lhs.value() - rhs.value(), lhs.dual() - rhs.dual());
  }

  /**
   * @brief Subtracts a dual number from a scalar.
   *
   * @tparam T scalar type.
   * @param lhs scalar minuend.
   * @param rhs dual subtrahend.
   * @return Dual<T> subtraction result.
   */
  template <typename T>
  [[nodiscard]] constexpr inline Dual<T> operator - ( T lhs, Dual<T> const & rhs ) noexcept {
    return Dual<T>(lhs - rhs.value(), -rhs.dual());
  }

  /**
   * @brief Subtracts a scalar from a dual number.
   *
   * @tparam T scalar type.
   * @param lhs dual minuend.
   * @param rhs scalar subtrahend.
   * @return Dual<T> subtraction result.
   */
  template <typename T>
  [[nodiscard]] constexpr inline Dual<T> operator - ( Dual<T> const & lhs, T rhs ) noexcept {
    return Dual<T>(lhs.value() - rhs, lhs.dual());
  }

  /**
   * @brief Multiplies two dual numbers.
   *
   * @tparam T scalar type.
   * @param lhs first factor.
   * @param rhs second factor.
   * @return Dual<T> product result.
   */
  template <typename T>
  [[nodiscard]] constexpr inline Dual<T> operator * ( Dual<T> const & lhs, Dual<T> const & rhs ) noexcept {
    return Dual<T>(
      lhs.value() * rhs.value(),
      lhs.dual() * rhs.value() + lhs.value() * rhs.dual()
    );
  }

  /**
   * @brief Multiplies a dual number by a scalar.
   *
   * @tparam T scalar type.
   * @param lhs dual factor.
   * @param rhs scalar factor.
   * @return Dual<T> product result.
   */
  template <typename T>
  [[nodiscard]] constexpr inline Dual<T> operator * ( Dual<T> const & lhs, T rhs ) noexcept {
    return Dual<T>(lhs.value() * rhs, lhs.dual() * rhs);
  }

  /**
   * @brief Multiplies a scalar by a dual number.
   *
   * @tparam T scalar type.
   * @param lhs scalar factor.
   * @param rhs dual factor.
   * @return Dual<T> product result.
   */
  template <typename T>
  [[nodiscard]] constexpr inline Dual<T> operator * ( T lhs, Dual<T> const & rhs ) noexcept {
    return Dual<T>(lhs * rhs.value(), lhs * rhs.dual());
  }

  /**
   * @brief Divides two dual numbers.
   *
   * @tparam T scalar type.
   * @param lhs dividend.
   * @param rhs divisor.
   * @return Dual<T> quotient result.
   */
  template <typename T>
  [[nodiscard]] constexpr inline Dual<T> operator / ( Dual<T> const & lhs, Dual<T> const & rhs ) noexcept {
    T const denom = rhs.value() * rhs.value();
    return Dual<T>(
      lhs.value() / rhs.value(),
      (lhs.dual() * rhs.value() - lhs.value() * rhs.dual()) / denom
    );
  }

  /**
   * @brief Divides a dual number by a scalar.
   *
   * @tparam T scalar type.
   * @param lhs dual dividend.
   * @param rhs scalar divisor.
   * @return Dual<T> quotient result.
   */
  template <typename T>
  [[nodiscard]] constexpr inline Dual<T> operator / ( Dual<T> const & lhs, T rhs ) noexcept {
    return Dual<T>(lhs.value() / rhs, lhs.dual() / rhs);
  }

  /**
   * @brief Divides a scalar by a dual number.
   *
   * @tparam T scalar type.
   * @param lhs scalar dividend.
   * @param rhs dual divisor.
   * @return Dual<T> quotient result.
   */
  template <typename T>
  [[nodiscard]] constexpr inline Dual<T> operator / ( T lhs, Dual<T> const & rhs ) noexcept {
    T const denom = rhs.value() * rhs.value();
    return Dual<T>(
      lhs / rhs.value(),
      -lhs * rhs.dual() / denom
    );
  }

  /**
   * @brief Prints a dual number in the format `a +/- b * eps`.
   *
   * @tparam CharT stream character type.
   * @tparam Traits stream traits type.
   * @tparam T dual scalar type.
   * @param stream output stream.
   * @param value dual number to print.
   * @return std::basic_ostream<CharT, Traits>& reference to the stream.
   */
  template <typename CharT, typename Traits, typename T>
  inline std::basic_ostream<CharT,Traits> &
  operator << (
    std::basic_ostream<CharT,Traits> & stream,
    Dual<T> const & value
  ) {
    using std::abs;
    stream << value.value()
           << (value.dual() >= T(0) ? " + " : " - ")
           << abs(value.dual())
           << " * eps";
    return stream;
  }

  /**
   * @brief Returns the absolute value.
   *
   * At `x = 0` the derivative is not defined; this implementation returns zero.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `abs(x)`.
   */
  template <typename T>
  inline Dual<T> abs( Dual<T> const & x ) {
    using std::abs;
    using std::copysign;
    T const primal = abs(x.value());
    T const slope  = x.value() == T(0) ? T(0) : copysign(T(1), x.value());
    return Dual<T>(primal, slope * x.dual());
  }

  /**
   * @brief Returns the sine function.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `sin(x)`.
   */
  template <typename T>
  inline Dual<T> sin( Dual<T> const & x ) {
    using std::cos;
    using std::sin;
    return Dual<T>(sin(x.value()), cos(x.value()) * x.dual());
  }

  /**
   * @brief Returns the cosine function.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `cos(x)`.
   */
  template <typename T>
  inline Dual<T> cos( Dual<T> const & x ) {
    using std::cos;
    using std::sin;
    return Dual<T>(cos(x.value()), -sin(x.value()) * x.dual());
  }

  /**
   * @brief Returns the tangent function.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `tan(x)`.
   */
  template <typename T>
  inline Dual<T> tan( Dual<T> const & x ) {
    using std::cos;
    using std::tan;
    T const c = cos(x.value());
    return Dual<T>(tan(x.value()), x.dual() / detail::square(c));
  }

  /**
   * @brief Returns the arcsine function.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `asin(x)`.
   */
  template <typename T>
  inline Dual<T> asin( Dual<T> const & x ) {
    using std::asin;
    using std::sqrt;
    return Dual<T>(
      asin(x.value()),
      x.dual() / sqrt(T(1) - detail::square(x.value()))
    );
  }

  /**
   * @brief Returns the arccosine function.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `acos(x)`.
   */
  template <typename T>
  inline Dual<T> acos( Dual<T> const & x ) {
    using std::acos;
    using std::sqrt;
    return Dual<T>(
      acos(x.value()),
      -x.dual() / sqrt(T(1) - detail::square(x.value()))
    );
  }

  /**
   * @brief Returns the arctangent function.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `atan(x)`.
   */
  template <typename T>
  inline Dual<T> atan( Dual<T> const & x ) {
    using std::atan;
    return Dual<T>(
      atan(x.value()),
      x.dual() / (T(1) + detail::square(x.value()))
    );
  }

  /**
   * @brief Returns the `atan2(y, x)` function.
   *
   * @tparam T scalar type.
   * @param y tangent numerator.
   * @param x tangent denominator.
   * @return Dual<T> `atan2(y, x)`.
   */
  template <typename T>
  inline Dual<T> atan2( Dual<T> const & y, Dual<T> const & x ) {
    using std::atan2;
    T const denom = detail::square(x.value()) + detail::square(y.value());
    return Dual<T>(
      atan2(y.value(), x.value()),
      (x.value() * y.dual() - y.value() * x.dual()) / denom
    );
  }

  /**
   * @brief Returns `atan2(y, x)` with scalar `x`.
   *
   * @tparam T scalar type.
   * @param y dual numerator.
   * @param x scalar denominator.
   * @return Dual<T> `atan2(y, x)`.
   */
  template <typename T>
  inline Dual<T> atan2( Dual<T> const & y, T x ) {
    return atan2(y, Dual<T>(x));
  }

  /**
   * @brief Returns `atan2(y, x)` with scalar `y`.
   *
   * @tparam T scalar type.
   * @param y scalar numerator.
   * @param x dual denominator.
   * @return Dual<T> `atan2(y, x)`.
   */
  template <typename T>
  inline Dual<T> atan2( T y, Dual<T> const & x ) {
    return atan2(Dual<T>(y), x);
  }

  /**
   * @brief Returns the hyperbolic sine.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `sinh(x)`.
   */
  template <typename T>
  inline Dual<T> sinh( Dual<T> const & x ) {
    using std::cosh;
    using std::sinh;
    return Dual<T>(sinh(x.value()), cosh(x.value()) * x.dual());
  }

  /**
   * @brief Returns the hyperbolic cosine.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `cosh(x)`.
   */
  template <typename T>
  inline Dual<T> cosh( Dual<T> const & x ) {
    using std::cosh;
    using std::sinh;
    return Dual<T>(cosh(x.value()), sinh(x.value()) * x.dual());
  }

  /**
   * @brief Returns the hyperbolic tangent.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `tanh(x)`.
   */
  template <typename T>
  inline Dual<T> tanh( Dual<T> const & x ) {
    using std::cosh;
    using std::tanh;
    T const c = cosh(x.value());
    return Dual<T>(tanh(x.value()), x.dual() / detail::square(c));
  }

  /**
   * @brief Returns the inverse hyperbolic sine.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `asinh(x)`.
   */
  template <typename T>
  inline Dual<T> asinh( Dual<T> const & x ) {
    using std::asinh;
    using std::sqrt;
    return Dual<T>(
      asinh(x.value()),
      x.dual() / sqrt(detail::square(x.value()) + T(1))
    );
  }

  /**
   * @brief Returns the inverse hyperbolic cosine.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `acosh(x)`.
   */
  template <typename T>
  inline Dual<T> acosh( Dual<T> const & x ) {
    using std::acosh;
    using std::sqrt;
    return Dual<T>(
      acosh(x.value()),
      x.dual() / (sqrt(x.value() - T(1)) * sqrt(x.value() + T(1)))
    );
  }

  /**
   * @brief Returns the inverse hyperbolic tangent.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `atanh(x)`.
   */
  template <typename T>
  inline Dual<T> atanh( Dual<T> const & x ) {
    using std::atanh;
    return Dual<T>(
      atanh(x.value()),
      x.dual() / (T(1) - detail::square(x.value()))
    );
  }

  /**
   * @brief Returns the natural exponential.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `exp(x)`.
   */
  template <typename T>
  inline Dual<T> exp( Dual<T> const & x ) {
    using std::exp;
    T const primal = exp(x.value());
    return Dual<T>(primal, primal * x.dual());
  }

  /**
   * @brief Returns `2^x`.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `exp2(x)`.
   */
  template <typename T>
  inline Dual<T> exp2( Dual<T> const & x ) {
    using std::exp2;
    using std::log;
    T const primal = exp2(x.value());
    return Dual<T>(primal, primal * log(T(2)) * x.dual());
  }

  /**
   * @brief Returns `exp(x) - 1`.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `expm1(x)`.
   */
  template <typename T>
  inline Dual<T> expm1( Dual<T> const & x ) {
    using std::exp;
    using std::expm1;
    return Dual<T>(expm1(x.value()), exp(x.value()) * x.dual());
  }

  /**
   * @brief Returns the natural logarithm.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `log(x)`.
   */
  template <typename T>
  inline Dual<T> log( Dual<T> const & x ) {
    using std::log;
    return Dual<T>(log(x.value()), x.dual() / x.value());
  }

  /**
   * @brief Returns the base-2 logarithm.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `log2(x)`.
   */
  template <typename T>
  inline Dual<T> log2( Dual<T> const & x ) {
    using std::log;
    using std::log2;
    return Dual<T>(log2(x.value()), x.dual() / (x.value() * log(T(2))));
  }

  /**
   * @brief Returns the base-10 logarithm.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `log10(x)`.
   */
  template <typename T>
  inline Dual<T> log10( Dual<T> const & x ) {
    using std::log;
    using std::log10;
    return Dual<T>(log10(x.value()), x.dual() / (x.value() * log(T(10))));
  }

  /**
   * @brief Returns `log(1 + x)`.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `log1p(x)`.
   */
  template <typename T>
  inline Dual<T> log1p( Dual<T> const & x ) {
    using std::log1p;
    return Dual<T>(log1p(x.value()), x.dual() / (T(1) + x.value()));
  }

  /**
   * @brief Returns the square root.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `sqrt(x)`.
   */
  template <typename T>
  inline Dual<T> sqrt( Dual<T> const & x ) {
    using std::sqrt;
    T const primal = sqrt(x.value());
    return Dual<T>(primal, x.dual() / (T(2) * primal));
  }

  /**
   * @brief Returns the cubic root.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `cbrt(x)`.
   */
  template <typename T>
  inline Dual<T> cbrt( Dual<T> const & x ) {
    using std::cbrt;
    T const primal = cbrt(x.value());
    return Dual<T>(primal, x.dual() / (T(3) * primal * primal));
  }

  /**
   * @brief Raises a dual number to a dual exponent.
   *
   * @tparam T scalar type.
   * @param base dual base.
   * @param exponent dual exponent.
   * @return Dual<T> `pow(base, exponent)`.
   */
  template <typename T>
  inline Dual<T> pow( Dual<T> const & base, Dual<T> const & exponent ) {
    using std::log;
    using std::pow;
    T const primal = pow(base.value(), exponent.value());
    return Dual<T>(
      primal,
      primal * (
        exponent.dual() * log(base.value()) +
        exponent.value() * base.dual() / base.value()
      )
    );
  }

  /**
   * @brief Raises a dual number to a scalar exponent.
   *
   * @tparam T scalar type.
   * @param base dual base.
   * @param exponent scalar exponent.
   * @return Dual<T> `pow(base, exponent)`.
   */
  template <typename T>
  inline Dual<T> pow( Dual<T> const & base, T exponent ) {
    using std::pow;
    T const primal = pow(base.value(), exponent);
    return Dual<T>(
      primal,
      exponent * pow(base.value(), exponent - T(1)) * base.dual()
    );
  }

  /**
   * @brief Raises a scalar base to a dual exponent.
   *
   * @tparam T scalar type.
   * @param base scalar base.
   * @param exponent dual exponent.
   * @return Dual<T> `pow(base, exponent)`.
   */
  template <typename T>
  inline Dual<T> pow( T base, Dual<T> const & exponent ) {
    using std::log;
    using std::pow;
    T const primal = pow(base, exponent.value());
    return Dual<T>(primal, primal * log(base) * exponent.dual());
  }

  /**
   * @brief Returns the `hypot(x, y)` function.
   *
   * @tparam T scalar type.
   * @param x first leg.
   * @param y second leg.
   * @return Dual<T> `hypot(x, y)`.
   */
  template <typename T>
  inline Dual<T> hypot( Dual<T> const & x, Dual<T> const & y ) {
    using std::hypot;
    T const primal = hypot(x.value(), y.value());
    return Dual<T>(
      primal,
      (x.value() * x.dual() + y.value() * y.dual()) / primal
    );
  }

  /**
   * @brief Returns `hypot(x, y)` with scalar `y`.
   *
   * @tparam T scalar type.
   * @param x first dual leg.
   * @param y second scalar leg.
   * @return Dual<T> `hypot(x, y)`.
   */
  template <typename T>
  inline Dual<T> hypot( Dual<T> const & x, T y ) {
    return hypot(x, Dual<T>(y));
  }

  /**
   * @brief Returns `hypot(x, y)` with scalar `x`.
   *
   * @tparam T scalar type.
   * @param x first scalar leg.
   * @param y second dual leg.
   * @return Dual<T> `hypot(x, y)`.
   */
  template <typename T>
  inline Dual<T> hypot( T x, Dual<T> const & y ) {
    return hypot(Dual<T>(x), y);
  }

  /**
   * @brief Returns the greatest integer not larger than the argument.
   *
   * The derivative is zero wherever the function is differentiable.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `floor(x)`.
   */
  template <typename T>
  inline Dual<T> floor( Dual<T> const & x ) {
    using std::floor;
    return Dual<T>(floor(x.value()), T(0));
  }

  /**
   * @brief Returns the smallest integer not smaller than the argument.
   *
   * The derivative is zero wherever the function is differentiable.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `ceil(x)`.
   */
  template <typename T>
  inline Dual<T> ceil( Dual<T> const & x ) {
    using std::ceil;
    return Dual<T>(ceil(x.value()), T(0));
  }

  /**
   * @brief Truncates the fractional part.
   *
   * The derivative is zero wherever the function is differentiable.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `trunc(x)`.
   */
  template <typename T>
  inline Dual<T> trunc( Dual<T> const & x ) {
    using std::trunc;
    return Dual<T>(trunc(x.value()), T(0));
  }

  /**
   * @brief Rounds to the nearest integer.
   *
   * The derivative is zero wherever the function is differentiable.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `round(x)`.
   */
  template <typename T>
  inline Dual<T> round( Dual<T> const & x ) {
    using std::round;
    return Dual<T>(round(x.value()), T(0));
  }

  /**
   * @brief Scales by `2^exponent`.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @param exponent integer exponent.
   * @return Dual<T> `ldexp(x, exponent)`.
   */
  template <typename T>
  inline Dual<T> ldexp( Dual<T> const & x, int exponent ) {
    using std::ldexp;
    return Dual<T>(ldexp(x.value(), exponent), ldexp(x.dual(), exponent));
  }

  /**
   * @brief Scales by `FLT_RADIX^exponent`.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @param exponent integer exponent.
   * @return Dual<T> `scalbn(x, exponent)`.
   */
  template <typename T>
  inline Dual<T> scalbn( Dual<T> const & x, int exponent ) {
    using std::scalbn;
    return Dual<T>(scalbn(x.value(), exponent), scalbn(x.dual(), exponent));
  }

  /**
   * @brief Scales by `FLT_RADIX^exponent` with a long exponent.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @param exponent long integer exponent.
   * @return Dual<T> `scalbln(x, exponent)`.
   */
  template <typename T>
  inline Dual<T> scalbln( Dual<T> const & x, long exponent ) {
    using std::scalbln;
    return Dual<T>(scalbln(x.value(), exponent), scalbln(x.dual(), exponent));
  }

  /**
   * @brief Returns `x * y + z`.
   *
   * @tparam T scalar type.
   * @param x first factor.
   * @param y second factor.
   * @param z final addend.
   * @return Dual<T> `x * y + z`.
   */
  template <typename T>
  inline Dual<T> fma( Dual<T> const & x, Dual<T> const & y, Dual<T> const & z ) {
    return x * y + z;
  }

  /**
   * @brief Returns the error function.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `erf(x)`.
   */
  template <typename T>
  inline Dual<T> erf( Dual<T> const & x ) {
    using std::erf;
    using std::exp;
    return Dual<T>(
      erf(x.value()),
      detail::erf_scale<T>() * exp(-detail::square(x.value())) * x.dual()
    );
  }

  /**
   * @brief Returns the complementary error function.
   *
   * @tparam T scalar type.
   * @param x dual argument.
   * @return Dual<T> `erfc(x)`.
   */
  template <typename T>
  inline Dual<T> erfc( Dual<T> const & x ) {
    using std::erfc;
    using std::exp;
    return Dual<T>(
      erfc(x.value()),
      -detail::erf_scale<T>() * exp(-detail::square(x.value())) * x.dual()
    );
  }

}

#endif
