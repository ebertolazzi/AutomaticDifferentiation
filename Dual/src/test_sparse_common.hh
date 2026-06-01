#pragma once

#ifndef TEST_SPARSE_COMMON_HH
#define TEST_SPARSE_COMMON_HH

#include <array>
#include <cmath>
#include <type_traits>

/**
 * @file test_sparse_common.hh
 * @brief Shared large sparse test functions used by the derivative benchmarks.
 */

namespace TestSparse {

  namespace detail {

    template <typename T, typename Enable = void>
    struct ConstantBuilder {
      static T make( double value ) {
        return T(value);
      }
    };

    template <typename T>
    struct ConstantBuilder<T, std::void_t<typename T::value_type>> {
      static T make( double value ) {
        return T(ConstantBuilder<typename T::value_type>::make(value));
      }
    };

    /**
     * @brief Builds a constant of the requested scalar type.
     *
     * @tparam T scalar type.
     * @param value passive floating-point value.
     * @return T scalar constant converted recursively.
     */
    template <typename T>
    T constant( double value ) {
      return ConstantBuilder<T>::make(value);
    }

  } // namespace detail

  //! Number of variables used in the large sparse test case.
  constexpr int kLargeVariableCount = 32;

  //! Number of residuals produced by the sparse vector map.
  constexpr int kLargeResidualCount = kLargeVariableCount - 2;

  //! Fixed-size variable array used by the sparse test case.
  template <typename T>
  using VariableArray = std::array<T, kLargeVariableCount>;

  //! Fixed-size residual array used by the sparse vector map.
  template <typename T>
  using ResidualArray = std::array<T, kLargeResidualCount>;

  //! Number of variables used in the huge sparse vector-map test case.
  constexpr int kHugeSparseVariableCount = 64;

  //! Number of residuals produced by the huge sparse vector map.
  constexpr int kHugeSparseResidualCount = kHugeSparseVariableCount - 3;

  //! Fixed-size variable array used by the huge sparse vector map.
  template <typename T>
  using HugeVariableArray = std::array<T, kHugeSparseVariableCount>;

  //! Fixed-size residual array used by the huge sparse vector map.
  template <typename T>
  using HugeResidualArray = std::array<T, kHugeSparseResidualCount>;

  //! Number of variables used by the dense multivariate test case.
  constexpr int kDenseVariableCount = 10;

  //! Number of residuals used by the dense vector map.
  constexpr int kDenseResidualCount = 10;

  //! Fixed-size variable array used by the dense multivariate test case.
  template <typename T>
  using DenseVariableArray = std::array<T, kDenseVariableCount>;

  //! Fixed-size residual array used by the dense vector map.
  template <typename T>
  using DenseResidualArray = std::array<T, kDenseResidualCount>;

  /**
   * @brief Returns one local scalar contribution depending on three neighboring variables.
   *
   * @tparam T scalar type.
   * @param index local term index.
   * @param x0 left variable.
   * @param x1 center variable.
   * @param x2 right variable.
   * @return T local scalar contribution.
   */
  template <typename T>
  T sparse_scalar_term(
    int       index,
    T const & x0,
    T const & x1,
    T const & x2
  ) {
    using std::exp;
    using std::log;
    using std::sin;

    T const diff01 = x0 - detail::constant<T>(0.35) * x1;
    T const diff02 = x0 - x2;
    T const mix    = x0 * x2;
    T const shift  =
      detail::constant<T>(1.0) +
      detail::constant<T>(0.05) * x0 * x0 +
      detail::constant<T>(0.04) * x1 * x1 +
      detail::constant<T>(0.03) * x2 * x2;

    return
      detail::constant<T>(0.08) * diff01 * diff01 +
      detail::constant<T>(0.10) * diff02 * diff02 +
      detail::constant<T>(0.02) * mix * mix +
      exp(detail::constant<T>(0.02) * (x0 + detail::constant<T>(0.5) * x2)) +
      sin(detail::constant<T>(0.30) * (x1 + detail::constant<T>(0.4) * x2) + detail::constant<T>(0.01 * (index + 1))) +
      log(shift);
  }

  /**
   * @brief Evaluates the large sparse scalar objective.
   *
   * Each local term depends only on three consecutive variables, so the Hessian
   * has a banded sparse structure with half-bandwidth two.
   *
   * @tparam T scalar type.
   * @param x input variables.
   * @return T scalar objective value.
   */
  template <typename T>
  T sparse_scalar_objective( VariableArray<T> const & x ) {
    T value{};
    for ( int i = 0; i < kLargeResidualCount; ++i ) {
      value += sparse_scalar_term(i, x[i], x[i + 1], x[i + 2]);
    }
    return value;
  }

  /**
   * @brief Returns one residual of the large sparse vector map.
   *
   * Every residual depends only on three consecutive variables, producing a
   * very sparse banded Jacobian with three structural nonzeros per row.
   *
   * @tparam T scalar type.
   * @param index residual index.
   * @param x0 left variable.
   * @param x1 center variable.
   * @param x2 right variable.
   * @return T residual value.
   */
  template <typename T>
  T sparse_residual_value(
    int       index,
    T const & x0,
    T const & x1,
    T const & x2
  ) {
    using std::sin;

    T const diff02 = x0 - x2;
    return
      sin(detail::constant<T>(0.25) * x0 + detail::constant<T>(0.15) * x1 + detail::constant<T>(0.01 * (index + 1))) +
      detail::constant<T>(0.10) * diff02 * diff02 +
      detail::constant<T>(0.05) * x1 * x2 +
      detail::constant<T>(0.02 * (index + 1)) * x0 * x2 +
      detail::constant<T>(0.30) * x1;
  }

  /**
   * @brief Returns one residual row evaluated on the full variable array.
   *
   * @tparam T scalar type.
   * @param index residual index.
   * @param x input variables.
   * @return T residual value.
   */
  template <typename T>
  T sparse_residual_row(
    int                       index,
    VariableArray<T> const  & x
  ) {
    return sparse_residual_value(index, x[index], x[index + 1], x[index + 2]);
  }

  /**
   * @brief Evaluates the full sparse vector map.
   *
   * @tparam T scalar type.
   * @param x input variables.
   * @return ResidualArray<T> vector of residual values.
   */
  template <typename T>
  ResidualArray<T> sparse_vector_map( VariableArray<T> const & x ) {
    ResidualArray<T> residuals{};
    for ( int i = 0; i < kLargeResidualCount; ++i ) {
      residuals[i] = sparse_residual_row(i, x);
    }
    return residuals;
  }

  /**
   * @brief Returns one residual of the huge sparse vector map.
   *
   * Every residual depends only on four neighboring variables. The resulting
   * Jacobian remains sparse, but the problem size is deliberately larger than
   * the baseline sparse case used elsewhere in the tests.
   *
   * @tparam T scalar type.
   * @param index residual index.
   * @param x0 first local variable.
   * @param x1 second local variable.
   * @param x2 third local variable.
   * @param x3 fourth local variable.
   * @return T residual value.
   */
  template <typename T>
  T huge_sparse_residual_value(
    int       index,
    T const & x0,
    T const & x1,
    T const & x2,
    T const & x3
  ) {
    using std::cos;
    using std::sin;

    T const diff03 = x0 - x3;
    T const mix12  = x1 * x2;
    T const local_sum =
      detail::constant<T>(0.18) * x0 +
      detail::constant<T>(0.11) * x1 +
      detail::constant<T>(0.07) * x2 +
      detail::constant<T>(0.05) * x3;

    return
      sin(local_sum + detail::constant<T>(0.013 * (index + 1))) +
      detail::constant<T>(0.035) * diff03 * diff03 +
      detail::constant<T>(0.022) * mix12 +
      detail::constant<T>(0.012) * x0 * x2 +
      detail::constant<T>(0.010) * x1 * x3 +
      cos(
        detail::constant<T>(0.09) * x0 -
        detail::constant<T>(0.06) * x2 +
        detail::constant<T>(0.04) * x3 +
        detail::constant<T>(0.007 * (index + 1))
      );
  }

  /**
   * @brief Returns one residual row evaluated on the full huge sparse array.
   *
   * @tparam T scalar type.
   * @param index residual index.
   * @param x input variables.
   * @return T residual value.
   */
  template <typename T>
  T huge_sparse_residual_row(
    int                           index,
    HugeVariableArray<T> const  & x
  ) {
    return huge_sparse_residual_value(index, x[index], x[index + 1], x[index + 2], x[index + 3]);
  }

  /**
   * @brief Evaluates the full huge sparse vector map.
   *
   * @tparam T scalar type.
   * @param x input variables.
   * @return HugeResidualArray<T> vector of residual values.
   */
  template <typename T>
  HugeResidualArray<T> huge_sparse_vector_map( HugeVariableArray<T> const & x ) {
    HugeResidualArray<T> residuals{};
    for ( int i = 0; i < kHugeSparseResidualCount; ++i ) {
      residuals[i] = huge_sparse_residual_row(i, x);
    }
    return residuals;
  }

  /**
   * @brief Evaluates a dense scalar objective with full Hessian coupling.
   *
   * @tparam T scalar type.
   * @param x input variables.
   * @return T scalar objective value.
   */
  template <typename T>
  T dense_scalar_objective( DenseVariableArray<T> const & x ) {
    using std::cos;
    using std::exp;
    using std::sin;

    T value{};
    T linear_combo{};
    for ( int i = 0; i < kDenseVariableCount; ++i ) {
      T const xi = x[i];
      value +=
        detail::constant<T>(0.04 * (i + 1)) * xi * xi +
        detail::constant<T>(0.12) * sin(detail::constant<T>(0.23 * (i + 1)) * xi + detail::constant<T>(0.01 * (i + 1))) +
        exp(detail::constant<T>(0.015 * (i + 1)) * xi);
      linear_combo += detail::constant<T>(0.06 + 0.01 * i) * xi;
    }

    for ( int i = 0; i < kDenseVariableCount; ++i ) {
      for ( int j = i + 1; j < kDenseVariableCount; ++j ) {
        value +=
          detail::constant<T>(0.016) * x[i] * x[j] +
          detail::constant<T>(0.018) *
            cos(
              detail::constant<T>(0.11) * x[i] -
              detail::constant<T>(0.08) * x[j] +
              detail::constant<T>(0.01 * (i + j + 1))
            );
      }
    }

    value += detail::constant<T>(0.003) * linear_combo * linear_combo * linear_combo;
    return value;
  }

  /**
   * @brief Returns one row of a dense vector map.
   *
   * Each residual depends on all variables, so the Jacobian is structurally
   * dense.
   *
   * @tparam T scalar type.
   * @param row residual row index.
   * @param x input variables.
   * @return T residual value.
   */
  template <typename T>
  T dense_residual_row(
    int                            row,
    DenseVariableArray<T> const  & x
  ) {
    using std::sin;

    T linear_combo{};
    T quadratic_sum{};
    for ( int j = 0; j < kDenseVariableCount; ++j ) {
      T const xj = x[j];
      linear_combo += detail::constant<T>(0.05 + 0.004 * ((row + j) % 5) + 0.002 * j) * xj;
      quadratic_sum += detail::constant<T>(0.008 + 0.001 * ((row + 2 * j) % 7)) * xj * xj;
    }

    T cross_sum{};
    for ( int j = 0; j < kDenseVariableCount - 1; ++j ) {
      cross_sum += detail::constant<T>(0.010 + 0.001 * ((row + j) % 4)) * x[j] * x[j + 1];
    }

    return
      detail::constant<T>(0.35) *
        sin(linear_combo + detail::constant<T>(0.02 * (row + 1))) +
      quadratic_sum +
      cross_sum +
      detail::constant<T>(0.004 * (row + 1)) * linear_combo * linear_combo;
  }

  /**
   * @brief Evaluates the full dense vector map.
   *
   * @tparam T scalar type.
   * @param x input variables.
   * @return DenseResidualArray<T> vector of residual values.
   */
  template <typename T>
  DenseResidualArray<T> dense_vector_map( DenseVariableArray<T> const & x ) {
    DenseResidualArray<T> residuals{};
    for ( int row = 0; row < kDenseResidualCount; ++row ) {
      residuals[row] = dense_residual_row(row, x);
    }
    return residuals;
  }

  /**
   * @brief Builds a deterministic sample point with positive entries.
   *
   * @param sample_index sample identifier.
   * @param amplitude oscillation amplitude.
   * @return VariableArray<double> sample point.
   */
  inline VariableArray<double>
  build_sparse_sample_point( int sample_index, double amplitude = 0.12 ) {
    VariableArray<double> x{};
    for ( int i = 0; i < kLargeVariableCount; ++i ) {
      double const base = 0.45 + 0.018 * i;
      double const osc1 = std::sin(0.13 * sample_index + 0.21 * i);
      double const osc2 = std::cos(0.07 * sample_index * (i + 1) + 0.17 * i);
      x[i] = base + amplitude * (0.65 * osc1 + 0.35 * osc2);
    }
    return x;
  }

  /**
   * @brief Builds a deterministic sample point for the huge sparse vector map.
   *
   * @param sample_index sample identifier.
   * @param amplitude oscillation amplitude.
   * @return HugeVariableArray<double> sample point.
   */
  inline HugeVariableArray<double>
  build_huge_sparse_sample_point( int sample_index, double amplitude = 0.10 ) {
    HugeVariableArray<double> x{};
    for ( int i = 0; i < kHugeSparseVariableCount; ++i ) {
      double const base = 0.32 + 0.011 * i;
      double const osc1 = std::sin(0.09 * sample_index + 0.14 * i);
      double const osc2 = std::cos(0.05 * sample_index * (i + 1) + 0.07 * i);
      x[i] = base + amplitude * (0.60 * osc1 + 0.40 * osc2);
    }
    return x;
  }

  /**
   * @brief Builds a deterministic sample point for the dense ten-variable case.
   *
   * @param sample_index sample identifier.
   * @param amplitude oscillation amplitude.
   * @return DenseVariableArray<double> sample point.
   */
  inline DenseVariableArray<double>
  build_dense_sample_point( int sample_index, double amplitude = 0.14 ) {
    DenseVariableArray<double> x{};
    for ( int i = 0; i < kDenseVariableCount; ++i ) {
      double const base = 0.28 + 0.045 * i;
      double const osc1 = std::sin(0.17 * sample_index + 0.31 * i);
      double const osc2 = std::cos(0.11 * sample_index * (i + 1) + 0.19 * i);
      x[i] = base + amplitude * (0.55 * osc1 + 0.45 * osc2);
    }
    return x;
  }

  /**
   * @brief Returns the structural number of Jacobian nonzeros.
   *
   * @return int expected Jacobian nonzero count.
   */
  inline int expected_sparse_jacobian_nonzeros() {
    return 3 * kLargeResidualCount;
  }

  /**
   * @brief Returns the structural number of Hessian nonzeros.
   *
   * The Hessian has nonzeros on the main diagonal and on the first two
   * off-diagonals on both sides.
   *
   * @return int expected Hessian nonzero count.
   */
  inline int expected_sparse_hessian_nonzeros() {
    return
      kLargeVariableCount +
      2 * (kLargeVariableCount - 1) +
      2 * (kLargeVariableCount - 2);
  }

  /**
   * @brief Returns the structural number of Jacobian nonzeros for the huge sparse map.
   *
   * @return int expected Jacobian nonzero count.
   */
  inline int expected_huge_sparse_jacobian_nonzeros() {
    return 4 * kHugeSparseResidualCount;
  }

  /**
   * @brief Returns the structural number of Jacobian nonzeros for the dense map.
   *
   * @return int expected dense Jacobian nonzero count.
   */
  inline int expected_dense_jacobian_nonzeros() {
    return kDenseResidualCount * kDenseVariableCount;
  }

  /**
   * @brief Returns the structural number of Hessian nonzeros for the dense scalar objective.
   *
   * @return int expected dense Hessian nonzero count.
   */
  inline int expected_dense_hessian_nonzeros() {
    return kDenseVariableCount * kDenseVariableCount;
  }

} // namespace TestSparse


#endif
