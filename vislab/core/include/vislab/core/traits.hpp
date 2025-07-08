#pragma once

#include <Eigen/Eigen>

#include <type_traits>

namespace vislab
{
    namespace detail
    {
        /**
         * @brief Is 'T' an Eigen matrix?
         * @tparam T Type to inspect.
         */
        template <typename T>
        using is_eigen_matrix = std::is_base_of<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>>;

        /**
         * @brief Value for is 'T' an Eigen matrix?
         * @tparam T Type to inspect.
         */
        template <typename T>
        constexpr bool is_eigen_matrix_v = is_eigen_matrix<T>::value;

        /**
         * @brief Enable if 'T' is an Eigen matrix
         * @tparam T Type to inspect.
         */
        template <typename T>
        using enable_if_eigen_matrix_t = std::enable_if_t<is_eigen_matrix_v<T>>;

        /**
         * @brief Enable if 'T' is not an Eigen matrix
         * @tparam T Type to inspect.
         */
        template <typename T>
        using enable_if_not_eigen_matrix_t = std::enable_if_t<!is_eigen_matrix_v<T>>;
    }

    namespace detail
    {
        /**
         * @brief Is 'T' a sparse Eigen matrix?
         * @tparam T Type to inspect.
         */
        template <typename T>
        using is_sparse_eigen_matrix = std::is_base_of<Eigen::SparseMatrixBase<std::decay_t<T>>, std::decay_t<T>>;

        /**
         * @brief Value for is 'T' a sparse Eigen matrix?
         * @tparam T Type to inspect.
         */
        template <typename T>
        constexpr bool is_sparse_eigen_matrix_v = is_sparse_eigen_matrix<T>::value;

        /**
         * @brief Enable if 'T' is a sparse Eigen matrix
         * @tparam T Type to inspect.
         */
        template <typename T>
        using enable_if_sparse_eigen_matrix_t = std::enable_if_t<is_sparse_eigen_matrix_v<T>>;

        /**
         * @brief Enable if 'T' is not a sparse Eigen matrix
         * @tparam T Type to inspect.
         */
        template <typename T>
        using enable_if_not_sparse_eigen_matrix_t = std::enable_if_t<!is_sparse_eigen_matrix_v<T>>;
    }

    namespace detail
    {
        /**
         * @brief Is 'T' an Eigen array?
         * @tparam T Type to inspect.
         */
        template <typename T>
        using is_eigen_array = std::is_base_of<Eigen::ArrayBase<std::decay_t<T>>, std::decay_t<T>>;

        /**
         * @brief Value for is 'T' an Eigen array?
         * @tparam T Type to inspect.
         */
        template <typename T>
        constexpr bool is_eigen_array_v = is_eigen_array<T>::value;

        /**
         * @brief Enable if 'T' is an Eigen array
         * @tparam T Type to inspect.
         */
        template <typename T>
        using enable_if_eigen_array_t = std::enable_if_t<is_eigen_array_v<T>>;

        /**
         * @brief Enable if 'T' is not an Eigen array
         * @tparam T Type to inspect.
         */
        template <typename T>
        using enable_if_not_eigen_array_t = std::enable_if_t<!is_eigen_array_v<T>>;
    }

    namespace detail
    {
        /**
         * @brief Is 'T' an Eigen array or Eigen matrix?
         * @tparam T Type to inspect.
         */
        template <typename T>
        using is_eigen = std::bool_constant<is_eigen_matrix_v<T> || is_sparse_eigen_matrix_v<T> || is_eigen_array_v<T>>;

        /**
         * @brief Value for is 'T' an Eigen array or Eigen matrix?
         * @tparam T Type to inspect.
         */
        template <typename T>
        constexpr bool is_eigen_v = is_eigen<T>::value;

        /**
         * @brief Enable if 'T' is an Eigen array or Eigen matrix
         * @tparam T Type to inspect.
         */
        template <typename T>
        using enable_if_eigen_t = std::enable_if_t<is_eigen_v<T>>;

        /**
         * @brief Enable if 'T' is not an Eigen array or Eigen matrix
         * @tparam T Type to inspect.
         */
        template <typename T>
        using enable_if_not_eigen_t = std::enable_if_t<!is_eigen_v<T>>;
    }

    namespace detail
    {
        /**
         * @brief Copies the modifiers of type S to type T (const/pointer/lvalue/rvalue reference)
         * @tparam S Source type.
         * @tparam T Target type.
         */
        template <typename S, typename T>
        struct copy_flags
        {
        private:
            using R  = std::remove_reference_t<S>;
            using T1 = std::conditional_t<std::is_const_v<R>, std::add_const_t<T>, T>;
            using T2 = std::conditional_t<std::is_pointer_v<S>, std::add_pointer_t<T1>, T1>;
            using T3 = std::conditional_t<std::is_lvalue_reference_v<S>, std::add_lvalue_reference_t<T2>, T2>;
            using T4 = std::conditional_t<std::is_rvalue_reference_v<S>, std::add_rvalue_reference_t<T3>, T3>;

        public:
            using type = T4;
        };

        /**
         * @brief Copies the modifiers of type S to type T (const/pointer/lvalue/rvalue reference)
         * @tparam S Source type.
         * @tparam T Target type.
         */
        template <typename S, typename T>
        using copy_flags_t = typename detail::copy_flags<S, T>::type;
    }

    namespace detail
    {
        // Replace the base scalar type
        template <typename T, typename Value, bool CopyFlags = true, typename = void>
        struct replace_scalar
        {
        };

        template <typename T, typename Value, bool CopyFlags>
        struct replace_scalar<T, Value, CopyFlags, std::enable_if_t<std::bool_constant</*!is_forward_dual_v<T> &&*/ !is_eigen_v<T> /*&& !is_autodiff_scalar_v<T>*/>::value>>
        {
            using type = std::conditional_t<CopyFlags, copy_flags_t<T, Value>, Value>;
        };

        // Replace the base scalar type (Eigen matrix specialization)
        template <typename T, typename Value, bool CopyFlags>
        struct replace_scalar<T, Value, CopyFlags, enable_if_eigen_matrix_t<T>>
        {
        private:
            using Matrix = Eigen::Matrix<Value, std::decay_t<T>::RowsAtCompileTime, std::decay_t<T>::ColsAtCompileTime, std::decay_t<T>::Options, std::decay_t<T>::MaxRowsAtCompileTime, std::decay_t<T>::MaxColsAtCompileTime>;

        public:
            using type = std::conditional_t<CopyFlags, copy_flags_t<T, Matrix>, Matrix>;
        };

        // Replace the base scalar type (Eigen array specialization)
        template <typename T, typename Value, bool CopyFlags>
        struct replace_scalar<T, Value, CopyFlags, enable_if_eigen_array_t<T>>
        {
        private:
            using Array = Eigen::Array<Value, std::decay_t<T>::RowsAtCompileTime, std::decay_t<T>::ColsAtCompileTime, std::decay_t<T>::Options, std::decay_t<T>::MaxRowsAtCompileTime, std::decay_t<T>::MaxColsAtCompileTime>;

        public:
            using type = std::conditional_t<CopyFlags, copy_flags_t<T, Array>, Array>;
        };

        // Replace the base scalar type
        template <typename T, typename Value, bool CopyFlags = true>
        using replace_scalar_t = typename replace_scalar<T, Value, CopyFlags>::type;
    }

    template <typename T, bool CopyFlags = true>
    using as_int32_t = detail::replace_scalar_t<T, int32_t, CopyFlags>;
    template <typename T, bool CopyFlags = true>
    using as_uint32_t = detail::replace_scalar_t<T, uint32_t, CopyFlags>;
    template <typename T, bool CopyFlags = true>
    using as_int64_t = detail::replace_scalar_t<T, int64_t, CopyFlags>;
    template <typename T, bool CopyFlags = true>
    using as_uint64_t = detail::replace_scalar_t<T, uint64_t, CopyFlags>;
    template <typename T, bool CopyFlags = true>
    using as_float_t = detail::replace_scalar_t<T, float, CopyFlags>;
    template <typename T, bool CopyFlags = true>
    using as_double_t = detail::replace_scalar_t<T, double, CopyFlags>;
    template <typename T, bool CopyFlags = true>
    using as_bool_t = detail::replace_scalar_t<T, bool, CopyFlags>;
    template <typename T, bool CopyFlags = true>
    using as_size_t = detail::replace_scalar_t<T, std::size_t, CopyFlags>;
}
