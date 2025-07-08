#pragma once

#include "types.hpp"

namespace vislab
{
    /**
     * @brief Base class for parameters that have a min/max value range.
     * @tparam TDerived Type of the derived class (CRTP).
     * @tparam TValueType Type of value that is stored in the parameter.
     */
    template <typename TValueType>
    class NumericParameter;

    /**
     * @brief Native float parameter.
     */
    using FloatParameter = NumericParameter<float>;

    /**
     * @brief Native double parameter.
     */
    using DoubleParameter = NumericParameter<double>;

    /**
     * @brief Native 32-bit integer parameter.
     */
    using Int32Parameter = NumericParameter<int32_t>;

    /**
     * @brief Native 64-bit integer parameter.
     */
    using Int64Parameter = NumericParameter<int64_t>;

    /**
     * @brief Two-component 32-bit integer parameter.
     */
    using Vec2iParameter = NumericParameter<Eigen::Vector2i>;

    /**
     * @brief Three-component 32-bit integer parameter.
     */
    using Vec3iParameter = NumericParameter<Eigen::Vector3i>;

    /**
     * @brief Four-component 32-bit integer parameter.
     */
    using Vec4iParameter = NumericParameter<Eigen::Vector4i>;

    /**
     * @brief Two-component 32-bit float parameter.
     */
    using Vec2fParameter = NumericParameter<Eigen::Vector2f>;

    /**
     * @brief Three-component 32-bit float parameter.
     */
    using Vec3fParameter = NumericParameter<Eigen::Vector3f>;

    /**
     * @brief Four-component 32-bit float parameter.
     */
    using Vec4fParameter = NumericParameter<Eigen::Vector4f>;

    /**
     * @brief Two-component 64-bit float parameter.
     */
    using Vec2dParameter = NumericParameter<Eigen::Vector2d>;

    /**
     * @brief Three-component 64-bit float parameter.
     */
    using Vec3dParameter = NumericParameter<Eigen::Vector3d>;

    /**
     * @brief Four-component 64-bit float parameter.
     */
    using Vec4dParameter = NumericParameter<Eigen::Vector4d>;

    /**
     * @brief Four-component float color parameter.
     */
    using ColorParameter = NumericParameter<Eigen::Vector4f>;
}
