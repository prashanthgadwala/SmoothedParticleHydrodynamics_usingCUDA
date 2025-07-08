#pragma once

#include "transfer_function_fwd.hpp"

namespace vislab
{
    /**
     * @brief Base class for a parameter.
     * @tparam TValueType Type of value that is stored in the parameter.
     */
    template <typename TValueType>
    class Parameter;

    /**
     * @brief Native boolean parameter.
     */
    using BoolParameter = Parameter<bool>;

    /**
     * @brief Native string parameter.
     */
    using StringParameter = Parameter<std::string>;

    /**
     * @brief TransferFunction1d parameter.
     */
    using TransferFunction1dParameter = Parameter<TransferFunction1d>;

    /**
     * @brief TransferFunction2d parameter.
     */
    using TransferFunction2dParameter = Parameter<TransferFunction2d>;

    /**
     * @brief TransferFunction3d parameter.
     */
    using TransferFunction3dParameter = Parameter<TransferFunction3d>;

    /**
     * @brief TransferFunction4d parameter.
     */
    using TransferFunction4dParameter = Parameter<TransferFunction4d>;

    /**
     * @brief TransferFunction1f parameter.
     */
    using TransferFunction1fParameter = Parameter<TransferFunction1f>;

    /**
     * @brief TransferFunction2f parameter.
     */
    using TransferFunction2fParameter = Parameter<TransferFunction2f>;

    /**
     * @brief TransferFunction3f parameter.
     */
    using TransferFunction3fParameter = Parameter<TransferFunction3f>;

    /**
     * @brief TransferFunction4f parameter.
     */
    using TransferFunction4fParameter = Parameter<TransferFunction4f>;
}
