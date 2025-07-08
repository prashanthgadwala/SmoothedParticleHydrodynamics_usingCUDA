#pragma once

#include <stdint.h>

namespace vislab
{
    /**
     * @brief Abstract base class for data arrays with a fixed number of components.
     * @tparam TDimensions Number of components per element.
     */
    template <int64_t TDimensions>
    class BaseArray;

    /**
     * @brief Abstract base array with one component per element.
     */
    using IArray1 = BaseArray<1>;

    /**
     * @brief Abstract base array with two component per element.
     */
    using IArray2 = BaseArray<2>;

    /**
     * @brief Abstract base array with three component per element.
     */
    using IArray3 = BaseArray<3>;

    /**
     * @brief Abstract base array with four component per element.
     */
    using IArray4 = BaseArray<4>;

    /**
     * @brief Abstract base array with nine components per element.
     */
    using IArray9 = BaseArray<9>;

    /**
     * @brief Abstract base array with sixteen components per element.
     */
    using IArray16 = BaseArray<16>;
}
