#pragma once

#include <stdint.h>

namespace vislab
{
    /**
     * @brief Transfer function with fixed number of output components.
     */
    template <int64_t TComponents>
    class BaseTransferFunction;

    /**
     * @brief Interface for transfer functions that map to one number.
     */
    using ITransferFunction1 = BaseTransferFunction<1>;

    /**
     * @brief Interface for transfer functions that map to two numbers.
     */
    using ITransferFunction2 = BaseTransferFunction<2>;

    /**
     * @brief Interface for transfer functions that map to three numbers.
     */
    using ITransferFunction3 = BaseTransferFunction<3>;

    /**
     * @brief Interface for transfer functions that map to four numbers.
     */
    using ITransferFunction4 = BaseTransferFunction<4>;
}
