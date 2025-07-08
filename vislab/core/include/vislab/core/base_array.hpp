#pragma once

#include "base_array_fwd.hpp"

#include "iarray.hpp"

namespace vislab
{
    /**
     * @brief Abstract base class for data arrays with a fixed number of components.
     * @tparam TDimensions Number of components per element.
     */
    template <int64_t TDimensions>
    class BaseArray : public Interface<BaseArray<TDimensions>, IArray>
    {
    public:
        /**
         * @brief Number of components in each of the element entries.
         */
        static const int64_t Dimensions = TDimensions;
    };
}
