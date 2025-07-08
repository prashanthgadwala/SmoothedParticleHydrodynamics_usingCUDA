#pragma once

#include <stdint.h>

namespace vislab
{
    /**
     * @brief Base class for fields with a specific storage format and certain input dimensions.
     * @tparam TValueType Type that the field is mapping to.
     * @tparam TSpatialDimensions Number of spatial dimensions.
     * @tparam TDomainDimensions Total number of dimensions.
     */
    template <typename TValueType, int64_t TSpatialDimensions, int64_t TDomainDimensions>
    class BaseField;
}
