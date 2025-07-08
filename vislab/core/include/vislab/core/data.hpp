#pragma once

#include "iserializable.hpp"
#include "object.hpp"

namespace vislab
{
    /**
     * @brief Base class for a data object.
     */
    class Data : public Interface<Data, Object, ISerializable>
    {
    public:
        /**
         * @brief Tests if the data object is fully initialized and ready for use.
         * @return True if the data object is in a valid state.
         */
        [[nodiscard]] virtual bool isValid() const
        {
            return true;
        }
    };
}
