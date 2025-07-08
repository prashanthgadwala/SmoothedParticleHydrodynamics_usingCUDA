#pragma once

#include "object.hpp"

namespace vislab {

    /**
     * @brief Interface for objects convertible to strings.
     */
    class IStringifiable : public Interface<IStringifiable>
    {
    public:

        /**
         * @brief Convert to string.
         *
         * @return std::string representing this.
         */
        [[nodiscard]] virtual std::string toString() const = 0;

        /**
         * @brief Initialize from string.
         *
         * @param from the string to initialize from.
         */
        virtual void fromString(const std::string& from) = 0;
    };

}
