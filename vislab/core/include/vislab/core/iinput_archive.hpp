#pragma once

#include "iarchive.hpp"

namespace vislab
{
    /**
     * @brief Abstract base class for input archive.
     */
    class IInputArchive : public Interface<IInputArchive, IArchive>
    {
    public:
        /**
         * @brief Reads a variable from the archive.
         * @tparam T Type of variable to read.
         * @param name Name of variable to read.
         * @return Value that was read.
         */
        template <typename T>
        inline T read(const char* name)
        {
            T value;
            this->operator()(name, value);
            return value;
        }
    };
}
