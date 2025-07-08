#pragma once

#include "iarchive.hpp"

namespace vislab
{
    /**
     * @brief Abstract base class for output archive.
     */
    class IOutputArchive : public Interface<IOutputArchive, IArchive>
    {
    public:
        /**
         * @brief Writes a variable into an archive.
         * @tparam T Type of variable to write.
         * @param name Name of variable to write.
         * @param value Value of variable to write.
         */
        template <typename T>
        inline void write(const char* name, const T& value)
        {
            T v = value;
            this->operator()(name, v);
        }
    };
}
