#pragma once

#include "data.hpp"
#include "iarchive.hpp"

namespace vislab
{
    /**
     * @brief Interface for output ports.
     */
    class IOutputPort : public Interface<IOutputPort, Object>
    {
    public:
        /**
         * @brief Constructor.
         */
        IOutputPort()
            : required(true)
        {
        }

        /**
         * @brief Gets the data that is stored on the port.
         * @return Shared pointer that might be nullptr if there is no data.
         */
        inline std::shared_ptr<Data> getData() { return getDataImpl(); }

        /**
         * @brief Set the data to this input port.
         *
         * @param data
         */
        void setData(std::shared_ptr<Data> data)
        {
            setDataImpl(std::move(data));
        }

        /**
         * @brief Gets the data type that is accepted by the port.
         * @return Accepted data type.
         */
        [[nodiscard]] virtual Type getDataType() const = 0;

        /**
         * @brief Flag that determines whether a valid output port is required for executing the algorithm.
         */
        bool required;

    private:
        /**
         * @brief Gets the data that is stored on the port.
         * @return Shared pointer that might be nullptr if there is no data set.
         */
        virtual std::shared_ptr<Data> getDataImpl() = 0;

        /**
         * @brief Sets the data that is stored on the port.
         * @param Shared pointer that might be nullptr if there is no data set.
         */
        virtual void setDataImpl(std::shared_ptr<Data> data) = 0;
    };
}
