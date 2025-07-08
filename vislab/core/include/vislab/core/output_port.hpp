#pragma once

#include "ioutput_port.hpp"

namespace vislab
{
    /**
     * @brief Strongly typed output port.
     * @tparam TData Data type that is accepted by the port.
     */
    template <typename TData>
    class OutputPort : public Concrete<OutputPort<TData>, IOutputPort>
    {
    public:
        /**
         * @brief Constructor.
         */
        OutputPort() = default;

        /**
         * @brief Gets the data that is stored on the port.
         * @return Weak pointer that might be nullptr if there is no data set.
         */
        [[nodiscard]] inline std::shared_ptr<TData> getData() { return mData; }

        /**
         * @brief Sets the data that is stored on the port.
         * @param data Data to be set on the port.
         */
        void setData(std::shared_ptr<TData> data) { this->mData = data; }

        /**
         * @brief Gets the data type that is accepted by the port.
         * @return Accepted data type.
         */
        [[nodiscard]] inline Type getDataType() const override { return TData::type(); }

        /**
         * @brief Gets the data type that is accepted by the port.
         * @return Accepted data type.
         */
        static Type getDataTypeOutput() { return TData::type(); }

    private:
        /**
         * @brief Data stored on the input port.
         */
        std::shared_ptr<TData> mData;

        /**
         * @brief Gets the data that is stored on the port.
         * @return Weak pointer that might be nullptr if there is no data set.
         */
        [[nodiscard]] inline std::shared_ptr<Data> getDataImpl() override { return mData; }

        void setDataImpl(std::shared_ptr<Data> data) override
        {
            std::shared_ptr<TData> cast = std::dynamic_pointer_cast<TData>(data);
            if (!cast)
            {
                throw std::invalid_argument("Data cannot be cast to concrete type!");
            }
            setData(cast);
        }
    };
}
