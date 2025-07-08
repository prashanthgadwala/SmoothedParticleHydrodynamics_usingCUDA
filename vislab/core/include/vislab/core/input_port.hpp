#pragma once

#include "iinput_port.hpp"

namespace vislab
{
    /**
     * @brief Strongly typed input port.
     * @tparam TData Data type that is accepted by the port.
     */
    template <typename TData>
    class InputPort : public Concrete<InputPort<TData>, IInputPort>
    {
    public:
        /**
         * @brief Default ctor.
         */
        InputPort() = default;

        /**
         * @brief Gets the data that is stored on the port.
         * @return Shared pointer that might be nullptr if there is no data set.
         */
        [[nodiscard]] inline std::shared_ptr<const TData> getData() const { return mData; }

        /**
         * @brief Sets the data that is stored on the port.
         * @param data Data to be set on the port.
         */
        void setData(std::shared_ptr<const TData> data)
        {
            if (this->mData != data)
            {
                this->mData = data;
                this->markChanged();
            }
        }

        /**
         * @brief Gets the data type that is accepted by the port.
         * @return Accepted data type.
         */
        [[nodiscard]] inline Type getDataType() const override { return TData::type(); }

        /**
         * @brief Gets the data type that is accepted by the port.
         * @return Accepted data type.
         */
        static Type getDataTypeInput() { return TData::type(); }

    protected:
        /**
         * @brief Gets the data that is stored on the port.
         * @return Shared pointer that might be nullptr if there is no data set.
         */
        [[nodiscard]] std::shared_ptr<const Data> getDataImpl() const override { return mData; }

        /**
         * @brief Set the data that is stored on the port.
         *
         * @param data data to be stored in the port.
         */
        void setDataImpl(std::shared_ptr<const Data> data) override
        {
            std::shared_ptr<const TData> cast = std::dynamic_pointer_cast<const TData>(data);
            if (!cast)
            {
                throw std::invalid_argument("Data cannot be cast to concrete type!");
            }
            setData(cast);
        }

    private:
        /**
         * @brief Data stored on the input port.
         */
        std::shared_ptr<const TData> mData;
    };
}
