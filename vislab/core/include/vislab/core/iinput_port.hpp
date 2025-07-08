#pragma once

#include <utility>

#include "data.hpp"
#include "iarchive.hpp"

namespace vislab
{
    /**
     * @brief Interface for input ports.
     */
    class IInputPort : public Interface<IInputPort, Object>
    {
    public:
        /**
         * @brief Constructor.
         */
        IInputPort()
            : required(true)
            , mChanged(false)
        {
        }

        /**
         * @brief Gets the data that is stored on the port.
         * @return Shared pointer that might be nullptr if there is no data set.
         */
        [[nodiscard]] inline std::shared_ptr<const Data> getData()
        {
            return getDataImpl();
        }

        /**
         * @brief Set the data to this input port.
         *
         * @param data
         */
        void setData(std::shared_ptr<const Data> data)
        {
            setDataImpl(std::move(data));
        }

        /**
         * @brief Gets the data type that is accepted by the port.
         * @return Accepted data type.
         */
        [[nodiscard]] virtual Type getDataType() const = 0;

        /**
         * @brief Flag that determines whether a valid input port is required for executing the algorithm.
         */
        bool required;

        /**
         * @brief Gets a flag that marks whether the input port has changed its value since the last call of resetChanged().
         * @return Flag that indicates that the input port has received new data.
         */
        [[nodiscard]] bool changed() const { return mChanged; }

        /**
         * @brief Resets the changed flag back to false.
         */
        void resetChanged() { mChanged = false; }

    protected:
        /**
         * @brief Sets flag that this parameter has changed.
         */
        void markChanged() { mChanged = true; }

        /**
         * @brief Gets the data that is stored on the port.
         * @return Shared pointer that might be nullptr if there is no data set.
         */
        [[nodiscard]] virtual std::shared_ptr<const Data> getDataImpl() const = 0;

        /**
         * @brief Sets the data that is stored on the port.
         * @param Shared pointer that might be nullptr if there is no data set.
         */
        virtual void setDataImpl(std::shared_ptr<const Data> data) = 0;

    private:
        /**
         * @brief Internal member that marks that the input port has changed its value since the last update. The algorithm class uses this to know what has changed. After the update call is completed, the Algorithm sets this flag back to false.
         */
        bool mChanged;
    };
}
