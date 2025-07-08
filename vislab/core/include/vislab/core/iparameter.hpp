#pragma once

#include "data.hpp"
#include <any>
#include <utility>

namespace vislab
{
    /**
     * @brief Basic interface for a parameter.
     */
    class IParameter : public Interface<IParameter, Data>
    {
    public:
        /**
         * @brief Constructor.
         */
        IParameter()
            : mChanged(false)
        {
        }

        /**
         * @brief Gets a flag that marks whether the parameter has changed its value since the last call of resetChanged().
         * @return Flag that indicates that the parameter has received new data.
         */
        [[nodiscard]] bool changed() const { return mChanged; }

        /**
         * @brief Resets the changed flag back to false.
         */
        void resetChanged() { mChanged = false; }

        /**
         * @brief Set the value of the parameter from any.
         *
         * @param value anytype representing value of the parameter.
         */
        void setValue(std::any value) { setValueImpl(std::move(value)); }

        /**
         * @brief Get the value of the parameter as any.
         *
         * @return parameter value wrapped into any.
         */
        std::any getValue() { return getValueImpl(); }

    protected:
        /**
         * @brief Sets flag that this parameter has changed.
         */
        void markChanged() { mChanged = true; }

        /**
         * @brief Implementation of getValue() with any return type.
         *
         * @return value wrapped in std::any.
         */
        virtual std::any getValueImpl() = 0;

        /**
         * @brief Implementation of setValue() with any parameter type.
         *
         * @param value the new value wrapped into a std::any.
         */
        virtual void setValueImpl(std::any value) = 0;

    private:
        /**
         * @brief Internal member that marks that the input port has changed its value since the last update. The algorithm class uses this to know what has changed. After the update call is completed, the Algorithm sets this flag back to false.
         */
        bool mChanged;
    };
}
