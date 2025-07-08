#pragma once

#include "data.hpp"
#include "event.hpp"
#include "iarchive.hpp"
#include "iparameter.hpp"
#include "parameter_fwd.hpp"
#include "traits.hpp"
#include "transfer_function.hpp"

namespace vislab
{
    /**
     * @brief Base class for a parameter.
     * @tparam TValueType Type of value that is stored in the parameter.
     */
    template <typename TValueType>
    class Parameter : public Concrete<Parameter<TValueType>, IParameter>
    {
    public:
        /**
         * @brief Type of the value stored in this parameter.
         */
        using ValueType = TValueType;

        /**
         * @brief Constructor.
         */
        Parameter()
            : Concrete<Parameter<TValueType>, IParameter>()
            , mHidden(false)
        {
            if constexpr (std::is_scalar_v<TValueType>)
                mValue = TValueType(0);
            else if constexpr (detail::is_eigen_v<TValueType>)
                mValue = TValueType::Zero();
            else
                mValue = TValueType();
        }

        /**
         * @brief Constructor.
         * @param value Initial value.
         */
        Parameter(const TValueType& value)
            : Concrete<Parameter<TValueType>, IParameter>()
            , mHidden(false)
            , mValue(value)
        {
        }

        /**
         * @brief Copy-assignment operator. Event listeners are not copied.
         * @param other Other parameter to copy from.
         * @return Reference to self.
         */
        Parameter& operator=(const Parameter& other)
        {
            mValue  = other.mValue;
            mHidden = other.mHidden;
            return *this;
        }

        /**
         * @brief Assignment operator for direct value assignment.
         * @param value New value to be set.
         */
        Parameter& operator=(const ValueType& value)
        {
            setValue(value);
            return *this;
        }

        /**
         * @brief Gets the value stored in this parameter.
         * @return Values stored in this parameter.
         */
        [[nodiscard]] inline const TValueType& getValue() const { return mValue; }

        /**
         * @brief Sets the value stored in this parameter.
         * @param value New value to be set.
         * @param notifyOnChange Flag that determines whether an event is raised to all listeners.
         */
        virtual void setValue(const TValueType& value, bool notifyOnChange = true)
        {
            if (mValue != value)
            {
                mValue = value;
                this->markChanged();
                if (notifyOnChange)
                    onChange.notify(this, &mValue);
            }
        }

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize from/into.
         */
        void serialize(IArchive& archive) override
        {
            archive("Value", mValue);
            archive("Hidden", mHidden);
        }

        /**
         * @brief Event that is raised when the value changes.
         */
        TEvent<Parameter, TValueType> onChange;

        /**
         * @brief Is the parameter hidden from the UI?
         * @return True if the parameter is hidden from the UI.
         */
        [[nodiscard]] inline bool isHidden() const { return mHidden; }

        /**
         * @brief Tests if another parameter stores the exact same value and whether they have the same visibilty.
         * @param other Other parameter to compare with.
         * @return True, if the values and visibility are the same.
         */
        [[nodiscard]] inline bool operator==(const Parameter& other) const
        {
            return mValue == other.mValue && mHidden == other.mHidden;
        }

        /**
         * @brief Hides this parameter.
         */
        void hide()
        {
            if (!mHidden)
            {
                mHidden = true;
                onHiddenChange.notify(this, &mHidden);
            }
        }

        /**
         * @brief Shows this parameter.
         */
        void show()
        {
            if (mHidden)
            {
                mHidden = false;
                onHiddenChange.notify(this, &mHidden);
            }
        }

        /**
         * @brief Event that is raised when the hidden flag changes.
         */
        BoolEvent onHiddenChange;

    protected:
        /**
         * @brief Data value stored by this parameter.
         */
        TValueType mValue;

        /**
         * @brief Flag that stored whether this parameter is hidden from the UI.
         */
        bool mHidden;

        void setValueImpl(std::any value) override
        {
            setValue(std::any_cast<TValueType>(value));
        }

        std::any getValueImpl() override
        {
            return std::any(mValue);
        }
    };
}
