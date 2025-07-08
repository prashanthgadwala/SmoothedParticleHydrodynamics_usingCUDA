#pragma once

#include "data.hpp"
#include "event.hpp"
#include "iarchive.hpp"
#include "numeric_parameter_fwd.hpp"
#include "parameter.hpp"
#include "traits.hpp"

namespace vislab
{
    /**
     * @brief Base class for parameters that have a min/max value range.
     * @tparam TValueType Type of value that is stored in the parameter.
     */
    template <typename TValueType>
    class NumericParameter : public Concrete<NumericParameter<TValueType>, Parameter<TValueType>>
    {
    public:
        /**
         * @brief Constructor.
         */
        NumericParameter()
            : Concrete<NumericParameter<TValueType>, Parameter<TValueType>>()
        {
            if constexpr (std::is_scalar_v<TValueType>)
            {
                this->mValue = TValueType(0);
                mMinValue    = -std::numeric_limits<TValueType>::max();
                mMaxValue    = std::numeric_limits<TValueType>::max();
            }
            else if constexpr (detail::is_eigen_v<TValueType>)
            {
                this->mValue = TValueType::Zero();
                mMinValue.setConstant(-std::numeric_limits<typename TValueType::Scalar>::max());
                mMaxValue.setConstant(std::numeric_limits<typename TValueType::Scalar>::max());
            }
            else
            {
                this->mValue = TValueType();
                mMinValue    = TValueType(-std::numeric_limits<TValueType>::max());
                mMaxValue    = TValueType(std::numeric_limits<TValueType>::max());
            }
        }

        /**
         * @brief Constructor.
         */
        NumericParameter(TValueType initialValue)
            : Concrete<NumericParameter<TValueType>, Parameter<TValueType>>(initialValue)
        {
            if constexpr (std::is_scalar_v<TValueType>)
            {
                this->mValue = initialValue;
                mMinValue    = -std::numeric_limits<TValueType>::max();
                mMaxValue    = std::numeric_limits<TValueType>::max();
            }
            else if constexpr (detail::is_eigen_v<TValueType>)
            {
                this->mValue = initialValue;
                mMinValue.setConstant(-std::numeric_limits<typename TValueType::Scalar>::max());
                mMaxValue.setConstant(std::numeric_limits<typename TValueType::Scalar>::max());
            }
            else
            {
                this->mValue = initialValue;
                mMinValue    = TValueType(-std::numeric_limits<TValueType>::max());
                mMaxValue    = TValueType(std::numeric_limits<TValueType>::max());
            }
        }

        /**
         * @brief Constructor.
         * @param value Initial value.
         * @param minValue Smallest possible value.
         * @param maxValue Largest possible value.
         */
        NumericParameter(const TValueType& value, const TValueType& minValue, const TValueType& maxValue)
            : Concrete<NumericParameter<TValueType>, Parameter<TValueType>>(value)
            , mMinValue(minValue)
            , mMaxValue(maxValue)
        {
        }

        /**
         * @brief Destructor.
         */
        virtual ~NumericParameter() = default;

        /**
         * @brief Copy-assignment operator. Event listeners are not copied.
         * @param other Other parameter to copy from.
         * @return Reference to self.
         */
        NumericParameter& operator=(const NumericParameter& other)
        {
            Parameter<TValueType>::operator=(other);
            mMinValue = other.mMinValue;
            mMaxValue = other.mMaxValue;
            return *this;
        }

        /**
         * @brief Sets the value stored in this parameter.
         * @param value New value to be set.
         * @param notifyOnChange Flag that determines whether an event is raised to all listeners.
         */
        void setValue(const TValueType& value, bool notifyOnChange = true) override
        {
            // clamp to the min/max range
            TValueType newValue;
            if constexpr (std::is_scalar_v<TValueType>)
            {
                newValue = std::min(std::max(mMinValue, value), mMaxValue);
            }
            else if constexpr (detail::is_eigen_v<TValueType>)
            {
                newValue = value.cwiseMin(mMaxValue).cwiseMax(mMinValue);
            }
            else
            {
                newValue = min(max(mMinValue, value), mMaxValue);
            }

            // update value if different
            if (this->mValue != newValue)
            {
                this->mValue = newValue;
                if (notifyOnChange)
                    this->onChange.notify(this, &this->mValue);
            }
        }

        /**
         * @brief Gets the smallest possible value. If none is set, this is -std::numeric_limits<TValueType>::max().
         * @return Smallest possible value of the numeric range.
         */
        [[nodiscard]] inline const TValueType& getMinValue() const { return mMinValue; }

        /**
         * @brief Sets the smallest possible value.
         * @param data New smallest possible value to be set.
         * @param notifyOnChange Flag that determines whether an event is raised to all listeners.
         */
        void setMinValue(const TValueType& data, bool notifyOnChange = true)
        {
            if (mMinValue != data)
            {
                mMinValue = data;
                if (notifyOnChange)
                    onMinChange.notify(this, &mMinValue);
            }
        }

        /**
         * @brief Gets the largest possible value. If none is set, this is std::numeric_limits<TValueType>::max().
         * @return Largest possible value of the numeric range.
         */
        [[nodiscard]] inline const TValueType& getMaxValue() const { return mMaxValue; }

        /**
         * @brief Sets the largest possible value.
         * @param data New largest possible value to be set.
         * @param notifyOnChange Flag that determines whether an event is raised to all listeners.
         */
        void setMaxValue(const TValueType& data, bool notifyOnChange = true)
        {
            if (mMaxValue != data)
            {
                mMaxValue = data;
                if (notifyOnChange)
                    onMaxChange.notify(this, &mMaxValue);
            }
        }

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize from/into.
         */
        void serialize(IArchive& archive) override
        {
            Parameter<TValueType>::serialize(archive);
            archive("MinValue", mMinValue);
            archive("MaxValue", mMaxValue);
        }

        /**
         * @brief Tests if another parameter stores the exact same values and whether they have the same visibilty.
         * @param other Other parameter to compare with.
         * @return True, if the values and visibility are the same.
         */
        bool operator==(const NumericParameter& other) const
        {
            return Parameter<TValueType>::operator==(other) && mMinValue == other.mMinValue && mMaxValue == other.mMaxValue;
        }

        /**
         * @brief Event that is raised when the smallest possible value changes.
         */
        TEvent<NumericParameter, TValueType> onMinChange;

        /**
         * @brief Event that is raised when the largest possible value changes.
         */
        TEvent<NumericParameter, TValueType> onMaxChange;

    protected:
        /**
         * @brief Smallest possible value stored by this parameter.
         */
        TValueType mMinValue;

        /**
         * @brief Largest possible value stored by this parameter.
         */
        TValueType mMaxValue;
    };
}
