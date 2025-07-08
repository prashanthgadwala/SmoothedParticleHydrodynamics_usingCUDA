#pragma once

#include "base_transfer_function.hpp"

#include "iarchive.hpp"
#include "transfer_function_fwd.hpp"

#include <map>

namespace vislab
{
    /**
     * @brief Discrete transfer function
     * @tparam TScalar Scalar value type of the output type.
     * @tparam TComponents Number of components that the transfer function maps to.
     */
    template <int64_t TComponents, typename TScalar>
    class TransferFunction : public Concrete<TransferFunction<TComponents, TScalar>, BaseTransferFunction<TComponents>>
    {
    public:
        /**
         * @brief Number of components that the transfer function maps to.
         */
        static const int64_t Components = TComponents;

        /**
         * @brief Scalar value type of the output type.
         */
        using Scalar = TScalar;

        /**
         * @brief Value type that is mapped to.
         */
        using ValueDouble = typename BaseTransferFunction<TComponents>::ValueDouble;

        /**
         * @brief Value type in which the transfer function is given.
         */
        using Value = Eigen::Vector<Scalar, TComponents>;

        /**
         * @brief Constructor with default transfer function from [0,1].
         */
        TransferFunction()
            : Concrete<TransferFunction<TComponents, TScalar>, BaseTransferFunction<TComponents>>()
        {
        }

        /**
         * @brief Constructor. Receives global minimal and maximal transfer function bounds.
         * @param minValue Minimum bound for the transfer function.
         * @param maxValue Maximum bound for the transfer function.
         */
        TransferFunction(double minValue, double maxValue)
            : Concrete<TransferFunction<TComponents, TScalar>, BaseTransferFunction<TComponents>>(minValue, maxValue)
        {
        }

        /**
         * @brief Applies the transfer function to a single value.
         * @param value Value from the domain of the transfer function.
         * @return Mapped value.
         */
        [[nodiscard]] ValueDouble map(double value) const override
        {
            // Handle trivial cases
            if (values.empty())
                return ValueDouble::Zero();
            if (values.size() == 1)
                return values.begin()->second.template cast<double>();

            // Apply global transfer function bounds
            double t = std::min(std::max(0.0, (value - this->minValue) / (this->maxValue - this->minValue)), 1.0);

            // auto lower = Values.lower_bound(t);
            auto lower = values.upper_bound(t);
            auto upper = lower;
            lower--;

            if (lower == values.end())
                return values.begin()->second.template cast<double>();
            if (upper == values.end())
                return values.rbegin()->second.template cast<double>();

            // Interpolate the value from the nearest entries
            double t0 = lower->first;
            double t1 = upper->first;

            assert(t0 <= t && t <= t1);

            t = std::min(std::max(0.0, (t - t0) / (t1 - t0)), 1.0);
            return lower->second.template cast<double>() + t * (upper->second - lower->second).template cast<double>();
        }

        /**
         * @brief Tests if two transfer functions are equal.
         * @param other Other transfer function to compare with.
         * @return True if equal.
         */
        [[nodiscard]] bool operator==(const TransferFunction& other) const
        {
            if (values.size() != other.values.size() || this->minValue != other.minValue || this->maxValue != other.maxValue)
                return false;
            return std::equal(values.begin(), values.end(), other.values.begin(), other.values.end());
        }

        /**
         * @brief Tests if two transfer functions are unequal.
         * @param other Other transfer function to compare with.
         * @return True if unequal.
         */
        [[nodiscard]] bool operator!=(const TransferFunction& other) const
        {
            return !operator==(other);
        }

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override
        {
            BaseTransferFunction<TComponents>::serialize(archive);
            archive("Values", values);
        }

        /**
         * @brief The discrete transfer function representation. The first value of the pair is the 'value' and the second value is the corresponding color.
         */
        std::map<double, Value> values;
    };
}
