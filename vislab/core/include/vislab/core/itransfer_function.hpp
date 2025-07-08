#pragma once

#include "data.hpp"

namespace vislab
{
    /**
     * @brief Interface for transfer functions.
     */
    class ITransferFunction : public Interface<ITransferFunction, Data>
    {
    public:
        /**
         * @brief Constructor with default transfer function from [0,1]. 
         */
        ITransferFunction();

        /**
         * @brief Constructor. Receives global minimal and maximal transfer function bounds.
         * @param minValue Minimum bound for the transfer function.
         * @param maxValue Maximum bound for the transfer function.
         */
        ITransferFunction(double minValue, double maxValue);

        /**
         * @brief Serializes the object into/from an archive.
         * @param archive Archive to serialize into/from.
         */
        void serialize(IArchive& archive) override;

        /**
         * @brief Global minimal bounds of the transfer function.
         */
        double minValue;

        /**
         * @brief Global maximal bounds of the transfer function.
         */
        double maxValue;
    };
}
