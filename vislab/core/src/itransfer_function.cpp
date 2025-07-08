#include <vislab/core/itransfer_function.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    ITransferFunction::ITransferFunction()
        : minValue(0)
        , maxValue(1)
    {
    }

    ITransferFunction::ITransferFunction(double minValue, double maxValue)
        : minValue(minValue)
        , maxValue(maxValue)
    {
    }

    void ITransferFunction::serialize(IArchive& archive)
    {
        archive("MinValue", minValue);
        archive("MaxValue", maxValue);
    }
}
