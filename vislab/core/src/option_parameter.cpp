#include <vislab/core/option_parameter.hpp>

#include <vislab/core/iarchive.hpp>

namespace vislab
{
    OptionParameter::OptionParameter()
        : Concrete<OptionParameter, Parameter<int32_t>>()
    {
    }

    OptionParameter::OptionParameter(int32_t value, const std::vector<std::string>& labels)
        : Concrete<OptionParameter, Parameter<int32_t>>(value)
    {
        setLabels(labels, false);
    }

    const std::vector<std::string>& OptionParameter::getLabels() const
    {
        return mLabels;
    }

    void OptionParameter::setLabels(const std::vector<std::string>& newLabels, bool notifyOnChange)
    {
        mLabels = newLabels;
        if (notifyOnChange)
            onLabelsChange.notify(this, &mLabels);
    }

    std::string OptionParameter::getCurrentLabel() const
    {
        return mLabels.at(mValue);
    }

    void OptionParameter::serialize(IArchive& archive)
    {
        Parameter<int32_t>::serialize(archive);
        archive("Labels", mLabels);
    }

    bool OptionParameter::operator==(const OptionParameter& other) const
    {
        return Parameter<int32_t>::operator==(other) && mLabels == other.mLabels;
    }
}
