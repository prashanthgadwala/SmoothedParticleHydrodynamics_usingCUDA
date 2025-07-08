#include <vislab/core/update_info.hpp>

namespace vislab
{
    UpdateInfo::UpdateInfo(EValidationState state, const std::string& string)
        : mValidationState(state)
        , mValidationError(string)
    {
    }

    UpdateInfo UpdateInfo::reportValid()
    {
        return UpdateInfo(EValidationState::Valid, "");
    }

    UpdateInfo UpdateInfo::reportWarning(const std::string& string)
    {
        return UpdateInfo(EValidationState::Warning, string);
    }

    UpdateInfo UpdateInfo::reportError(const std::string& string)
    {
        return UpdateInfo(EValidationState::Error, string);
    }

    UpdateInfo::EValidationState UpdateInfo::getState() const
    {
        return mValidationState;
    }

    const std::string& UpdateInfo::getError() const
    {
        return mValidationError;
    }

    bool UpdateInfo::success() const
    {
        return getState() == UpdateInfo::EValidationState::Valid;
    }
}
