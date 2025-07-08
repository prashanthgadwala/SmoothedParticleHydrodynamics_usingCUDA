#include "vislab/core/ialgorithm.hpp"

#include "vislab/core/iarchive.hpp"

#include <stdexcept>
#include <thread>

namespace vislab
{

    IAlgorithm::IAlgorithm()
    {
        active.setValue(true, false);
    }

    void IAlgorithm::serialize(IArchive& archive)
    {
        archive("Active", active);
        // parameters
        for (auto p : getParameters())
            archive(p.first.c_str(), *p.second);
    }

    UpdateInfo IAlgorithm::update(ProgressInfo& progressInfo)
    {
        // Check if the input ports are ready.
        for (auto port : getInputPorts())
            if (port.second->required && (port.second->getData() == nullptr || !port.second->getData()->isValid()))
                return UpdateInfo::reportError("The required input port '" + port.first + "' was not set or is invalid.");

        // Check if the output ports are ready.
        for (auto port : getOutputPorts())
            if (port.second->required && (port.second->getData() == nullptr || !port.second->getData()->isValid()))
                return UpdateInfo::reportError("The output port '" + port.first + "' was not set or is invalid.");

        // Check if the parameters are ready.
        for (auto param : getParameters())
            if (!param.second->isValid())
                return UpdateInfo::reportError("The parameter '" + param.first + "' is not valid.");

        progressInfo.noJobsDone();
        // call internal update
        UpdateInfo updateInfo = this->internalUpdate(progressInfo);

        // if the computation was successful (valid) set all jobs to done (this has no effect if already done by the algorithm)
        // if not set no jobs done (also has no effect if already done by algorithm)
        if (updateInfo.success())
        {
            // Check if the output ports are valid after the update.
            for (auto port : getOutputPorts())
                if (port.second->getData() == nullptr || !port.second->getData()->isValid())
                    return UpdateInfo::reportError("The output port '" + port.first + "' was not set or is invalid after the update.");

            progressInfo.allJobsDone();
        }
        else
        {
            progressInfo.noJobsDone();
        }

        // reset the dirty flags
        for (auto port : getInputPorts())
            port.second->resetChanged();
        for (auto param : getParameters())
            param.second->resetChanged();

        return updateInfo;
    }

    std::future<UpdateInfo> IAlgorithm::updateAsync(ProgressInfo& progress)
    {
        return std::async(
            static_cast<UpdateInfo (IAlgorithm::*)(ProgressInfo&)>(&IAlgorithm::update),
            this,
            std::ref(progress));
    }

    UpdateInfo IAlgorithm::update()
    {
        ProgressInfo progressInfo;
        return update(progressInfo);
    }

    void IAlgorithm::allocateOutputs()
    {
        for (auto& [name, outputPort] : getOutputPorts())
        {
            auto constructed = outputPort->getDataType().construct_shared<Data>();
            if (!constructed)
            {
                throw std::runtime_error("Default allocation of outputs not possible! Type does not have a default constructor.");
            }
            outputPort->setData(constructed);
        }
    }
}