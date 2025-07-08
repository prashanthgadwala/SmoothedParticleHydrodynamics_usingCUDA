#include <vislab/core/iarchive.hpp>
#include <iostream>

namespace vislab
{
    std::shared_ptr<Object> IArchive::allocate(const std::string& uid)
    {
        return Factory::create(uid);
    }

    void IArchive::beginState(const char* name, IArchive::EState state)
    {
        // push new state to stacks
        mStates.push(state);
        mNames.emplace(name);
        // notify child class
        stateStarted(name, state);
    }

    void IArchive::endState(const char* name, IArchive::EState state)
    {
        // check if everything is as expected
        if (state != mStates.top())
        {
            throw std::runtime_error("Invalid sequence of state changes while archiving!");
        }
        if (std::string(name) != mNames.top())
        {
            throw std::runtime_error("Invalid sequence of state changes while archiving!");
        }
        // pop state from stacks
        mStates.pop();
        mNames.pop();
        // notify child class
        stateEnded(name, state);
    }

    IArchive::IArchive(IArchive::EState initialState)
        : mStates({ initialState })
        , mNames({ "root" })
    {
    }
    IArchive::IArchive()
        : mStates({ Group })
        , mNames({ "root" })
    {
    }
}
