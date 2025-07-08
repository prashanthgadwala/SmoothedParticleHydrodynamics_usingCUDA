#include <vislab/core/progress_info.hpp>

#include <stdexcept>

namespace vislab
{
    ProgressInfo::ProgressInfo()
        : mJobs(1)
        , mJobsDone(0)
    {
    }

    ProgressInfo::ProgressInfo(const ProgressInfo& other)
    : mJobs(other.mJobs)
    , mJobsDone(other.mJobsDone.load())
    {
    }



    double ProgressInfo::getProgress() const
    {
        return mJobsDone / static_cast<double>(mJobs);
    }

    void ProgressInfo::jobDone()
    {
        mJobsDone++;
    }

    void ProgressInfo::jobsDone(uint64_t jobsDone)
    {
        mJobsDone += jobsDone;
    }

    void ProgressInfo::allJobsDone()
    {
        mJobsDone = mJobs;
    }

    void ProgressInfo::setTotalJobs(uint64_t jobs)
    {
        if (jobs == 0)
        {
            throw std::invalid_argument("Number of jobs must be at least 1.");
        }
        mJobs = jobs;
    }

    void ProgressInfo::onReport(double progress)
    {
    }
    void ProgressInfo::noJobsDone()
    {
        mJobsDone = 0;
    }
    ProgressInfo& ProgressInfo::operator=(const ProgressInfo& other)
    {
        mJobs = other.mJobs;
        mJobsDone = other.mJobsDone.load();
        return *this;
    }
}
