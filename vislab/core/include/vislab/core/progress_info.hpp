#pragma once

#include <atomic>

namespace vislab
{
    /**
     * @brief Class that logs the progress of an update. Derive from this class to handle reporting or logging.
     */
    class ProgressInfo
    {
    public:
        /**
         * @brief Constructor.
         */
        ProgressInfo();

        /**
         * @brief Copy constructor to handle copy of atomic value.
         *
         * @param other
         */
        ProgressInfo(const ProgressInfo& other);

        /**
         * @brief Copy assignment operator to handle copy of atomic value.
         * @param other
         * @return
         */
        ProgressInfo& operator=(const ProgressInfo& other);

        /**
         * @brief Gets the current progress of the computation in [0,1].
         * @return Progress in [0=nothing done, 1=complete].
         */
        [[nodiscard]] double getProgress() const;

        /**
         * @brief Tells the progress info that one job was completed.
         */
        void jobDone();

        /**
         * @brief Tells the progress info that multiple jobs were completed.
         * @param jobsDone Number of completed jobs.
         */
        void jobsDone(uint64_t jobsDone);

        /**
         * @brief Tells the progress info that all jobs are completed.
         */
        void allJobsDone();

        /**
         * @brief Tells the progress info that none of the jobs are completed.
         */
        void noJobsDone();

        /**
         * @brief Sets the total number of jobs.
         * @param jobs Total number of jobs.
         */
        void setTotalJobs(uint64_t jobs);

    protected:
        /**
         * @brief Virtual function that is called when an update occurred. Overwrite this function to handle the reports.
         * @param progress Current progress.
         */
        virtual void onReport(double progress);

    private:
        /**
         * @brief Total number of jobs that need to be finished.
         */
        uint64_t mJobs;

        /**
         * @brief Current number of completed jobs.
         */
        std::atomic_uint64_t mJobsDone;
    };
}
