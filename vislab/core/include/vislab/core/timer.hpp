#pragma once

#include <chrono>

namespace vislab
{
    /**
     * @brief A basic timer class to measure elapsed CPU time in milliseconds.
     */
    class Timer
    {
    public:
        /**
         * @brief Begins the measurement.
         */
        void tic();

        /**
         * @brief Ends the measurement.
         * @return The resulting time in milliseconds.
         */
        [[nodiscard]] double toc() const;

    private:
        /**
         * @brief Beginning of the measurement.
         */
        std::chrono::high_resolution_clock::time_point t_start;
    };
}
