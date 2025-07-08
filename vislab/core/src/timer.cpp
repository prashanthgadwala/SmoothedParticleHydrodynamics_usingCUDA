#include <vislab/core/timer.hpp>

namespace vislab
{
    void Timer::tic()
    {
        t_start = std::chrono::high_resolution_clock::now();
    }

    double Timer::toc() const
    {
        auto t_end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t_end - t_start).count();
    }
}
