#include "mimir/framelimit.hpp"

#include <time.h> // timespec
#include <chrono> // std::chrono::nanoseconds

namespace mimir
{

timespec nanotimeToTimespec(int64_t time)
{
    constexpr int64_t billion = 1000000000;
    timespec ts;
    ts.tv_nsec = time % billion;
    ts.tv_sec = time / billion;
    return ts;
}

int64_t timespecToNanotime(timespec *ts)
{
    constexpr int64_t billion = 1000000000;
    return static_cast<int64_t>(ts->tv_sec) * billion + ts->tv_nsec;
}

int64_t getNanotime()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return timespecToNanotime(&ts);
}

int64_t getElapsedTime(int64_t old_time)
{
    return getNanotime() - old_time;
}

int64_t getSleepTime(int64_t old_time, int64_t target)
{
    return target - getElapsedTime(old_time);
}

int frameSleep(uint64_t sleep_time)
{
    if (sleep_time <= 0) return 0;
    struct timespec ts = nanotimeToTimespec(sleep_time);
    return nanosleep(&ts, nullptr);
}

void frameStall(int64_t target_rate)
{
    static int64_t old_time = 0;
    static int64_t overhead = 0;

    if (target_rate <= 0) return;
    auto start = getNanotime();
    auto sleep_time = getSleepTime(old_time, target_rate);
    if (sleep_time > overhead)
    {
        auto adjusted_time = sleep_time - overhead;
        frameSleep(adjusted_time);
        overhead = (getElapsedTime(start) - adjusted_time + overhead * 99) / 100;
    }
    old_time = getNanotime();
}

int64_t getTargetFrameTime(bool enable, int target_fps)
{
    return (enable && target_fps > 0)? int64_t(1000000000.0 / target_fps) : 0;
}

} // namespace mimir