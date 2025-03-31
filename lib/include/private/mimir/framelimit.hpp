#pragma once

#include <cstdint> // int64_t

namespace mimir
{

void frameStall(int64_t target_rate);
int64_t getTargetFrameTime(bool enable, int target_fps);

} // namespace mimir