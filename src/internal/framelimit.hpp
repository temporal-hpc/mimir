#pragma once

namespace mimir
{

void frameStall(unsigned target_rate);
int getTargetFrameTime(bool enable, int target_fps);

} // namespace mimir