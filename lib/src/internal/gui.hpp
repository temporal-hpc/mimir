#pragma once

#include <mimir/mimir.hpp>

namespace mimir
{

void addViewObjectGui(InteropView *view_ptr, int uid);
void addViewObjectGui(std::shared_ptr<InteropView2> view, int uid);

} // namespace mimir