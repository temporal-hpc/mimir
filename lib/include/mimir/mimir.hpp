#pragma once

#include <mimir/engine/engine.hpp>

#include <memory> // std::unique_ptr

namespace mimir
{

struct MimirEngine;

typedef std::unique_ptr<MimirEngine, void(*)(MimirEngine*)> EngineHandle;

EngineHandle make(int width, int height);
EngineHandle make(ViewerOptions opts);

} // namespace mimir