#include <mimir/mimir.hpp>

namespace mimir
{

void deleteEngine(MimirEngine *handle)
{
    handle->exit();
    delete handle;
}

EngineHandle make(int width, int height)
{
    auto engine_ptr = new MimirEngine(MimirEngine::make(width, height));
    return EngineHandle(engine_ptr);
}

EngineHandle make(ViewerOptions opts)
{
    auto engine_ptr = new MimirEngine(MimirEngine::make(opts));
    return EngineHandle(engine_ptr);
}

} // namespace mimir