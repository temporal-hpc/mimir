#include <mimir/mimir.hpp>

#include <mimir/engine/engine.hpp>

namespace mimir
{

void createEngine(ViewerOptions opts, Engine *engine)
{
    // TODO: Check and forward results
    *engine = new MimirEngine(MimirEngine::make(opts));
}

void createEngine(int width, int height, Engine *engine)
{
    // TODO: Check and forward results
    *engine = new MimirEngine(MimirEngine::make(width, height));
}

void destroyEngine(Engine engine)
{
    engine->exit(); // TODO: Check and forward results
    delete engine;
}

void allocLinear(Engine engine, void **dev_ptr, size_t size, Allocation *alloc)
{
    *alloc = engine->allocLinear(dev_ptr, size);
}

void createView(Engine engine, ViewParams params, View *view)
{
    *view = engine->createView(params);
}

void display(Engine engine, std::function<void(void)> func, size_t iter_count)
{
    engine->display(func, iter_count);
}

void displayAsync(Engine engine)
{
    engine->displayAsync();
}

void prepareViews(Engine engine)
{
    engine->prepareViews();
}

void updateViews(Engine engine)
{
    engine->updateViews();
}

AttributeParams makeStructuredGrid(Engine engine, uint3 size, float3 start)
{
    return engine->makeStructuredGrid(size, start);
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