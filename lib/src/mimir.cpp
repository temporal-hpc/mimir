#include <mimir/mimir.hpp>

#include "internal/engine.hpp"

namespace mimir
{

void createEngine(ViewerOptions opts, Engine *engine)
{
    *engine = new MimirEngine(MimirEngine::make(opts));
}

void createEngine(int width, int height, Engine *engine)
{
    *engine = new MimirEngine(MimirEngine::make(width, height));
}

void destroyEngine(Engine engine)
{
    engine->exit();
    delete engine;
}

bool isRunning(Engine engine) { return engine->running; }

void allocLinear(Engine engine, void **dev_ptr, size_t size, AllocHandle *alloc)
{
    *alloc = engine->allocLinear(dev_ptr, size);
}

void allocMipmap(Engine engine, cudaMipmappedArray_t *dev_arr, const cudaChannelFormatDesc *desc,
    cudaExtent extent, unsigned int num_levels, AllocHandle *alloc)
{
    *alloc = engine->allocMipmap(dev_arr, desc, extent, num_levels);
}

void createView(Engine engine, ViewDescription *desc, ViewHandle *handle)
{
    *handle = engine->createView(desc);
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

void setGuiCallback(Engine engine, std::function<void(void)> callback)
{
    engine->setGuiCallback(callback);
}

AttributeDescription makeStructuredGrid(Engine engine, ViewExtent extent, float3 start)
{
    return engine->makeStructuredGrid(extent, start);
}

AttributeDescription makeImageFrame(Engine engine)
{
    return engine->makeImageDomain();
}

void copyTextureData(Engine engine, TextureDescription tex_desc, void *data, size_t memsize)
{
    engine->loadTexture(tex_desc, data, memsize);
}

void makeTexture(Engine engine, TextureDescription desc, TextureHandle *texture)
{
    *texture = engine->makeTexture(desc);
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