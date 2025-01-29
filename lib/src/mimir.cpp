#include <mimir/mimir.hpp>

#include "mimir/engine.hpp"

namespace mimir
{

void createEngine(ViewerOptions opts, EngineHandle *engine)
{
    *engine = new MimirEngine(MimirEngine::make(opts));
}

void createEngine(int width, int height, EngineHandle *engine)
{
    *engine = new MimirEngine(MimirEngine::make(width, height));
}

void destroyEngine(EngineHandle engine)
{
    engine->exit();
    delete engine;
}

bool isRunning(EngineHandle engine) { return engine->running; }

void allocLinear(EngineHandle engine, void **dev_ptr, size_t size, AllocHandle *alloc)
{
    *alloc = engine->allocLinear(dev_ptr, size);
}

void allocMipmap(EngineHandle engine, cudaMipmappedArray_t *dev_arr, const cudaChannelFormatDesc *desc,
    cudaExtent extent, unsigned int num_levels, AllocHandle *alloc)
{
    *alloc = engine->allocMipmap(dev_arr, desc, extent, num_levels);
}

void createView(EngineHandle engine, ViewDescription *desc, ViewHandle *handle)
{
    *handle = engine->createView(desc);
}

void display(EngineHandle engine, std::function<void(void)> func, size_t iter_count)
{
    engine->display(func, iter_count);
}

void displayAsync(EngineHandle engine)
{
    engine->displayAsync();
}

void prepareViews(EngineHandle engine)
{
    engine->prepareViews();
}

void updateViews(EngineHandle engine)
{
    engine->updateViews();
}

void setGuiCallback(EngineHandle engine, std::function<void(void)> callback)
{
    engine->setGuiCallback(callback);
}

AttributeDescription makeStructuredGrid(EngineHandle engine, ViewExtent extent, float3 start)
{
    return engine->makeStructuredGrid(extent, start);
}

AttributeDescription makeImageFrame(EngineHandle engine)
{
    return engine->makeImageDomain();
}

void copyTextureData(EngineHandle engine, TextureDescription tex_desc, void *data, size_t memsize)
{
    engine->loadTexture(tex_desc, data, memsize);
}

void makeTexture(EngineHandle engine, TextureDescription desc, TextureHandle *texture)
{
    *texture = engine->makeTexture(desc);
}

void exit(EngineHandle engine)
{
    engine->exit();
}

void getMetrics(EngineHandle engine)
{
    engine->showMetrics();
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