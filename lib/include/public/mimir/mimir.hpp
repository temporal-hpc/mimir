#pragma once

#include <mimir/options.hpp>
#include <mimir/view.hpp>

#include <cuda_runtime_api.h>

#include <functional> // std::function
#include <memory> // std::unique_ptr

namespace mimir
{

// Forward declarations

struct MimirEngine;
struct View;
struct Texture;

// C-style API

typedef struct MimirEngine* Engine;
typedef struct Allocation* AllocHandle;
typedef struct View* ViewHandle;
typedef struct Texture* TextureHandle;

void createEngine(ViewerOptions opts, Engine *engine);
void createEngine(int width, int height, Engine *engine);
void destroyEngine(Engine engine);
bool isRunning(Engine engine);

void display(Engine engine, std::function<void(void)> func, size_t iter_count);
void displayAsync(Engine engine);
void prepareViews(Engine engine);
void updateViews(Engine engine);

void allocLinear(Engine engine, void **dev_ptr, size_t size, AllocHandle *alloc);
void allocMipmap(Engine engine, cudaMipmappedArray_t *dev_arr, const cudaChannelFormatDesc *desc,
    cudaExtent extent, unsigned int num_levels, AllocHandle *alloc
);
void createView(Engine engine, ViewDescription *desc, ViewHandle *view);
void setGuiCallback(Engine engine, std::function<void(void)> callback);
AttributeDescription makeStructuredGrid(Engine engine, ViewExtent extent, float3 start={0.f,0.f,0.f});
AttributeDescription makeImageFrame(Engine engine);
void copyTextureData(Engine engine, TextureDescription tex_desc, void *data, size_t memsize);
void getMetrics(Engine engine);
void exit(Engine engine);

// C++ API

typedef std::unique_ptr<MimirEngine> EngineHandle;

EngineHandle make(int width, int height);
EngineHandle make(ViewerOptions opts);

} // namespace mimir