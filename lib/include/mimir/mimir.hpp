#pragma once

#include <mimir/options.hpp>
#include <mimir/view.hpp>

#include <functional> // std::function
#include <memory> // std::unique_ptr

namespace mimir
{

// Forward declarations

struct MimirEngine;
struct View;

// C-style API

typedef struct MimirEngine* Engine;
typedef struct Allocation* AllocHandle;
typedef struct View* ViewHandle;

void createEngine(ViewerOptions opts, Engine *engine);
void createEngine(int width, int height, Engine *engine);
void destroyEngine(Engine engine);

void display(Engine engine, std::function<void(void)> func, size_t iter_count);
void displayAsync(Engine engine);
void prepareViews(Engine engine);
void updateViews(Engine engine);

void allocLinear(Engine engine, void **dev_ptr, size_t size, AllocHandle *alloc);
void createView(Engine engine, ViewDescription *desc, ViewHandle *view);
void setGuiCallback(Engine engine, std::function<void(void)> callback);
AttributeDescription makeStructuredGrid(Engine engine, ViewExtent extent, float3 start={0.f,0.f,0.f});
AttributeDescription makeImageFrame(Engine engine);

// C++ API

typedef std::unique_ptr<MimirEngine> EngineHandle;

EngineHandle make(int width, int height);
EngineHandle make(ViewerOptions opts);

} // namespace mimir