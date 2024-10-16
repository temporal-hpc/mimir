#pragma once

#include <mimir/options.hpp>
#include <mimir/interop_view.hpp>

#include <functional> // std::function
#include <memory> // std::unique_ptr

namespace mimir
{

// Forward declarations

struct MimirEngine;
struct InteropView;

// C-style API

typedef struct MimirEngine* Engine;
typedef struct DeviceAllocation* Allocation;
typedef struct InteropView* View;

void createEngine(ViewerOptions opts, Engine *engine);
void createEngine(int width, int height, Engine *engine);
void destroyEngine(Engine engine);

void display(Engine engine, std::function<void(void)> func, size_t iter_count);
void displayAsync(Engine engine);
void prepareViews(Engine engine);
void updateViews(Engine engine);

void allocLinear(Engine engine, void **dev_ptr, size_t size, Allocation *alloc);
void createView(Engine engine, ViewParams params, View *view);
AttributeParams makeStructuredGrid(Engine engine, uint3 size, float3 start={0.f,0.f,0.f});

// C++ API

typedef std::unique_ptr<MimirEngine> EngineHandle;

EngineHandle make(int width, int height);
EngineHandle make(ViewerOptions opts);

} // namespace mimir