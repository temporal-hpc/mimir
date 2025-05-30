#pragma once

#include <mimir/options.hpp>
#include <mimir/view.hpp>

#include <cuda_runtime_api.h>

#include <functional> // std::function

namespace mimir
{

// Forward declarations
struct MimirInstance;
struct View;
struct Texture;

// Opaque handle declarations
typedef struct MimirInstance* InstanceHandle;
typedef struct View* ViewHandle;
typedef struct Texture* TextureHandle;

// Initializes a Mimir engine instance with a window of the specified dimensions.
// Resources managed by this instance must be cleaned up by calling destroyInstance().
void createInstance(int width, int height, InstanceHandle *engine);

// Initializes a Mimir engine instance with additional option values passed as argument.
void createInstance(ViewerOptions opts, InstanceHandle *engine);

// Destroys an engine instance created with createInstance().
// The destroyed engine handle cannot be used after this call.
void destroyInstance(InstanceHandle engine);

// Query to check if a visualization window created by this engine is open.
bool isRunning(InstanceHandle engine);

// Starts display and blocks program execution until the display window closes
// The function passed as argument can perform updates over interop-mapped memory,
// as it is
void display(InstanceHandle engine, std::function<void(void)> func, size_t iter_count);

// Starts display and returns immediately, allowing program execution to continue.
// After calling this function, it is possible to write to interop-mapped memory
// by calling prepareViews and updateViews.
void displayAsync(InstanceHandle engine);

// Starts a GPU interop critical section.
// Code between this call and updateViews() is considered CUDA compute work,
// so Vulkan cannot read interop-mapped data during this period.
void prepareViews(InstanceHandle engine);

// Ends a GPU interop critical section started by prepareViews().
void updateViews(InstanceHandle engine);

// Allocates linear interop-mapped memory as per cudaMalloc().
void allocLinear(InstanceHandle engine, void **dev_ptr, size_t size, AllocHandle *alloc);

// Allocates opaque interop-mapped memory as per cudaMallocMipmappedArray().
void allocMipmap(InstanceHandle engine, cudaMipmappedArray_t *dev_arr,
    const cudaChannelFormatDesc *desc, cudaExtent extent, unsigned int num_levels,
    AllocHandle *alloc
);

// Creates a view structure and registers it with an existing engine instance
// The returned handle to the created view can be used to modify its parameters after creating it.
void createView(InstanceHandle engine, ViewDescription *desc, ViewHandle *view);

// Switches view state between visible and invisible; does not modify underlying data.
bool toggleVisibility(ViewHandle view);

// Sets the default color for the elements in this view.
void setViewDefaultColor(ViewHandle view, float4 color);

// Scales the elements of a view by a factor for each cartesian axis (X,Y,Z).
void scaleView(ViewHandle view, float3 scale);

// Translates elements of a view.
void translateView(ViewHandle view, float3 pos);

// Rotates elements of a view using angles in radians.
void rotateView(ViewHandle view, float3 rot);

// Translates camera to the specified position.
void setCameraPosition(InstanceHandle handle, float3 pos);

// Rotates camera to the specified angle.
void setCameraRotation(InstanceHandle handle, float3 rot);

// Adds a GUI callback function that gets called after the engine GUI function (if enabled).
// The callback function can be used to call ImGUI functions to display additional GUI elements.
void setGuiCallback(InstanceHandle engine, std::function<void(void)> callback);

// Helper function to generate a regular grid
// The returned attribute description contains all values needed to use the generated data
// inside a view description.
AttributeDescription makeStructuredGrid(InstanceHandle engine, Layout extent,
    float3 start={0.f,0.f,0.f}
);

// Helper function to generate a square frame for placing an image.
AttributeDescription makeImageFrame(InstanceHandle engine);

// Helper function to copy data from a linear memory array to an interop texture defined
// in the texture description parameter.
void copyTextureData(InstanceHandle engine, TextureDescription tex_desc, void *data, size_t memsize);

// Closes the display window if open.
void exit(InstanceHandle engine);

// Prints metrics for the current engine.
PerformanceMetrics getMetrics(InstanceHandle engine);

} // namespace mimir