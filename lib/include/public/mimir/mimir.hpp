#pragma once

#include <mimir/options.hpp>
#include <mimir/remote_protocol.hpp>
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

// Total size (bytes) of the selected GPU's Vulkan DEVICE_LOCAL memory heaps -- i.e. the VRAM the
// renderer can actually allocate from. On some datacenter GPUs this is smaller than the CUDA-
// reported total, so it's the honest ceiling for mimir's allocations.
size_t deviceLocalMemory(InstanceHandle engine);

// Starts display and blocks program execution until the display window closes
// The function passed as argument can perform updates over interop-mapped memory,
// as it is
void display(InstanceHandle engine, std::function<void(void)> func, size_t iter_count);

// Starts display and returns immediately, allowing program execution to continue.
// After calling this function, it is possible to write to interop-mapped memory
// by calling prepareViews and updateViews.
void displayAsync(InstanceHandle engine);

// Renders frames with no window (requires RenderMode::Headless). The function passed as
// argument runs before each of the iter_count frames, e.g. to advance a simulation over
// interop-mapped memory; it must synchronize its own CUDA work before returning.
// The last rendered frame can be written to disk with saveFrame().
void renderHeadless(InstanceHandle engine, std::function<void(void)> func, size_t iter_count);

// Writes the most recently rendered headless frame to a binary PPM (P6) image file.
void saveFrame(InstanceHandle engine, const char *path);

// Runs the workload continuously and streams rendered frames to a connected client, applying
// the control events it sends back (camera, pause). Requires RenderMode::Headless. The
// simulation is sovereign: it advances whether or not a viewer is connected (while unwatched,
// rendering/encoding are skipped and compute free-runs at full speed), so a days-long job can
// be visited briefly, left, and revisited later. Serves one client at a time; clients may
// connect and disconnect at any time. Returns when max_iters compute steps have elapsed
// (0 = run forever). 'func' advances the workload before each step (as in display()). When
// use_h264 is true and the library was built with ffmpeg support, frames are H.264-encoded
// before sending; otherwise raw frames are streamed. The client is told the actual codec in
// the Hello handshake. 'kind' selects the transport: TCP (default, works everywhere incl.
// ssh -L tunnels) or QUIC (UDP, TLS + congestion control, for direct connections; H.264 frames
// then ride unreliable QUIC datagrams). 'token' is an optional shared secret the client must
// present (empty = accept any client).
// 'bitrate_kbps' is the H.264 target bitrate (ignored for raw frames); temporally noisy content
// such as undenoised path tracing needs far more than the 8000 default to avoid ghosting.
// 'stats_csv' (optional) writes the per-second server telemetry to a CSV file
// (time_s,frame,fps,steps_s,kbps,encode_ms) for benchmarking; nullptr/empty disables it.
// 'fps' > 0 caps the streamed FRAME rate at that rate and sets the encoder's rate-control
// framerate (so bitrate_kbps is honored at that cadence); 0 = uncapped, sessions run at the
// natural render+encode+send rate, paced only by the link and the client.
// 'steps_per_frame' selects how the simulation relates to frame production:
//   0 (default) = decoupled: the sim runs on its own thread at full speed, and each streamed
//                 frame samples the latest state (monitoring; the viewer never slows the run,
//                 at the cost of a torn-latest read). 'fps' caps pixels-on-the-wire only.
//   N >= 1      = lockstep: advance exactly N sim steps, then render one frame, sequentially
//                 (tear-free, deterministic; good for recording/reproducing). N=1 is the
//                 classic 1-step-per-frame behavior. Here 'fps' paces both frames AND steps.
void serveRemote(InstanceHandle engine, unsigned short port,
    std::function<void(void)> func, size_t max_iters, bool use_h264 = false,
    remote::TransportKind kind = remote::TransportKind::Tcp,
    const char *token = "", int bitrate_kbps = 8000, const char *stats_csv = nullptr,
    int fps = 0, int steps_per_frame = 0
);

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