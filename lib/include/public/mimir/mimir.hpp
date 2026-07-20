#pragma once

#include <mimir/options.hpp>
#include <mimir/remote_protocol.hpp>
#include <mimir/view.hpp>

#include <cuda_runtime_api.h>

#include <cstdint> // uint64_t (DeviceBufferLimits)
#include <functional> // std::function
#include <string>  // GpuCapabilities::name, gpuBanner

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

// Per-buffer size limits (bytes) that cap a SINGLE Vulkan buffer/allocation, independent of how
// much VRAM is free. A storage-buffer binding cannot exceed max_storage_buffer_range; a buffer
// object cannot exceed max_buffer_size (VK_KHR_maintenance4/Vulkan 1.3, 0 if unreported); one
// memory allocation cannot exceed max_memory_allocation_size. These -- not total VRAM -- are what
// make vkCreateBuffer fail with VK_ERROR_OUT_OF_DEVICE_MEMORY for a multi-GiB buffer on a card with
// hundreds of GB free (e.g. the path-tracing instance buffer at >~2^24 particles). 0 = unreported.
struct DeviceBufferLimits
{
    uint64_t max_storage_buffer_range;
    uint64_t max_buffer_size;
    uint64_t max_memory_allocation_size;
    // Max instances in a single top-level acceleration structure and max primitives in a single
    // bottom-level one (ray tracing). mimir's path tracer packs all particles as AABB primitives in
    // one BLAS (one TLAS instance), so max_primitive_count -- not max_instance_count (~2^24) -- is the
    // ray-tracing particle ceiling; NVIDIA reports ~2^29 for it. 0 if the device lacks acceleration
    // structures.
    uint64_t max_instance_count;
    uint64_t max_primitive_count;
};
DeviceBufferLimits deviceBufferLimits(InstanceHandle engine);

// Maximum width/height (texels) of a single 2D VkImage on the selected device
// (VkPhysicalDeviceLimits::maxImageDimension2D) -- the analog of OpenGL's GL_MAX_TEXTURE_SIZE. A
// ViewType::Image cannot present a grid larger than this in any dimension regardless of free VRAM;
// callers must downsample or tile past it. 0 if no device is selected.
uint32_t maxImageDimension2D(InstanceHandle engine);

// ---- Device capability + memory-budget helpers (for a startup banner / pre-flight) -------------
// These need no InstanceHandle -- they query the CUDA device directly -- so a sample can report the
// GPU and reject an over-large workload BEFORE creating an instance, without re-deriving the CUDA/NVML
// boilerplate. Device ordinal is in the CUDA-visible set (i.e. 0 after a cudaSetDevice / --dev N).

// GPU capabilities for a banner / diagnostics. Core counts come from the compute-capability tables;
// the RT-core count is a best-effort estimate (NOT CUDA-queryable) -- 0 means no hardware RT (datacenter
// parts) and thus software BVH. NVENC/NVDEC presence is probed via NVML.
struct GpuCapabilities
{
    std::string name;
    size_t   vram_total_bytes    = 0;
    double   mem_bandwidth_gbps   = 0.0; // theoretical peak HBM/GDDR bandwidth (2 * clock * bus / 8)
    int      sm_count             = 0;
    int      cuda_cores           = 0;
    int      tensor_cores         = 0;
    int      rt_cores             = 0;   // 0 => software BVH
    bool     nvenc                = false;
    bool     nvdec                = false;
    double   power_usage_w        = 0.0; // instantaneous board draw at query time (0 if unreported)
    double   power_limit_w        = 0.0; // enforced power cap ("max"), 0 if unreported
};
GpuCapabilities queryGpuCapabilities(int device = 0);
// One-line human banner assembled from the caps, e.g.
//   "device 0 (NVIDIA B300 SXM6 AC) | 268 GB | 7672 GB/s mem BW | 148 SMs | 18944 CUDA cores | ..."
std::string gpuBanner(int device, const GpuCapabilities& caps);

// Instantaneous board power draw and the enforced power cap ("max"), in watts, via NVML. Unlike the
// static fields of queryGpuCapabilities, usage_w changes with load, so sample this periodically (e.g.
// once per telemetry second) for a live reading. Both 0 if the GPU/driver has no power telemetry.
struct GpuPower { double usage_w = 0.0; double limit_w = 0.0; };
GpuPower gpuPower(int device = 0);

// Device bytes/particle of mimir's OWN interop allocations: the position buffer (12 B, always) plus a
// per-particle AABB (24 B) only under path tracing WITHOUT LOD (LOD builds the BVH over occupied cells,
// not particles). Callers add any per-particle data of their own (a sim's attribute arrays) on top,
// then pass the sum to memoryBudget().
uint64_t interopBytesPerParticle(LightModel light_model, bool lod_active);

// GPU memory budget for `particle_count` at `bytes_per_particle`, from the CURRENTLY-FREE VRAM on the
// device -- a pre-flight so a too-large count is rejected cleanly before Vulkan OOMs mid-setup.
struct MemoryBudget
{
    size_t   free_bytes    = 0;
    size_t   total_bytes   = 0;
    uint64_t max_particles = 0;     // free_bytes / bytes_per_particle
    bool     fits          = false; // particle_count <= max_particles
};
MemoryBudget memoryBudget(uint64_t particle_count, uint64_t bytes_per_particle, int device = 0);

// Row-pitch alignment (in texels) the device requires for a LINEAR-tiled image of the given format
// -- e.g. 128 for R8_UNORM on NVIDIA. An interop Image view aliases a buffer to such an image, so a
// buffer whose width is NOT a multiple of this alignment renders sheared. Present through a buffer
// whose width is a multiple of the returned value. Returns 1 if unknown (no shear constraint).
uint32_t linearImageRowAlignment(InstanceHandle engine, FormatDescription format);

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

// Places the camera at `eye` looking toward `center`, with `up` the world up direction
// (usually {0,1,0}). This is the unambiguous way to aim the camera: no sign conventions to
// reverse-engineer -- the camera ends up at `eye` and looks straight at `center`.
//
// Convention (right-handed): at zero position/rotation the camera sits at the origin looking
// down +z, with +x to the right and +y up. setCameraPosition/setCameraRotation operate in that
// Euler frame; setCameraLookAt is the direct alternative when you know where the camera should be
// and what it should look at (e.g. framing a grid centered at the origin: eye behind it on -z,
// center = {0,0,0}).
void setCameraLookAt(InstanceHandle handle, float3 eye, float3 center, float3 up);

// Adds a GUI callback function that gets called after the engine GUI function (if enabled).
// The callback function can be used to call ImGUI functions to display additional GUI elements.
void setGuiCallback(InstanceHandle engine, std::function<void(void)> callback);

// Sets a block of text shown in the built-in HUD overlay (requires ViewerOptions::show_hud). Use it
// to surface a sample's own metrics -- e.g. a benchmark's compute/transfer/energy numbers -- straight
// from plain C++/CUDA/NVML code, with NO ImGui dependency or GUI code in the sample. Typically called
// once per frame with a freshly formatted, possibly multi-line string; it appears below the built-in
// FPS/render lines. Thread-safe: the engine copies the text under a lock, so the compute thread may
// call it while the render thread draws.
void setHudText(InstanceHandle engine, const char *text);

// Input API -- lets a sample handle scroll/keyboard without depending on GLFW or ImGui.
// setScrollCallback registers a function called (on the render thread, during event processing) with
// the mouse-wheel delta (dx, dy). isKeyDown returns whether a key is currently held; isKeyPressed
// returns true once per physical press (it consumes the latched press), for edge-triggered actions.
void setScrollCallback(InstanceHandle engine, std::function<void(double dx, double dy)> callback);
bool isKeyDown(InstanceHandle engine, Key key);
bool isKeyPressed(InstanceHandle engine, Key key);

// Built-in pause / single-step of the simulation. While paused the viewport keeps rendering (camera,
// HUD, etc. stay live) but the simulation is held. Space toggles pause and '.' queues one step at
// runtime; these functions are the programmatic equivalents. display() applies them automatically.
// For samples that run their own compute loop after displayAsync(), gate the sim advance on
// shouldStep(): `if (shouldStep(engine)) { launchKernel(); }` -- it returns true when not paused and
// otherwise consumes one queued step. isPaused() is a side-effect-free query.
void setPaused(InstanceHandle engine, bool paused);
bool isPaused(InstanceHandle engine);
void requestStep(InstanceHandle engine);
bool shouldStep(InstanceHandle engine);

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