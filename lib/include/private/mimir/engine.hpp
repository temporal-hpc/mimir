#pragma once

#include <vulkan/vulkan.h>
#include <cuda_runtime_api.h>

#include <functional> // std::function
#include <string> // std::string
#include <mutex> // std::mutex (HUD text guard)
#include <thread> // std::thread
#include <array> // std::array (keyboard state)
#include <cstdint> // uint8_t
#include <chrono> // std::chrono (camera frame timing)
#include <vector> // std::vector

#include <mimir/options.hpp>
#include <mimir/remote_protocol.hpp>
#include <mimir/view.hpp>

#include "api.hpp"
#include "camera.hpp"
#include "deletion_queue.hpp"
#include "device.hpp"
#include "framebuffer.hpp"
#include "interop.hpp"
#include "lod.hpp"
#include "metrics.hpp"
#include "pipeline.hpp"
#include "raytracing.hpp"
#include "swapchain.hpp"
#include "window.hpp"

namespace mimir
{

namespace
{
    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;
    // Timeout value for frame acquisition and synchronization structures
    // To remove the timeout, use std::numeric_limits<uint64_t>::max();
    static constexpr uint64_t frame_timeout = 2000000000;
}

struct AllocatedBuffer
{
    VkBuffer buffer;
    VkDeviceMemory memory;
};

struct SyncData
{
    VkFence frame_fence;
    VkSemaphore image_acquired;
    VkSemaphore render_complete;
};

struct VulkanQueue
{
    uint32_t family_index;
    VkQueue queue;
};

struct MimirInstance
{
    ViewerOptions options;

    VkInstance instance;
    PhysicalDevice physical_device;
    VulkanQueue graphics, present;
    VkDevice device;
    VkCommandPool command_pool;

    VkRenderPass render_pass;
    VkDescriptorSetLayout descriptor_layout;
    VkPipelineLayout pipeline_layout;
    VkDescriptorPool descriptor_pool;
    VkSurfaceKHR surface;

    Swapchain swapchain;
    PipelineBuilder pipeline_builder;
    //VmaAllocator allocator = nullptr;
    //VmaPool interop_pool   = nullptr;

    Framebuffer framebuffers;
    std::vector<VkCommandBuffer> command_buffers;
    std::vector<VkDescriptorSet> descriptor_sets;
    std::function<void(void)> gui_callback;
    // Input API (setScrollCallback / isKeyDown / isKeyPressed). scroll_callback is invoked on the
    // render thread during event processing. The key arrays are indexed by (int)Key: key_down is 1
    // while held; key_pressed latches on press and is cleared by isKeyPressed(). Plain byte arrays
    // (not std::atomic) so MimirInstance stays movable; accessed cross-thread via std::atomic_ref.
    std::function<void(double, double)> scroll_callback = {};
    std::array<uint8_t, static_cast<size_t>(Key::Count)> key_down{};
    std::array<uint8_t, static_cast<size_t>(Key::Count)> key_pressed{};

    // Depth buffer
    VkImage depth_image;
    VkDeviceMemory depth_memory;
    VkImageView depth_view;

    // Offscreen color targets used in headless mode (in place of swapchain images).
    std::vector<VkImage> offscreen_images;
    std::vector<VkDeviceMemory> offscreen_memory;
    // Index of the most recently rendered image (used to read back headless frames).
    uint32_t last_image_idx;

    // Persistent device-local buffer (BGRA, packed) exported to CUDA, used by mapFrameToCuda()
    // for zero-copy NVENC: the rendered image is copied here GPU->GPU and handed to the encoder
    // as a CUDA device pointer, so pixels never touch host memory. Created lazily.
    VkBuffer            frame_cuda_buf_    = VK_NULL_HANDLE;
    VkDeviceMemory      frame_cuda_mem_    = VK_NULL_HANDLE;
    cudaExternalMemory_t frame_cuda_extmem_ = nullptr;
    void               *frame_cuda_ptr_    = nullptr;

    // Host-visible staging buffer reused by readFrameBytes across frames (the host readback path).
    // Allocating/freeing it per frame is a heavyweight driver call that dominated readback; kept
    // here and recreated only when the resolution changes.
    VkBuffer            readback_buf_      = VK_NULL_HANDLE;
    VkDeviceMemory      readback_mem_      = VK_NULL_HANDLE;
    VkDeviceSize        readback_size_     = 0;

    // Synchronization structures
    std::array<SyncData, MAX_FRAMES_IN_FLIGHT> sync_data;
    interop::Barrier interop;

    uint64_t render_timeline;
    // One-shot latches for the createView geometry/shading resolution warnings, so a program with
    // many marker views logs each mismatch once instead of once per view.
    bool warned_pt_geometry = false;
    bool warned_sprite_shading = false;
    // Set by displayAsync() before the render thread starts: rendering runs on its own thread while
    // the caller drives the sim through prepareViews/updateViews. display() and renderHeadless leave
    // it false (single-threaded). Read by lodRasterOnComputeThread; never mutated after startup.
    bool threaded_display = false;
    // render_timeline value when the graphics (and its metrics query pool) were last (re)built.
    // Metrics results are only read once enough frames have accumulated since then, so a freshly
    // rebuilt (empty) query pool isn't read with WAIT and blocked on after a resize.
    uint64_t graphics_epoch = 0;
    bool running;
    // Interop lockstep handshake between the compute thread and the async render thread.
    // prepareViews() bumps render_request once per compute iteration to ask the render thread
    // for exactly one interop-synchronized frame; the render thread renders one such frame per
    // request. This keeps the count of GPU interop submissions equal to the number of compute
    // iterations (the CUDA-Vulkan timeline ping-pong requires strict 1:1 alternation), so no
    // orphaned interop-gated submit is ever left waiting on a signal that will never come.
    // Accessed cross-thread via std::atomic_ref (kept a plain member so MimirInstance stays
    // movable, since it is returned by value from make()).
    uint64_t render_request;
    // Optional sample-supplied HUD text (setHudText), rendered in the built-in overlay so a sample
    // can show its own metrics without touching ImGui. Heap-held behind a pointer so MimirInstance
    // stays movable (std::mutex is not movable); written by the compute thread, read by the render
    // thread under the mutex. Default-initialized so make()'s aggregate need not list it.
    struct HudPanel { std::mutex mutex; std::string text; };
    HudPanel* hud_panel = nullptr;
    // Built-in pause / single-step, accessed cross-thread via std::atomic_ref (default-initialized
    // so make()'s designated-initializer list need not mention them, like graphics_epoch). When
    // paused the sim is held (frames keep rendering); each queued step advances it exactly once.
    bool paused = false;
    uint64_t pending_steps = 0;
    std::thread rendering_thread;

    std::vector<AllocatedBuffer> uniform_buffers;
    std::vector<View*> views;
    GlfwContext window_context;
    Camera camera;
    // Wall-clock timestamp of the previous updateCamera() call, for frame-rate-independent
    // fly movement / auto-orbit. Zero-initialized; the first update seeds it and moves nothing.
    std::chrono::steady_clock::time_point last_camera_time{};

    // Deletion queues organized by lifetime
    struct {
        DeletionQueue context;
        DeletionQueue graphics;
        DeletionQueue views;
    } deletors;

    // Benchmarking
    metrics::GraphicsMonitor graphics_monitor;
    metrics::ComputeMonitor compute_monitor;

    // Path tracing (Shading::PathTraced). rt_enabled is true only when the instance
    // requested path tracing AND the device is RT-capable; otherwise the engine renders
    // the raster fallback (createView warns). The context is built once in initVulkan; its
    // extent-dependent frame resources are (re)built in initGraphics.
    bool rt_enabled = false;
    RayTracingContext raytracing{};

    // ---- Voxel box path tracing (CA3D-voxels): trace the LIVING cells as boxes, O(living) ----
    // Each frame the engine compacts the visible Voxels view's fine int-state grid into a positions
    // buffer (voxelCompactLiving) and hands the RT the per-frame count; the RT traces boxes (see
    // RayTracingContext::voxel_boxes). Set up in prepare() when a Voxels view is path-traced; empty
    // otherwise. state_cuda is the view's fine-state CUDA device pointer (the color-index interop buffer).
    struct VoxelPtView { struct View* view = nullptr; void* state_cuda = nullptr; };
    std::vector<VoxelPtView> voxel_pt_views{};
    void*           voxel_pt_positions_cuda = nullptr; // compacted living-cell centers (interop, CUDA side)
    VkDeviceAddress voxel_pt_positions_addr = 0;        // ... same buffer, read by the AABB writer (BDA)
    uint32_t*       voxel_pt_count_dev = nullptr;       // device counter for the compaction
    uint32_t        voxel_pt_N = 0;
    float3          voxel_pt_origin{0.f, 0.f, 0.f};
    float           voxel_pt_spacing = 1.f;
    float           voxel_pt_radius = 0.5f;             // box half-extent (world)
    // --lod M for voxel PT: max-pool the fine N^3 state to M^3 and trace the living COARSE cells as
    // bigger boxes (O(living coarse)). voxel_pt_M = 0 traces the fine living cells.
    uint32_t        voxel_pt_M = 0;
    int*            voxel_pt_coarse_state = nullptr;    // M^3 max-pooled state (device)
    float3          voxel_pt_coarse_origin{0.f, 0.f, 0.f};
    float           voxel_pt_coarse_spacing = 1.f;
    // Compact the visible Voxels view's living cells and set raytracing.voxel_prim_count. Called each
    // frame before recordUpdateScene when voxel PT is active. No-op if no PT voxel view is visible.
    void updateVoxelPtScene();
    // Shared LOD data-reduction stage (lib/src/lod.cpp). Engine-owned; initialized when
    // options.pt_lod_cells > 0. Path tracing consumes it via raytracing.lod; the raster point modes
    // (none/phong) reduce inline in the frame cmd and draw the reduced set via vkCmdDrawIndirect.
    LodContext lod_context{};

    // ---- In-shader Voxels grid-coarsening LOD (voxel_lod.slang) ----
    // Coarsens a ViewType::Voxels grid without materializing any coarse data: a dedicated pipeline
    // draws M^3 procedural points whose vertex stage max-pools each coarse cell's fine block on the
    // fly. Push constants mirror LodPush in voxel_lod.slang.
    // Layout MUST match voxel_lod.slang's LodPush under std430 push-constant rules: the two leading
    // uints are followed by 8 bytes of padding so each float4 lands on its 16-byte alignment
    // (gridOrigin @16, gridSpacing @32; total 48). A tight 40-byte struct would trip
    // VUID-VkGraphicsPipelineCreateInfo-layout-10069 (block not contained in the push range).
    struct VoxelLodPush {
        uint32_t fineN, coarseM;
        uint32_t _pad0 = 0, _pad1 = 0;
        float gridOrigin[4];   // fine grid world origin (mirrors makeStructuredGrid's `start`)
        float gridSpacing[4];  // fine grid world spacing per cell (makeStructuredGrid uses 1)
    };
    // Dedicated coarse-voxel pipeline. layout binds set 0 = the shared descriptor_layout (uniforms +
    // colorbuf) and set 1 = set_layout (the fine-state SSBO), with a vertex-stage push-constant range.
    struct VoxelLodPipeline {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    };
    VoxelLodPipeline voxel_lod_pipeline{};
    // Per-view record: the fine-state SSBO (the color-index interop buffer), its set-1 descriptor and
    // the push constants. One per qualifying Voxels view; drawElements routes these to the LOD path.
    struct VoxelLodView {
        View* view = nullptr;
        VkBuffer fine_state = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        VoxelLodPush push{};
    };
    std::vector<VoxelLodView> voxel_lod_views{};
    // Build the coarse-voxel pipeline (empty vertex input, point topology; geometry entry per domain).
    VoxelLodPipeline makeVoxelLodPipeline(DomainType domain);

    // Raster LOD (none/phong): the interop position buffer address + full particle count captured at
    // init, used each frame to record the reduction inline. Set only when the raster point-mode LOD
    // path is active (rt_enabled uses raytracing's own address instead).
    VkDeviceAddress lod_raster_pos_addr = 0;
    uint64_t        lod_raster_count    = 0;
    // CUDA-mapped device pointer aliasing the SAME positions as lod_raster_pos_addr, set at bind time
    // only when lod_context.usesCuda(). Fed to LodContext::reduceCuda on the CUDA reduction path (the
    // Vulkan N^3 accumulator is not allocated there, so recordReduction must NOT be used). nullptr on
    // the Vulkan-fallback path.
    const void*     lod_raster_pos_cuda = nullptr;
    // CPU wall-clock of the raster reduction (recordLodRaster's reduceCuda + cudaStreamSynchronize),
    // mirroring RayTracingContext::last_lod_ms. Only meaningful on the CUDA path: the Vulkan-fallback
    // reduction runs async, in-cmd, with no host stall, so a CPU timer there would be measuring
    // nothing but cmd-buffer recording overhead -- left at 0.0 (see recordLodRaster).
    double          last_lod_raster_ms  = 0.0;
    // True when the active raster-LOD view is an instanced mesh marker (phong-mesh / SphereMesh, incl.
    // voxel-LOD lit views routed to a cube mesh): the reduction feeds per-INSTANCE positions (binding 1)
    // and the draw is vkCmdDrawIndexedIndirect (fixed indexCount = lod_raster_index_count, varying
    // instanceCount). False for point modes (none/phong-sphere).
    bool            lod_raster_mesh     = false;
    // Template index count for the active raster-LOD mesh view (icosphere or cube). The indirect-args
    // build writes this as the draw's fixed indexCount -- must NOT be hardcoded to sphere_index_count,
    // which is 0 when the view uses the cube template (ensureCubeMesh, sphere never built).
    uint32_t        lod_raster_index_count = 0;
    // Records the per-frame reduction + indirect-args build for raster point modes, BEFORE the render
    // pass (compute cannot run inside one). No-op when the raster LOD path is inactive.
    void recordLodRaster(VkCommandBuffer cmd, uint32_t frame_idx);
    // True when the raster LOD's CUDA reduction must be issued from the COMPUTE thread (updateViews)
    // instead of inline while the render thread records the frame. Required whenever the interop
    // handshake is live on the threaded display path: the reduction runs on the interop stream, which
    // by then holds the NEXT step's cudaWaitExternalSemaphoresAsync, so a render-thread
    // cudaStreamSynchronize there waits for a Vulkan signal that only the frame being recorded can
    // produce -- a deadlock. Same rule the path tracer follows (see updateViews / reduceLodCompute).
    // False on the decoupled path (its reduction has a dedicated stream) and for the single-threaded
    // display() loop (which submits its frame before the handshake's wait is enqueued).
    bool lodRasterOnComputeThread() const;
    // Slot the raster LOD reduction writes and the draw reads. The compute-thread reduction uses a
    // single slot: the interop handshake lets exactly one step's CUDA work be in flight, so the next
    // reduction cannot start until the frame that reads this slot has completed. The inline path
    // keeps per-frame slots (nothing serializes the frames there).
    uint32_t lodRasterSlot(uint32_t frame_idx) const;
    // Runs the raster LOD CUDA reduction on the compute thread (called from updateViews, after the
    // sim kernel and before signalKernelFinish). No-op unless lodRasterOnComputeThread().
    void reduceLodRasterCompute();
    // Refresh copy-path interop Image views from their shared buffers (vkCmdCopyBufferToImage) before
    // the render pass. No-op for zero-copy aliased image views. See the ViewType::Image creation path.
    void recordImageCopies(VkCommandBuffer cmd);

    // Shared template icosphere for the instanced mesh marker mode (Geometry::Mesh /
    // MarkerOptions::RenderMode::SphereMesh). Built lazily the first time a mesh marker view is
    // created; every such view instances this one unit-sphere mesh, transformed per particle in the
    // vertex shader. index_count == 0 means it has not been built yet.
    VkBuffer sphere_vbo = VK_NULL_HANDLE;   // interleaved {position, normal} unit icosphere vertices
    VkBuffer sphere_ibo = VK_NULL_HANDLE;   // triangle indices, uint32
    uint32_t sphere_index_count = 0;
    void ensureSphereMesh(); // build sphere_vbo/ibo once (uses ViewerOptions::mesh_subdivisions)

    // Shared template unit cube for voxel LOD (instanced like the icosphere; 24 verts w/ per-face
    // normals, 36 indices). Built lazily the first time a voxel-LOD lit mesh view is set up.
    VkBuffer cube_vbo = VK_NULL_HANDLE;   // interleaved {position, normal}, 24 vertices
    VkBuffer cube_ibo = VK_NULL_HANDLE;   // 36 uint32 indices (12 triangles)
    uint32_t cube_index_count = 0;
    void ensureCubeMesh(); // build cube_vbo/ibo once

    // Temporal-accumulation state (path tracing). pt_accum_frame is the running count of frames
    // already averaged into the accumulator; it resets to 0 whenever the scene may have changed --
    // a new simulation iteration (advance_interop) or any camera movement (detected by comparing
    // the view matrix / fov against the previous frame). See renderFrame.
    uint32_t pt_accum_frame = 0;
    glm::mat4 pt_last_view{0.f};
    float pt_last_fov = -1.f;
    // Set by the host-lockstep loops (renderHeadless, serveRemote) after running a compute step,
    // since they render with advance_interop=false and would otherwise never report a scene
    // change to the accumulator. Consumed (cleared) by the next renderFrame.
    bool pt_scene_dirty = false;
    // Nanoseconds a serveRemote() compute() callback has reported spending on its optional "sort"
    // sub-stage (see mimir::reportSortTimeNs), accumulated since process start. serveRemote reads a
    // delta across each stats window (same pattern as its own local compute_ns_total) to report a
    // mean sort time/step, separate from the generic compute time. Plain member (not std::atomic) so
    // MimirInstance stays movable, like running/render_request above; accessed cross-thread via
    // std::atomic_ref.
    uint64_t sort_ns_total = 0;

    static MimirInstance make(ViewerOptions opts);
    static MimirInstance make(int width, int height);

    // Allocates linear device memory, equivalent to cudaMalloc(dev_ptr, size)
    LinearAlloc *allocLinear(void **dev_ptr, size_t size);
    // Allocates opaque device memory, equivalent to cudaMallocMipmappedArray()
    OpaqueAlloc *allocMipmap(cudaMipmappedArray_t *dev_arr,
         const cudaChannelFormatDesc *desc, cudaExtent extent, unsigned int num_levels = 1
    );

    // Allocates device memory initialized for representing a structured domain
    AttributeDescription makeStructuredGrid(Layout size, float3 start={0.f,0.f,0.f});
    AttributeDescription makeImageDomain();

    // View creation
    View *createView(ViewDescription *desc);
    VkBuffer createAttributeBuffer(VkDeviceSize size,
        VkBufferUsageFlags usage, VkDeviceMemory memory
    );

    void display(std::function<void(void)> func, size_t iter_count);
    void displayAsync();
    // Returns whether the simulation should advance this iteration: true if not paused, otherwise
    // consumes one queued single-step (if any). Drives both display()'s loop and the public
    // shouldStep() used by samples that run their own compute loop after displayAsync().
    bool consumeStep();
    void prepareViews();
    void updateViews();
    void deinit();
    void exit();
    PerformanceMetrics getMetrics();

    // True when this instance renders offscreen with no window/surface.
    bool isHeadless() const { return options.render_mode != RenderMode::Local; }

    // Renders iter_count frames with no window, calling func before each frame
    // (e.g. to advance a simulation). The last frame can be read back with saveFrameToPpm().
    void renderHeadless(std::function<void(void)> func, size_t iter_count);
    // Copies the most recently rendered offscreen frame into out (B8G8R8A8 bytes).
    void readFrameBytes(std::vector<unsigned char>& out);
    // Copies the most recently rendered offscreen frame into a persistent CUDA-mapped device
    // buffer (BGRA, packed, stride = width*4) entirely on the GPU and returns its CUDA device
    // pointer (nullptr on failure). For zero-copy NVENC; the buffer is created on first use.
    void *mapFrameToCuda();
    // Releases the zero-copy frame buffer so the next mapFrameToCuda() rebuilds it (e.g. after a
    // resolution change).
    void freeFrameCudaBuffer();
    // Writes the most recently rendered offscreen frame to a binary PPM (P6) file.
    void saveFrameToPpm(const char *path);
    // Runs the workload continuously (with or without a viewer) and streams rendered frames
    // to a connected client, applying the control events it sends back (camera, pause).
    // Renders headless; returns once max_iters compute steps elapse (0 = run forever).
    void serveRemote(uint16_t port, std::function<void(void)> compute, size_t max_iters,
        bool use_h264 = false,
        remote::TransportKind kind = remote::TransportKind::Tcp,
        std::string token = {}, int bitrate_kbps = 8000, std::string stats_csv = {},
        int target_fps = 0, int steps_per_frame = 0, size_t pause_at = 0);

    void setGuiCallback(std::function<void(void)> callback) { gui_callback = callback; };
    void setScrollCallback(std::function<void(double, double)> cb) { scroll_callback = cb; };

    void initVulkan();
    void prepare();
    // Renders and submits a single frame. When advance_interop is true, the submission carries
    // the CUDA-Vulkan interop timeline wait/signal (used for one frame per compute iteration in
    // synchronized mode); otherwise it is a plain frame that does not touch the interop timeline.
    void renderFrame(bool advance_interop = false);
    // Advance the interactive camera once per frame: WASD fly movement (Fly mode) or the scripted
    // auto-orbit (orbit_speed > 0). Computes its own frame dt. No-op without a window.
    void updateCamera();
    void drawElements(uint32_t image_idx);
    void waitKernelStart();
    void signalKernelFinish();
    void waitTimelineHost();

    // Vulkan core-related functions
    void createInstance();
    void createSyncObjects();
    void updateDescriptorSets();

    // Swapchain-related functions
    void initGraphics();
    void cleanupGraphics();
    void recreateGraphics();
    // Creates the offscreen color image ring used in headless mode.
    void createOffscreenTarget(int width, int height);
    void createViewPipelines();
    void initUniformBuffers();
    void updateUniformBuffers(uint32_t image_idx);

    void immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function);
    void loadTexture(TextureDescription desc, void *img_data, size_t memsize);
    void copyBufferToTexture(VkBuffer buffer, VkImage image, VkExtent3D extent);
    void generateMipmaps(VkImage image, VkFormat img_format,
        int img_width, int img_height, int mip_levels
    );
    void transitionImageLayout(VkImage image,
        VkImageLayout old_layout, VkImageLayout new_layout
    );
};

static_assert(std::is_default_constructible_v<MimirInstance>);
//static_assert(std::is_nothrow_default_constructible_v<MimirInstance>);
//static_assert(std::is_trivially_default_constructible_v<MimirInstance>);

} // namespace mimir