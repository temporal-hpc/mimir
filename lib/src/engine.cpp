#include "mimir/engine.hpp"
#include "mimir/mimir.hpp"

#include <spdlog/cfg/env.h>

#include "mimir/api.hpp"
#include "mimir/framelimit.hpp"
#include "mimir/gui.hpp"
#include "mimir/resources.hpp"
#include "mimir/validation.hpp"
#include "mimir/shader_types.hpp"

#include <glm/gtc/matrix_transform.hpp> // glm::lookAt (fly camera world-to-view)

#include <atomic> // std::atomic_ref
#include <iostream>
#include <fstream> // std::ofstream
#include <algorithm> // std::max
#include <chrono> // std::chrono
#include <set> // std::set
#include <unordered_map> // std::unordered_map (icosphere edge cache)
#include <cmath> // std::sqrt
#include <cstring> // std::memcpy

namespace mimir
{

VkPresentModeKHR getDesiredPresentMode(PresentMode opts)
{
    switch (opts)
    {
        case PresentMode::Immediate:       return VK_PRESENT_MODE_IMMEDIATE_KHR;
        case PresentMode::VSync:           return VK_PRESENT_MODE_FIFO_KHR;
        case PresentMode::TripleBuffering: return VK_PRESENT_MODE_MAILBOX_KHR;
        default:                           return VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
}

uint32_t getAlignedSize(size_t original_size, size_t min_alignment)
{
	// Calculate required alignment based on minimum device offset alignment
	size_t aligned_size = original_size;
	if (min_alignment > 0)
    {
		aligned_size = (aligned_size + min_alignment - 1) & ~(min_alignment - 1);
	}
	return aligned_size;
}

// Creates a camera initialized with sensible defaults
Camera defaultCamera(int width, int height, float fov)
{
    auto camera = Camera::make();
    camera.type           = Camera::CameraType::LookAt;
    camera.rotation_speed = 0.5f;
    camera.setPosition(glm::vec3(0.f, 0.f, -2.85f));
    camera.setRotation(glm::vec3(0.f, 0.f, 0.f));
    camera.setPerspective(fov, (float)width / (float)height, 0.1f, 10000.f);
    return camera;
}

MimirInstance MimirInstance::make(ViewerOptions opts)
{
    MimirInstance engine{
        .options           = opts,
        .instance          = VK_NULL_HANDLE,
        .physical_device   = {},
        .graphics          = { .family_index = ~0u, .queue = VK_NULL_HANDLE },
        .present           = { .family_index = ~0u, .queue = VK_NULL_HANDLE },
        .device            = VK_NULL_HANDLE,
        .command_pool      = VK_NULL_HANDLE,
        .render_pass       = VK_NULL_HANDLE,
        .descriptor_layout = VK_NULL_HANDLE,
        .pipeline_layout   = VK_NULL_HANDLE,
        .descriptor_pool   = VK_NULL_HANDLE,
        .surface           = VK_NULL_HANDLE,
        .swapchain         = {},
        .pipeline_builder  = {},
        .framebuffers      = {},
        .command_buffers   = { VK_NULL_HANDLE },
        .descriptor_sets   = { VK_NULL_HANDLE },
        .gui_callback      = []() { return; },
        .depth_image       = VK_NULL_HANDLE,
        .depth_memory      = VK_NULL_HANDLE,
        .depth_view        = VK_NULL_HANDLE,
        .offscreen_images  = {},
        .offscreen_memory  = {},
        .last_image_idx    = 0,
        .sync_data         = { SyncData{
            .frame_fence = VK_NULL_HANDLE,
            .image_acquired = VK_NULL_HANDLE,
            .render_complete = VK_NULL_HANDLE
        } },
        .interop           = {
            .timeline_value = 0,
            .vk_semaphore   = VK_NULL_HANDLE,
            .cuda_semaphore = nullptr,
            .cuda_stream    = 0,
        },
        .render_timeline   = 0,
        .running           = false,
        .render_request    = 0,
        .rendering_thread  = {},
        .uniform_buffers   = {},
        .views             = {},
        .window_context    = {},
        .camera            = {},
        .deletors          = {},
        .graphics_monitor  = {},
        .compute_monitor   = {},
    };

#ifdef NDEBUG
    spdlog::set_level(spdlog::level::off);
#else
    spdlog::set_level(spdlog::level::trace);
#endif
    // SPDLOG_LEVEL=<level> overrides the build-type default (e.g. to get slang
    // compile diagnostics out of a release build).
    spdlog::cfg::load_env_levels();
    spdlog::set_pattern("[%H:%M:%S] [%l] %v");

    engine.options.present.target_frame_time = getTargetFrameTime(
        engine.options.present.enable_fps_limit, engine.options.present.target_fps
    );

    auto width  = engine.options.window.size.x;
    auto height = engine.options.window.size.y;
    // Headless instances render offscreen and create no window or surface.
    if (engine.options.render_mode == RenderMode::Local)
    {
        engine.window_context = GlfwContext::make(engine.options.window, &engine);
        engine.deletors.context.add([&] { engine.window_context.clean(); });
    }
    engine.camera = defaultCamera(width, height, engine.options.camera_fov);

    engine.initVulkan();

    return engine;
}

MimirInstance MimirInstance::make(int width, int height)
{
    ViewerOptions opts;
    opts.window.size = {width, height};
    return MimirInstance::make(opts);
}

void MimirInstance::deinit()
{
    if (rendering_thread.joinable())
    {
        rendering_thread.join();
    }
    if (interop.cuda_stream != nullptr)
    {
        validation::checkCuda(cudaStreamSynchronize(interop.cuda_stream));
    }

    vkDeviceWaitIdle(device);
    freeFrameCudaBuffer();
    if (readback_buf_ != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(device, readback_buf_, nullptr);
        vkFreeMemory(device, readback_mem_, nullptr);
        readback_buf_ = VK_NULL_HANDLE;
    }
    cleanupGraphics();
    if (!isHeadless())
    {
        gui::shutdown();
        window_context.exit();
    }
    deletors.views.flush();
    deletors.context.flush();
}

void MimirInstance::exit()
{
    window_context.exit();
}

void MimirInstance::prepare()
{
    initUniformBuffers();
    createViewPipelines();
    updateDescriptorSets();

    // Path tracing: bind the scene to the first Markers view's interop position buffer so the
    // per-frame TLAS follows the live particles. Done once, after views/pipelines exist.
    if (rt_enabled && !raytracing.scene_bound)
    {
        for (auto* view : views)
        {
            if (view->desc.type == ViewType::Markers && view->vb_count > 0)
            {
                auto c = view->desc.default_color; // particle albedo for PT (also --pcolor)
                // Positions are read by buffer-device-address in the AABB writer (no storage-range
                // cap), so hand bindScene the buffer's device address rather than the VkBuffer.
                VkBufferDeviceAddressInfo addr_info{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                    .pNext = nullptr, .buffer = view->vbo[0],
                };
                VkDeviceAddress pos_addr = vkGetBufferDeviceAddress(device, &addr_info);
                raytracing.lod_cells = options.pt_lod_cells;
                raytracing.bindScene(pos_addr, view->draw_count, view->desc.default_size,
                    glm::vec4(c.x, c.y, c.z, c.w));
                break;
            }
        }
        if (!raytracing.scene_bound)
        {
            spdlog::warn("Path tracing: no Markers view found to bind; RT frames will be empty");
        }
    }

    // Fly camera starts with the cursor captured for immediate mouse-look (TAB frees it for the
    // HUD). Skipped in headless (no window).
    if (window_context.window && options.camera_control == CameraControl::Fly)
    {
        window_context.cursor_captured = true;
        window_context.first_mouse = true;
        glfwSetInputMode(window_context.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }
}

void MimirInstance::displayAsync()
{
    prepare();
    std::atomic_ref<bool>(running).store(true, std::memory_order_release);
    rendering_thread = std::thread([&,this]()
    {
        // Number of interop-synchronized frames this thread has already produced. In sync mode
        // the compute thread bumps render_request once per iteration and we render exactly one
        // interop frame per outstanding request, keeping the GPU/CUDA timeline ping-pong 1:1.
        uint64_t served = 0;
        while(!window_context.shouldClose())
        {
            window_context.processEvents();
            updateCamera();
            if (options.present.enable_interop_sync)
            {
                auto requested = std::atomic_ref<uint64_t>(render_request).load(std::memory_order_acquire);
                if (requested > served)
                {
                    gui::draw(camera, options, views, gui_callback);
                    renderFrame(/*advance_interop=*/true);
                    served++;
                }
                else
                {
                    // No compute step pending: keep the window responsive without producing a
                    // frame that would read the shared buffer outside the interop handshake.
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                }
            }
            else
            {
                // Unsynchronized: free-run plain frames (no interop timeline participation).
                gui::draw(camera, options, views, gui_callback);
                renderFrame(/*advance_interop=*/false);
            }
        }
        std::atomic_ref<bool>(running).store(false, std::memory_order_release);
        vkDeviceWaitIdle(device);
    });
}

void MimirInstance::updateCamera()
{
    if (window_context.window == nullptr) { return; } // headless: nothing to drive

    auto now = std::chrono::steady_clock::now();
    if (last_camera_time.time_since_epoch().count() == 0)
    {
        last_camera_time = now; // seed on the first call; move nothing this frame
        return;
    }
    float dt = std::chrono::duration<float>(now - last_camera_time).count();
    last_camera_time = now;
    dt = std::min(dt, 0.1f); // clamp so a stall (resize, breakpoint) can't fling the camera

    // Scripted auto-orbit: circle the scene origin, always looking at it. Overrides manual input.
    if (options.orbit_speed > 0.f)
    {
        const glm::vec3 center(0.f);
        glm::vec3 rel = camera.position - center;
        float ang = glm::radians(options.orbit_speed * dt);
        float c = std::cos(ang), s = std::sin(ang);
        glm::vec3 rot(rel.x * c + rel.z * s, rel.y, -rel.x * s + rel.z * c); // rotate about +Y
        camera.setLookAt(center + rot, center, glm::vec3(0.f, 1.f, 0.f));
        return;
    }

    // Fly camera: WASD moves along the current view basis (mouse-look is in the cursor callback).
    // Only while the cursor is captured, so interacting with the HUD never drifts the camera.
    if (options.camera_control == CameraControl::Fly && window_context.cursor_captured)
    {
        auto* w = window_context.window;
        glm::vec3 fwd = glm::vec3(camera.matrices.view[2]); // world look direction
        const glm::vec3 world_up(0.f, 1.f, 0.f);
        // Screen-right in world space. NOT matrices.view[0]: setLookAt stores right = up x fwd,
        // which is the OPPOSITE of the glm::lookAt basis the raster now renders with, so using
        // it would invert A/D. cross(fwd, up) matches what the viewer sees (pitch is clamped to
        // +-89.9 deg, so fwd never parallels world_up and the cross is well-defined).
        glm::vec3 right = glm::normalize(glm::cross(fwd, world_up));

        glm::vec3 dir(0.f);
        if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) { dir += fwd; }
        if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) { dir -= fwd; }
        if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) { dir += right; }
        if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) { dir -= right; }
        if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS
         || glfwGetKey(w, GLFW_KEY_SPACE) == GLFW_PRESS) { dir += world_up; }
        if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS
         || glfwGetKey(w, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) { dir -= world_up; }

        if (glm::dot(dir, dir) > 0.f)
        {
            // Move the eye, then rebuild the roll-free FPS view (translate() would use the euler
            // path and reintroduce horizon roll). setFlyLook keeps the same yaw/pitch orientation.
            camera.position += glm::normalize(dir) * (options.camera_move_speed * dt);
            camera.setFlyLook();
        }
    }
}

void MimirInstance::prepareViews()
{
    if (options.present.enable_interop_sync && std::atomic_ref<bool>(running).load(std::memory_order_acquire))
    {
        // Request exactly one interop-synchronized frame from the render thread for this step.
        std::atomic_ref<uint64_t>(render_request).fetch_add(1, std::memory_order_release);
        waitKernelStart();
        compute_monitor.startWatch();
    }
}

void MimirInstance::waitKernelStart()
{
    static uint64_t wait_value = 1;
    cudaExternalSemaphoreWaitParams wait_params{};
    wait_params.flags = 0;
    wait_params.params.fence.value = wait_value;
    // Wait for Vulkan to complete its work
    validation::checkCuda(cudaWaitExternalSemaphoresAsync(
        &interop.cuda_semaphore, &wait_params, 1, interop.cuda_stream)
    );
    wait_value += 2;
}

void MimirInstance::updateViews()
{
    if (options.present.enable_interop_sync && std::atomic_ref<bool>(running).load(std::memory_order_acquire))
    {
        compute_monitor.stopWatch();
        signalKernelFinish();
    }
}

void MimirInstance::signalKernelFinish()
{
    static uint64_t signal_value = 2;
    cudaExternalSemaphoreSignalParams signal_params{};
    signal_params.flags = 0;
    signal_params.params.fence.value = signal_value;
    // Signal Vulkan to continue with the updated buffers
    validation::checkCuda(cudaSignalExternalSemaphoresAsync(
        &interop.cuda_semaphore, &signal_params, 1, interop.cuda_stream)
    );
    signal_value += 2;
}

void MimirInstance::display(std::function<void(void)> func, size_t iter_count)
{
    prepare();

    std::atomic_ref<bool>(running).store(true, std::memory_order_release);
    size_t iter_idx = 0;
    // Single-threaded lockstep: exactly one interop-synchronized frame per simulation step.
    bool interop = options.present.enable_interop_sync;
    while(!window_context.shouldClose())
    {
        window_context.processEvents();
        updateCamera();
        gui::draw(camera, options, views, gui_callback);
        renderFrame(/*advance_interop=*/interop);

        if (std::atomic_ref<bool>(running).load(std::memory_order_acquire)) waitKernelStart();
        if (iter_idx < iter_count)
        {
            func(); // Advance the simulation
            iter_idx++;
        }
        if (std::atomic_ref<bool>(running).load(std::memory_order_acquire)) signalKernelFinish();
    }
    std::atomic_ref<bool>(running).store(false, std::memory_order_release);
    vkDeviceWaitIdle(device);
}

void MimirInstance::renderHeadless(std::function<void(void)> func, size_t iter_count)
{
    prepare();
    // 'running' is left false so renderFrame() skips the interop timeline handshake:
    // here compute and rendering are serialized on the host (func is expected to
    // synchronize its own CUDA work before returning).
    auto frames = std::max<size_t>(iter_count, 1);
    for (size_t i = 0; i < frames; ++i)
    {
        if (i < iter_count) { func(); pt_scene_dirty = true; }
        renderFrame();
    }
    vkDeviceWaitIdle(device);
}

void MimirInstance::readFrameBytes(std::vector<unsigned char>& out)
{
    auto width  = swapchain.extent.width;
    auto height = swapchain.extent.height;
    VkDeviceSize memsize = static_cast<VkDeviceSize>(width) * height * 4;

    // Reuse a cached host-visible staging buffer across frames. Allocating + freeing one per frame
    // (vkCreateBuffer/vkAllocateMemory/... /vkFreeMemory, heavyweight driver calls) dominated the
    // readback cost; recreate only when the resolution (and so memsize) changes.
    if (readback_size_ != memsize)
    {
        if (readback_buf_ != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(device, readback_buf_, nullptr);
            vkFreeMemory(device, readback_mem_, nullptr);
        }
        readback_buf_ = createBuffer(device, memsize, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        VkMemoryRequirements mem_req{};
        vkGetBufferMemoryRequirements(device, readback_buf_, &mem_req);
        // Prefer HOST_CACHED: the readback does a large CPU-side memcpy *out* of this buffer, and
        // HOST_COHERENT memory is typically write-combined on NVIDIA -> uncached CPU reads that are
        // many times slower. Fall back to coherent if the device exposes no cached host type.
        const auto& memprops = physical_device.memory.memoryProperties;
        VkMemoryPropertyFlags want =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        if (findMemoryType(memprops, mem_req.memoryTypeBits, want) == ~0u)
        {
            want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }
        readback_mem_ = allocateMemory(device, memprops, mem_req, want);
        validation::checkVulkan(vkBindBufferMemory(device, readback_buf_, readback_mem_, 0));
        readback_size_ = memsize;
    }

    // The offscreen image is already in TRANSFER_SRC layout (render pass final layout)
    VkImage src = offscreen_images[last_image_idx];
    immediateSubmit([=, this](VkCommandBuffer cmd)
    {
        VkBufferImageCopy region{
            .bufferOffset      = 0,
            .bufferRowLength   = 0,
            .bufferImageHeight = 0,
            .imageSubresource  = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .imageOffset       = { 0, 0, 0 },
            .imageExtent       = { width, height, 1 },
        };
        vkCmdCopyImageToBuffer(cmd, src,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback_buf_, 1, &region
        );
    });

    unsigned char *data = nullptr;
    validation::checkVulkan(vkMapMemory(device, readback_mem_, 0, memsize, 0, (void**)&data));
    // Make the GPU's writes visible to this CPU read (a no-op on coherent memory, required on the
    // cached, non-coherent type preferred above).
    VkMappedMemoryRange range{ .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .pNext = nullptr, .memory = readback_mem_, .offset = 0, .size = VK_WHOLE_SIZE };
    vkInvalidateMappedMemoryRanges(device, 1, &range);
    out.resize(static_cast<size_t>(memsize));
    std::memcpy(out.data(), data, static_cast<size_t>(memsize));
    vkUnmapMemory(device, readback_mem_);
}

void MimirInstance::freeFrameCudaBuffer()
{
    // Release the zero-copy NVENC frame buffer (CUDA mapping first, then Vulkan memory). Safe to
    // call when nothing was allocated; mapFrameToCuda() lazily recreates it at the current size.
    if (frame_cuda_extmem_ != nullptr)
    {
        validation::checkCuda(cudaDestroyExternalMemory(frame_cuda_extmem_));
        frame_cuda_extmem_ = nullptr;
    }
    if (frame_cuda_buf_ != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(device, frame_cuda_buf_, nullptr);
        vkFreeMemory(device, frame_cuda_mem_, nullptr);
        frame_cuda_buf_ = VK_NULL_HANDLE;
    }
    frame_cuda_ptr_ = nullptr;
}

void *MimirInstance::mapFrameToCuda()
{
    auto width  = swapchain.extent.width;
    auto height = swapchain.extent.height;
    VkDeviceSize memsize = static_cast<VkDeviceSize>(width) * height * 4;

    // Lazily create a device-local buffer backed by CUDA-importable memory (same OPAQUE_FD
    // export mechanism as allocLinear), then keep reusing it across frames.
    if (frame_cuda_buf_ == VK_NULL_HANDLE)
    {
        VkExternalMemoryBufferCreateInfo extmem_info{
            .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
            .pNext       = nullptr,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
        };
        frame_cuda_buf_ = createBuffer(device, memsize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, &extmem_info);
        VkMemoryRequirements memreq{};
        vkGetBufferMemoryRequirements(device, frame_cuda_buf_, &memreq);

        VkExportMemoryAllocateInfoKHR export_info{
            .sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
            .pNext       = nullptr,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
        };
        auto available = physical_device.memory.memoryProperties;
        frame_cuda_mem_ = allocateMemory(device, available, memreq,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &export_info);
        validation::checkVulkan(vkBindBufferMemory(device, frame_cuda_buf_, frame_cuda_mem_, 0));

        frame_cuda_extmem_ = interop::importCudaExternalMemory(frame_cuda_mem_, memreq.size, device);
        cudaExternalMemoryBufferDesc buffer_desc{
            .offset = 0, .size = memsize, .flags = 0, .reserved = {},
        };
        validation::checkCuda(cudaExternalMemoryGetMappedBuffer(
            &frame_cuda_ptr_, frame_cuda_extmem_, &buffer_desc));
    }

    // Copy the rendered image (already in TRANSFER_SRC layout) into the CUDA-mapped buffer. This
    // is a GPU->GPU copy; immediateSubmit blocks until it completes, so CUDA can read it after.
    VkImage src = offscreen_images[last_image_idx];
    immediateSubmit([=, this](VkCommandBuffer cmd)
    {
        VkBufferImageCopy region{
            .bufferOffset      = 0,
            .bufferRowLength   = 0,
            .bufferImageHeight = 0,
            .imageSubresource  = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .imageOffset       = { 0, 0, 0 },
            .imageExtent       = { width, height, 1 },
        };
        vkCmdCopyImageToBuffer(cmd, src,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, frame_cuda_buf_, 1, &region);
    });
    return frame_cuda_ptr_;
}

void MimirInstance::saveFrameToPpm(const char *path)
{
    auto width  = swapchain.extent.width;
    auto height = swapchain.extent.height;
    std::vector<unsigned char> data;
    readFrameBytes(data);

    // Write a binary PPM (P6), converting the B8G8R8A8 image to RGB
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    for (uint32_t i = 0; i < width * height; ++i)
    {
        file.put(static_cast<char>(data[i * 4 + 2])); // R
        file.put(static_cast<char>(data[i * 4 + 1])); // G
        file.put(static_cast<char>(data[i * 4 + 0])); // B
    }
    spdlog::info("Saved headless frame ({}x{}) to {}", width, height, path);
}

uint32_t getVertexRate(ViewType type)
{
    switch (type)
    {
        case ViewType::Edges: { return 3; } // AKA TriangleMesh
        case ViewType::Boxes: { return 2; }
        default: return 1;
    }
}

constexpr VkIndexType getIndexBufferType(int bytesize)
{
    switch (bytesize)
    {
        case 2: return VK_INDEX_TYPE_UINT16;
        case 4: return VK_INDEX_TYPE_UINT32;
        // TODO: Add VK_INDEX_TYPE_UINT8_EXT for char and VK_INDEX_TYPE_NONE_KHR for default
        default: return VK_INDEX_TYPE_NONE_KHR;
    }
}

void initGridCoords(float3 *data, Layout size, float3 start)
{
    auto slice_size = size.x * size.y;
    for (uint32_t z = 0; z < size.z; ++z)
    {
        auto rz = start.z + static_cast<float>(z);
        for (uint32_t y = 0; y < size.y; ++y)
        {
            auto ry = start.y + static_cast<float>(y);
            for (uint32_t x = 0; x < size.x; ++x)
            {
                auto rx = start.x + static_cast<float>(x);
                data[slice_size * z + size.x * y + x] = float3{rx, ry, rz};
            }
        }
    }
}

AttributeDescription MimirInstance::makeStructuredGrid(Layout size, float3 start)
{
    assert(size.x > 0 || size.y > 0 || size.z > 0);
    auto memsize = sizeof(float3) * size.x * size.y * size.z;

    // Create test buffer for querying the desired memory properties
    auto domain_buffer = createBuffer(device, memsize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    VkMemoryRequirements memreq{};
    vkGetBufferMemoryRequirements(device, domain_buffer, &memreq);

    auto available = physical_device.memory.memoryProperties;
    auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    auto vk_memory = allocateMemory(device, available, memreq, flags);
    validation::checkVulkan(vkBindBufferMemory(device, domain_buffer, vk_memory, 0));
    float3 *data = nullptr;
    vkMapMemory(device, vk_memory, 0, memsize, 0, (void**)&data);
    initGridCoords(data, size, start);
    vkUnmapMemory(device, vk_memory);
    auto grid_alloc = new LinearAlloc({memreq.size, vk_memory, nullptr});
    deletors.context.add([=,this]{ delete grid_alloc; });

    // Add deletors to queue for later cleanup
    deletors.views.add([=,this]{
        spdlog::trace("Free structured domain memory");
        vkFreeMemory(device, vk_memory, nullptr);
        vkDestroyBuffer(device, domain_buffer, nullptr);
    });

    return AttributeDescription{
        .source   = grid_alloc,
        .size     = size.x * size.y * size.z,
        .format   = FormatDescription::make<float3>(),
        .indexing = {},
    };
}

AttributeDescription MimirInstance::makeImageDomain()
{
    const std::vector<Vertex> vertices{
        { {  1.f,  1.f, 0.f }, { 1.f, 1.f } },
        { { -1.f,  1.f, 0.f }, { 0.f, 1.f } },
        { { -1.f, -1.f, 0.f }, { 0.f, 0.f } },
        { {  1.f, -1.f, 0.f }, { 1.f, 0.f } }
    };
    // Indices for a single uv-view quad made from two triangles
    const std::vector<uint16_t> indices{ 0, 1, 2, 2, 3, 0 };//, 4, 5, 6, 6, 7, 4 };

    uint32_t vert_memsize = sizeof(Vertex) * vertices.size();
    uint32_t ids_memsize = sizeof(uint16_t) * indices.size();

    auto vbo = createBuffer(device, vert_memsize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    VkMemoryRequirements memreq{};
    vkGetBufferMemoryRequirements(device, vbo, &memreq);
    // Allocate memory and bind it to buffers
    auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    auto available = physical_device.memory.memoryProperties;
    auto vbo_mem = allocateMemory(device, available, memreq, flags);
    validation::checkVulkan(vkBindBufferMemory(device, vbo, vbo_mem, 0));
    auto vbo_alloc = new LinearAlloc({memreq.size, vbo_mem, nullptr});

    auto ibo = createBuffer(device, ids_memsize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    vkGetBufferMemoryRequirements(device, ibo, &memreq);
    // Allocate memory and bind it to buffers
    flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    auto ibo_mem = allocateMemory(device, available, memreq, flags);
    validation::checkVulkan(vkBindBufferMemory(device, ibo, ibo_mem, 0));
    auto ibo_alloc = new LinearAlloc({memreq.size, ibo_mem, nullptr});

    // Init image quad coords and indices
    char *vert_data = nullptr;
    vkMapMemory(device, vbo_mem, 0, vert_memsize, 0, (void**)&vert_data);
    std::memcpy(vert_data, vertices.data(), vert_memsize);
    vkUnmapMemory(device, vbo_mem);

    char *ids_data = nullptr;
    vkMapMemory(device, ibo_mem, 0, ids_memsize, 0, (void**)&ids_data);
    std::memcpy(ids_data, indices.data(), ids_memsize);
    vkUnmapMemory(device, ibo_mem);

    deletors.views.add([=,this]{
        vkFreeMemory(device, vbo_mem, nullptr);
        vkDestroyBuffer(device, vbo, nullptr);
        vkFreeMemory(device, ibo_mem, nullptr);
        vkDestroyBuffer(device, ibo, nullptr);
    });

    return AttributeDescription{
        .source   = vbo_alloc,
        .size     = (uint32_t)vertices.size(),
        .format   = FormatDescription::make<float3>(),
        .indexing = {
            .source     = ibo_alloc,
            .size       = (uint32_t)indices.size(),
            .index_size = sizeof(uint16_t),
        }
    };
}

void MimirInstance::ensureSphereMesh()
{
    if (sphere_index_count > 0) { return; } // already built (persists for the instance lifetime)

    // Unit icosphere (vertex positions double as normals). Same midpoint-subdivision geometry the
    // path tracer builds for its BLAS, so mesh raster and path tracing render matching spheres.
    const float t = (1.f + std::sqrt(5.f)) / 2.f;
    std::vector<glm::vec3> verts = {
        {-1,t,0},{1,t,0},{-1,-t,0},{1,-t,0}, {0,-1,t},{0,1,t},
        {0,-1,-t},{0,1,-t}, {t,0,-1},{t,0,1},{-t,0,-1},{-t,0,1},
    };
    for (auto& v : verts) { v = glm::normalize(v); }
    std::vector<glm::uvec3> faces = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11}, {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9}, {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1},
    };
    for (uint32_t s = 0; s < options.pt_subdivisions; ++s)
    {
        std::unordered_map<uint64_t, uint32_t> cache;
        auto midpoint = [&](uint32_t a, uint32_t b) -> uint32_t {
            uint64_t key = a < b ? ((uint64_t)a << 32 | b) : ((uint64_t)b << 32 | a);
            auto it = cache.find(key);
            if (it != cache.end()) { return it->second; }
            auto m = glm::normalize((verts[a] + verts[b]) * 0.5f);
            auto idx = static_cast<uint32_t>(verts.size());
            verts.push_back(m); cache.emplace(key, idx); return idx;
        };
        std::vector<glm::uvec3> next; next.reserve(faces.size() * 4);
        for (const auto& f : faces) {
            uint32_t a = midpoint(f.x, f.y), b = midpoint(f.y, f.z), c = midpoint(f.z, f.x);
            next.push_back({f.x,a,c}); next.push_back({f.y,b,a});
            next.push_back({f.z,c,b}); next.push_back({a,b,c});
        }
        faces.swap(next);
    }
    std::vector<uint32_t> indices; indices.reserve(faces.size() * 3);
    for (const auto& f : faces) { indices.insert(indices.end(), {f.x, f.y, f.z}); }

    VkDeviceSize vsize = verts.size() * sizeof(glm::vec3);
    VkDeviceSize isize = indices.size() * sizeof(uint32_t);
    auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    auto available = physical_device.memory.memoryProperties;
    VkMemoryRequirements memreq{};

    sphere_vbo = createBuffer(device, vsize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    vkGetBufferMemoryRequirements(device, sphere_vbo, &memreq);
    auto vbo_mem = allocateMemory(device, available, memreq, flags);
    validation::checkVulkan(vkBindBufferMemory(device, sphere_vbo, vbo_mem, 0));

    sphere_ibo = createBuffer(device, isize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    vkGetBufferMemoryRequirements(device, sphere_ibo, &memreq);
    auto ibo_mem = allocateMemory(device, available, memreq, flags);
    validation::checkVulkan(vkBindBufferMemory(device, sphere_ibo, ibo_mem, 0));

    void* p = nullptr;
    vkMapMemory(device, vbo_mem, 0, vsize, 0, &p); std::memcpy(p, verts.data(), vsize);
    vkUnmapMemory(device, vbo_mem);
    vkMapMemory(device, ibo_mem, 0, isize, 0, &p); std::memcpy(p, indices.data(), isize);
    vkUnmapMemory(device, ibo_mem);

    sphere_index_count = static_cast<uint32_t>(indices.size());
    VkBuffer vbo = sphere_vbo, ibo = sphere_ibo;
    deletors.views.add([=,this]{
        vkFreeMemory(device, vbo_mem, nullptr); vkDestroyBuffer(device, vbo, nullptr);
        vkFreeMemory(device, ibo_mem, nullptr); vkDestroyBuffer(device, ibo, nullptr);
    });
    spdlog::info("Mesh markers: icosphere subdiv {} ({} tris) built for instanced raster",
        options.pt_subdivisions, sphere_index_count / 3);
}

LinearAlloc *MimirInstance::allocLinear(void **dev_ptr, size_t size)
{
    assert(size > 0);

    VkBufferUsageFlags usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    // Path tracing reads the interop positions by buffer-device-address (in the AABB writer), so the
    // buffer needs the SHADER_DEVICE_ADDRESS usage and its memory the DEVICE_ADDRESS alloc flag. Only
    // on RT-capable devices, where bufferDeviceAddress is enabled (see device.cpp).
    if (rt_enabled) { usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT; }
    // Create temporary buffer for querying the desired memory properties
    VkExternalMemoryBufferCreateInfo extmem_info{
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    };
    auto query_buf = createBuffer(device, size, usage, &extmem_info);
    VkMemoryRequirements memreq{};
    vkGetBufferMemoryRequirements(device, query_buf, &memreq);

    // Allocate external device memory. When RT is on, chain the DEVICE_ADDRESS alloc flag so the
    // bound buffer can expose a device address for the BDA position read.
    VkExportMemoryAllocateInfoKHR export_info{
        .sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    };
    VkMemoryAllocateFlagsInfo addr_flags{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .pNext = &export_info,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, .deviceMask = 0,
    };
    auto available = physical_device.memory.memoryProperties;
    auto memflags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const void *alloc_chain = rt_enabled ? static_cast<const void*>(&addr_flags)
                                         : static_cast<const void*>(&export_info);
    auto vk_memory = allocateMemory(device, available, memreq, memflags, alloc_chain);
    // The real allocated amount is determined by the memory requirements structure
    spdlog::debug("Allocated {} bytes for interop ({} requested)", memreq.size, size);

    // Export and map the external memory to CUDA
    auto cuda_extmem = interop::importCudaExternalMemory(vk_memory, memreq.size, device);

    // Add deletors to queue for later cleanup
    deletors.views.add([=,this]{
        spdlog::trace("Free interop memory");
        validation::checkCuda(cudaDestroyExternalMemory(cuda_extmem));
        vkFreeMemory(device, vk_memory, nullptr);
    });
    vkDestroyBuffer(device, query_buf, nullptr);

    LinearAlloc alloc{
        .size        = memreq.size,
        .vk_mem      = vk_memory,
        .cuda_extmem = cuda_extmem
    };
    cudaExternalMemoryBufferDesc buffer_desc{ .offset = 0, .size = size, .flags = 0, .reserved = {} };
    validation::checkCuda(cudaExternalMemoryGetMappedBuffer(
        dev_ptr, alloc.cuda_extmem, &buffer_desc)
    );
    auto alloc_ptr = new LinearAlloc(alloc);
    deletors.context.add([=,this]{ delete alloc_ptr; });
    return alloc_ptr;
}

FormatDescription getFormatFromCuda(const cudaChannelFormatDesc *desc)
{
    // Convert format kind from CUDA enum to library enum class
    FormatKind kind;
    switch (desc->f)
    {
        case cudaChannelFormatKindSigned:   { kind = FormatKind::Signed; break; }
        case cudaChannelFormatKindUnsigned: { kind = FormatKind::Unsigned; break; }
        case cudaChannelFormatKindFloat: default: { kind = FormatKind::Float; break; }
    }
    // Get channel size in bits, assuming that all channels are the same (for now)
    int size = desc->x / 8;
    // A channel exists if its size is greater than zero
    int components = (desc->x > 0) + (desc->y > 0) + (desc->z > 0) + (desc->w > 0);
    return { .kind = kind, .size = size, .components = components };
}

VkExtent3D getExtentFromCuda(cudaExtent extent)
{
    return VkExtent3D
    {
        .width  = extent.width  > 0? (uint32_t)extent.width  : 1,
        .height = extent.height > 0? (uint32_t)extent.height : 1,
        .depth  = extent.depth  > 0? (uint32_t)extent.depth  : 1,
    };
}

OpaqueAlloc *MimirInstance::allocMipmap(cudaMipmappedArray_t *dev_arr,
    const cudaChannelFormatDesc *desc, cudaExtent extent, unsigned int num_levels)
{
    auto format = getFormatFromCuda(desc);
    ImageParams img_params{
        .type   = getImageType(Layout::make(extent.width, extent.height, extent.depth)),
        .format = getVulkanFormat(format),
        .extent = getExtentFromCuda(extent),
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage  = VK_IMAGE_USAGE_SAMPLED_BIT,
        .levels = num_levels,
    };
    // Create temporary image for querying the desired memory properties
    VkExternalMemoryImageCreateInfo extmem_info{
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    auto query_img = createImage(device, physical_device.handle, img_params, &extmem_info);
    VkMemoryRequirements memreq{};
    vkGetImageMemoryRequirements(device, query_img, &memreq);

    // Allocate external device memory
    VkExportMemoryAllocateInfoKHR export_info{
        .sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    };
    auto available = physical_device.memory.memoryProperties;
    auto memflags  = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    auto vk_memory = allocateMemory(device, available, memreq, memflags, &export_info);

    // Export and map the external memory to CUDA
    auto cuda_extmem = interop::importCudaExternalMemory(vk_memory, memreq.size, device);

    // Add deletors to queue for later cleanup
    deletors.views.add([=,this]{
        spdlog::trace("Free interop mipmapped image");
        validation::checkCuda(cudaDestroyExternalMemory(cuda_extmem));
        vkFreeMemory(device, vk_memory, nullptr);
    });
    vkDestroyImage(device, query_img, nullptr);

    auto alloc = OpaqueAlloc{memreq.size, vk_memory, cuda_extmem};
    cudaExternalMemoryMipmappedArrayDesc array_desc{
        .offset     = 0,
        .formatDesc = *desc,
        .extent     = extent,
        .flags      = 0,
        .numLevels  = num_levels,
        .reserved   = {},
    };
    validation::checkCuda(cudaExternalMemoryGetMappedMipmappedArray(
        dev_arr, cuda_extmem, &array_desc)
    );

    auto alloc_ptr = new OpaqueAlloc(alloc);
    deletors.context.add([=,this]{ delete alloc_ptr; });
    return alloc_ptr;
}

VkBuffer MimirInstance::createAttributeBuffer(VkDeviceSize memsize,
    VkBufferUsageFlags usage, VkDeviceMemory memory)
{
    // Create and bind buffer
    VkExternalMemoryBufferCreateInfo extmem_info{
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    };
    auto attr_buffer = createBuffer(device, memsize, usage, &extmem_info);
    deletors.views.add([=,this]{ vkDestroyBuffer(device, attr_buffer, nullptr); });
    validation::checkVulkan(vkBindBufferMemory(device, attr_buffer, memory, 0));
    return attr_buffer;
}

// TODO: Implement as use descriptive enum instead of bool
// TODO: Add more validations
bool validateViewDescription(ViewDescription *desc)
{
    bool has_elements = desc->layout.getTotalCount() > 0;
    bool has_position_attr = false;
    for (auto &[type, attr] : desc->attributes)
    {
        has_position_attr |= type == AttributeType::Position;
    }
    return has_elements && has_position_attr;
}

uint32_t getDrawCount(ViewDescription *desc)
{
    auto& pos_attr = desc->attributes[AttributeType::Position];
    return hasIndexing(pos_attr)? pos_attr.indexing.size : pos_attr.size;
}

ViewOptions initOptions(ViewType type) {
    ViewOptions options;
    switch (type)
    {
        // Initialize option to known defaults for the matching view type
        case ViewType::Markers: { options = MarkerOptions::defaults(); break; }
        case ViewType::Edges:   { options = MeshOptions::defaults(); break; }
        // Default: don't know (or care) about options
        default:                { break; }
    }
    return options;
}

View *MimirInstance::createView(ViewDescription *desc)
{
    if (!validateViewDescription(desc))
    {
        spdlog::error("Invalid view");
        return nullptr;
    }

    View view{
        .pipeline    = VK_NULL_HANDLE,
        .draw_count  = getDrawCount(desc),
        .vb_count    = 0,
        .vbo         = {VK_NULL_HANDLE},
        .offsets     = {0},
        .use_ibo     = false,
        .ibo         = VK_NULL_HANDLE,
        .index_type  = VK_INDEX_TYPE_NONE_KHR,
        .tex_count   = 0,
        .textures    = {},
        .ssbo_count  = 0,
        .storage     = {VK_NULL_HANDLE},
        .translation = glm::mat4(1.f),
        .rotation    = glm::mat4(1.f),
        .scale       = glm::mat4(1.f),
        .desc        = *desc,
    };

    // If no option value is set (the variant is default-initialized to std::monostate)
    if (view.desc.options.index() == 0)
    {
        // Generate default options for the current view type
        view.desc.options = initOptions(view.desc.type);
    }

    // The instance-wide light model decides how markers are shaded; the per-view
    // MarkerOptions::render_mode is derived from it here (see ViewerOptions::light_model).
    if (view.desc.type == ViewType::Markers
        && std::holds_alternative<MarkerOptions>(view.desc.options))
    {
        auto& marker_opts = std::get<MarkerOptions>(view.desc.options);
        switch (options.light_model)
        {
            case LightModel::None:
                marker_opts.render_mode = MarkerOptions::RenderMode::Flat2D;
                break;
            case LightModel::Phong:
                marker_opts.render_mode = MarkerOptions::RenderMode::Sphere3D;
                break;
            case LightModel::PhongMesh:
                marker_opts.render_mode = MarkerOptions::RenderMode::SphereMesh;
                break;
            case LightModel::PathTracing:
                // Markers are path-traced via the RT pipeline; the Sphere3D raster mode is
                // kept as the pipeline built for this view so RT-incapable devices still
                // render (drawElements is skipped for the RT path, see renderFrame).
                if (!rt_enabled)
                {
                    spdlog::warn("LightModel::PathTracing requested but device is not "
                                 "RT-capable; rendering with Phong raster instead");
                }
                marker_opts.render_mode = MarkerOptions::RenderMode::Sphere3D;
                break;
        }
    }

    translateView(&view, desc->position);
    rotateView(&view, desc->rotation);
    scaleView(&view, desc->scale);

    // Create attribute buffers
    for (auto &[type, attr] : desc->attributes)
    {
        spdlog::trace("Processing {} attribute", getAttributeType(type));
        if (type == AttributeType::Color && desc->type == ViewType::Image)
        {
            ImageParams params{
                .type   = getImageType(desc->layout),
                .format = getVulkanFormat(attr.format),
                .extent = getVulkanExtent(desc->layout),
                .tiling = getImageTiling(attr.source),
                .usage  = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                .levels = 1,
            };
            VkExternalMemoryImageCreateInfo extmem_info{
                .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                .pNext       = nullptr,
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
            };
            auto teximg = createImage(device, physical_device.handle, params, &extmem_info);
            vkBindImageMemory(device, teximg, getMemoryVulkan(attr.source), 0);

            Texture tex{
                .image    = teximg,
                .img_view = createImageView(device, tex.image, params, VK_IMAGE_ASPECT_COLOR_BIT),
                .sampler  = createSampler(device, VK_FILTER_LINEAR, false),
                .format   = params.format,
                .extent   = params.extent,
            };

            transitionImageLayout(tex.image,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            );

            deletors.views.add([=,this]{
                spdlog::trace("Destroying texture");
                vkDestroyImageView(device, tex.img_view, nullptr);
                vkDestroyImage(device, tex.image, nullptr);
                vkDestroySampler(device, tex.sampler, nullptr);
            });

            view.textures[view.tex_count++] = tex;
        }
        // Handle image quad vertex buffer size
        else if (type == AttributeType::Position && desc->type == ViewType::Image)
        {
            VkDeviceSize vb_size = sizeof(Vertex) * attr.size;
            VkBufferUsageFlags vb_usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            VkDeviceMemory vb_mem = getMemoryVulkan(attr.source);
            // TODO: Get if there is still space remaining (or maybe do it in validation)
            view.vbo[view.vb_count] = createAttributeBuffer(vb_size, vb_usage, vb_mem);
            view.vb_count++;
        }
        // Map source to a vertex buffer when accessing its elements directly
        // Source is always mapped this way for position attributes
        else if (type == AttributeType::Position || !hasIndexing(attr))
        {
            // 64-bit multiply: getSize() and attr.size are both 32-bit, so their product overflows
            // for large point counts -- at n = 2^30 float3 positions, 12 * 2^30 = 3 * 2^32 wraps to
            // exactly 0, creating a zero-size vertex buffer that renders nothing (silently blank
            // frame). Widen before multiplying.
            VkDeviceSize vb_size = static_cast<VkDeviceSize>(attr.format.getSize()) * attr.size;
            spdlog::trace("Position vertex buffer created for {} bytes ({} elements)",
                vb_size, getSourceSize(attr.source)
            );
            VkBufferUsageFlags vb_usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            // Path tracing reads positions by buffer-device-address in the AABB-writer compute shader,
            // so the position buffer needs the SHADER_DEVICE_ADDRESS usage (its interop memory already
            // carries the DEVICE_ADDRESS alloc flag from allocLinear).
            if (rt_enabled) { vb_usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT; }
            VkDeviceMemory vb_mem = getMemoryVulkan(attr.source);
            // TODO: Get if there is still space remaining (or maybe do it in validation)
            view.vbo[view.vb_count] = createAttributeBuffer(vb_size, vb_usage, vb_mem);
            view.vb_count++;
        }
        // If a non-position attribute uses indirect mapping, its source is mapped to a storage buffer
        else
        {
            VkDeviceSize sb_size = static_cast<VkDeviceSize>(attr.format.getSize()) * attr.size;
            spdlog::trace("Position storage buffer created for {} bytes ({} elements)",
                sb_size, getSourceSize(attr.source)
            );
            VkBufferUsageFlags sb_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            VkDeviceMemory sb_mem = getMemoryVulkan(attr.source);
            view.storage[view.ssbo_count++] = createAttributeBuffer(sb_size, sb_usage, sb_mem);
        }

        // If there is no indirect source access, the attribute is now fully processed
        if (!hasIndexing(attr)) { continue; }

        // Create indirect buffers as index buffer for position attributes,
        // and as vertex buffers for all other attributes
        VkDeviceMemory memory = getMemoryVulkan(attr.indexing.source);
        VkDeviceSize memsize = static_cast<VkDeviceSize>(attr.indexing.index_size) * attr.indexing.size;
        spdlog::trace("Attribute buffer created for {} bytes", memsize);
        if (type == AttributeType::Position)
        {
            VkBufferUsageFlags ib_usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
            view.ibo = createAttributeBuffer(memsize, ib_usage, memory);
            view.index_type = getIndexBufferType(attr.indexing.index_size);
            view.use_ibo = true;
        }
        else
        {
            VkBufferUsageFlags vb_usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            view.vbo[view.vb_count++] = createAttributeBuffer(memsize, vb_usage, memory);
        }
    }

    // Instanced mesh markers: rebind so the shared template icosphere is the per-vertex geometry
    // (binding 0) and the interop particle positions -- currently vbo[0] -- become per-instance data
    // (binding 1). The draw becomes one indexed instance of the icosphere per particle. This is the
    // sample's mesh-sphere path (LightModel::PhongMesh); the positions stay the same zero-copy
    // interop buffer, so nothing extra is streamed per frame.
    if (view.desc.type == ViewType::Markers
        && std::holds_alternative<MarkerOptions>(view.desc.options)
        && std::get<MarkerOptions>(view.desc.options).render_mode
               == MarkerOptions::RenderMode::SphereMesh
        && view.vb_count >= 1)
    {
        ensureSphereMesh();
        VkBuffer instance_positions = view.vbo[0]; // interop particle centers (per-instance)
        uint32_t particle_count     = view.draw_count;
        view.vbo[0]        = sphere_vbo;           // binding 0: unit icosphere vertices
        view.offsets[0]    = 0;
        view.vbo[1]        = instance_positions;   // binding 1: particle centers
        view.offsets[1]    = 0;
        view.vb_count      = 2;
        view.ibo           = sphere_ibo;
        view.index_type    = VK_INDEX_TYPE_UINT32;
        view.use_ibo       = true;
        view.draw_count    = sphere_index_count;   // template icosphere indices
        view.instance_count = particle_count;      // one instance per particle
    }

    auto handle = new View(view);
    deletors.views.add([=,this]{ delete handle; });
    views.push_back(handle);
    return handle;
}

VkDescriptorSetLayoutBinding descriptorLayoutBinding(
    uint32_t binding, VkDescriptorType type, VkShaderStageFlags flags)
{
    return VkDescriptorSetLayoutBinding{
        .binding            = binding,
        .descriptorType     = type,
        .descriptorCount    = 1,
        .stageFlags         = flags,
        .pImmutableSamplers = nullptr,
    };
}

void MimirInstance::initVulkan()
{
    createInstance();
    // In headless mode no surface is created; pickPhysicalDevice treats a null
    // surface as a request for a headless (no-presentation) device.
    if (!isHeadless())
    {
        window_context.createSurface(instance, &surface);
        deletors.context.add([=,this](){
            vkDestroySurfaceKHR(instance, surface, nullptr);
        });
    }
    physical_device = pickPhysicalDevice(instance, surface);

    if (isHeadless())
    {
        findGraphicsQueueFamily(physical_device.handle, graphics.family_index);
        present.family_index = graphics.family_index;
    }
    else
    {
        findQueueFamilies(physical_device.handle, surface,
            graphics.family_index, present.family_index
        );
    }
    std::set unique_queue_families{ graphics.family_index, present.family_index };
    std::vector<uint32_t> queue_families(unique_queue_families.begin(), unique_queue_families.end());
    device = createLogicalDevice(physical_device.handle, queue_families, isHeadless());
    vkGetDeviceQueue(device, graphics.family_index, 0, &graphics.queue);
    vkGetDeviceQueue(device, present.family_index, 0, &present.queue);

    command_pool = createCommandPool(device, graphics.family_index,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    );

    deletors.context.add([=,this](){
        vkDestroyCommandPool(device, command_pool, nullptr);
        vkDestroyDevice(device, nullptr);
    });

    // Create VMA handle
    /*
    auto memtypes = physical_device.memory.memoryProperties.memoryTypes;
    auto memtype_count = physical_device.memory.memoryProperties.memoryTypeCount;
    std::vector<VkExternalMemoryHandleTypeFlagsKHR> external_memtypes(memtype_count, 0);
    for (uint32_t i = 0; i < memtype_count; ++i)
    {
        auto memtype = memtypes[i];
        if (memtype.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        {
            external_memtypes[i] = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        }
    }

    VmaAllocatorCreateInfo allocator_info{
        .flags                          = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT,
        .physicalDevice                 = physical_device.handle,
        .device                         = device,
        .preferredLargeHeapBlockSize    = 0,
        .pAllocationCallbacks           = nullptr,
        .pDeviceMemoryCallbacks         = nullptr,
        .pHeapSizeLimit                 = nullptr,
        .pVulkanFunctions               = nullptr,
        .instance                       = instance,
        .vulkanApiVersion               = VK_API_VERSION_1_2,
        .pTypeExternalMemoryHandleTypes = external_memtypes.data(),
    };
    validation::checkVulkan(vmaCreateAllocator(&allocator_info, &allocator));
    deletors.context.add([=,this](){ vmaDestroyAllocator(allocator); });

    // Create VMA pool for external (interop) memory allocations
    VmaPoolCreateInfo pool_info{
        .memoryTypeIndex        = 0, // TODO
        .flags                  = 0,
        .blockSize              = 0,
        .minBlockCount          = 0,
        .maxBlockCount          = 0,
        .priority               = 0.f, // Ignored
        .minAllocationAlignment = 0,
        .pMemoryAllocateNext    = nullptr,
    };
    validation::checkVulkan(vmaCreatePool(allocator, &pool_info, &interop_pool));
    deletors.context.add([=,this](){ vmaDestroyPool(allocator, interop_pool); });
*/
    // Create descriptor pool
    std::vector<VkDescriptorPoolSize> pool_sizes{
        { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
    };
    descriptor_pool = createDescriptorPool(device, pool_sizes);

    // Create descriptor set and pipeline layouts
    VkShaderStageFlags all_stages =
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    std::vector<VkDescriptorSetLayoutBinding> layout_bindings{
        descriptorLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, all_stages),
        descriptorLayoutBinding(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, all_stages),
        descriptorLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, all_stages),
        descriptorLayoutBinding(3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT),
        descriptorLayoutBinding(4, VK_DESCRIPTOR_TYPE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
        descriptorLayoutBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, all_stages)
    };
    descriptor_layout = createDescriptorSetLayout(device, layout_bindings);
    pipeline_layout = createPipelineLayout(device, descriptor_layout);

    deletors.context.add([=,this]{
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    });

    // Path tracing (LightModel::PathTracing): build the resolution-independent RT context
    // (BLAS/TLAS/pipeline/SBT) once, before initGraphics() so the per-frame storage images
    // and composite pipeline are created there. Requires an RT-capable device; otherwise the
    // instance silently renders the raster fallback (createView emits the warning).
    rt_enabled = (options.light_model == LightModel::PathTracing)
              && supportsRayTracing(physical_device.handle);
    if (rt_enabled)
    {
        auto submit = [this](std::function<void(VkCommandBuffer)> fn) {
            immediateSubmit(std::move(fn));
        };
        // Register the multi-material SBT. Material 0 is the particle surface (bindScene sets its
        // albedo from the view color); material 1 is a spare emissive slot that demonstrates the
        // library supports several materials in one SBT. The particles keep instance_material_count
        // == 1 (every instance uses material 0), so the extra slot costs one tiny idle hit record.
        std::vector<MaterialData> materials = {
            MaterialData{ .albedo = { 0.82f, 0.82f, 0.88f }, .emission = 0.f }, // diffuse (particles)
            MaterialData{ .albedo = { 1.00f, 0.85f, 0.55f }, .emission = 4.f }, // spare emissive light
        };
        // int64 buffer atomics were enabled at device creation iff supported (see
        // createLogicalDevice); mirror that query so the RT context knows whether the LOD-centroid
        // int64 BDA accumulator is available (else --lod falls back to cell-center placement).
        bool int64_atomics = supportsInt64Atomics(physical_device.handle);
        raytracing = RayTracingContext::make(device, physical_device.handle,
            physical_device.memory.memoryProperties, submit,
            options.pt_subdivisions, /*max_recursion=*/2, int64_atomics, std::move(materials)
        );
        deletors.context.add([this]{ raytracing.destroy(); });
    }

    initGraphics();
    createSyncObjects();
    // CUDA compute events are context-lifetime (not swapchain-lifetime): they outlive
    // swapchain rebuilds and must not be destroyed/recreated in cleanupGraphics().
    compute_monitor = metrics::ComputeMonitor::make(0);
    deletors.context.add([=,this]{
        cudaEventDestroy(compute_monitor.start);
        cudaEventDestroy(compute_monitor.stop);
    });
    // After command pool and render pass are created.
    // The GUI uses the GLFW/window backend, so it is only initialized for on-screen instances.
    if (!isHeadless())
    {
        gui::init(instance, physical_device.handle, device,
            descriptor_pool, render_pass, graphics, window_context
        );
    }
    descriptor_sets = createDescriptorSets(device,
        descriptor_pool, descriptor_layout, swapchain.image_count
    );
}

void MimirInstance::createInstance()
{
    if (validation::enable_layers && !validation::checkValidationLayerSupport())
    {
        spdlog::error("validation layers requested, but not supported");
    }

    VkApplicationInfo app_info{
        .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext              = nullptr,
        .pApplicationName   = "Mimir",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName        = "Mimir",
        .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion         = VK_API_VERSION_1_2,
    };

    // List additional required validation layers.
    // Headless instances need no window-system (surface) extensions.
    auto extensions = isHeadless()?
        std::vector<const char*>{} : GlfwContext::getRequiredExtensions();
    if (validation::enable_layers)
    {
        // Enable debugging message extension
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    VkInstanceCreateInfo instance_info{
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext                   = nullptr,
        .flags                   = 0,
        .pApplicationInfo        = &app_info,
        .enabledLayerCount       = 0,
        .ppEnabledLayerNames     = nullptr,
        .enabledExtensionCount   = (uint32_t)extensions.size(),
        .ppEnabledExtensionNames = extensions.data(),
    };

    VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};
    // Include validation layer names if they are enabled
    if (validation::enable_layers)
    {
        debug_create_info = validation::debugMessengerCreateInfo();
        instance_info.pNext               = &debug_create_info;
        instance_info.enabledLayerCount   = validation::layers.size();
        instance_info.ppEnabledLayerNames = validation::layers.data();
    }
    validation::checkVulkan(vkCreateInstance(&instance_info, nullptr, &instance));
    deletors.context.add([=,this]{ vkDestroyInstance(instance, nullptr); });

    if (validation::enable_layers)
    {
        VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;
        validation::checkVulkan(validation::CreateDebugUtilsMessengerEXT(
            instance, &debug_create_info, nullptr, &debug_messenger)
        );
        deletors.context.add([=,this]{
            validation::DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
        });
    }
}

void MimirInstance::createSyncObjects()
{
    //images_inflight.resize(swap->image_count, VK_NULL_HANDLE);
    for (auto& sync : sync_data)
    {
        sync.frame_fence = createFence(device, VK_FENCE_CREATE_SIGNALED_BIT);
        sync.image_acquired = createSemaphore(device);
        sync.render_complete = createSemaphore(device);
        deletors.context.add([=,this]{
            vkDestroyFence(device, sync.frame_fence, nullptr);
            vkDestroySemaphore(device, sync.image_acquired, nullptr);
            vkDestroySemaphore(device, sync.render_complete, nullptr);
        });
    }
    interop = interop::Barrier::make(device);
    deletors.context.add([=,this]{
        validation::checkCuda(cudaDestroyExternalSemaphore(interop.cuda_semaphore));
        vkDestroySemaphore(device, interop.vk_semaphore, nullptr);
    });
}

void MimirInstance::cleanupGraphics()
{
    vkDeviceWaitIdle(device);
    //vkFreeCommandBuffers(device, command_pool, command_buffers.size(), command_buffers.data());
    deletors.graphics.flush();
}

void MimirInstance::initGraphics()
{
    // Mark a fresh metrics epoch: the query pool below starts empty, so renderFrame() must let a
    // few frames accumulate before reading results with WAIT (otherwise it blocks after a resize).
    graphics_epoch = render_timeline;

    // Determine render target size. Headless instances have no window to query,
    // so the configured window size is used as the offscreen target extent.
    int width, height;
    if (isHeadless())
    {
        width  = options.window.size.x;
        height = options.window.size.y;
        createOffscreenTarget(width, height);
    }
    else
    {
        window_context.getFramebufferSize(width, height);
        auto present_mode = getDesiredPresentMode(options.present.mode);
        std::vector queue_indices{graphics.family_index, present.family_index};
        swapchain = Swapchain::make(device, physical_device.handle,
            surface, width, height, present_mode, queue_indices
        );
    }

    // Create one command buffer per swapchain/offscreen image
    command_buffers = createCommandBuffers(device, command_pool, swapchain.image_count);

    // Initialize graphics metrics monitoring (recreated on swapchain rebuild)
    auto timestamp_period = physical_device.general.properties.limits.timestampPeriod;
    graphics_monitor = metrics::GraphicsMonitor::make(device, 2 * command_buffers.size(), timestamp_period, 240);

    deletors.graphics.add([=,this]{
        if (!isHeadless()) { vkDestroySwapchainKHR(device, swapchain.current, nullptr); }
        vkDestroyQueryPool(device, graphics_monitor.query_pool, nullptr);
    });

    // Create depth image and image view
    ImageParams depth_params{
        .type   = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_D32_SFLOAT,
        .extent = { swapchain.extent.width, swapchain.extent.height, 1 },
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage  = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .levels = 1,
    };
    depth_image = createImage(device, physical_device.handle, depth_params);

    auto available = physical_device.memory.memoryProperties;
    VkMemoryRequirements mem_req{};
    vkGetImageMemoryRequirements(device, depth_image, &mem_req);
    depth_memory = allocateMemory(device, available, mem_req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    validation::checkVulkan(vkBindImageMemory(device, depth_image, depth_memory, 0));
    depth_view = createImageView(device, depth_image, depth_params, VK_IMAGE_ASPECT_DEPTH_BIT);

    // Create render pass with color and depth attachments
    VkAttachmentDescription color{
        .flags          = 0,
        .format         = swapchain.format,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        // On-screen frames are presented; headless frames are copied out for readback/encoding.
        .finalLayout    = isHeadless()?
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };
    VkAttachmentDescription depth{
        .flags          = 0, // Can be VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT
        .format         = depth_params.format,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp        = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    render_pass = createRenderPass(device, color, depth);

    deletors.graphics.add([=,this]{
        vkDestroyImageView(device, depth_view, nullptr);
        vkDestroyImage(device, depth_image, nullptr);
        vkFreeMemory(device, depth_memory, nullptr);
        vkDestroyRenderPass(device, render_pass, nullptr);
    });

    if (isHeadless())
    {
        std::span<const VkImage> imgs{offscreen_images};
        framebuffers = Framebuffer::make(device, render_pass, imgs,
            swapchain.format, swapchain.extent, depth_view
        );
    }
    else
    {
        framebuffers = Framebuffer::make(device, render_pass, swapchain, depth_view);
    }
    for (uint32_t i = 0; i < swapchain.image_count; ++i)
    {
        deletors.graphics.add([=,this]{
            vkDestroyImageView(device, framebuffers.image_views[i], nullptr);
            vkDestroyFramebuffer(device, framebuffers.handles[i], nullptr);
        });
    }

    // Path-tracing frame resources (storage images + composite pipeline) depend on the
    // render target extent/render pass, so they are (re)built here and freed on rebuild.
    if (rt_enabled)
    {
        raytracing.createFrameResources(swapchain.extent, render_pass);
        deletors.graphics.add([this]{ raytracing.destroyFrameResources(); });
    }
}

void MimirInstance::recreateGraphics()
{
    cleanupGraphics();
    initGraphics();
    createViewPipelines();
}

void MimirInstance::createOffscreenTarget(int width, int height)
{
    // Reuse the Swapchain struct as the common render-target descriptor (format/extent/count)
    // so the rest of the engine treats headless and on-screen targets uniformly.
    swapchain.current     = VK_NULL_HANDLE;
    swapchain.old         = VK_NULL_HANDLE;
    swapchain.format      = VK_FORMAT_B8G8R8A8_UNORM;
    swapchain.extent      = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
    swapchain.image_count = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    offscreen_images.clear();
    offscreen_memory.clear();

    auto available = physical_device.memory.memoryProperties;
    for (uint32_t i = 0; i < swapchain.image_count; ++i)
    {
        ImageParams params{
            .type   = VK_IMAGE_TYPE_2D,
            .format = swapchain.format,
            .extent = { swapchain.extent.width, swapchain.extent.height, 1 },
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            // Color attachment for rendering; transfer source for frame readback/encoding.
            .usage  = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .levels = 1,
        };
        auto image = createImage(device, physical_device.handle, params);
        VkMemoryRequirements mem_req{};
        vkGetImageMemoryRequirements(device, image, &mem_req);
        auto memory = allocateMemory(device, available, mem_req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        validation::checkVulkan(vkBindImageMemory(device, image, memory, 0));
        offscreen_images.push_back(image);
        offscreen_memory.push_back(memory);
    }

    // Free with the graphics-lifetime deletors so recreateGraphics() rebuilds the target.
    auto images = offscreen_images;
    auto memories = offscreen_memory;
    deletors.graphics.add([=,this]{
        for (auto image : images)  { vkDestroyImage(device, image, nullptr); }
        for (auto memory : memories) { vkFreeMemory(device, memory, nullptr); }
    });
}

void MimirInstance::updateDescriptorSets()
{
    for (size_t i = 0; i < descriptor_sets.size(); ++i)
    {
        // Write MVP matrix, scene info and texture samplers
        std::vector<VkWriteDescriptorSet> updates;

        VkDescriptorBufferInfo mvp_info{
            .buffer = uniform_buffers[i].buffer,
            .offset = 0,
            .range  = sizeof(ModelViewProjection),
        };
        VkWriteDescriptorSet write_buf{
            .sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext            = nullptr,
            .dstSet           = descriptor_sets[i],
            .dstBinding       = 0,
            .dstArrayElement  = 0,
            .descriptorCount  = 1,
            .descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .pImageInfo       = nullptr,
            .pBufferInfo      = &mvp_info,
            .pTexelBufferView = nullptr,
        };
        updates.push_back(write_buf);

        VkDescriptorBufferInfo scene_info{
            .buffer = uniform_buffers[i].buffer,
            .offset = 0,
            .range  = sizeof(SceneUniforms),
        };
        write_buf.dstBinding  = 1;
        write_buf.pBufferInfo = &scene_info;
        updates.push_back(write_buf);

        VkDescriptorBufferInfo view_info{
            .buffer = uniform_buffers[i].buffer,
            .offset = 0,
            .range  = sizeof(ViewUniforms),
        };
        write_buf.dstBinding  = 2;
        write_buf.pBufferInfo = &view_info;
        updates.push_back(write_buf);

        for (const auto& view : views)
        {
            for (uint32_t k = 0; k < view->tex_count; ++k)
            {
                auto tex = view->textures[k];

                VkDescriptorImageInfo img_info{
                    .sampler     = tex.sampler,
                    .imageView   = tex.img_view,
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                };
                VkWriteDescriptorSet write_img{
                    .sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext            = nullptr,
                    .dstSet           = descriptor_sets[i],
                    .dstBinding       = 3,
                    .dstArrayElement  = 0,
                    .descriptorCount  = 1,
                    .descriptorType   = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    .pImageInfo       = &img_info,
                    .pBufferInfo      = nullptr,
                    .pTexelBufferView = nullptr,
                };
                updates.push_back(write_img);

                VkDescriptorImageInfo samp_info{
                    .sampler     = tex.sampler,
                    .imageView   = tex.img_view,
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                };
                write_img.dstBinding     = 4;
                write_img.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                write_img.pImageInfo     = &samp_info;
                updates.push_back(write_img);
            }
            for (uint32_t k = 0; k < view->ssbo_count; ++k)
            {
                auto ssbo = view->storage[k];

                VkDescriptorBufferInfo ssbo_info{
                    .buffer = ssbo,
                    .offset = 0,
                    .range  = VK_WHOLE_SIZE,
                };
                write_buf.dstBinding     = 5;
                write_buf.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                write_buf.pBufferInfo    = &ssbo_info;
                updates.push_back(write_buf);
            }
        }
        vkUpdateDescriptorSets(device, updates.size(), updates.data(), 0, nullptr);
    }
}

void MimirInstance::waitTimelineHost()
{
    VkSemaphoreWaitInfo wait_info{
        .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext          = nullptr,
        .flags          = 0,
        .semaphoreCount = 1,
        .pSemaphores    = &interop.vk_semaphore,
        .pValues        = &interop.timeline_value,
    };
    validation::checkVulkan(vkWaitSemaphores(device, &wait_info, frame_timeout));
}

void MimirInstance::renderFrame(bool advance_interop)
{
    // Get frame index from the inflight frames array
    auto frame_idx = render_timeline % MAX_FRAMES_IN_FLIGHT;

    // Retrieve synchronization data for frame i
    auto frame_sync = sync_data[frame_idx];
    auto fence = frame_sync.frame_fence;

    // CPU-side phase breakdown of this frame (see GraphicsMonitor::last_*_ms). The "wait" phase
    // spans the fence wait + swapchain acquire, where the render thread blocks on GPU / present
    // backpressure -- typically the dominant, and otherwise unmeasured, part of a frame.
    auto phase_clock = std::chrono::steady_clock::now();

    // Wait for fence of frame i to end, then immediately reset it for further use
    validation::checkVulkan(vkWaitForFences(device, 1, &fence, VK_TRUE, frame_timeout));
    validation::checkVulkan(vkResetFences(device, 1, &fence));

    // Start measuring frame time
    graphics_monitor.startFrameWatch();

    static uint64_t wait_value = 0;
    static uint64_t signal_value = 1;

    bool advance_timeline = false;
    // Interop-semaphore value the GPU submit waits on (CUDA's compute-done signal). Captured so
    // the GPU-frame measurement below can tell whether CUDA was already finished at submit time.
    uint64_t interop_wait_target = 0;
    // On-screen frames synchronize with swapchain acquire/present through binary semaphores;
    // headless frames have neither, relying on the frame fence (and the interop timeline).
    std::vector<VkSemaphore> waits;
    std::vector<VkPipelineStageFlags> stages;
    std::vector<VkSemaphore> signals;
    std::vector<uint64_t> wait_values;
    std::vector<uint64_t> signal_values;
    if (!isHeadless())
    {
        waits.push_back(frame_sync.image_acquired);
        stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        signals.push_back(frame_sync.render_complete);
        wait_values.push_back(0);
        signal_values.push_back(0);
    }

    if (advance_interop)
    {
        // GPU waits for CUDA's signal directly at vertex-input stage so the vertex shader
        // cannot read positions before CUDA has finished writing them.  The old CPU-side
        // vkWaitSemaphores call is removed: it was redundant because the GPU semaphore wait
        // provides the same guarantee, and it was stalling the render thread unnecessarily.
        // interop.timeline_value was always 0 (never updated), so the GPU wait was a no-op;
        // use wait_value here to pass the correct expected CUDA signal value.
        waits.push_back(interop.vk_semaphore);
        // The first GPU consumer of the interop positions is the vertex shader for raster, but
        // the instance-writer compute for path tracing; gate the wait on the right stage.
        stages.push_back(rt_enabled
            ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
        signals.push_back(interop.vk_semaphore);
        wait_values.push_back(wait_value);
        signal_values.push_back(signal_value);
        interop_wait_target = wait_value;
        advance_timeline = true;
    }

    // Select the target image. On-screen mode acquires from the swapchain (signaling a
    // semaphore when ready); headless mode renders directly into this frame's offscreen image.
    uint32_t image_idx = static_cast<uint32_t>(frame_idx);
    VkResult result = VK_SUCCESS;
    if (!isHeadless())
    {
        result = vkAcquireNextImageKHR(device, swapchain.current,
            frame_timeout, frame_sync.image_acquired, VK_NULL_HANDLE, &image_idx
        );
        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateGraphics();
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        {
            spdlog::error("Failed to acquire swapchain image");
            return;
        }
    }
    last_image_idx = image_idx;

    // if (images_inflight[image_idx] != VK_NULL_HANDLE)
    // {
    //     vkWaitForFences(device, 1, &images_inflight[image_idx], VK_TRUE, timeout);
    // }
    // images_inflight[image_idx] = frame.render_fence;

    if (render_timeline - graphics_epoch > MAX_FRAMES_IN_FLIGHT)
    {
        graphics_monitor.getRenderTimeResults(device, frame_idx);
        // Read back this frame_idx's PT timestamps (written last time it was in flight) before
        // recordUpdateScene resets them below.
        if (rt_enabled) { raytracing.readTimings(frame_idx); }
    }

    // End of the wait phase; the record phase covers command-buffer recording below.
    {
        auto now = std::chrono::steady_clock::now();
        graphics_monitor.last_wait_ms =
            std::chrono::duration<float, std::milli>(now - phase_clock).count();
        phase_clock = now;
    }

    // Retrieve a command buffer and start recording to it
    auto cmd = command_buffers[frame_idx];
    validation::checkVulkan(vkResetCommandBuffer(cmd, 0));
    VkCommandBufferBeginInfo cmd_info{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    validation::checkVulkan(vkBeginCommandBuffer(cmd, &cmd_info));
    graphics_monitor.startRenderWatch(device, cmd, frame_idx);

    // Path tracing: trace primary rays into this frame's storage image BEFORE the raster
    // render pass (vkCmdTraceRaysKHR cannot run inside a render pass). The composite pass
    // below samples the result. Raster light models skip this entirely.
    if (rt_enabled)
    {
        RtPushConstants pc{};
        // The raygen needs the world-space basis of the camera the raster pass shows on screen
        // (screen-right, screen-up, view direction, eye). matrices.view holds different things
        // depending on which path last wrote it, so derive the basis per mode:
        //  - Fly and scripted auto-orbit write it via setLookAt: camera-to-world with col3 = eye,
        //    col2 = forward, col1 = screen-up -- but col0 = up x fwd is the OPPOSITE of
        //    screen-right (the same trap as the WASD handler in updateCamera), so negate it.
        //  - Manual orbit (trackball) writes translate(pos) * rotmat, which the raster consumes
        //    directly as world-to-view. The camera basis is its inverse: rows of R are the
        //    world axes of view space, eye = -R^T * pos, and the view direction is -z in view
        //    space (the projection maps clip.w = -z_view).
        const auto& v = camera.matrices.view;
        glm::vec3 eye, cam_right, cam_up, cam_fwd;
        if (options.camera_control == CameraControl::Fly || options.orbit_speed > 0.f)
        {
            eye       =  glm::vec3(v[3]);
            cam_right = -glm::vec3(v[0]);
            cam_up    =  glm::vec3(v[1]);
            cam_fwd   =  glm::vec3(v[2]);
        }
        else
        {
            glm::mat3 rt = glm::transpose(glm::mat3(v)); // R^T: its columns are the rows of R
            eye       = -(rt * glm::vec3(v[3]));
            cam_right =  rt[0];
            cam_up    =  rt[1];
            cam_fwd   = -rt[2];
        }
        // Particle albedo is no longer in the push constants: it lives in the SBT material record
        // (bindScene copies the view color into material 0). The basis w lanes instead carry
        // ViewerOptions::light_color, which scales the PT sun (pathtrace.slang's
        // SUN_RADIANCE_PER_UNIT) so the same light knob drives raster and path-traced modes.
        auto lc = options.light_color;
        pc.cam_pos     = glm::vec4(eye, 1.f);
        pc.cam_right   = glm::vec4(glm::normalize(cam_right), lc.x);
        pc.cam_up      = glm::vec4(glm::normalize(cam_up), lc.y);
        pc.cam_forward = glm::vec4(glm::normalize(cam_fwd), lc.z);
        pc.tan_half_fov = std::tan(glm::radians(camera.fov) * 0.5f);
        pc.aspect       = (float)swapchain.extent.width / (float)swapchain.extent.height;
        auto lp = options.light_pos;
        auto bg = options.background_color;
        pc.sun_dir     = glm::vec4(lp.x, lp.y, lp.z, 0.f);
        // Path-traced sky/environment = the instance background color (w = intensity), so a
        // simulation controls the backdrop (incl. black) with the same knob as the raster modes.
        pc.sky_color   = glm::vec4(bg.x, bg.y, bg.z, 1.0f);
        pc.frame_index = static_cast<uint32_t>(render_timeline);
        pc.spp         = options.pt_samples_per_pixel;
        pc.bounces     = options.pt_max_bounces;

        // Temporal accumulation: restart the running mean from zero whenever the scene may have
        // changed -- a new simulation iteration (advance_interop consumes one compute step;
        // pt_scene_dirty is the equivalent signal from the host-lockstep headless/remote loops) or
        // any camera motion (view matrix or fov differs from last frame). Otherwise keep
        // accumulating so a static view converges. A resize recreates the accumulator too.
        bool cam_moved = camera.matrices.view != pt_last_view || camera.fov != pt_last_fov;
        pt_last_view = camera.matrices.view;
        pt_last_fov  = camera.fov;
        // The particle positions changed (a new sim iteration) iff advance_interop or the host-loop
        // dirty flag is set; a camera move alone does NOT move geometry. Only a geometry change needs
        // the AABB buffer + BLAS/TLAS (re)built -- when it is unchanged (paused sim, or a static view
        // being accumulated) recordUpdateScene skips the whole build phase and reuses the AS.
        //
        // Camera motion does NOT invalidate the acceleration structure: the BVH is built in WORLD
        // space and is independent of the viewpoint. Moving the camera only changes the rays the
        // raygen shoots (pc.cam_* below) through the same unchanged geometry, so we re-trace the
        // existing BVH -- never rebuild it. A camera move DOES invalidate the accumulated image
        // (those samples were for the old view), so it resets the accumulator; it just does not
        // touch the BVH. Hence rebuild is gated on geo_changed only, while the accumulator reset
        // below is gated on geo_changed || cam_moved.
        bool geo_changed = advance_interop || pt_scene_dirty;
        if (geo_changed || cam_moved) { pt_accum_frame = 0; }
        pt_scene_dirty = false;
        pc.accum_frame = pt_accum_frame;
        pt_accum_frame++;

        // (Re)build/refit this frame's AS from the live interop positions (only when they changed),
        // then trace it. When denoising, the trace leaves the display image in GENERAL and
        // recordDenoise writes the filtered result.
        raytracing.recordUpdateScene(cmd, frame_idx, /*rebuild=*/geo_changed);
        raytracing.recordTrace(cmd, frame_idx, pc, /*leave_image_general=*/options.pt_denoise);
        if (options.pt_denoise) { raytracing.recordDenoise(cmd, frame_idx); }
    }

    // Set clear color and depth stencil value
    std::array<VkClearValue, 2> clear_values{};
    std::memcpy(clear_values[0].color.float32, &options.background_color.x, sizeof(options.background_color));
    clear_values[1].depthStencil = { .depth = 0.f, .stencil = 0 };

    VkRenderPassBeginInfo render_pass_info{
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext           = nullptr,
        .renderPass      = render_pass,
        .framebuffer     = framebuffers.handles[image_idx],
        .renderArea      = { {0, 0}, swapchain.extent },
        .clearValueCount = (uint32_t)clear_values.size(),
        .pClearValues    = clear_values.data(),
    };

    // Render pass
    vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

    // Path tracing composites the ray-traced storage image over a fullscreen triangle;
    // raster light models draw the view geometry. The ImGui HUD renders on top of both.
    if (rt_enabled) { raytracing.recordComposite(cmd, frame_idx); }
    else            { drawElements(frame_idx); }
    if (!isHeadless()) { gui::render(cmd); }
    vkCmdEndRenderPass(cmd);

    graphics_monitor.stopRenderWatch(cmd, frame_idx);
    // Finalize command buffer recording, so it can be executed
    validation::checkVulkan(vkEndCommandBuffer(cmd));

    updateUniformBuffers(frame_idx);
    render_timeline++;
    if (advance_timeline)
    {
        wait_value += 2;
        signal_value += 2;
    }

    // End of the record phase; the submit phase covers vkQueueSubmit + vkQueuePresentKHR below.
    {
        auto now = std::chrono::steady_clock::now();
        graphics_monitor.last_record_ms =
            std::chrono::duration<float, std::milli>(now - phase_clock).count();
        phase_clock = now;
    }

    // Fill submit waits & signals info. The timeline submit info is only needed when this
    // frame carries the interop timeline semaphore (advance_timeline); plain frames submit
    // with binary semaphores alone.
    VkTimelineSemaphoreSubmitInfo *extra = nullptr;
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    if (advance_timeline)
    {
        timeline_info = VkTimelineSemaphoreSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreValueCount   = (uint32_t)wait_values.size(),
            .pWaitSemaphoreValues      = wait_values.data(),
            .signalSemaphoreValueCount = (uint32_t)signal_values.size(),
            .pSignalSemaphoreValues    = signal_values.data(),
        };
        extra = &timeline_info;
    }

    VkSubmitInfo submit_info{
        .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext                = extra,
        .waitSemaphoreCount   = (uint32_t)waits.size(),
        .pWaitSemaphores      = waits.data(),
        .pWaitDstStageMask    = stages.data(),
        .commandBufferCount   = 1,
        .pCommandBuffers      = &cmd,
        .signalSemaphoreCount = (uint32_t)signals.size(),
        .pSignalSemaphores    = signals.data(),
    };

    // Is CUDA already finished at submit time? If the interop semaphore has already reached the
    // value this submit waits on, the GPU will not idle on it, so the GPU-frame measurement below
    // is pure render work with no CUDA wait folded in. (Read before submit; non-blocking.)
    bool cuda_ready_at_submit = true;
    if (advance_interop)
    {
        uint64_t interop_now = 0;
        validation::checkVulkan(
            vkGetSemaphoreCounterValue(device, interop.vk_semaphore, &interop_now));
        cuda_ready_at_submit = interop_now >= interop_wait_target;
    }

    // Execute command buffer using image as attachment in framebuffer
    validation::checkVulkan(vkQueueSubmit(graphics.queue, 1, &submit_info, fence));

    // Return image result back to swapchain for presentation on screen.
    // Headless frames are not presented; they stay in TRANSFER_SRC layout for readback/encoding.
    if (!isHeadless())
    {
        VkPresentInfoKHR present_info{
            .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext              = nullptr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &frame_sync.render_complete,
            .swapchainCount     = 1,
            .pSwapchains        = &swapchain.current,
            .pImageIndices      = &image_idx,
            .pResults           = nullptr,
        };
        result = vkQueuePresentKHR(present.queue, &present_info);
        // Resize should be done after presentation to ensure semaphore consistency
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || window_context.resize_requested)
        {
            recreateGraphics();
            window_context.resize_requested = false;
        }
    }

    // Close the submit phase (before any optional fps-limit stall, which is not real render work).
    graphics_monitor.last_submit_ms =
        std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - phase_clock).count();

    // In lockstep interop mode the render thread produces exactly one frame per compute step and
    // then blocks polling for the next request, so waiting on this frame's fence here is free and
    // does not perturb render_ms (the GPU->CUDA interop signal is GPU-side, independent of this
    // host wait). It yields the true end-to-end GPU frame latency -- the honest measure of where
    // Render's wall time goes, which the narrow render-pass timestamp can under-report. The fence
    // is left signaled; the next reuse of this frame_idx waits (instantly) and resets it as usual.
    //
    // Only record the sample when CUDA was already done at submit (cuda_ready_at_submit): otherwise
    // the GPU would have idled on the interop semaphore waiting for the compute kernel, and that
    // wait would be folded into the measurement. Skipping keeps GPU frame pure render work, never
    // CUDA-wait time. In steady-state lockstep the render displays the *previous* step's positions,
    // so the kernel is already finished and samples are essentially always recorded.
    if (advance_interop)
    {
        validation::checkVulkan(vkWaitForFences(device, 1, &fence, VK_TRUE, frame_timeout));
        if (cuda_ready_at_submit)
        {
            graphics_monitor.last_gpu_ms =
                std::chrono::duration<float, std::milli>(
                    std::chrono::steady_clock::now() - phase_clock).count();
        }
    }

    // Limit frame if it was configured
    if (options.present.enable_fps_limit) { frameStall(options.present.target_frame_time); }
    graphics_monitor.stopFrameWatch();
    //spdlog::trace("frame {} finished", render_timeline-1);
}

void MimirInstance::drawElements(uint32_t image_idx)
{
    auto min_alignment = physical_device.getUboOffsetAlignment();
    auto size_mvp = getAlignedSize(sizeof(ModelViewProjection), min_alignment);
    auto size_view = getAlignedSize(sizeof(ViewUniforms), min_alignment);
    auto size_scene = getAlignedSize(sizeof(SceneUniforms), min_alignment);
    auto size_ubo = size_mvp + size_view + size_scene;

    auto cmd = command_buffers[image_idx];
    for (uint32_t i = 0; i < views.size(); ++i)
    {
        // Do not draw anything if visibility is turned off
        if (!views[i]->desc.visible) { continue; }
        auto& view = views[i];

        // Bind descriptor set and pipeline
        std::vector<uint32_t> offsets = {
            i * size_ubo,
            i * size_ubo + size_mvp + size_view,
            i * size_ubo + size_mvp
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout, 0, 1, &descriptor_sets[image_idx], offsets.size(), offsets.data()
        );
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, view->pipeline);
        vkCmdBindVertexBuffers(cmd, 0, view->vb_count, view->vbo, view->offsets);

        if (view->use_ibo) // Index buffer exists, bind it and perform indexed draw
        {
            // instance_count > 1 for SphereMesh markers (one icosphere instance per particle).
            vkCmdBindIndexBuffer(cmd, view->ibo, 0, view->index_type);
            vkCmdDrawIndexed(cmd, view->draw_count, view->instance_count, 0, 0, 0);
        }
        else // Perform regular draw with bound vertex buffers
        {
            uint32_t first_vertex = 0;
            vkCmdDraw(cmd, view->draw_count, view->instance_count, first_vertex, 0);
        }
    }
}

void MimirInstance::createViewPipelines(/*std::span<std::shared_ptr<InteropView>> views*/)
{
    auto start = std::chrono::steady_clock::now();

    pipeline_builder = PipelineBuilder::make(pipeline_layout, swapchain.extent);
    for (auto& view : views)
    {
        pipeline_builder.addPipeline(view->desc, device);
    }
    auto pipelines = pipeline_builder.createPipelines(device, render_pass);
    for (size_t i = 0; i < pipelines.size(); ++i)
    {
        auto& view = views[i];
        view->pipeline = pipelines[i];
        deletors.graphics.add([=,this]{ vkDestroyPipeline(device, view->pipeline, nullptr); });
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    spdlog::trace("Created {} pipeline object(s) in {} ms", pipelines.size(), elapsed);
}

void MimirInstance::initUniformBuffers()
{
    auto min_alignment = physical_device.getUboOffsetAlignment();
    auto size_mvp = getAlignedSize(sizeof(ModelViewProjection), min_alignment);
    auto size_view = getAlignedSize(sizeof(ViewUniforms), min_alignment);
    auto size_scene = getAlignedSize(sizeof(SceneUniforms), min_alignment);
    auto size_ubo = (size_mvp + size_view + size_scene) * views.size();

    uniform_buffers.resize(swapchain.image_count);
    auto available = physical_device.memory.memoryProperties;
    for (auto& ubo : uniform_buffers)
    {
        ubo.buffer = createBuffer(device, size_ubo, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        VkMemoryRequirements memreq{};
        vkGetBufferMemoryRequirements(device, ubo.buffer, &memreq);
        auto mem_usage = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        ubo.memory = allocateMemory(device, available, memreq, mem_usage);
        validation::checkVulkan(vkBindBufferMemory(device, ubo.buffer, ubo.memory, 0));
        deletors.context.add([=,this]{
            vkDestroyBuffer(device, ubo.buffer, nullptr);
            vkFreeMemory(device, ubo.memory, nullptr);
        });
    }
}

// Update uniform buffers for view at index [view_idx] for frame [image_idx]
void MimirInstance::updateUniformBuffers(uint32_t image_idx)
{
    auto min_alignment = physical_device.getUboOffsetAlignment();
    auto size_mvp = getAlignedSize(sizeof(ModelViewProjection), min_alignment);
    auto size_view = getAlignedSize(sizeof(ViewUniforms), min_alignment);
    auto size_scene = getAlignedSize(sizeof(SceneUniforms), min_alignment);
    auto size_ubo = size_mvp + size_view + size_scene;
    auto memory = uniform_buffers[image_idx].memory;

    for (size_t view_idx = 0; view_idx < views.size(); ++view_idx)
    {
        auto& view = views[view_idx];
        if (!view->desc.visible) { continue; }

        // The raster pipeline consumes matrices.view as a trackball (orientation pivots the scene
        // about the world origin). For the fly camera we instead feed it a proper world-to-view so
        // rotation pivots about the eye = free-look. camera.matrices.view is camera-to-world with
        // columns right/up/fwd and eye in column 3; glm::lookAt gives the matching world-to-view
        // (translation in column 3, where the shaders read it). The path tracer reads
        // camera.matrices.view itself (not this UBO copy), so it is unaffected.
        //
        // glm::lookAt is a proper (non-reflected) world-to-view, so world chirality is preserved:
        // an eye at (0,0,4) looking down -z sees +x on screen right, exactly like the orbit
        // trackball at its home view (and like datoviz). The fly input paths (mouse yaw in
        // window.cpp, WASD strafe in updateCamera) are written against this same convention.
        glm::mat4 raster_view = camera.matrices.view;
        glm::mat4 raster_proj = camera.matrices.perspective;
        if (options.camera_control == CameraControl::Fly)
        {
            glm::vec3 fwd = glm::vec3(camera.matrices.view[2]);
            glm::vec3 eye = glm::vec3(camera.matrices.view[3]);
            raster_view = glm::lookAt(eye, eye + fwd, glm::vec3(0.f, 1.f, 0.f));
        }

        ModelViewProjection mvp{
            .model = view->translation * view->rotation * view->scale,
            .view  = raster_view,
            .proj  = raster_proj,
            .all   = mvp.proj * mvp.view * mvp.model,
            .inv_model = glm::inverse(mvp.model),
            .inv_view  = glm::inverse(mvp.view),
        };

        auto color = view->desc.default_color;
        ViewUniforms vu{
            .color     = glm::vec4(color.x, color.y, color.z, color.w),
            .size      = view->desc.default_size,
            .linewidth = view->desc.linewidth,
            .antialias = view->desc.antialias,
        };

        auto bg = options.background_color;
        auto extent = view->desc.layout;
        // light_pos needs no per-mode adjustment: the fly camera's glm::lookAt view matches the
        // orbit trackball view at the shared home pose (eye (0,0,4) looking -z => identity
        // rotation), so the marker shader's world->view light transform lights both the same.
        auto lp = options.light_pos;
        auto lc = options.light_color;
        auto sc = options.specular_color;
        SceneUniforms su{
            .background_color = glm::vec4(bg.x, bg.y, bg.z, bg.w),
            .extent           = glm::ivec3{extent.x, extent.y, extent.z},
            .resolution       = glm::ivec2{options.window.size.x, options.window.size.y},
            .camera_pos       = camera.position,
            .light_pos        = glm::vec3(lp.x, lp.y, lp.z),
            .light_color      = glm::vec3{lc.x, lc.y, lc.z},
            .specular_color   = glm::vec3{sc.x, sc.y, sc.z},
            .specular_power   = options.specular_power,
            .ambient_strength = options.ambient_strength,
        };

        char *data = nullptr;
        auto offset = size_ubo * view_idx;
        validation::checkVulkan(vkMapMemory(device, memory, offset, size_ubo, 0, (void**)&data));
        std::memcpy(data, &mvp, sizeof(mvp));
        std::memcpy(data + size_mvp, &vu, sizeof(vu));
        std::memcpy(data + size_mvp + size_view, &su, sizeof(su));
        vkUnmapMemory(device, memory);
    }
}

struct ConvertedMemory
{
    float data;
    std::string units;
};

ConvertedMemory formatMemory(uint64_t memsize)
{
    constexpr float kilobyte = 1024.f;
    constexpr float megabyte = kilobyte * 1024.f;
    constexpr float gigabyte = megabyte * 1024.f;

    ConvertedMemory converted{};
    converted.data = static_cast<float>(memsize) / gigabyte;
    converted.units = "GB";

    return converted;
}

std::string readMemoryHeapFlags(VkMemoryHeapFlags flags)
{
    switch (flags)
    {
        case VK_MEMORY_HEAP_DEVICE_LOCAL_BIT: return "Device local bit";
        case VK_MEMORY_HEAP_MULTI_INSTANCE_BIT: return "Multiple instance bit";
        default: return "Host local heap memory";
    }
    return "";
}

PerformanceMetrics MimirInstance::getMetrics()
{
    // Update memory usage stats
    auto memory = physical_device.getMemoryStats();

    return PerformanceMetrics{
        .frame_rate = graphics_monitor.getFramerate(),
        .times = {
            .compute  = compute_monitor.total_compute_time,
            .graphics = graphics_monitor.total_graphics_time,
            .pipeline = (float)graphics_monitor.total_pipeline_time,
            .tlas_build = rt_enabled ? (float)raytracing.last_build_ms : 0.f,
            .trace      = rt_enabled ? (float)raytracing.last_trace_ms : 0.f,
            .wait   = graphics_monitor.last_wait_ms,
            .record = graphics_monitor.last_record_ms,
            .submit = graphics_monitor.last_submit_ms,
            .gpu    = graphics_monitor.last_gpu_ms,
        },
        .devmem = {
            .usage  = formatMemory(memory.usage).data,
            .budget = formatMemory(memory.budget).data,
        }
    };
}

void MimirInstance::immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function)
{
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    auto alloc_info = VkCommandBufferAllocateInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext              = nullptr,
        .commandPool        = command_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    validation::checkVulkan(vkAllocateCommandBuffers(device, &alloc_info, &cmd));

    // Begin command buffer recording with a only-one-use buffer
    VkCommandBufferBeginInfo cmd_info{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext            = nullptr,
        .flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    validation::checkVulkan(vkBeginCommandBuffer(cmd, &cmd_info));
    function(cmd);
    validation::checkVulkan(vkEndCommandBuffer(cmd));

    VkSubmitInfo submit_info{
        .sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext                = nullptr,
        .waitSemaphoreCount   = 0,
        .pWaitSemaphores      = nullptr,
        .pWaitDstStageMask    = nullptr,
        .commandBufferCount   = 1,
        .pCommandBuffers      = &cmd,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores    = nullptr,
    };
    auto queue = graphics.queue;
    validation::checkVulkan(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE));
    validation::checkVulkan(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(device, command_pool, 1, &cmd);
}

void MimirInstance::loadTexture(TextureDescription desc, void *data, size_t memsize)
{
    ImageParams params{
        .type   = getImageType(desc.extent),
        .format = getVulkanFormat(desc.format),
        .extent = getVulkanExtent(desc.extent),
        .tiling = getImageTiling(desc.source),
        .usage  = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .levels = desc.levels,
    };
    VkExternalMemoryImageCreateInfo extmem_info{
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    auto image = createImage(device, physical_device.handle, params, &extmem_info);
    validation::checkVulkan(vkBindImageMemory(device, image, getMemoryVulkan(desc.source), 0));

    // Create staging buffer to copy image data
    VkDeviceSize staging_size = getSourceSize(desc.source);
    auto staging_buffer = createBuffer(device, staging_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    auto available = physical_device.memory.memoryProperties;
    VkMemoryRequirements staging_req{};
    vkGetBufferMemoryRequirements(device, staging_buffer, &staging_req);
    auto staging_memory = allocateMemory(device, available, staging_req,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    vkBindBufferMemory(device, staging_buffer, staging_memory, 0);

    char *mapped = nullptr;
    validation::checkVulkan(vkMapMemory(device, staging_memory, 0, memsize, 0, (void**)&mapped));
    memcpy(mapped, data, static_cast<size_t>(memsize));
    vkUnmapMemory(device, staging_memory);

    transitionImageLayout(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToTexture(staging_buffer, image, params.extent);

    generateMipmaps(image, params.format, params.extent.width, params.extent.height, desc.levels);
    validation::checkCuda(cudaDeviceSynchronize());

    vkDestroyBuffer(device, staging_buffer, nullptr);
    vkFreeMemory(device, staging_memory, nullptr);
    vkDestroyImage(device, image, nullptr);
}

void MimirInstance::copyBufferToTexture(VkBuffer buffer, VkImage image, VkExtent3D extent)
{
    VkImageSubresourceLayers subres{
        .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
        .mipLevel       = 0,
        .baseArrayLayer = 0,
        .layerCount     = 1
    };
    VkBufferImageCopy region{
        .bufferOffset      = 0,
        .bufferRowLength   = 0,
        .bufferImageHeight = 0,
        .imageSubresource  = subres,
        .imageOffset       = {0, 0, 0},
        .imageExtent       = extent
    };
    immediateSubmit([=](VkCommandBuffer cmd)
    {
        auto layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        vkCmdCopyBufferToImage(cmd, buffer, image, layout, 1, &region);
    });
}

void MimirInstance::generateMipmaps(VkImage image, VkFormat format,
    int img_width, int img_height, int mip_levels)
{
    auto props = getImageFormatProperties(physical_device.handle, format);
    auto blit_support = VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    if (!(props.optimalTilingFeatures & blit_support))
    {
        spdlog::error("texture image format does not support linear blitting!");
    }

    immediateSubmit([=](VkCommandBuffer cmd)
    {
        VkImageMemoryBarrier barrier{
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext               = nullptr,
            .srcAccessMask       = 0,
            .dstAccessMask       = 0,
            .oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout           = VK_IMAGE_LAYOUT_UNDEFINED,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = image,
            .subresourceRange = VkImageSubresourceRange{
                .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel   = 0,
                .levelCount     = 1,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            }
        };

        int32_t mip_width  = img_width;
        int32_t mip_height = img_height;

        for (uint32_t i = 1; i < static_cast<uint32_t>(mip_levels); i++)
        {
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                                nullptr, 1, &barrier);

            int32_t mip_x = mip_width > 1 ? mip_width / 2 : 1;
            int32_t mip_y = mip_height > 1 ? mip_height / 2 : 1;
            VkImageBlit blit{
                .srcSubresource = VkImageSubresourceLayers{
                    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel       = i - 1,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
                .srcOffsets = { {0, 0, 0}, {mip_width, mip_height, 1} },
                .dstSubresource = VkImageSubresourceLayers{
                    .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel       = i,
                    .baseArrayLayer = 0,
                    .layerCount     = 1,
                },
                .dstOffsets = { {0, 0, 0}, {mip_x, mip_y, 1} },
            };

            vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                            VK_FILTER_LINEAR);

            barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                                0, nullptr, 1, &barrier);

            if (mip_width > 1) mip_width /= 2;
            if (mip_height > 1) mip_height /= 2;
        }

        barrier.subresourceRange.baseMipLevel = mip_levels - 1;
        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
            0, nullptr, 1, &barrier
        );
    });
}

void MimirInstance::transitionImageLayout(VkImage image,
    VkImageLayout old_layout, VkImageLayout new_layout)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = old_layout;
    barrier.newLayout           = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = image;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel   = 0;
    barrier.subresourceRange.levelCount     = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount     = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = 0;

    VkPipelineStageFlags src_stage, dst_stage;
    if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED)
    {
        barrier.srcAccessMask = 0;
        src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    }
    else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (old_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else
    {
        spdlog::error("unsupported layout transition");
        return;
    }

    if (new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dst_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else
    {
        spdlog::error("unsupported layout transition");
        return;
    }

    immediateSubmit([=](VkCommandBuffer cmd)
    {
        vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    });
}

} // namespace mimir