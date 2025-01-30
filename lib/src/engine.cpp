#include "mimir/engine.hpp"

#include "mimir/api.hpp"
#include "mimir/framelimit.hpp"
#include "mimir/gui.hpp"
#include "mimir/resources.hpp"
#include "mimir/validation.hpp"
#include "mimir/shader_types.hpp"

#include <iostream>
#include <chrono> // std::chrono
#include <set> // std::set

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
Camera defaultCamera(int width, int height)
{
    auto camera = Camera::make();
    camera.type           = Camera::CameraType::LookAt;
    camera.rotation_speed = 0.5f;
    camera.flip_y         = true;
    camera.setPosition(glm::vec3(0.f, 0.f, -2.85f));
    camera.setRotation(glm::vec3(0.f, 0.f, 0.f));
    camera.setPerspective(60.f, (float)width / (float)height, 0.1f, 256.f);
    return camera;
}

MimirEngine MimirEngine::make(ViewerOptions opts)
{
    MimirEngine engine{
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
        .compute_active    = false,
        .rendering_thread  = {},
        .uniform_buffers   = {},
        .views             = {},
        .window_context    = {},
        .camera            = {},
        .deletors          = {},
        .graphics_monitor  = {},
        .compute_monitor   = {},
    };

    spdlog::set_level(spdlog::level::trace);
    spdlog::set_pattern("[%H:%M:%S] [%l] %v");

    engine.options.present.max_fps = engine.options.present.mode == PresentMode::VSync? 60 : 300;
    engine.options.present.target_frame_time = getTargetFrameTime(
        engine.options.present.enable_fps_limit, engine.options.present.target_fps
    );

    auto width  = engine.options.window.size.x;
    auto height = engine.options.window.size.y;
    engine.window_context = GlfwContext::make(width, height, engine.options.window.title.c_str(), &engine);
    engine.deletors.context.add([&] { engine.window_context.clean(); });
    engine.camera = defaultCamera(width, height);

    engine.initVulkan();

    return engine;
}

MimirEngine MimirEngine::make(int width, int height)
{
    ViewerOptions opts;
    opts.window.size = {width, height};
    return MimirEngine::make(opts);
}

void MimirEngine::exit()
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
    cleanupGraphics();
    gui::shutdown();
    window_context.exit();
    deletors.views.flush();
    deletors.context.flush();
}

void MimirEngine::prepare()
{
    initUniformBuffers();
    createViewPipelines();
    updateDescriptorSets();
}

void MimirEngine::displayAsync()
{
    prepare();
    running = true;
    rendering_thread = std::thread([&,this]()
    {
        while(!window_context.shouldClose())
        {
            window_context.processEvents();
            gui::draw(camera, options, views, gui_callback);
            renderFrame();
        }
        running = false;
        vkDeviceWaitIdle(device);
    });
}

void MimirEngine::prepareViews()
{
    if (options.present.enable_sync && running)
    {
        waitKernelStart();
        compute_monitor.startWatch();
    }
}

void MimirEngine::waitKernelStart()
{
    static uint64_t wait_value = 1;
    cudaExternalSemaphoreWaitParams wait_params{};
    wait_params.flags = 0;
    wait_params.params.fence.value = wait_value;
    //spdlog::trace("kernel waits for render {}", wait_params.params.fence.value);

    // Wait for Vulkan to complete its work
    validation::checkCuda(cudaWaitExternalSemaphoresAsync(
        &interop.cuda_semaphore, &wait_params, 1, interop.cuda_stream)
    );
    wait_value += 2;
    compute_active = true;
}

void MimirEngine::updateViews()
{
    if (options.present.enable_sync && running)
    {
        compute_monitor.stopWatch();
        signalKernelFinish();
        compute_active = false;
    }
}

void MimirEngine::signalKernelFinish()
{
    static uint64_t signal_value = 2;
    cudaExternalSemaphoreSignalParams signal_params{};
    signal_params.flags = 0;
    signal_params.params.fence.value = signal_value;
    //spdlog::trace("kernel signals iteration {}", signal_params.params.fence.value);

    // Signal Vulkan to continue with the updated buffers
    validation::checkCuda(cudaSignalExternalSemaphoresAsync(
        &interop.cuda_semaphore, &signal_params, 1, interop.cuda_stream)
    );
    signal_value += 2;
}

void MimirEngine::display(std::function<void(void)> func, size_t iter_count)
{
    prepare();

    running = true;
    compute_active = true;
    size_t iter_idx = 0;
    while(!window_context.shouldClose())
    {
        window_context.processEvents();
        gui::draw(camera, options, views, gui_callback);
        renderFrame();

        if (iter_idx < iter_count)
        {
            waitKernelStart();
            func(); // Advance the simulation
            iter_idx++;
            signalKernelFinish();
        }
    }
    running = false;
    compute_active = false;
    vkDeviceWaitIdle(device);
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

void initGridCoords(float3 *data, ViewExtent size, float3 start)
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

AttributeDescription MimirEngine::makeStructuredGrid(ViewExtent size, float3 start)
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
    auto grid_alloc = new Allocation({AllocationType::Linear, memreq.size, vk_memory, nullptr});
    deletors.context.add([=,this]{ delete grid_alloc; });

    // Add deletors to queue for later cleanup
    deletors.views.add([=,this]{
        spdlog::trace("Free structured domain memory");
        vkFreeMemory(device, vk_memory, nullptr);
        vkDestroyBuffer(device, domain_buffer, nullptr);
    });

    return AttributeDescription{
        .source     = grid_alloc,
        .size       = size.x * size.y * size.z,
        .format     = FormatDescription::make<float3>(),
        .indices    = nullptr,
        .index_size = 0,
    };
}

AttributeDescription MimirEngine::makeImageDomain()
{
    const std::vector<Vertex> vertices{
        { {  1.f,  1.f, 0.f }, { 1.f, 1.f } },
        { { -1.f,  1.f, 0.f }, { 0.f, 1.f } },
        { { -1.f, -1.f, 0.f }, { 0.f, 0.f } },
        { {  1.f, -1.f, 0.f }, { 1.f, 0.f } }
    };
    // Indices for a single uv-view quad made from two triangles
    const std::vector<uint16_t> indices{ 0, 1, 2, 2, 3, 0 };//, 4, 5, 6, 6, 7, 4 };

    uint32_t vert_size = sizeof(Vertex) * vertices.size();
    uint32_t ids_size = sizeof(uint16_t) * indices.size();

    auto vbo = createBuffer(device, vert_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    VkMemoryRequirements memreq{};
    vkGetBufferMemoryRequirements(device, vbo, &memreq);
    // Allocate memory and bind it to buffers
    auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    auto available = physical_device.memory.memoryProperties;
    auto vbo_mem = allocateMemory(device, available, memreq, flags);
    validation::checkVulkan(vkBindBufferMemory(device, vbo, vbo_mem, 0));
    auto vbo_alloc = new Allocation({AllocationType::Linear, memreq.size, vbo_mem, nullptr});

    auto ibo = createBuffer(device, ids_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    vkGetBufferMemoryRequirements(device, ibo, &memreq);
    // Allocate memory and bind it to buffers
    flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    auto ibo_mem = allocateMemory(device, available, memreq, flags);
    validation::checkVulkan(vkBindBufferMemory(device, ibo, ibo_mem, 0));
    auto ibo_alloc = new Allocation({AllocationType::Linear, memreq.size, ibo_mem, nullptr});

    // Init image quad coords and indices
    char *vert_data = nullptr;
    vkMapMemory(device, vbo_mem, 0, vert_size, 0, (void**)&vert_data);
    std::memcpy(vert_data, vertices.data(), vert_size);
    vkUnmapMemory(device, vbo_mem);

    char *ids_data = nullptr;
    vkMapMemory(device, ibo_mem, 0, ids_size, 0, (void**)&ids_data);
    std::memcpy(ids_data, indices.data(), ids_size);
    vkUnmapMemory(device, ibo_mem);

    deletors.views.add([=,this]{
        vkFreeMemory(device, vbo_mem, nullptr);
        vkDestroyBuffer(device, vbo, nullptr);
        vkFreeMemory(device, ibo_mem, nullptr);
        vkDestroyBuffer(device, ibo, nullptr);
    });

    return AttributeDescription{
        .source     = vbo_alloc,
        .size       = static_cast<uint32_t>(vertices.size()),
        .format     = FormatDescription::make<float3>(),
        .indices    = ibo_alloc,
        .index_size = sizeof(uint16_t),
    };
}

Texture *MimirEngine::makeTexture(TextureDescription desc)
{
    ImageParams params{
        .type   = getImageType(desc.extent),
        .format = getVulkanFormat(desc.format),
        .extent = getVulkanExtent(desc.extent),
        .tiling = getImageTiling(desc.source->type),
        .usage  = VK_IMAGE_USAGE_SAMPLED_BIT,
        .levels = desc.levels,
    };
    VkExternalMemoryImageCreateInfo extmem_info{
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    auto teximg = createImage(device, physical_device.handle, params, &extmem_info);
    validation::checkVulkan(vkBindImageMemory(device, teximg, desc.source->vk_mem, 0));

    Texture texture{
        .image    = teximg,
        .img_view = createImageView(device, texture.image, params, VK_IMAGE_ASPECT_COLOR_BIT),
        .sampler  = createSampler(device, VK_FILTER_LINEAR, false),
        .format   = params.format,
        .extent   = params.extent,
    };

    transitionImageLayout(texture.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );

    deletors.context.add([=,this]{
        spdlog::trace("Destroying texture");
        vkDestroyImageView(device, texture.img_view, nullptr);
        vkDestroyImage(device, texture.image, nullptr);
        vkDestroySampler(device, texture.sampler, nullptr);
    });

    auto tex_ptr = new Texture(texture);
    deletors.context.add([=,this]{ delete tex_ptr; });
    return tex_ptr;
}

Allocation *MimirEngine::allocLinear(void **dev_ptr, size_t size)
{
    assert(size > 0);

    auto usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    // Create temporary buffer for querying the desired memory properties
    VkExternalMemoryBufferCreateInfo extmem_info{
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    };
    auto query_buf = createBuffer(device, size, usage, &extmem_info);
    VkMemoryRequirements memreq{};
    vkGetBufferMemoryRequirements(device, query_buf, &memreq);

    // Allocate external device memory
    VkExportMemoryAllocateInfoKHR export_info{
        .sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    };
    auto available = physical_device.memory.memoryProperties;
    auto memflags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    auto vk_memory = allocateMemory(device, available, memreq, memflags, &export_info);
    spdlog::debug("Allocated {} bytes for interop ({} requested)", size, memreq.size);

    // Export and map the external memory to CUDA
    auto cuda_extmem = interop::importCudaExternalMemory(vk_memory, memreq.size, device);

    // Add deletors to queue for later cleanup
    deletors.views.add([=,this]{
        spdlog::trace("Free interop memory");
        validation::checkCuda(cudaDestroyExternalMemory(cuda_extmem));
        vkFreeMemory(device, vk_memory, nullptr);
    });
    vkDestroyBuffer(device, query_buf, nullptr);

    Allocation alloc{
        .type        = AllocationType::Linear,
        .size        = memreq.size,
        .vk_mem      = vk_memory,
        .cuda_extmem = cuda_extmem
    };
    cudaExternalMemoryBufferDesc buffer_desc{ .offset = 0, .size = size, .flags = 0 };
    validation::checkCuda(cudaExternalMemoryGetMappedBuffer(
        dev_ptr, alloc.cuda_extmem, &buffer_desc)
    );
    auto alloc_ptr = new Allocation(alloc);
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
    // For now, assume that all channels have same size
    // TODO: Handle case when channels are different
    int size = desc->x / 8;
    // Assume that a component exists if its size is greater than zero
    int components = (desc->x > 0) + (desc->y > 0) + (desc->z > 0) + (desc->w > 0);
    return { .kind = kind, .size = size, .components = components };
}

VkExtent3D getExtentFromCuda(cudaExtent extent)
{
    return VkExtent3D
    {
        .width  = extent.width > 0? (uint32_t)extent.width : 1,
        .height = extent.height > 0? (uint32_t)extent.height : 1,
        .depth  = extent.depth > 0? (uint32_t)extent.depth : 1,
    };
}

Allocation *MimirEngine::allocMipmap(cudaMipmappedArray_t *dev_arr,
    const cudaChannelFormatDesc *desc, cudaExtent extent, unsigned int num_levels)
{
    auto format = getFormatFromCuda(desc);
    ImageParams img_params{
        .type   = getImageType(ViewExtent::make(extent.width, extent.height, extent.depth)),
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

    auto alloc = Allocation{AllocationType::Opaque, memreq.size, vk_memory, cuda_extmem};
    cudaExternalMemoryMipmappedArrayDesc array_desc{
        .offset     = 0,
        .formatDesc = *desc,
        .extent     = extent,
        .flags      = 0,
        .numLevels  = num_levels,
    };
    validation::checkCuda(cudaExternalMemoryGetMappedMipmappedArray(
        dev_arr, cuda_extmem, &array_desc)
    );

    auto alloc_ptr = new Allocation(alloc);
    deletors.context.add([=,this]{ delete alloc_ptr; });
    return alloc_ptr;
}

VkBuffer MimirEngine::createAttributeBuffer(VkDeviceSize memsize,
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
    bool has_position_attr = false;
    for (auto &[type, attr] : desc->attributes)
    {
        has_position_attr |= type == AttributeType::Position;
    }
    return has_position_attr;
}

View *MimirEngine::createView(ViewDescription *desc)
{
    if (!validateViewDescription(desc))
    {
        spdlog::error("Invalid view");
        return nullptr;
    }

    ViewDetails detail{
        .pipeline   = VK_NULL_HANDLE,
        .draw_count = desc->element_count,
        .vb_count   = 0,
        .vbo        = {VK_NULL_HANDLE},
        .offsets    = {0},
        .use_ibo    = false,
        .ibo        = VK_NULL_HANDLE,
        .index_type = VK_INDEX_TYPE_NONE_KHR,
        .tex_count  = 0,
        .textures   = {},
        .ssbo_count = 0,
        .storage    = {VK_NULL_HANDLE},
        .desc       = *desc,
    };

    // Create attribute buffers
    for (auto &[type, attr] : desc->attributes)
    {
        spdlog::trace("Processing {} attribute", getAttributeType(type));
        if (type == AttributeType::Color && desc->view_type == ViewType::Image)
        {
            ImageParams params{
                .type   = getImageType(desc->extent),
                .format = getVulkanFormat(attr.format),
                .extent = getVulkanExtent(desc->extent),
                .tiling = getImageTiling(attr.source->type),
                .usage  = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                .levels = 1,
            };
            VkExternalMemoryImageCreateInfo extmem_info{
                .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                .pNext       = nullptr,
                .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
            };
            auto teximg = createImage(device, physical_device.handle, params, &extmem_info);
            vkBindImageMemory(device, teximg, attr.source->vk_mem, 0);

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

            detail.textures[detail.tex_count++] = tex;
        }
        // Handle image quad vertex buffer size
        else if (type == AttributeType::Position && desc->view_type == ViewType::Image)
        {
            VkDeviceSize vb_size = sizeof(Vertex) * attr.size;
            VkBufferUsageFlags vb_usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            VkDeviceMemory vb_mem = attr.source->vk_mem;
            // TODO: Get if there is still space remaining (or maybe do it in validation)
            detail.vbo[detail.vb_count] = createAttributeBuffer(vb_size, vb_usage, vb_mem);
            detail.vb_count++;
        }
        // Map source to a vertex buffer when accessing its elements directly
        // Source is always mapped this way for position attributes
        else if (type == AttributeType::Position || attr.indices == nullptr)
        {
            VkDeviceSize vb_size = attr.format.getSize() * attr.size; // sizeof(Vertex) * attr.size;
            spdlog::trace("Position vertex buffer created with {} bytes ({} available)",
                vb_size, attr.source->size
            );
            VkBufferUsageFlags vb_usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            VkDeviceMemory vb_mem = attr.source->vk_mem;
            // TODO: Get if there is still space remaining (or maybe do it in validation)
            detail.vbo[detail.vb_count] = createAttributeBuffer(vb_size, vb_usage, vb_mem);
            detail.vb_count++;
        }
        // If a non-position attribute uses indirect mapping, its source is mapped to a storage buffer
        else
        {
            VkDeviceSize sb_size = attr.format.getSize() * attr.size;
            spdlog::trace("Position storage buffer created with {} bytes ({} available)",
                sb_size, attr.source->size
            );
            VkBufferUsageFlags sb_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            VkDeviceMemory sb_mem = attr.source->vk_mem;
            detail.storage[detail.ssbo_count++] = createAttributeBuffer(sb_size, sb_usage, sb_mem);
        }

        // If there is no indirect source access, the attribute is now fully processed
        if (attr.indices == nullptr) { continue; }

        // Create indirect buffers as index buffer for position attributes,
        // and as vertex buffers for all other attributes
        VkDeviceMemory memory = attr.indices->vk_mem;
        VkDeviceSize memsize = attr.index_size * desc->element_count;
        spdlog::trace("Attribute buffer created with {} bytes", memsize);
        if (type == AttributeType::Position)
        {
            VkBufferUsageFlags ib_usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
            detail.ibo = createAttributeBuffer(memsize, ib_usage, memory);
            detail.index_type = getIndexBufferType(attr.index_size);
            detail.use_ibo = true;
        }
        else
        {
            VkBufferUsageFlags vb_usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            detail.vbo[detail.vb_count++] = createAttributeBuffer(memsize, vb_usage, memory);
        }
    }

    auto inner = new ViewDetails(detail);
    auto handle = new View({
        .detail        = inner,
        .visible       = true,
        .default_color = { 0.f, 0.f, 0.f, 1.f },
        .default_size  = 10.f,
    });
    deletors.views.add([=,this]{ delete inner; delete handle; });
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

void MimirEngine::initVulkan()
{
    createInstance();
    window_context.createSurface(instance, &surface);
    deletors.context.add([=,this](){
        vkDestroySurfaceKHR(instance, surface, nullptr);
    });
    physical_device = pickPhysicalDevice(instance, surface);

    findQueueFamilies(physical_device.handle, surface, graphics.family_index, present.family_index);
    std::set unique_queue_families{ graphics.family_index, present.family_index };
    std::vector<uint32_t> queue_families(unique_queue_families.begin(), unique_queue_families.end());
    device = createLogicalDevice(physical_device.handle, queue_families);
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

    initGraphics();
    createSyncObjects();
    // After command pool and render pass are created
    gui::init(instance, physical_device.handle, device,
        descriptor_pool, render_pass, graphics, window_context
    );
    descriptor_sets = createDescriptorSets(device,
        descriptor_pool, descriptor_layout, swapchain.image_count
    );
}

void MimirEngine::createInstance()
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

    // List additional required validation layers
    auto extensions = GlfwContext::getRequiredExtensions();
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

void MimirEngine::createSyncObjects()
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

void MimirEngine::cleanupGraphics()
{
    vkDeviceWaitIdle(device);
    //vkFreeCommandBuffers(device, command_pool, command_buffers.size(), command_buffers.data());
    deletors.graphics.flush();
}

void MimirEngine::initGraphics()
{
    // Initialize swapchain
    int width, height;
    window_context.getFramebufferSize(width, height);
    auto present_mode = getDesiredPresentMode(options.present.mode);
    std::vector queue_indices{graphics.family_index, present.family_index};
    swapchain = Swapchain::make(device, physical_device.handle,
        surface, width, height, present_mode, queue_indices
    );

    // Create one command buffer per swapchain image
    command_buffers = createCommandBuffers(device, command_pool, swapchain.image_count);

    // Initialize metrics monitoring
    auto timestamp_period = physical_device.general.properties.limits.timestampPeriod;
    graphics_monitor = metrics::GraphicsMonitor::make(device, 2 * command_buffers.size(), timestamp_period, 240);
    compute_monitor = metrics::ComputeMonitor::make(0);

    deletors.graphics.add([=,this]{
        vkDestroySwapchainKHR(device, swapchain.current, nullptr);
        vkDestroyQueryPool(device, graphics_monitor.query_pool, nullptr);
        cudaEventDestroy(compute_monitor.start);
        cudaEventDestroy(compute_monitor.stop);
    });

    // Create depth image and image view from available formats
    std::vector<VkFormat> candidate_formats{
        VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT
    };
    auto depth_format = findSupportedImageFormat(physical_device.handle, candidate_formats,
        VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
    ImageParams params{
        .type   = VK_IMAGE_TYPE_2D,
        .format = depth_format,
        .extent = { swapchain.extent.width, swapchain.extent.height, 1 },
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage  = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .levels = 1,
    };
    depth_image = createImage(device, physical_device.handle, params);

    auto available = physical_device.memory.memoryProperties;
    VkMemoryRequirements mem_req{};
    vkGetImageMemoryRequirements(device, depth_image, &mem_req);
    depth_memory = allocateMemory(device, available, mem_req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    validation::checkVulkan(vkBindImageMemory(device, depth_image, depth_memory, 0));
    depth_view = createImageView(device, depth_image, params, VK_IMAGE_ASPECT_DEPTH_BIT);

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
        .finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };
    VkAttachmentDescription depth{
        .flags          = 0, // Can be VK_ATTACHMENT_DESCRIPTION_MAY_ALIAS_BIT
        .format         = depth_format,
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE,
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

    framebuffers = Framebuffer::make(device, render_pass, swapchain, depth_view);
    for (uint32_t i = 0; i < swapchain.image_count; ++i)
    {
        deletors.graphics.add([=,this]{
            vkDestroyImageView(device, framebuffers.image_views[i], nullptr);
            vkDestroyFramebuffer(device, framebuffers.handles[i], nullptr);
        });
    }
}

void MimirEngine::recreateGraphics()
{
    cleanupGraphics();
    initGraphics();
    createViewPipelines();
}

void MimirEngine::updateDescriptorSets()
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
            for (uint32_t k = 0; k < view->detail->tex_count; ++k)
            {
                auto tex = view->detail->textures[k];

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
            for (uint32_t k = 0; k < view->detail->ssbo_count; ++k)
            {
                auto ssbo = view->detail->storage[k];

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

void MimirEngine::waitTimelineHost()
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

void MimirEngine::renderFrame()
{
    // Get frame index from the inflight frames array
    auto frame_idx = render_timeline % MAX_FRAMES_IN_FLIGHT;

    // Retrieve synchronization data for frame i
    auto frame_sync = sync_data[frame_idx];
    auto fence = frame_sync.frame_fence;

    // Wait for fence of frame i to end, then immediately reset it for further use
    validation::checkVulkan(vkWaitForFences(device, 1, &fence, VK_TRUE, frame_timeout));
    validation::checkVulkan(vkResetFences(device, 1, &fence));

    // Start measuring frame time
    graphics_monitor.startFrameWatch();

    static uint64_t wait_value = 0;
    static uint64_t signal_value = 1;

    bool advance_timeline = false;
    std::vector<VkSemaphore> waits           = {frame_sync.image_acquired};
    std::vector<VkPipelineStageFlags> stages = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    std::vector<VkSemaphore> signals         = {frame_sync.render_complete};
    std::vector<uint64_t> wait_values        = {0};
    std::vector<uint64_t> signal_values      = {0};

    if (compute_active && options.present.enable_sync)
    {
        VkSemaphoreWaitInfo wait_info{
            .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .pNext          = nullptr,
            .flags          = 0,
            .semaphoreCount = 1,
            .pSemaphores    = &interop.vk_semaphore,
            .pValues        = &wait_value,
        };
        validation::checkVulkan(vkWaitSemaphores(device, &wait_info, frame_timeout));

        waits.push_back(interop.vk_semaphore);
        stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        signals.push_back(interop.vk_semaphore);
        wait_values.push_back(interop.timeline_value);
        signal_values.push_back(signal_value);
        advance_timeline = true;
    }

    // Acquire image from swap chain, signaling to the semaphore when it is ready for use
    uint32_t image_idx;
    auto result = vkAcquireNextImageKHR(device, swapchain.current,
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

    // if (images_inflight[image_idx] != VK_NULL_HANDLE)
    // {
    //     vkWaitForFences(device, 1, &images_inflight[image_idx], VK_TRUE, timeout);
    // }
    // images_inflight[image_idx] = frame.render_fence;

    if (render_timeline > MAX_FRAMES_IN_FLIGHT)
    {
        graphics_monitor.getRenderTimeResults(device, frame_idx);
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

    // Set clear color and depth stencil value
    std::array<VkClearValue, 2> clear_values{};
    std::memcpy(clear_values[0].color.float32, &options.background_color.x, sizeof(options.background_color));
    clear_values[1].depthStencil = {1.f, 0};

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

    drawElements(frame_idx);
    gui::render(cmd);
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

    // Fill submit waits & signals info
    VkTimelineSemaphoreSubmitInfo *extra = nullptr;
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    if (running && options.present.enable_sync)
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

    // Execute command buffer using image as attachment in framebuffer
    validation::checkVulkan(vkQueueSubmit(graphics.queue, 1, &submit_info, fence));

    // Return image result back to swapchain for presentation on screen
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

    // Limit frame if it was configured
    if (options.present.enable_fps_limit) { frameStall(options.present.target_frame_time); }
    graphics_monitor.stopFrameWatch();
    //spdlog::trace("frame {} finished", render_timeline-1);

    /*if (options.report_period > 0 && frame_time > options.report_period)
    {
        printf("Report at %d seconds:\n", options.report_period);
        showMetrics();
        last_time = current_time;
    }*/
}

void MimirEngine::drawElements(uint32_t image_idx)
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
        if (!views[i]->visible) { continue; }
        auto& view = views[i]->detail;

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
            vkCmdBindIndexBuffer(cmd, view->ibo, 0, view->index_type);
            vkCmdDrawIndexed(cmd, view->draw_count, 1, 0, 0, 0);
        }
        else // Perform regular draw with bound vertex buffers
        {
            // Instanced rendering is not currently supported
            uint32_t instance_count = 1;
            uint32_t first_vertex = 0;
            // uint32_t scenario_count = view->params.offsets.size();
            // uint32_t scenario_idx = view->params.options.scenario_index;
            // if (scenario_idx < scenario_count)
            // {
            //     vertex_count = view->params.sizes[scenario_idx];
            //     first_vertex = view->params.offsets[scenario_idx];
            // }
            vkCmdDraw(cmd, view->draw_count, instance_count, first_vertex, 0);
        }
    }
}

void MimirEngine::createViewPipelines(/*std::span<std::shared_ptr<InteropView>> views*/)
{
    auto start = std::chrono::steady_clock::now();

    pipeline_builder = PipelineBuilder::make(pipeline_layout, swapchain.extent);
    for (auto& view : views)
    {
        pipeline_builder.addPipeline(view->detail->desc, device);
    }
    auto pipelines = pipeline_builder.createPipelines(device, render_pass);
    for (size_t i = 0; i < pipelines.size(); ++i)
    {
        auto& v = views[i]->detail;
        v->pipeline = pipelines[i];
        deletors.graphics.add([=,this]{ vkDestroyPipeline(device, v->pipeline, nullptr); });
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    spdlog::trace("Created {} pipeline object(s) in {} ms", pipelines.size(), elapsed);
}

void MimirEngine::initUniformBuffers()
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
void MimirEngine::updateUniformBuffers(uint32_t image_idx)
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

        ModelViewProjection mvp{
            .model = glm::mat4(1.f),
            .view  = camera.matrices.view,
            .proj  = camera.matrices.perspective,
        };

        auto dc = view->default_color;
        ViewUniforms vu{
            .color         = glm::vec4(dc[0], dc[1], dc[2], dc[3]),
            .size          = view->default_size,
            .depth         = 0.f,
            .element_count = 0,
        };

        auto bg = options.background_color;
        auto extent = view->detail->desc.extent;
        SceneUniforms su{
            .background_color    = glm::vec4(bg.x, bg.y, bg.z, bg.w),
            .extent      = glm::ivec3{extent.x, extent.y, extent.z},
            .resolution  = glm::ivec2{options.window.size.x, options.window.size.y},
            .camera_pos  = camera.position,
            .light_pos   = glm::vec3(0,0,0),
            .light_color = glm::vec4(0,0,0,0),
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

void MimirEngine::showMetrics()
{
    int w, h;
    window_context.getFramebufferSize(w, h);
    std::string label;
    if (w == 0 && h == 0) label = "None";
    else if (w == 1920 && h == 1080) label = "FHD";
    else if (w == 2560 && h == 1440) label = "QHD";
    else if (w == 3840 && h == 2160) label = "UHD";

    auto framerate = graphics_monitor.getFramerate();

    auto stats = physical_device.getMemoryStats();
    auto gpu_usage  = formatMemory(stats.usage);
    auto gpu_budget = formatMemory(stats.budget);

    printf("%s,%d,%f,%f,%lf,%f,%f,%f,", label.c_str(), options.present.target_fps,
        framerate,compute_monitor.total_compute_time,graphics_monitor.total_pipeline_time,
        graphics_monitor.total_graphics_time,gpu_usage.data,gpu_budget.data
    );

    //auto fps = ImGui::GetIO().Framerate; printf("\nFPS %f\n", fps);
    //getTimeResults();
    /*printf("Framebuffer size: %dx%d\n", w, h);
    printf("Average frame rate over 120 frames: %.2f FPS\n", framerate);

    dev.updateMemoryProperties();
    auto gpu_usage = dev.formatMemory(dev.props.gpu_usage);
    printf("GPU memory usage: %.2f %s\n", gpu_usage.data, gpu_usage.units.c_str());
    auto gpu_budget = dev.formatMemory(dev.props.gpu_budget);
    printf("GPU memory budget: %.2f %s\n", gpu_budget.data, gpu_budget.units.c_str());
    //this->exit();

    auto props = dev.budget_properties;
    for (int i = 0; i < static_cast<int>(dev.props.heap_count); ++i)
    {
        auto heap_usage = dev.formatMemory(props.heapUsage[i]);
        printf("Heap %d usage: %.2f %s\n", i, heap_usage.data, heap_usage.units.c_str());
        auto heap_budget = dev.formatMemory(props.heapBudget[i]);
        printf("Heap %d budget: %.2f %s\n", i, heap_budget.data, heap_budget.units.c_str());
        auto heap_flags = dev.memory_properties2.memoryProperties.memoryHeaps[i].flags;
        printf("Heap %d flags: %s\n", i, dev.readMemoryHeapFlags(heap_flags).c_str());
    }*/
}

void MimirEngine::immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function)
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

void MimirEngine::loadTexture(TextureDescription desc, void *data, size_t memsize)
{
    ImageParams params{
        .type   = getImageType(desc.extent),
        .format = getVulkanFormat(desc.format),
        .extent = getVulkanExtent(desc.extent),
        .tiling = getImageTiling(desc.source->type),
        .usage  = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .levels = desc.levels,
    };
    VkExternalMemoryImageCreateInfo extmem_info{
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    auto image = createImage(device, physical_device.handle, params, &extmem_info);
    validation::checkVulkan(vkBindImageMemory(device, image, desc.source->vk_mem, 0));

    // Create staging buffer to copy image data
    VkDeviceSize staging_size = desc.source->size;
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

void MimirEngine::copyBufferToTexture(VkBuffer buffer, VkImage image, VkExtent3D extent)
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

void MimirEngine::generateMipmaps(VkImage image, VkFormat format,
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

void MimirEngine::transitionImageLayout(VkImage image,
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