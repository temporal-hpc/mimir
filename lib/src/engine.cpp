#include <mimir/mimir.hpp>
#include <mimir/engine/vk_framebuffer.hpp>

#include "internal/camera.hpp"
#include "internal/framelimit.hpp"
#include "internal/gui.hpp"
#include "internal/interop.hpp"
#include "internal/validation.hpp"
#include "internal/vk_pipeline.hpp"
#include "internal/vk_properties.hpp"
#include "internal/window.hpp"

#include <dlfcn.h> // dladdr
#include <chrono> // std::chrono
#include <filesystem> // std::filesystem

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

// Setup the shader path so that the library can actually load them
// Hackish and Linux-only, but works for now
std::string getDefaultShaderPath()
{
    // If shaders are installed in library path, set working directory there
    Dl_info dl_info;
    dladdr((void*)getDefaultShaderPath, &dl_info);
    auto lib_pathname = dl_info.dli_fname;
    if (lib_pathname != nullptr)
    {
        std::filesystem::path lib_path(lib_pathname);
        return lib_path.parent_path().string();
    }
    else // Use executable path as working dir
    {
        auto exe_folder = std::filesystem::read_symlink("/proc/self/exe").remove_filename();
        return exe_folder;
    }
}

MimirEngine::MimirEngine():
    camera{ std::make_unique<Camera>() },
    shader_path{ getDefaultShaderPath() }
{}

MimirEngine::~MimirEngine()
{
    if (rendering_thread.joinable())
    {
        rendering_thread.join();
    }
    vkDeviceWaitIdle(dev.logical_device);

    if (interop->cuda_stream != nullptr)
    {
        validation::checkCuda(cudaStreamSynchronize(interop->cuda_stream));
    }

    cleanupGraphics();
    gui::shutdown();
    deletors.views.flush();
    deletors.context.flush();
}

void MimirEngine::init(ViewerOptions opts)
{
    spdlog::set_level(spdlog::level::trace);
    spdlog::set_pattern("[%H:%M:%S] [%l] %v");

    options = opts;
    options.present.max_fps = options.present.mode == PresentMode::VSync? 60 : 300;
    target_frame_time = getTargetFrameTime(options.present.enable_fps_limit, options.present.target_fps);

    auto width  = options.window.size.x;
    auto height = options.window.size.y;
    window_context = std::make_unique<GlfwContext>();
    window_context->init(width, height, options.window.title.c_str(), this);
    deletors.context.add([=,this] {
        window_context->clean();
    });

    initVulkan();

    camera->type = Camera::CameraType::LookAt;
    //camera->flipY = true;
    camera->setPosition(glm::vec3(0.f, 0.f, -2.85f)); //(glm::vec3(0.f, 0.f, -3.75f));
    camera->setRotation(glm::vec3(0.f, 0.f, 0.f)); //(glm::vec3(15.f, 0.f, 0.f));
    camera->setRotationSpeed(0.5f);
    camera->setPerspective(60.f, (float)width / (float)height, 0.1f, 256.f);
}

void MimirEngine::init(int width, int height)
{
    ViewerOptions opts;
    opts.window.size = {width, height};
    init(opts);
}

void MimirEngine::exit()
{
    window_context->exit();
}

void MimirEngine::prepare()
{
    initUniformBuffers();
    createViewPipelines();
    updateDescriptorSets();
    updateLinearTextures();
}

void MimirEngine::displayAsync()
{
    prepare();
    running = true;
    rendering_thread = std::thread([this]()
    {
        while(!window_context->shouldClose())
        {
            window_context->processEvents();
            gui::draw(camera.get(), options, views, gui_callback);
            renderFrame();
        }
        running = false;
        vkDeviceWaitIdle(dev.logical_device);
    });
}

void MimirEngine::prepareViews()
{
    if (options.present.enable_sync && running)
    {
        kernel_working = true;
        waitKernelStart();
        perf.startCuda();
    }
}

void MimirEngine::waitKernelStart()
{
    cudaExternalSemaphoreWaitParams wait_params{};
    wait_params.flags = 0;
    wait_params.params.fence.value = render_timeline+1;
    //printf("kernel waits for render %llu\n", wait_params.params.fence.value);

    // Wait for Vulkan to complete its work
    validation::checkCuda(cudaWaitExternalSemaphoresAsync(
        &interop->cuda_semaphore, &wait_params, 1, interop->cuda_stream)
    );
    updateLinearTextures();
}

void MimirEngine::updateViews()
{
    if (options.present.enable_sync && running)
    {
        perf.endCuda();
        signalKernelFinish();
        kernel_working = false;
    }
}

void MimirEngine::signalKernelFinish()
{
    interop->timeline_value++;
    cudaExternalSemaphoreSignalParams signal_params{};
    signal_params.flags = 0;
    signal_params.params.fence.value = interop->timeline_value;
    //printf("kernel signals iteration %llu\n", signal_params.params.fence.value);

    // Signal Vulkan to continue with the updated buffers
    validation::checkCuda(cudaSignalExternalSemaphoresAsync(
        &interop->cuda_semaphore, &signal_params, 1, interop->cuda_stream)
    );
}

void MimirEngine::display(std::function<void(void)> func, size_t iter_count)
{
    prepare();
    running = true;
    kernel_working = true;
    size_t iter_idx = 0;
    while(!window_context->shouldClose())
    {
        window_context->processEvents();
        gui::draw(camera.get(), options, views, gui_callback);
        renderFrame();

        if (running) waitKernelStart();
        if (iter_idx < iter_count)
        {
            func(); // Advance the simulation
            iter_idx++;
        }
        if (running) signalKernelFinish();
    }
    kernel_working = false;
    running = false;
    vkDeviceWaitIdle(dev.logical_device);
}

void MimirEngine::updateLinearTextures()
{
    /*for (auto& view : views)
    {
        // TODO: Reimplement this ugly loop
        if (view->params.view_type == ViewType::Image)
        {
            for (auto &[attr, memory] : view->params.attributes)
            {
                if (memory.params.resource_type == ResourceType::LinearTexture)
                {
                    dev.updateLinearTexture(memory);
                }
            }
        }
    }*/
}

void initImplicitCoords(VkDevice dev, VkDeviceMemory mem, VkDeviceSize memsize, uint3 size)
{
    float3 *data = nullptr;
    vkMapMemory(dev, mem, 0, memsize, 0, (void**)&data);
    auto slice_size = size.x * size.y;
    for (uint32_t z = 0; z < size.z; ++z)
    {
        auto rz = static_cast<float>(z) / size.z;
        rz = 2 * rz - 1;
        for (uint32_t y = 0; y < size.y; ++y)
        {
            auto ry = static_cast<float>(y) / size.y;
            ry = 2 * ry - 1;
            for (uint32_t x = 0; x < size.x; ++x)
            {
                auto rx = static_cast<float>(x) / size.x;
                rx = 2 * rx - 1;
                data[slice_size * z + size.x * y + x] = float3{rx, ry, rz};
            }
        }
    }
    vkUnmapMemory(dev, mem);
}

AttributeParams MimirEngine::makeStructuredDomain(StructuredDomainParams params)
{
    auto sz = params.size;
    assert(sz.x > 0 || sz.y > 0 || sz.z > 0);
    auto memsize = sizeof(float3) * sz.x * sz.y * sz.z;

    // Create test buffer for querying the desired memory properties
    auto domain_buffer = dev.createBuffer(memsize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    VkMemoryRequirements memreq{};
    vkGetBufferMemoryRequirements(dev.logical_device, domain_buffer, &memreq);

    auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    auto vk_memory = dev.allocateMemory(memreq, flags);
    vkBindBufferMemory(dev.logical_device, domain_buffer, vk_memory, 0);
    initImplicitCoords(dev.logical_device, vk_memory, memreq.size, params.size);

    // Add deletors to queue for later cleanup
    deletors.views.add([=,this]{
        spdlog::trace("Free structured domain memory");
        vkFreeMemory(dev.logical_device, vk_memory, nullptr);
        vkDestroyBuffer(dev.logical_device, domain_buffer, nullptr);
    });

    return AttributeParams{
        .allocation = std::make_shared<Allocation>(memreq.size, vk_memory, nullptr),
        .format     = { .type = DataType::float32, .components = 3 },
        .offset     = 0,
    };
}

std::shared_ptr<Allocation> MimirEngine::allocLinear(void **dev_ptr, size_t size)
{
    assert(size > 0);

    auto usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    auto alloc = allocExtmemBuffer(size, usage);
    cudaExternalMemoryBufferDesc buffer_desc{ .offset = 0, .size = size, .flags = 0 };
    validation::checkCuda(cudaExternalMemoryGetMappedBuffer(
        dev_ptr, alloc.cuda_extmem, &buffer_desc)
    );

    // Assemble the external memory handle
    return std::make_shared<Allocation>(alloc);
}

std::shared_ptr<Allocation> MimirEngine::allocMipmap(cudaMipmappedArray_t *dev_arr,
    const cudaChannelFormatDesc *desc, cudaExtent extent, unsigned int num_levels)
{
    assert(extent.width > 0 && extent.height > 0 && extent.depth > 0 && num_levels > 0);
    // TODO: Validate parameters against driver limits

    VkImageType type = extent.depth > 1? VK_IMAGE_TYPE_3D : extent.height > 1? VK_IMAGE_TYPE_2D : VK_IMAGE_TYPE_1D;
    VkFormat format  = VK_FORMAT_UNDEFINED; // TODO: Determine

    VkExternalMemoryImageCreateInfo extmem_info{
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkImageCreateInfo info{
        .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext       = &extmem_info,
        .flags       = 0,
        .imageType   = type,
        .format      = format,
        .extent      = VkExtent3D{ (uint32_t)extent.width, (uint32_t)extent.height, (uint32_t)extent.depth },
        .mipLevels   = num_levels,
        .arrayLayers = 1,
        .samples     = VK_SAMPLE_COUNT_1_BIT,
        .tiling      = VK_IMAGE_TILING_OPTIMAL,
        .usage       = VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage test_image = VK_NULL_HANDLE;
    validation::checkVulkan(vkCreateImage(dev.logical_device, &info, nullptr, &test_image));
    VkMemoryRequirements memreq{};
    vkGetImageMemoryRequirements(dev.logical_device, test_image, &memreq);

    auto memflags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkExportMemoryAllocateInfoKHR export_info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
        .pNext = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    auto vk_memory = dev.allocateMemory(memreq, memflags, &export_info);
    auto cuda_extmem = interop::importCudaExternalMemory(
        vk_memory, memreq.size, dev.logical_device
    );

    vkDestroyImage(dev.logical_device, test_image, nullptr);
    Allocation alloc{memreq.size, vk_memory, cuda_extmem};

    cudaExternalMemoryMipmappedArrayDesc array_desc{
        .offset     = 0,
        .formatDesc = *desc,
        .extent     = extent,
        .flags      = 0,
        .numLevels  = num_levels,
    };
    validation::checkCuda(cudaExternalMemoryGetMappedMipmappedArray(
        dev_arr, alloc.cuda_extmem, &array_desc)
    );

    return std::make_shared<Allocation>(alloc);
}

Allocation MimirEngine::allocExtmemBuffer(size_t size, VkBufferUsageFlags usage)
{
    // Create test buffer for querying the desired memory properties
    VkExternalMemoryBufferCreateInfo extmem_info{
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    };
    auto test_buffer = dev.createBuffer(size, usage, &extmem_info);
    VkMemoryRequirements memreq{};
    vkGetBufferMemoryRequirements(dev.logical_device, test_buffer, &memreq);

    // Allocate external device memory
    VkExportMemoryAllocateInfoKHR export_info{
        .sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    };
    auto memflags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    auto vk_memory = dev.allocateMemory(memreq, memflags, &export_info);

    // Export and map the external memory to CUDA
    auto cuda_extmem = interop::importCudaExternalMemory(
        vk_memory, memreq.size, dev.logical_device
    );

    // Add deletors to queue for later cleanup
    deletors.views.add([=,this]{
        spdlog::trace("Free interop memory");
        validation::checkCuda(cudaDestroyExternalMemory(cuda_extmem));
        vkFreeMemory(dev.logical_device, vk_memory, nullptr);
    });

    vkDestroyBuffer(dev.logical_device, test_buffer, nullptr);
    return Allocation{memreq.size, vk_memory, cuda_extmem};
}

VkBuffer MimirEngine::createAttributeBuffer(const AttributeParams attr, size_t element_count, VkBufferUsageFlags usage)
{
    // Get and validate buffer size against allocation size
    VkDeviceSize memsize = getBytesize(attr.format) * element_count;
    assert(memsize + attr.offset <= attr.memory->size);

    // Create and bind buffer
    VkExternalMemoryBufferCreateInfo extmem_info{
        .sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .pNext       = nullptr,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
    };
    auto attr_buffer = dev.createBuffer(memsize, usage, &extmem_info);
    deletors.views.add([=,this]{ vkDestroyBuffer(dev.logical_device, attr_buffer, nullptr); });
    vkBindBufferMemory(dev.logical_device, attr_buffer, attr.allocation->vk_mem, 0);
    return attr_buffer;
}

std::shared_ptr<InteropView> MimirEngine::createView(ViewParams params)
{
    ViewResources res;
    res.vbo.handles.reserve(params.attributes.size());
    res.vbo.offsets.reserve(params.attributes.size());

    // Create attribute buffers
    for (const auto &[type, attr] : params.attributes)
    {
        auto usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        auto attr_buffer = createAttributeBuffer(attr, params.element_count, usage);
        // Register buffer info in attribute array
        res.vbo.handles.push_back(attr_buffer);
        res.vbo.offsets.push_back(attr.offset);
        res.vbo.count++;
    }

    // Create index buffer if its attribute was set
    if (params.indexing.allocation != nullptr)
    {
        auto usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        auto index_buffer = createAttributeBuffer(params.indexing, params.element_count, usage);
        // Register index buffer info
        res.ibo.handle = index_buffer;
        res.ibo.type   = getIndexType(params.indexing.format.type);
    }

    // TODO: Add uniform buffer support

    auto mem_handle = std::make_shared<InteropView>(params, res, VK_NULL_HANDLE);
    views.push_back(mem_handle);
    return mem_handle;
}

InteropMemory *MimirEngine::createBuffer(void **dev_ptr, MemoryParams params)
{
    auto mem_handle = new InteropMemory();
    mem_handle->params = params;
    deletors.views.add([=,this]{ delete mem_handle; });

    switch (params.resource_type)
    {
        case ResourceType::Texture:
        {
            dev.initMemoryImage(*mem_handle);
            deletors.views.add([=,this]{
                vkDestroyImage(dev.logical_device, mem_handle->image, nullptr);
                validation::checkCuda(cudaDestroyExternalMemory(mem_handle->cuda_extmem));
                vkFreeMemory(dev.logical_device, mem_handle->image_memory, nullptr);
                vkDestroyImageView(dev.logical_device, mem_handle->vk_view, nullptr);
                validation::checkCuda(cudaFreeMipmappedArray(mem_handle->mipmap_array));
                vkDestroySampler(dev.logical_device, mem_handle->vk_sampler, nullptr);
            });
            break;
        }
        case ResourceType::LinearTexture:
        {
            dev.initMemoryImageLinear(*mem_handle);
            deletors.views.add([=,this]{
                vkDestroyImage(dev.logical_device, mem_handle->image, nullptr);
                vkFreeMemory(dev.logical_device, mem_handle->image_memory, nullptr);
                vkDestroyImageView(dev.logical_device, mem_handle->vk_view, nullptr);
                vkDestroySampler(dev.logical_device, mem_handle->vk_sampler, nullptr);
                vkDestroyBuffer(dev.logical_device, mem_handle->data_buffer, nullptr);
                validation::checkCuda(cudaDestroyExternalMemory(mem_handle->cuda_extmem));
                vkFreeMemory(dev.logical_device, mem_handle->memory, nullptr);
            });
            break;
        }
        default:
        {
            dev.initMemoryBuffer(*mem_handle);
            deletors.views.add([=,this]{
                vkDestroyBuffer(dev.logical_device, mem_handle->data_buffer, nullptr);
                validation::checkCuda(cudaDestroyExternalMemory(mem_handle->cuda_extmem));
                vkFreeMemory(dev.logical_device, mem_handle->memory, nullptr);
            });
        }
    }

    *dev_ptr = mem_handle->cuda_ptr;
    allocations.push_back(mem_handle);
    return allocations.back();
}

InteropViewOld *MimirEngine::createView(ViewParamsOld params)
{
    auto view_handle = new InteropViewOld();
    view_handle->params = params;
    deletors.views.add([=,this]{ delete view_handle; });

    if (params.view_type == ViewType::Image)
    {
        dev.initViewImage(*view_handle);
        deletors.views.add([=,this]{
            vkDestroyBuffer(dev.logical_device, view_handle->aux_buffer, nullptr);
            vkFreeMemory(dev.logical_device, view_handle->aux_memory, nullptr);
        });
    }
    else
    {
        dev.initViewBuffer(*view_handle);
        deletors.views.add([=,this]{
            vkDestroyBuffer(dev.logical_device, view_handle->aux_buffer, nullptr);
            vkFreeMemory(dev.logical_device, view_handle->aux_memory, nullptr);
        });
    }

    //views.push_back(view_handle);
    //return views.back();
    return nullptr;
}

void MimirEngine::loadTexture(InteropMemory *interop, void *data)
{
    dev.loadTexture(interop, data);
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

void MimirEngine::listExtensions()
{
    uint32_t ext_count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> available(ext_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, available.data());

    printf("Available extensions:\n");
    for (const auto& extension : available)
    {
        printf("  %s\n", extension.extensionName);
    }
}

void MimirEngine::initVulkan()
{
    createInstance();
    window_context->createSurface(instance, &surface);
    deletors.context.add([=,this](){
        vkDestroySurfaceKHR(instance, surface, nullptr);
    });
    pickPhysicalDevice();
    dev.initLogicalDevice(surface);
    deletors.context.add([=,this](){
        vkDestroyCommandPool(dev.logical_device, dev.command_pool, nullptr);
        vkDestroyDevice(dev.logical_device, nullptr);
    });

    // Create VMA handle
    /*
    auto memtypes = dev.physical_device.memory.memoryProperties.memoryTypes;
    auto memtype_count = dev.physical_device.memory.memoryProperties.memoryTypeCount;
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
        .physicalDevice                 = dev.physical_device.handle,
        .device                         = dev.logical_device,
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
    descriptor_pool = dev.createDescriptorPool({
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
    });

    // Create descriptor set and pipeline layouts
    std::vector<VkDescriptorSetLayoutBinding> layout_bindings{
        descriptorLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
        ),
        descriptorLayoutBinding(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
        ),
        descriptorLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_FRAGMENT_BIT
        ),
        descriptorLayoutBinding(3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            VK_SHADER_STAGE_FRAGMENT_BIT
        ),
        descriptorLayoutBinding(4, VK_DESCRIPTOR_TYPE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT
        )
    };
    descriptor_layout = dev.createDescriptorSetLayout(layout_bindings);
    pipeline_layout = dev.createPipelineLayout(descriptor_layout);

    deletors.context.add([=,this]{
        vkDestroyDescriptorPool(dev.logical_device, descriptor_pool, nullptr);
        vkDestroyDescriptorSetLayout(dev.logical_device, descriptor_layout, nullptr);
        vkDestroyPipelineLayout(dev.logical_device, pipeline_layout, nullptr);
    });

    initGraphics();
    createSyncObjects();
    // After command pool and render pass are created
    gui::init(dev, instance, descriptor_pool, render_pass, window_context.get());

    descriptor_sets = dev.createDescriptorSets(
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
    auto extensions = window_context->getRequiredExtensions();
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
    deletors.context.add([=,this]{
        vkDestroyInstance(instance, nullptr);
    });

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

void MimirEngine::pickPhysicalDevice()
{
    int cuda_dev_count = 0;
    validation::checkCuda(cudaGetDeviceCount(&cuda_dev_count));
    if (cuda_dev_count == 0)
    {
        spdlog::error("could not find devices supporting CUDA");
    }

    auto all_devices = getDevices(instance);
    printf("Enumerating CUDA devices:\n");
    for (int dev_id = 0; dev_id < cuda_dev_count; ++dev_id)
    {
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, dev_id);
        printf("* ID: %d\n  Name: %s\n  Capability: %d.%d\n",
            dev_id, dev_prop.name, dev_prop.major, dev_prop.minor
        );
    }
    printf("Enumerating Vulkan devices:\n");
    for (const auto& dev : all_devices)
    {
        auto props = dev.general.properties;
        printf("* ID: %u\n  Name: %s\n", props.deviceID, props.deviceName);
    }

    int curr_device = 0, prohibited_count = 0;
    while (curr_device < cuda_dev_count)
    {
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, curr_device);
        if (dev_prop.computeMode == cudaComputeModeProhibited)
        {
            prohibited_count++;
            curr_device++;
            continue;
        }
        for (const auto& device : all_devices)
        {
            auto matching = memcmp((void*)&dev_prop.uuid, device.id_props.deviceUUID, VK_UUID_SIZE) == 0;
            if (matching && props::isDeviceSuitable(device.handle, surface))
            {
                validation::checkCuda(cudaSetDevice(curr_device));
                dev.physical_device = device;
                spdlog::info("Selected interop device {}: {}",
                    curr_device, device.general.properties.deviceName
                );
                break;
            }
        }
        curr_device++;
    }

    if (prohibited_count == cuda_dev_count)
    {
        spdlog::error("No CUDA-Vulkan interop device was found");
    }
}

void MimirEngine::createSyncObjects()
{
    //images_inflight.resize(swap->image_count, VK_NULL_HANDLE);
    for (auto& sync : sync_data)
    {
        sync.frame_fence = dev.createFence(VK_FENCE_CREATE_SIGNALED_BIT);
        sync.image_acquired = dev.createSemaphore();
        sync.render_complete = dev.createSemaphore();
        deletors.context.add([=,this]{
            vkDestroyFence(dev.logical_device, sync.frame_fence, nullptr);
            vkDestroySemaphore(dev.logical_device, sync.image_acquired, nullptr);
            vkDestroySemaphore(dev.logical_device, sync.render_complete, nullptr);
        });
    }
    interop = std::make_unique<InteropBarrier>(dev.createInteropBarrier());
    deletors.context.add([=,this]{
        validation::checkCuda(cudaDestroyExternalSemaphore(interop->cuda_semaphore));
        vkDestroySemaphore(dev.logical_device, interop->vk_semaphore, nullptr);
    });
}

void MimirEngine::cleanupGraphics()
{
    vkDeviceWaitIdle(dev.logical_device);
    /*vkFreeCommandBuffers(dev.logical_device, dev.command_pool,
        command_buffers.size(), command_buffers.data()
    );*/
    deletors.graphics.flush();
    fbs.clear();
}

void MimirEngine::initGraphics()
{
    // Initialize swapchain
    int width, height;
    window_context->getFramebufferSize(width, height);
    auto present_mode = getDesiredPresentMode(options.present.mode);
    std::vector queue_indices{dev.graphics.family_index, dev.present.family_index};
    swapchain = Swapchain::make(dev.logical_device, dev.physical_device.handle,
        surface, width, height, present_mode, queue_indices
    );

    render_pass = createRenderPass();
    command_buffers = dev.createCommandBuffers(swapchain.image_count);
    query_pool = dev.createQueryPool(2 * command_buffers.size());
    deletors.graphics.add([=,this]{
        vkDestroyRenderPass(dev.logical_device, render_pass, nullptr);
        vkDestroySwapchainKHR(dev.logical_device, swapchain.current, nullptr);
        vkDestroyQueryPool(dev.logical_device, query_pool, nullptr);
    });

    auto depth_format = findDepthFormat();
    VkImageCreateInfo depth_img_info{
        .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext       = nullptr,
        .flags       = 0,
        .imageType   = VK_IMAGE_TYPE_2D,
        .format      = depth_format,
        .extent      = { swapchain.extent.width, swapchain.extent.height, 1 },
        .mipLevels   = 1,
        .arrayLayers = 1,
        .samples     = VK_SAMPLE_COUNT_1_BIT,
        .tiling      = VK_IMAGE_TILING_OPTIMAL,
        .usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    validation::checkVulkan(
        vkCreateImage(dev.logical_device, &depth_img_info, nullptr, &depth_image)
    );

    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(dev.logical_device, depth_image, &mem_req);
    VkMemoryAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = mem_req.size,
        .memoryTypeIndex = dev.physical_device.findMemoryType(
            mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        ),
    };
    validation::checkVulkan(
        vkAllocateMemory(dev.logical_device, &alloc_info, nullptr, &depth_memory)
    );
    vkBindImageMemory(dev.logical_device, depth_image, depth_memory, 0);

    VkImageViewCreateInfo depth_view_info{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext    = nullptr,
        .flags    = 0,
        .image    = depth_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format   = depth_format,
        // Default mapping of all color channels
        .components = VkComponentMapping{
            .r = VK_COMPONENT_SWIZZLE_R,
            .g = VK_COMPONENT_SWIZZLE_G,
            .b = VK_COMPONENT_SWIZZLE_B,
            .a = VK_COMPONENT_SWIZZLE_A,
        },
        // Describe image purpose and which part of it should be accesssed
        .subresourceRange = VkImageSubresourceRange{
            .aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        }
    };
    validation::checkVulkan(
        vkCreateImageView(dev.logical_device, &depth_view_info, nullptr, &depth_view)
    );

    deletors.graphics.add([=,this]{
        vkDestroyImageView(dev.logical_device, depth_view, nullptr);
        vkDestroyImage(dev.logical_device, depth_image, nullptr);
        vkFreeMemory(dev.logical_device, depth_memory, nullptr);
    });

    fbs.resize(swapchain.image_count);
    auto sc_images = swapchain.getImages(dev.logical_device);
    for (uint32_t i = 0; i < swapchain.image_count; ++i)
    {
        // Create a basic image view to be used as color target
        fbs[i].addAttachment(dev.logical_device, sc_images[i], swapchain.format);
        fbs[i].create(dev.logical_device, render_pass, swapchain.extent, depth_view);
        deletors.graphics.add([=,this]{
            vkDestroyImageView(dev.logical_device, fbs[i].attachments[0].view, nullptr);
            vkDestroyFramebuffer(dev.logical_device, fbs[i].framebuffer, nullptr);
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
/*
        for (const auto& view : views)
        {
            for (const auto &[attr, memory] : view->params.attributes)
            {
                // TODO: Use increasing binding indices for additional texture memory
                if (memory.params.resource_type == ResourceType::Texture ||
                    memory.params.resource_type == ResourceType::LinearTexture)
                {
                    VkDescriptorImageInfo img_info{
                        .sampler     = memory.vk_sampler,
                        .imageView   = memory.vk_view,
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
                        .sampler     = memory.vk_sampler,
                        .imageView   = memory.vk_view,
                        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    };
                    write_img.dstBinding     = 4;
                    write_img.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                    write_img.pImageInfo     = &samp_info;
                    updates.push_back(write_img);
                }
            }
        }*/
        vkUpdateDescriptorSets(dev.logical_device, updates.size(), updates.data(), 0, nullptr);
    }
}

void MimirEngine::waitTimelineHost()
{
    VkSemaphoreWaitInfo wait_info{
        .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext          = nullptr,
        .flags          = 0,
        .semaphoreCount = 1,
        .pSemaphores    = &interop->vk_semaphore,
        .pValues        = &interop->timeline_value,
    };
    vkWaitSemaphores(dev.logical_device, &wait_info, frame_timeout);
}

void MimirEngine::renderFrame()
{
    auto frame_idx = render_timeline % MAX_FRAMES_IN_FLIGHT;
    //printf("frame %lu waits for %lu and signals %lu\n", render_timeline, interop->timeline_value, render_timeline+1);

    // Wait for frame fence and reset it after waiting
    auto frame_sync = sync_data[frame_idx];
    auto fence = frame_sync.frame_fence;
    validation::checkVulkan(vkWaitForFences(dev.logical_device, 1, &fence, VK_TRUE, frame_timeout));
    validation::checkVulkan(vkResetFences(dev.logical_device, 1, &fence));

    static chrono_tp start_time = std::chrono::high_resolution_clock::now();
    chrono_tp current_time = std::chrono::high_resolution_clock::now();
    if (render_timeline == 0)
    {
        last_time = start_time;
    }
    float frame_time = std::chrono::duration<float, std::chrono::seconds::period>(current_time - last_time).count();

    //waitTimelineHost();

    // Acquire image from swap chain, signaling to the image_ready semaphore
    // when the image is ready for use
    uint32_t image_idx;
    auto result = vkAcquireNextImageKHR(dev.logical_device, swapchain.current,
        frame_timeout, frame_sync.image_acquired, VK_NULL_HANDLE, &image_idx
    );
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateGraphics();
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        spdlog::error("Failed to acquire swapchain image");
    }

    /*if (images_inflight[image_idx] != VK_NULL_HANDLE)
    {
        vkWaitForFences(dev.logical_device, 1, &images_inflight[image_idx], VK_TRUE, timeout);
    }
    images_inflight[image_idx] = frame.render_fence;
    if (render_timeline > MAX_FRAMES_IN_FLIGHT)
    {
        total_pipeline_time += getRenderTimeResults(frame_idx);
    }*/

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

    //vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, frame_idx * 2);

    // Set clear color and depth stencil value
    std::array<VkClearValue, 2> clear_values{};
    std::memcpy(clear_values[0].color.float32, &options.bg_color.x, sizeof(options.bg_color));
    clear_values[1].depthStencil = {1.f, 0};

    VkRenderPassBeginInfo render_pass_info{
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext           = nullptr,
        .renderPass      = render_pass,
        .framebuffer     = fbs[image_idx].framebuffer,
        .renderArea      = { {0, 0}, swapchain.extent },
        .clearValueCount = (uint32_t)clear_values.size(),
        .pClearValues    = clear_values.data(),
    };

    // Render pass
    vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

    drawElements(frame_idx);
    gui::render(cmd);

    // End of render pass and timestamp query
    vkCmdEndRenderPass(cmd);
    //vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, frame_idx * 2 + 1);

    // Finalize command buffer recording, so it can be executed
    validation::checkVulkan(vkEndCommandBuffer(cmd));

    updateUniformBuffers(frame_idx);
    render_timeline++;

    // Fill submit waits & signals info
    std::vector<VkSemaphore> waits           = {frame_sync.image_acquired};
    std::vector<VkPipelineStageFlags> stages = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    std::vector<VkSemaphore> signals         = {frame_sync.render_complete};
    std::vector<uint64_t> wait_values        = {0};
    std::vector<uint64_t> signal_values      = {0};
    VkTimelineSemaphoreSubmitInfo *extra     = nullptr;
    VkTimelineSemaphoreSubmitInfo timeline_info{};
    if (kernel_working && options.present.enable_sync)
    {
        waits.push_back(interop->vk_semaphore);
        stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        signals.push_back(interop->vk_semaphore);
        wait_values.push_back(interop->timeline_value);
        signal_values.push_back(render_timeline+1);

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
    validation::checkVulkan(vkQueueSubmit(dev.graphics.queue, 1, &submit_info, fence));

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
    result = vkQueuePresentKHR(dev.present.queue, &present_info);
    // Resize should be done after presentation to ensure semaphore consistency
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || should_resize)
    {
        recreateGraphics();
        should_resize = false;
    }

    // Limit frame if it was configured
    if (options.present.enable_fps_limit) frameStall(target_frame_time);

    total_frame_count++;
    total_graphics_time += frame_time;
    frame_times[render_timeline % frame_times.size()] = frame_time;
    last_time = current_time;

    /*if (options.report_period > 0 && frame_time > options.report_period)
    {
        printf("Report at %d seconds:\n", options.report_period);
        showMetrics();
        last_time = current_time;
    }*/
}

void MimirEngine::drawElements(uint32_t image_idx)
{
    auto min_alignment = dev.physical_device.general.properties.limits.minUniformBufferOffsetAlignment;
    auto size_mvp = getAlignedSize(sizeof(ModelViewProjection), min_alignment);
    auto size_view = getAlignedSize(sizeof(ViewUniforms), min_alignment);
    auto size_scene = getAlignedSize(sizeof(SceneUniforms), min_alignment);
    auto size_ubo = size_mvp + size_view + size_scene;

    auto cmd = command_buffers[image_idx];
    for (uint32_t i = 0; i < views.size(); ++i)
    {
        auto& view = views[i];
        // Do not draw anything if view visibility is turned off
        if (!view->params.options.visible) continue;

        // Bind descriptor set and pipeline
        std::vector<uint32_t> offsets = {
            i * size_ubo,
            i * size_ubo + size_mvp + size_view,
            i * size_ubo + size_mvp
        };
        // NOTE: Second parameter can be also used to bind a compute pipeline
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout, 0, 1, &descriptor_sets[image_idx], offsets.size(), offsets.data()
        );
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, view->pipeline);

        auto& vbo = view->resources.vbo;
        vkCmdBindVertexBuffers(cmd, 0, vbo.count, vbo.handles.data(), vbo.offsets.data());
        auto vertex_count = view->params.element_count;

        auto& ibo = view->resources.ibo;
        if (ibo.handle != nullptr) // If index buffer exists, bind it and perform indexed draw
        {
            vkCmdBindIndexBuffer(cmd, ibo.handle, 0, ibo.type);
            vkCmdDrawIndexed(cmd, vertex_count, 1, 0, 0, 0);
        }
        else // Perform regular draw with bound vertex buffers
        {
            auto instance_count = 1;
            auto first_vertex = vertex_count * view->params.options.instance_index;
            vkCmdDraw(cmd, vertex_count, instance_count, first_vertex, 0);
        }

        // case ViewType::Image:
        // {
        //     vkCmdBindIndexBuffer(cmd, view->aux_buffer, view->index_offset, view->idx_type);
        //     vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
        //     break;
        // }
    }
}

bool MimirEngine::hasStencil(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

VkFormat MimirEngine::findDepthFormat()
{
    return dev.findSupportedImageFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}

VkRenderPass MimirEngine::createRenderPass()
{
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
        .format         = findDepthFormat(),
        .samples        = VK_SAMPLE_COUNT_1_BIT,
        .loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    std::array<VkAttachmentDescription, 2> attachments{ color, depth };

    VkAttachmentReference color_ref{
        .attachment = 0,
        .layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    VkAttachmentReference depth_ref{
        .attachment = 1,
        .layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    VkSubpassDescription subpass{
        .flags                   = 0, // Specify subpass usage
        .pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount    = 0,
        .pInputAttachments       = nullptr,
        .colorAttachmentCount    = 1,
        .pColorAttachments       = &color_ref,
        .pResolveAttachments     = nullptr,
        .pDepthStencilAttachment = &depth_ref,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments    = nullptr,
    };

    // Specify memory and execution dependencies between subpasses
    VkPipelineStageFlags stage_mask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    VkAccessFlags access_mask =
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    VkSubpassDependency dependency{
        .srcSubpass      = VK_SUBPASS_EXTERNAL,
        .dstSubpass      = 0,
        .srcStageMask    = stage_mask,
        .dstStageMask    = stage_mask,
        .srcAccessMask   = 0, // TODO: Change to VK_ACCESS_NONE in 1.3
        .dstAccessMask   = access_mask,
        .dependencyFlags = 0,
    };

    VkRenderPassCreateInfo pass_info{
        .sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext           = nullptr,
        .flags           = 0, // Can be VK_RENDER_PASS_CREATE_TRANSFORM_BIT_QCOM
        .attachmentCount = (uint32_t)attachments.size(),
        .pAttachments    = attachments.data(),
        .subpassCount    = 1,
        .pSubpasses      = &subpass,
        .dependencyCount = 1,
        .pDependencies   = &dependency,
    };

    spdlog::debug("Render pass created with {} attachments", attachments.size());
    VkRenderPass render_pass = VK_NULL_HANDLE;
    validation::checkVulkan(
        vkCreateRenderPass(dev.logical_device, &pass_info, nullptr, &render_pass)
    );
    return render_pass;
}

void MimirEngine::createViewPipelines()
{
    auto start = std::chrono::steady_clock::now();
    auto orig_path = std::filesystem::current_path();
    std::filesystem::current_path(shader_path);

    PipelineBuilder builder(pipeline_layout, swapchain.extent);

    for (auto& view : views)
    {
        builder.addPipeline(view->params, dev.logical_device);
    }
    auto pipelines = builder.createPipelines(dev.logical_device, render_pass);
    for (size_t i = 0; i < pipelines.size(); ++i)
    {
        views[i]->pipeline = pipelines[i];
        deletors.graphics.add([=,this]{
            vkDestroyPipeline(dev.logical_device, views[i]->pipeline, nullptr);
        });
    }

    // Restore original working directory
    std::filesystem::current_path(orig_path);
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    spdlog::trace("Created {} pipeline object(s) in {} ms", pipelines.size(), elapsed);
}

void MimirEngine::rebuildPipeline(InteropViewOld& view)
{
    auto orig_path = std::filesystem::current_path();
    std::filesystem::current_path(shader_path);

    PipelineBuilder builder(pipeline_layout, swapchain.extent);
    //builder.addPipeline(view.params, dev.logical_device);
    auto pipelines = builder.createPipelines(dev.logical_device, render_pass);
    // Destroy the old view pipeline and assign the new one
    vkDestroyPipeline(dev.logical_device, view.pipeline, nullptr);
    view.pipeline = pipelines[0];

    std::filesystem::current_path(orig_path);
}

void MimirEngine::initUniformBuffers()
{
    auto min_alignment = dev.physical_device.general.properties.limits.minUniformBufferOffsetAlignment;
    auto size_mvp = getAlignedSize(sizeof(ModelViewProjection), min_alignment);
    auto size_view = getAlignedSize(sizeof(ViewUniforms), min_alignment);
    auto size_scene = getAlignedSize(sizeof(SceneUniforms), min_alignment);
    auto size_ubo = (size_mvp + size_view + size_scene) * views.size();

    uniform_buffers.resize(swapchain.image_count);
    for (auto& ubo : uniform_buffers)
    {
        ubo.buffer = dev.createBuffer(size_ubo, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        VkMemoryRequirements memreq;
        vkGetBufferMemoryRequirements(dev.logical_device, ubo.buffer, &memreq);
        auto mem_usage = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        ubo.memory = dev.allocateMemory(memreq, mem_usage);
        vkBindBufferMemory(dev.logical_device, ubo.buffer, ubo.memory, 0);
        deletors.context.add([=,this]{
            vkDestroyBuffer(dev.logical_device, ubo.buffer, nullptr);
            vkFreeMemory(dev.logical_device, ubo.memory, nullptr);
        });
    }
}

// Update uniform buffers for view at index [view_idx] for frame [image_idx]
void MimirEngine::updateUniformBuffers(uint32_t image_idx)
{
    auto min_alignment = dev.physical_device.getUboOffsetAlignment();
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
            .view  = camera->matrices.view,
            .proj  = camera->matrices.perspective,
        };

        auto dc = view->params.options.default_color;
        ViewUniforms vu{
            .color         = glm::vec4(dc.x, dc.y, dc.z, dc.w),
            .size          = view->params.options.default_size,
            .depth         = view->params.options.depth,
            .element_count = view->params.options.custom_val,
        };

        auto bg = options.bg_color;
        auto extent = view->params.extent;
        SceneUniforms su{
            .bg_color    = glm::vec4(bg.x, bg.y, bg.z, bg.w),
            .extent      = glm::ivec3{extent.x, extent.y, extent.z},
            .resolution  = glm::ivec2{options.window.size.x, options.window.size.y},
            .camera_pos  = camera->position,
            .light_pos   = glm::vec3(0,0,0),
            .light_color = glm::vec4(0,0,0,0),
        };

        char *data = nullptr;
        auto offset = size_ubo * view_idx;
        vkMapMemory(dev.logical_device, memory, offset, size_ubo, 0, (void**)&data);
        std::memcpy(data, &mvp, sizeof(mvp));
        std::memcpy(data + size_mvp, &vu, sizeof(vu));
        std::memcpy(data + size_mvp + size_view, &su, sizeof(su));
        vkUnmapMemory(dev.logical_device, memory);
    }
}

double MimirEngine::getRenderTimeResults(uint32_t cmd_idx)
{
    auto timestamp_period = dev.physical_device.general.properties.limits.timestampPeriod;
    const double seconds_per_tick = static_cast<double>(timestamp_period) / 1e9;

    uint64_t buffer[2];
    validation::checkVulkan(vkGetQueryPoolResults(dev.logical_device, query_pool,
        2 * cmd_idx, 2, 2 * sizeof(uint64_t), buffer, sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)
    );
    vkResetQueryPool(dev.logical_device, query_pool, cmd_idx * 2, 2);
    // TODO: apply time &= timestamp_mask;
    return static_cast<double>(buffer[1] - buffer[0]) * seconds_per_tick;
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

void MimirEngine::showMetrics()
{
    int w, h;
    window_context->getFramebufferSize(w, h);
    std::string label;
    if (w == 0 && h == 0) label = "None";
    else if (w == 1920 && h == 1080) label = "FHD";
    else if (w == 2560 && h == 1440) label = "QHD";
    else if (w == 3840 && h == 2160) label = "UHD";

    auto frame_sample_size = std::min(frame_times.size(), total_frame_count);
    float total_frame_time = 0;
    for (size_t i = 0; i < frame_sample_size; ++i) total_frame_time += frame_times[i];
    auto framerate = frame_times.size() / total_frame_time;

    auto stats = dev.physical_device.getMemoryStats();
    auto gpu_usage  = formatMemory(stats.usage);
    auto gpu_budget = formatMemory(stats.budget);

    printf("%s,%d,%f,%f,%lf,%f,%f,%f,", label.c_str(), options.present.target_fps,
        framerate,perf.total_compute_time,total_pipeline_time,
        total_graphics_time,gpu_usage.data,gpu_budget.data
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

} // namespace mimir