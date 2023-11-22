#include <mimir/mimir.hpp>
#include <mimir/validation.hpp>
#include <mimir/engine/vk_framebuffer.hpp>
#include <mimir/engine/vk_swapchain.hpp>

#include "internal/camera.hpp"
#include "internal/framelimit.hpp"
#include "internal/vk_initializers.hpp"
#include "internal/vk_pipeline.hpp"
#include "internal/vk_properties.hpp"

#include <dlfcn.h> // dladdr
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <chrono> // std::chrono
#include <filesystem> // std::filesystem

namespace mimir
{

void setColor(float *vk_color, float4 color)
{
    vk_color[0] = color.x;
    vk_color[1] = color.y;
    vk_color[2] = color.z;
    vk_color[3] = color.w;
}

glm::vec4 getColor(float4 color)
{
    glm::vec4 colorvec;
    colorvec.x = color.x;
    colorvec.y = color.y;
    colorvec.z = color.z;
    colorvec.w = color.w;
    return colorvec;
}

// Setup the shader path so that the library can actually load them
// Hack-ish, but works for now
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

CudaviewEngine::CudaviewEngine():
    shader_path{ getDefaultShaderPath() },
    camera{ std::make_unique<Camera>() }
{}

CudaviewEngine::~CudaviewEngine()
{
    if (rendering_thread.joinable())
    {
        rendering_thread.join();
    }
    if (dev) vkDeviceWaitIdle(dev->logical_device);

    if (interop.cuda_stream != nullptr)
    {
        validation::checkCuda(cudaStreamSynchronize(interop.cuda_stream));
    }
    for (auto& ubo : uniform_buffers)
    {
        vkDestroyBuffer(dev->logical_device, ubo.buffer, nullptr);
        vkFreeMemory(dev->logical_device, ubo.memory, nullptr);
    }

    if (dev)
    {
        cleanupSwapchain();
        ImGui_ImplVulkan_Shutdown();
    }

    swap.reset();
    dev.reset();
}

void CudaviewEngine::init(ViewerOptions opts)
{
    options = opts;
    max_fps = options.present == PresentOptions::VSync? 60 : 300;
    target_frame_time = getTargetFrameTime(options.enable_fps_limit, options.target_fps);

    auto width  = options.window_size.x;
    auto height = options.window_size.y;
    
    // Initialize GLFW context and window
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);
    //glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window = glfwCreateWindow(width, height, options.window_title.c_str(), nullptr, nullptr);
    //glfwSetWindowSize(window, width, height);
    deletors.add([=,this] {
        //printf("Terminating GLFW\n");
        glfwDestroyWindow(window);
        glfwTerminate();
    });

    // Set GLFW action callbacks
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetWindowCloseCallback(window, windowCloseCallback);

    initVulkan();

    camera->type = Camera::CameraType::LookAt;
    //camera->flipY = true;
    camera->setPosition(glm::vec3(0.f, 0.f, -2.85f)); //(glm::vec3(0.f, 0.f, -3.75f));
    camera->setRotation(glm::vec3(1.5f, -2.5f, 0.f)); //(glm::vec3(15.f, 0.f, 0.f));
    camera->setRotationSpeed(0.5f);
    camera->setPerspective(60.f, (float)width / (float)height, 0.1f, 256.f);
}

void CudaviewEngine::init(int width, int height)
{
    ViewerOptions opts;
    opts.window_size = {width, height};
    init(opts);
}

void CudaviewEngine::exit()
{
    //printf("Exiting...\n");
    glfwSetWindowShouldClose(window, GL_TRUE);
    glfwPollEvents();
}

void CudaviewEngine::prepare()
{
    initUniformBuffers();
    createGraphicsPipelines();
    updateDescriptorSets();
}

void CudaviewEngine::displayAsync()
{
    prepare();
    running = true;
    rendering_thread = std::thread([this]()
    {
        while(!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            drawGui();
            renderFrame();
        }
        running = false;
        vkDeviceWaitIdle(dev->logical_device);
    });
}

void CudaviewEngine::prepareViews()
{
    if (options.enable_sync && running)
    {
        kernel_working = true;
        waitKernelStart();
        perf.startCuda();
    }
}

void CudaviewEngine::waitKernelStart()
{
    static uint64_t wait_value = 1;
    cudaExternalSemaphoreWaitParams wait_params{};
    wait_params.flags = 0;
    wait_params.params.fence.value = wait_value;

    // Wait for Vulkan to complete its work
    //printf("Waiting for value %lu to vulkan to finish\n", wait_value);
    validation::checkCuda(cudaWaitExternalSemaphoresAsync(
        &interop.cuda_semaphore, &wait_params, 1, interop.cuda_stream)
    );
    wait_value += 2;

    /*for (auto& view : views)
    {
        if (view->params.resource_type == ResourceType::Buffer &&
            view->params.element_type == ElementType::Image)
        {
            dev->updateTexture(view.get());
        }
    }*/
}

void CudaviewEngine::updateViews()
{
    if (options.enable_sync && running)
    {
        //printf("Kernel has ended\n");
        perf.endCuda();
        signalKernelFinish();
        kernel_working = false;
    }
}

void CudaviewEngine::signalKernelFinish()
{
    static uint64_t signal_value = 2;
    cudaExternalSemaphoreSignalParams signal_params{};
    signal_params.flags = 0;
    signal_params.params.fence.value = signal_value;

    // Signal Vulkan to continue with the updated buffers
    //printf("Signaling with value %lu that CUDA has ended\n", signal_value);
    validation::checkCuda(cudaSignalExternalSemaphoresAsync(
        &interop.cuda_semaphore, &signal_params, 1, interop.cuda_stream)
    );
    signal_value += 2;
}

void CudaviewEngine::display(std::function<void(void)> func, size_t iter_count)
{
    prepare();    
    running = true;
    kernel_working = true;
    size_t iter_idx = 0;
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        drawGui();
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
    vkDeviceWaitIdle(dev->logical_device);
}

InteropMemory *CudaviewEngine::createBuffer(void **dev_ptr, MemoryParams params)
{
    auto mem_handle = std::unique_ptr<InteropMemory>(new InteropMemory());
    mem_handle->params = params;

    dev->initMemoryBuffer(*mem_handle);
    *dev_ptr = mem_handle->cuda_ptr;
    allocations.push_back(std::move(mem_handle));
    return allocations.back().get();
}

InteropView2 *CudaviewEngine::createView(ViewParams2 params)
{
    auto view_handle = std::unique_ptr<InteropView2>(new InteropView2());
    view_handle->params = params;

    dev->initViewBuffer(*view_handle);
    views2.push_back(std::move(view_handle));
    return views2.back().get();
}

InteropView *CudaviewEngine::createView(void **ptr_devmem, ViewParams params)
{
    /*auto view_handle = std::unique_ptr<InteropView>(new InteropView());
    view_handle->params = params;

    dev->initView(*view_handle);
    *ptr_devmem = view_handle->cuda_ptr;
    views.push_back(std::move(view_handle));
    return views.back().get();*/
    return nullptr;
}

void CudaviewEngine::loadTexture(InteropView *view, void *data)
{
    dev->loadTexture(view, data);
}

void CudaviewEngine::drawGui()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    if (show_demo_window) { ImGui::ShowDemoWindow(); }
    if (options.show_metrics) { ImGui::ShowMetricsWindow(); }
    displayEngineGUI(); // Display the builtin GUI  
    gui_callback(); // Display user-provided addons
    ImGui::Render();
}

void CudaviewEngine::initVulkan()
{
    createInstance();
    swap = std::make_unique<VulkanSwapchain>();
    swap->initSurface(instance, window);
    pickPhysicalDevice();
    dev->initLogicalDevice(swap->surface);

    // Create descriptor pool
    descriptor_pool = dev->createDescriptorPool({
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
        vkinit::descriptorLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT
        ),
        vkinit::descriptorLayoutBinding(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT
        ),
        vkinit::descriptorLayoutBinding(2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT
        ),
        vkinit::descriptorLayoutBinding(3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 
            VK_SHADER_STAGE_FRAGMENT_BIT
        ),
        vkinit::descriptorLayoutBinding(4, VK_DESCRIPTOR_TYPE_SAMPLER,
            VK_SHADER_STAGE_FRAGMENT_BIT
        )
    };
    descriptor_layout = dev->createDescriptorSetLayout(layout_bindings);
    pipeline_layout = dev->createPipelineLayout(descriptor_layout);

    initSwapchain();
    initImgui(); // After command pool and render pass are created
    createSyncObjects();

    descriptor_sets = dev->createDescriptorSets(
        descriptor_pool, descriptor_layout, swap->image_count
    );
}

void CudaviewEngine::createInstance()
{
    if (validation::enable_layers && !validation::checkValidationLayerSupport())
    {
        throw std::runtime_error("validation layers requested, but not supported");
    }

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName   = "Mimir";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName        = "Mimir";
    app_info.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion         = VK_API_VERSION_1_2;

    uint32_t glfw_ext_count = 0;
    // List required GLFW extensions and additional required validation layers
    const char **glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
    std::vector<const char*> extensions(glfw_exts, glfw_exts + glfw_ext_count);
    if (validation::enable_layers)
    {
        // Enable debugging message extension
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
    extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    VkInstanceCreateInfo instance_info{};
    instance_info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pNext                   = nullptr;
    instance_info.flags                   = 0;
    instance_info.pApplicationInfo        = &app_info;
    instance_info.enabledLayerCount       = 0;
    instance_info.ppEnabledLayerNames     = nullptr;
    instance_info.enabledExtensionCount   = extensions.size();
    instance_info.ppEnabledExtensionNames = extensions.data();

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
    deletors.add([=,this]{ vkDestroyInstance(instance, nullptr); });

    if (validation::enable_layers)
    {
        // Details about the debug messenger and its callback
        validation::checkVulkan(validation::CreateDebugUtilsMessengerEXT(
            instance, &debug_create_info, nullptr, &debug_messenger)
        );
        deletors.add([=,this]{
            validation::DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
        });
    }
}

void CudaviewEngine::pickPhysicalDevice()
{
    int cuda_dev_count = 0;
    validation::checkCuda(cudaGetDeviceCount(&cuda_dev_count));
    if (cuda_dev_count == 0)
    {
        throw std::runtime_error("could not find devices supporting CUDA");
    }
    /*printf("Enumerating CUDA devices:\n");
    for (int dev_id = 0; dev_id < cuda_dev_count; ++dev_id)
    {
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, dev_id);
        printf("* ID: %d\n  Name: %s\n  Capability: %d.%d\n",
            dev_id, dev_prop.name, dev_prop.major, dev_prop.minor
        );
    }*/

    uint32_t vulkan_dev_count = 0;
    vkEnumeratePhysicalDevices(instance, &vulkan_dev_count, nullptr);
    if (vulkan_dev_count == 0)
    {
        throw std::runtime_error("could not find devices supporting Vulkan");
    }
    std::vector<VkPhysicalDevice> devices(vulkan_dev_count);
    vkEnumeratePhysicalDevices(instance, &vulkan_dev_count, devices.data());
    /*printf("Enumerating Vulkan devices:\n");
    for (const auto& dev : devices)
    {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(dev, &props);
        printf("* ID: %u\n  Name: %s\n", props.deviceID, props.deviceName);
    }*/

    auto fpGetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2");
    if (fpGetPhysicalDeviceProperties2 == nullptr)
    {
        throw std::runtime_error("could not find proc address for \"vkGetPhysicalDeviceProperties2KHR\","
            "which is needed for finding an interop-capable device");
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
        for (const auto& ph_dev : devices)
        {
            VkPhysicalDeviceIDProperties id_props{};
            id_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
            id_props.pNext = nullptr;

            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &id_props;

            fpGetPhysicalDeviceProperties2(ph_dev, &props2);
            auto matching = memcmp((void*)&dev_prop.uuid, id_props.deviceUUID, VK_UUID_SIZE) == 0;
            if (matching && props::isDeviceSuitable(ph_dev, swap->surface))
            {
                validation::checkCuda(cudaSetDevice(curr_device));
                VkPhysicalDeviceProperties props{};
                vkGetPhysicalDeviceProperties(ph_dev, &props);
                dev = std::make_unique<InteropDevice>(ph_dev);
                //printf("Selected CUDA-Vulkan device %d: %s\n\n", curr_device, dev_prop.name);
                break;
            }
        }
        curr_device++;
    }

    if (prohibited_count == cuda_dev_count)
    {
        throw std::runtime_error("No CUDA-Vulkan interop device was found");
    }
}

void CudaviewEngine::initImgui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance       = instance;
    init_info.PhysicalDevice = dev->physical_device;
    init_info.Device         = dev->logical_device;
    init_info.Queue          = dev->graphics.queue;
    init_info.DescriptorPool = descriptor_pool;
    init_info.MinImageCount  = 3; // TODO: Check if this is true
    init_info.ImageCount     = 3;
    init_info.MSAASamples    = VK_SAMPLE_COUNT_1_BIT;
    ImGui_ImplVulkan_Init(&init_info, render_pass);

    dev->immediateSubmit([=](VkCommandBuffer cmd) {
        ImGui_ImplVulkan_CreateFontsTexture(cmd);
    });
    ImGui_ImplVulkan_DestroyFontUploadObjects();
}

void CudaviewEngine::createSyncObjects()
{
    //images_inflight.resize(swap->image_count, VK_NULL_HANDLE);
    for (auto& fence : frame_fences)
    {
        fence = dev->createFence(VK_FENCE_CREATE_SIGNALED_BIT);
    }
    interop = dev->createInteropBarrier();
    present_semaphore = dev->createSemaphore();
}

void CudaviewEngine::cleanupSwapchain()
{
    vkDeviceWaitIdle(dev->logical_device);
    for (auto& view : views2)
    {
        vkDestroyPipeline(dev->logical_device, view->pipeline, nullptr);
    }
    /*vkFreeCommandBuffers(dev->logical_device, dev->command_pool,
        command_buffers.size(), command_buffers.data()
    );*/
    vkDestroyRenderPass(dev->logical_device, render_pass, nullptr);
    swap->cleanup();
    fbs.clear();
}

void CudaviewEngine::initSwapchain()
{
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    uint32_t width = w;
    uint32_t height = h;
    std::vector queue_indices{dev->graphics.family_index, dev->present.family_index};
    swap->create(width, height, options.present, queue_indices, dev->physical_device, dev->logical_device);
    render_pass = createRenderPass();
    command_buffers = dev->createCommandBuffers(swap->image_count);
    query_pool = dev->createQueryPool(2 * command_buffers.size());

    auto images = swap->createImages(dev->logical_device);

    auto depth_fmt = findDepthFormat();
    VkExtent3D extent{ width, height, 1 };
    auto type = VK_IMAGE_TYPE_2D;
    auto usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    auto image_info = vkinit::imageCreateInfo(type, depth_fmt, extent, usage);
    validation::checkVulkan(
        vkCreateImage(dev->logical_device, &image_info, nullptr, &depth_image)
    );

    VkMemoryRequirements mem_req;
    vkGetImageMemoryRequirements(dev->logical_device, depth_image, &mem_req);
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.pNext = nullptr;
    alloc_info.allocationSize = mem_req.size;
    alloc_info.memoryTypeIndex = dev->findMemoryType(
        mem_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    validation::checkVulkan(
        vkAllocateMemory(dev->logical_device, &alloc_info, nullptr, &depth_memory)
    );
    vkBindImageMemory(dev->logical_device, depth_image, depth_memory, 0);

    auto view_info = vkinit::imageViewCreateInfo(
        depth_image, VK_IMAGE_VIEW_TYPE_2D, depth_fmt, VK_IMAGE_ASPECT_DEPTH_BIT
    );
    validation::checkVulkan(
        vkCreateImageView(dev->logical_device, &view_info, nullptr, &depth_view)
    );

    swap->aux_deletors.add([=,this]{
        vkDestroyImageView(dev->logical_device, depth_view, nullptr);
        vkDestroyImage(dev->logical_device, depth_image, nullptr);
        vkFreeMemory(dev->logical_device, depth_memory, nullptr);
    });

    fbs.resize(swap->image_count);
    for (size_t i = 0; i < swap->image_count; ++i)
    {
        // Create a basic image view to be used as color target
        fbs[i].addAttachment(dev->logical_device, images[i], swap->color_format);
        fbs[i].create(dev->logical_device, render_pass, swap->extent, depth_view);
    }
}

void CudaviewEngine::recreateSwapchain()
{
    //printf("Recreating swapchain\n");
    cleanupSwapchain();
    initSwapchain();
    createGraphicsPipelines();
}

void CudaviewEngine::updateDescriptorSets()
{
    for (size_t i = 0; i < descriptor_sets.size(); ++i)
    {
        // Write MVP matrix, scene info and texture samplers
        std::vector<VkWriteDescriptorSet> updates;
        auto& set = descriptor_sets[i];

        VkDescriptorBufferInfo mvp_info{};
        mvp_info.buffer = uniform_buffers[i].buffer;
        mvp_info.offset = 0;
        mvp_info.range  = sizeof(ModelViewProjection);
        auto write_mvp = vkinit::writeDescriptorBuffer(set, 0, 
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, &mvp_info
        );
        updates.push_back(write_mvp);

        VkDescriptorBufferInfo primitive_info{};
        primitive_info.buffer = uniform_buffers[i].buffer;
        primitive_info.offset = 0;
        primitive_info.range  = sizeof(PrimitiveParams);
        auto write_primitive = vkinit::writeDescriptorBuffer(set, 2, 
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, &primitive_info
        );
        updates.push_back(write_primitive);

        VkDescriptorBufferInfo scene_info{};
        scene_info.buffer = uniform_buffers[i].buffer;
        scene_info.offset = 0;
        scene_info.range  = sizeof(SceneParams);
        auto write_scene = vkinit::writeDescriptorBuffer(set, 1, 
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, &scene_info
        );
        updates.push_back(write_scene);

        for (const auto& view : views2)
        {
            if (view->params.view_type == ViewType::Image)
            {
                /*VkDescriptorImageInfo img_info{};
                img_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                img_info.imageView   = view->vk_view;
                img_info.sampler     = view->vk_sampler;

                auto write_img = vkinit::writeDescriptorImage(set,
                    3, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &img_info
                );
                updates.push_back(write_img);

                VkDescriptorImageInfo samp_info{};
                samp_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                samp_info.imageView   = view->vk_view;
                samp_info.sampler     = view->vk_sampler;

                auto write_samp = vkinit::writeDescriptorImage(set,
                    4, VK_DESCRIPTOR_TYPE_SAMPLER, &samp_info
                );
                updates.push_back(write_samp);*/
            }
        }

        vkUpdateDescriptorSets(dev->logical_device, updates.size(), updates.data(), 0, nullptr);
    }
}

void CudaviewEngine::renderFrame()
{
    static chrono_tp start_time = std::chrono::high_resolution_clock::now();
    chrono_tp current_time = std::chrono::high_resolution_clock::now();
    if (current_frame == 0)
    {
        last_time = start_time;
    }
    float frame_time = std::chrono::duration<float, std::chrono::seconds::period>(current_time - last_time).count();

    constexpr auto timeout = 1000000000; //std::numeric_limits<uint64_t>::max();
    static uint64_t wait_value = 0;
    static uint64_t signal_value = 1;
    auto frame_idx = current_frame % MAX_FRAMES_IN_FLIGHT;

    bool advance_timeline = false;
    std::vector<VkSemaphore> waits;
    std::vector<VkSemaphore> signals;
    if (options.enable_sync && kernel_working)
    {
        VkSemaphoreWaitInfo wait_info{};
        wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait_info.pSemaphores = &interop.vk_semaphore;
        wait_info.semaphoreCount = 1;
        wait_info.pValues = &wait_value;
        //printf("Frame %lu will wait for semaphore value %lu\n", frame_idx, wait_value);
        vkWaitSemaphores(dev->logical_device, &wait_info, timeout);

        waits.push_back(interop.vk_semaphore);
        signals.push_back(interop.vk_semaphore);
        //printf("Frame %lu will signal semaphore value %lu\n", frame_idx, signal_value);
        advance_timeline = true;
    }

    // Acquire image from swap chain
    uint32_t image_idx;
    auto result = vkAcquireNextImageKHR(dev->logical_device, swap->swapchain,
        timeout, present_semaphore, VK_NULL_HANDLE, &image_idx
    );
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreateSwapchain();
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        throw std::runtime_error("Failed to acquire swapchain image");
    }

    /*if (images_inflight[image_idx] != VK_NULL_HANDLE)
    {
        vkWaitForFences(dev->logical_device, 1, &images_inflight[image_idx], VK_TRUE, timeout);
    }
    images_inflight[image_idx] = frame.render_fence;*/

    // Wait for fences 
    auto fence = frame_fences[frame_idx];
    //printf("Frame %lu will wait for fence\n", frame_idx);
    validation::checkVulkan(vkWaitForFences(dev->logical_device, 1, &fence, VK_TRUE, timeout));
    //printf("Frame %lu passed fence\n", frame_idx);
    if (current_frame > MAX_FRAMES_IN_FLIGHT)
    {
        total_pipeline_time += getRenderTimeResults(frame_idx);
    }

    // Retrieve a command buffer and start recording to it
    auto cmd = command_buffers[frame_idx];
    validation::checkVulkan(vkResetCommandBuffer(cmd, 0));
    auto cmd_flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    auto begin_info = vkinit::commandBufferBeginInfo(cmd_flags);
    validation::checkVulkan(vkBeginCommandBuffer(cmd, &begin_info));

    auto render_pass_info = vkinit::renderPassBeginInfo(
        render_pass, fbs[image_idx].framebuffer, swap->extent
    );
    std::array<VkClearValue, 2> clear_values{};
    setColor(clear_values[0].color.float32, bg_color);
    clear_values[1].depthStencil = {1.f, 0};
    render_pass_info.clearValueCount = clear_values.size();
    render_pass_info.pClearValues    = clear_values.data();

    // Start of render pass and timestamp query
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool, frame_idx * 2);
    vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

    drawElements(frame_idx);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    // End of render pass and timestamp query
    vkCmdEndRenderPass(cmd);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool, frame_idx * 2 + 1);

    // Finalize command buffer recording, so it can be executed
    validation::checkVulkan(vkEndCommandBuffer(cmd));

    updateUniformBuffers(frame_idx);

    // Fill out command buffer submission info
    std::vector<VkPipelineStageFlags> stages;
    stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    auto interop_sync_info = vkinit::timelineSubmitInfo(&wait_value, &signal_value);
    auto submit_info = vkinit::submitInfo(&cmd, waits, stages, signals, &interop_sync_info);

    // Clear fence before placing it again
    validation::checkVulkan(vkResetFences(dev->logical_device, 1, &fence));
    // Execute command buffer using image as attachment in framebuffer
    validation::checkVulkan(vkQueueSubmit(dev->graphics.queue, 1, &submit_info, fence));

    // Return image result back to swapchain for presentation on screen
    auto present_info = vkinit::presentInfo(&image_idx, &swap->swapchain, &present_semaphore);
    result = vkQueuePresentKHR(dev->present.queue, &present_info);
    // Resize should be done after presentation to ensure semaphore consistency
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || should_resize)
    {
        recreateSwapchain();
        should_resize = false;
    }

    // Limit frame if it was configured
    if (options.enable_fps_limit) frameStall(target_frame_time);

    total_frame_count++;
    total_graphics_time += frame_time;
    frame_times[current_frame % frame_times.size()] = frame_time;
    last_time = current_time;

    current_frame++;
    if (advance_timeline)
    {
        wait_value += 2;
        signal_value += 2;
    }

    /*if (options.report_period > 0 && frame_time > options.report_period)
    {
        printf("Report at %d seconds:\n", options.report_period);
        showMetrics();
        last_time = current_time;
    }*/
}

void CudaviewEngine::drawElements(uint32_t image_idx)
{
    auto min_alignment = dev->properties.limits.minUniformBufferOffsetAlignment;
    auto size_mvp = getAlignedSize(sizeof(ModelViewProjection), min_alignment);
    auto size_primitive = getAlignedSize(sizeof(PrimitiveParams), min_alignment);
    auto size_scene = getAlignedSize(sizeof(SceneParams), min_alignment);
    auto size_ubo = size_mvp + size_primitive + size_scene;

    auto cmd = command_buffers[image_idx];
    for (uint32_t i = 0; i < views2.size(); ++i)
    {
        auto& view = views2[i];
        if (!view->params.options.visible) continue;
        std::vector<uint32_t> offsets = {
            i * size_ubo,
            i * size_ubo + size_mvp + size_primitive,
            i * size_ubo + size_mvp
        };
        // NOTE: Second parameter can be also used to bind a compute pipeline
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout, 0, 1, &descriptor_sets[image_idx], offsets.size(), offsets.data()
        );
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, view->pipeline);

        switch (view->params.view_type)
        {
            case ViewType::Markers:
            case ViewType::Voxels:
            {
                vkCmdBindVertexBuffers(cmd, 0, view->vert_buffers.size(),
                    view->vert_buffers.data(), view->buffer_offsets.data()
                );
                vkCmdDraw(cmd, view->params.element_count, 1, 0, 0);
                break;
            }
            case ViewType::Edges:
            {
                vkCmdBindVertexBuffers(cmd, 0, view->vert_buffers.size(),
                    view->vert_buffers.data(), view->buffer_offsets.data()
                );
                vkCmdBindIndexBuffer(cmd, view->idx_buffer, 0, view->idx_type);
                vkCmdDrawIndexed(cmd, 3 * view->params.element_count, 1, 0, 0, 0);
                break;
            }
            case ViewType::Image:
            {
                printf("TODO draw image");
                break;
            }
            default: break;
        }

        /*if (view->params.resource_type == ResourceType::Texture ||
            view->params.element_type == ElementType::Image)
        {
            if (view->params.element_type == ElementType::Voxels)
            {
                VkBuffer vertex_buffers[] = { view->data_buffer };
                VkDeviceSize offsets[] = { 0 };
                auto binding_count = sizeof(vertex_buffers) / sizeof(vertex_buffers[0]);
                vkCmdBindVertexBuffers(cmd, 0, binding_count, vertex_buffers, offsets);
                vkCmdDraw(cmd, view->params.element_count, 1, 0, 0);
            }
            else
            {
                VkDeviceSize offsets[1] = {0};
                vkCmdBindVertexBuffers(cmd, 0, 1, &view->aux_buffer, offsets);
                vkCmdBindIndexBuffer(cmd, view->aux_buffer, view->index_offset, VK_INDEX_TYPE_UINT16);
                vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
            }
        }
        else if (view->params.resource_type == ResourceType::Buffer &&
                 view->params.domain_type == DomainType::Structured)
        {
            if (view->params.element_type == ElementType::Voxels)
            {
                VkBuffer vertex_buffers[] = { view->aux_buffer, view->data_buffer };
                VkDeviceSize offsets[] = { 0, 0 };
                auto binding_count = sizeof(vertex_buffers) / sizeof(vertex_buffers[0]);
                vkCmdBindVertexBuffers(cmd, 0, binding_count, vertex_buffers, offsets);
                vkCmdDraw(cmd, view->params.element_count, 1, 0, 0);
            }
        }
        else if (view->params.resource_type == ResourceType::Buffer &&
                 view->params.domain_type == DomainType::Unstructured)
        {
            switch (view->params.element_type)
            {
                case ElementType::Edges:
                {
                    VkBuffer vertexBuffers[] = { views[0]->data_buffer };
                    VkDeviceSize offsets[] = {0};
                    vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
                    vkCmdBindIndexBuffer(cmd, view->data_buffer, 0, VK_INDEX_TYPE_UINT32);
                    vkCmdDrawIndexed(cmd, 3 * view->params.element_count, 1, 0, 0, 0);
                    break;
                }
                case ElementType::Voxels:
                {
                    // TODO: Check what to do here
                }
            }
        }*/
    }
}

bool CudaviewEngine::hasStencil(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

VkFormat CudaviewEngine::findDepthFormat()
{
    return dev->findSupportedImageFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}

VkRenderPass CudaviewEngine::createRenderPass()
{
    auto depth = vkinit::attachmentDescription(findDepthFormat());
    depth.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_ref{};
    depth_ref.attachment = 1;
    depth_ref.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    auto color = vkinit::attachmentDescription(swap->color_format);
    color.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_ref{};
    color_ref.attachment = 0;
    color_ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    auto subpass = vkinit::subpassDescription(VK_PIPELINE_BIND_POINT_GRAPHICS);
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &color_ref;
    subpass.pDepthStencilAttachment = &depth_ref;

    // Specify memory and execution dependencies between subpasses
    auto stage_mask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | 
                      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    auto access_mask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                       VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    auto dependency = vkinit::subpassDependency();
    dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass    = 0;
    dependency.srcStageMask  = stage_mask;
    dependency.dstStageMask  = stage_mask;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask = access_mask;

    std::array<VkAttachmentDescription, 2> attachments{ color, depth };
    auto pass_info = vkinit::renderPassCreateInfo(attachments, &subpass, &dependency);

    //printf("render pass attachment count: %lu\n", attachments.size());
    VkRenderPass render_pass = VK_NULL_HANDLE;
    validation::checkVulkan(
        vkCreateRenderPass(dev->logical_device, &pass_info, nullptr, &render_pass)
    );
    return render_pass;
}

void CudaviewEngine::createGraphicsPipelines()
{
    //auto start = std::chrono::steady_clock::now();
    auto orig_path = std::filesystem::current_path();
    std::filesystem::current_path(shader_path);

    PipelineBuilder builder(pipeline_layout, swap->extent);

    // Iterate through views, generating the corresponding pipelines
    // TODO: This does not allow adding views at runtime
    for (auto& view : views2)
    {
        builder.addPipeline(view->params, dev.get());
    }
    auto pipelines = builder.createPipelines(dev->logical_device, render_pass);
    //printf("%lu pipeline(s) created\n", pipelines.size());
    for (size_t i = 0; i < pipelines.size(); ++i)
    {
        views2[i]->pipeline = pipelines[i];
    }

    // Restore original working directory
    std::filesystem::current_path(orig_path);
    //auto end = std::chrono::steady_clock::now();
    //auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    //printf("Creation time for all pipelines: %lu ms\n", elapsed);
}

void CudaviewEngine::rebuildPipeline(InteropView& view)
{
    auto orig_path = std::filesystem::current_path();
    std::filesystem::current_path(shader_path);

    PipelineBuilder builder(pipeline_layout, swap->extent);
    builder.addPipeline(view.params, dev.get());
    auto pipelines = builder.createPipelines(dev->logical_device, render_pass);
    // Destroy the old view pipeline and assign the new one
    vkDestroyPipeline(dev->logical_device, view.pipeline, nullptr);
    view.pipeline = pipelines[0];

    std::filesystem::current_path(orig_path);
}

void CudaviewEngine::initUniformBuffers()
{
    auto min_alignment = dev->properties.limits.minUniformBufferOffsetAlignment;
    auto size_mvp = getAlignedSize(sizeof(ModelViewProjection), min_alignment);
    auto size_primitive = getAlignedSize(sizeof(PrimitiveParams), min_alignment);
    auto size_scene = getAlignedSize(sizeof(SceneParams), min_alignment);
    auto size_ubo = (size_mvp + size_primitive + size_scene) * views2.size();
    
    uniform_buffers.resize(swap->image_count);
    for (auto& ubo : uniform_buffers)
    {
        ubo.buffer = dev->createBuffer(size_ubo, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        VkMemoryRequirements memreq;
        vkGetBufferMemoryRequirements(dev->logical_device, ubo.buffer, &memreq);
        auto mem_usage = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        ubo.memory = dev->allocateMemory(memreq, mem_usage);
        vkBindBufferMemory(dev->logical_device, ubo.buffer, ubo.memory, 0);
    }
}

// Update uniform buffers for view at index [view_idx] for frame [image_idx]
void CudaviewEngine::updateUniformBuffers(uint32_t image_idx)
{
    auto min_alignment = dev->properties.limits.minUniformBufferOffsetAlignment;
    auto size_mvp = getAlignedSize(sizeof(ModelViewProjection), min_alignment);
    auto size_primitive = getAlignedSize(sizeof(PrimitiveParams), min_alignment);
    auto size_scene = getAlignedSize(sizeof(SceneParams), min_alignment);
    auto size_ubo = size_mvp + size_primitive + size_scene;
    auto memory = uniform_buffers[image_idx].memory;

    for (size_t view_idx = 0; view_idx < views2.size(); ++view_idx)
    {
        auto& view = views2[view_idx];

        ModelViewProjection mvp{};
        mvp.model = glm::mat4(1.f);
        mvp.view  = camera->matrices.view;
        mvp.proj  = camera->matrices.perspective;

        PrimitiveParams primitive{};
        primitive.color = getColor(view->params.options.default_color);
        primitive.size = view->params.options.default_size;
        //primitive.depth = view->params.options.depth;

        SceneParams scene{};
        auto extent = view->params.extent;
        scene.bg_color = getColor(bg_color);
        scene.extent = glm::ivec3{extent.x, extent.y, extent.z};
        scene.resolution = glm::ivec2{options.window_size.x, options.window_size.y};
        scene.camera_pos = camera->position;

        char *data = nullptr;
        auto offset = size_ubo * view_idx;
        vkMapMemory(dev->logical_device, memory, offset, size_ubo, 0, (void**)&data);
        std::memcpy(data, &mvp, sizeof(mvp));
        std::memcpy(data + size_mvp, &primitive, sizeof(primitive));
        std::memcpy(data + size_mvp + size_primitive, &scene, sizeof(scene));
        vkUnmapMemory(dev->logical_device, memory);
    }
}

double CudaviewEngine::getRenderTimeResults(uint32_t cmd_idx)
{
    auto timestamp_period = dev->properties.limits.timestampPeriod;
    const double seconds_per_tick = static_cast<double>(timestamp_period) / 1e9;

    uint64_t buffer[2];
    validation::checkVulkan(vkGetQueryPoolResults(dev->logical_device, query_pool,
        2 * cmd_idx, 2, 2 * sizeof(uint64_t), buffer, sizeof(uint64_t), 
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)
    );
    vkResetQueryPool(dev->logical_device, query_pool, cmd_idx * 2, 2);
    // TODO: apply time &= timestamp_mask;
    return static_cast<double>(buffer[1] - buffer[0]) * seconds_per_tick;
}

} // namespace mimir