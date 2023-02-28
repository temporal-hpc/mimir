#include "cudaview/vk_engine.hpp"
#include "cudaview/io.hpp"

#include "internal/camera.hpp"
#include "internal/color.hpp"
#include "internal/utils.hpp"
#include "internal/validation.hpp"
#include "internal/vk_initializers.hpp"
#include "internal/vk_pipeline.hpp"
#include "internal/vk_properties.hpp"
#include "cudaview/engine/vk_cudadevice.hpp"
#include "cudaview/engine/vk_framebuffer.hpp"
#include "cudaview/engine/vk_swapchain.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <chrono> // std::chrono
#include <filesystem> // std::filesystem
#include <glm/gtx/string_cast.hpp>

VulkanEngine::VulkanEngine(cudaStream_t cuda_stream):
  stream(cuda_stream),
  shader_path(io::getDefaultShaderPath()),
  camera(std::make_unique<Camera>())
{}

VulkanEngine::~VulkanEngine()
{
  if (rendering_thread.joinable())
  {
    rendering_thread.join();
  }
  vkDeviceWaitIdle(dev->logical_device);

  if (stream != nullptr)
  {
    validation::checkCuda(cudaStreamSynchronize(stream));
  }

  cleanupSwapchain();

  //ImGui_ImplVulkan_Shutdown();
  deletors.flush();
  fbs.clear();
  swap.reset();
  dev.reset();
  if (instance != VK_NULL_HANDLE)
  {
    vkDestroyInstance(instance, nullptr);
  }
  if (window != nullptr)
  {
    glfwDestroyWindow(window);
  }
  glfwTerminate();
}

void VulkanEngine::init(int width, int height)
{
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  //glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  window = glfwCreateWindow(width, height, "Vulkan test", nullptr, nullptr);
  glfwSetWindowUserPointer(window, this);
  glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
  glfwSetCursorPosCallback(window, cursorPositionCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);
  glfwSetScrollCallback(window, scrollCallback);

  initVulkan();

  camera->type = Camera::CameraType::LookAt;
  //camera->flipY = true;
  camera->setPosition(glm::vec3(0.f, 0.f, -3.75f));
  camera->setRotation(glm::vec3(15.f, 0.f, 0.f));
  camera->setRotationSpeed(0.5f);
  camera->setPerspective(60.f, (float)width / (float)height, 0.1f, 256.f);
}

void VulkanEngine::displayAsync()
{
  rendering_thread = std::thread([this]()
  {
    while(!glfwWindowShouldClose(window))
    {
      glfwPollEvents(); // TODO: Move to main thread
      //drawGui();

      std::unique_lock<std::mutex> ul(mutex);
      cond.wait(ul, [&]{ return device_working == false; });
      renderFrame();
      ul.unlock();
    }
    vkDeviceWaitIdle(dev->logical_device);
  });
}

void VulkanEngine::prepareWindow()
{
  device_working = true;
  std::unique_lock<std::mutex> ul(mutex);
  waitKernelStart();
  ul.unlock();
  cond.notify_one();
}

void VulkanEngine::updateWindow()
{
  std::unique_lock<std::mutex> ul(mutex);
  for (auto view : views)
  {
    if (view.params.resource_type == ResourceType::TextureLinear)
      dev->updateTexture(view);
  }
  device_working = false;
  signalKernelFinish();
  ul.unlock();
  cond.notify_one();
}

void VulkanEngine::display(std::function<void(void)> func, size_t iter_count)
{
  size_t iteration_idx = 0;
  device_working = true;
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    //drawGui();
    renderFrame();

    waitKernelStart();
    if (iteration_idx < iter_count)
    {
      // Advance the simulation
      func();
      for (auto view : views)
      {
        if (view.params.resource_type == ResourceType::TextureLinear)
          dev->updateTexture(view);
      }
      iteration_idx++;
    }
    signalKernelFinish();
  }
  device_working = false;
  vkDeviceWaitIdle(dev->logical_device);
}

CudaView VulkanEngine::addView(void **ptr_devmem, const ViewParams params)
{
  auto view = dev->createView(params);
  /*data_extent.x = std::max(data_extent.x, params.extent.x);
  data_extent.y = std::max(data_extent.y, params.extent.y);
  data_extent.z = std::max(data_extent.z, params.extent.z);*/
  dev->createUniformBuffers(view, swap->image_count);
  views.push_back(view);
  updateDescriptorSets();
  *ptr_devmem = view.cuda_ptr;
  return view;
}

void VulkanEngine::initVulkan()
{
  createInstance();
  swap = std::make_unique<VulkanSwapchain>();
  swap->initSurface(instance, window);
  pickPhysicalDevice();
  dev = std::make_unique<VulkanCudaDevice>(physical_device);
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

  // Create descriptor set layout
  descriptor_layout = dev->createDescriptorSetLayout({
    vkinit::descriptorLayoutBinding(0, // binding
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_GEOMETRY_BIT
    ),
    vkinit::descriptorLayoutBinding(1, // binding
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT
    ),
    vkinit::descriptorLayoutBinding(2, // binding
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT
    ),
    vkinit::descriptorLayoutBinding(5, // binding
      VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT
    ),
    vkinit::descriptorLayoutBinding(6, // binding
      VK_DESCRIPTOR_TYPE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT
    )
  });

  // Create pipeline layout
  std::vector<VkDescriptorSetLayout> layouts{descriptor_layout};
  auto pipeline_layout_info = vkinit::pipelineLayoutCreateInfo(layouts);
  validation::checkVulkan(vkCreatePipelineLayout(
    dev->logical_device, &pipeline_layout_info, nullptr, &pipeline_layout)
  );

  initSwapchain();

  //initImgui(); // After command pool and render pass are created
  createSyncObjects();
}

void VulkanEngine::createInstance()
{
  if (validation::enable_layers &&
     !validation::checkValidationLayerSupport())
  {
    throw std::runtime_error("validation layers requested, but not supported");
  }

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName   = "Vulkan test";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName        = "No engine";
  app_info.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion         = VK_API_VERSION_1_3;

  uint32_t glfw_ext_count = 0;
  // List required GLFW extensions and additional required validation layers
  const char **glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);
  std::vector<const char*> extensions(glfw_exts, glfw_exts + glfw_ext_count);
  if (validation::enable_layers)
  {
    // Enable debugging message extension
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }
  extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

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

  if (validation::enable_layers)
  {
    // Details about the debug messenger and its callback
    validation::checkVulkan(validation::CreateDebugUtilsMessengerEXT(
      instance, &debug_create_info, nullptr, &debug_messenger)
    );
    deletors.pushFunction([=]{
      validation::DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
    });
  }

  /*uint32_t extension_count = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
  std::vector<VkExtensionProperties> available_exts(extension_count);
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, available_exts.data());

  std::cout << "Available extensions:\n";
  for (const auto& extension : available_exts)
  {
    std::cout << '\t' << extension.extensionName << '\n';
  }*/
}

void VulkanEngine::pickPhysicalDevice()
{
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
  if (device_count == 0)
  {
    throw std::runtime_error("failed to find GPUs with Vulkan support");
  }
  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

  for (const auto& dev : devices)
  {
    if (props::isDeviceSuitable(dev, swap->surface))
    {
      physical_device = dev;
      break;
    }
  }
  if (physical_device == VK_NULL_HANDLE)
  {
    throw std::runtime_error("failed to find a suitable GPU!");
  }
}

void VulkanEngine::initImgui()
{
  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForVulkan(window, true);

  ImGui_ImplVulkan_InitInfo init_info{};
  init_info.Instance       = instance;
  init_info.PhysicalDevice = physical_device;
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

void VulkanEngine::getWaitFrameSemaphores(std::vector<VkSemaphore>& wait,
  std::vector<VkPipelineStageFlags>& wait_stages) const
{
  // Wait semaphore has not been initialized on the first frame
  if (current_frame != 0 && device_working == true)
  {
    // Vulkan waits until Cuda is done with the display buffer before rendering
    wait.push_back(kernel_finish.vk_semaphore);
    // Cuda will wait until all pipeline commands are complete
    wait_stages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
  }
}

void VulkanEngine::getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const
{
  // Vulkan signals this semaphore once the device array is ready for Cuda use
  signal.push_back(kernel_start.vk_semaphore);
}

void VulkanEngine::waitKernelStart()
{
  cudaExternalSemaphoreWaitParams wait_params{};
  wait_params.flags = 0;
  wait_params.params.fence.value = 0;
  // Wait for Vulkan to complete its work
  validation::checkCuda(cudaWaitExternalSemaphoresAsync(//&cuda_timeline_semaphore
    &kernel_start.cuda_semaphore, &wait_params, 1, stream)
  );
}

void VulkanEngine::signalKernelFinish()
{
  cudaExternalSemaphoreSignalParams signal_params{};
  signal_params.flags = 0;
  signal_params.params.fence.value = 0;

  // Signal Vulkan to continue with the updated buffers
  validation::checkCuda(cudaSignalExternalSemaphoresAsync(//&cuda_timeline_semaphore
    &kernel_finish.cuda_semaphore, &signal_params, 1, stream)
  );
}

void VulkanEngine::createSyncObjects()
{
  images_inflight.resize(swap->image_count, VK_NULL_HANDLE);

  auto semaphore_info = vkinit::semaphoreCreateInfo();
  auto fence_info = vkinit::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
  for (auto& frame : frames)
  {
    validation::checkVulkan(vkCreateSemaphore(
      dev->logical_device, &semaphore_info, nullptr, &frame.present_semaphore)
    );
    validation::checkVulkan(vkCreateSemaphore(
      dev->logical_device, &semaphore_info, nullptr, &frame.render_semaphore)
    );
    validation::checkVulkan(vkCreateFence(
      dev->logical_device, &fence_info, nullptr, &frame.render_fence)
    );
    deletors.pushFunction([=]{
      vkDestroySemaphore(dev->logical_device, frame.present_semaphore, nullptr);
      vkDestroySemaphore(dev->logical_device, frame.render_semaphore, nullptr);
      vkDestroyFence(dev->logical_device, frame.render_fence, nullptr);
    });
  }

  // Vulkan signal will be CUDA wait
  kernel_start = dev->createInteropBarrier();
  // CUDA signal will be Vulkan wait
  kernel_finish = dev->createInteropBarrier();

  /*validation::checkVulkan(vkCreateSemaphore(
    device, &semaphore_info, nullptr, &vk_presentation_semaphore)
  );
  createExternalSemaphore(vk_timeline_semaphore);
  importCudaExternalSemaphore(cuda_timeline_semaphore, vk_timeline_semaphore);
  if (cuda_timeline_semaphore != nullptr)
  {
    validation::checkCuda(cudaDestroyExternalSemaphore(cuda_timeline_semaphore));
  }
  if (vk_presentation_semaphore != VK_NULL_HANDLE)
  {
    vkDestroySemaphore(device, vk_presentation_semaphore, nullptr);
  }
  if (vk_timeline_semaphore != VK_NULL_HANDLE)
  {
    vkDestroySemaphore(device, vk_timeline_semaphore, nullptr);
  }*/
}

void VulkanEngine::cleanupSwapchain()
{
  for (auto& view : views)
  {
    vkDestroyBuffer(dev->logical_device, view.ubo_buffer, nullptr);
    vkFreeMemory(dev->logical_device, view.ubo_memory, nullptr);
  }
  for (auto pipeline : pipelines)
  {
    vkDestroyPipeline(dev->logical_device, pipeline, nullptr);
  }
  vkDestroyPipelineLayout(dev->logical_device, pipeline_layout, nullptr);
  vkDestroyRenderPass(dev->logical_device, render_pass, nullptr);
  vkFreeCommandBuffers(dev->logical_device, dev->command_pool,
    command_buffers.size(), command_buffers.data()
  );
  swap->cleanup();
}

void VulkanEngine::initSwapchain()
{
  int w, h;
  glfwGetFramebufferSize(window, &w, &h);
  uint32_t width = w;
  uint32_t height = h;
  std::vector queue_indices{dev->graphics.family_index, dev->present.family_index};
  swap->create(width, height, queue_indices, dev->physical_device, dev->logical_device);
  command_buffers = dev->createCommandBuffers(swap->image_count);
  render_pass = createRenderPass();

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

  deletors.pushFunction([=]{
    vkDestroyImageView(dev->logical_device, depth_view, nullptr);
    vkDestroyImage(dev->logical_device, depth_image, nullptr);
    vkFreeMemory(dev->logical_device, depth_memory, nullptr);
  });

  fbs.resize(swap->image_count);
  for (size_t i = 0; i < swap->image_count; ++i)
  {
    // Create a basic image view to be used as color target
    fbs[i].addAttachment(dev->logical_device, images[i], swap->color_format);
    fbs[i].create(dev->logical_device, render_pass, swap->swapchain_extent, depth_view);
  }

  createGraphicsPipelines();
  descriptor_sets = dev->createDescriptorSets(
    descriptor_pool, descriptor_layout, swap->image_count
  );
  //updateDescriptorSets();
}

void VulkanEngine::recreateSwapchain()
{
  vkDeviceWaitIdle(dev->logical_device);

  cleanupSwapchain();
  initSwapchain();
}

void VulkanEngine::updateDescriptorSets()
{
  auto min_alignment = dev->properties.limits.minUniformBufferOffsetAlignment;
  auto size_mvp = getAlignedSize(sizeof(ModelViewProjection), min_alignment);
  auto size_colors = getAlignedSize(sizeof(PrimitiveParams), min_alignment);
  auto size_scene = getAlignedSize(sizeof(SceneParams), min_alignment);
  auto size_ubo = 2 * size_mvp + size_colors + size_scene;

  for (size_t i = 0; i < descriptor_sets.size(); ++i)
  {
    // Write MVP matrix, scene info and texture samplers
    std::vector<VkWriteDescriptorSet> desc_writes;
    //desc_writes.reserve(3 + views_structured.size());
    for (const auto& view : views)
    {
      VkDescriptorBufferInfo mvp_info{};
      mvp_info.buffer = view.ubo_buffer;
      mvp_info.offset = i * size_ubo;
      mvp_info.range  = sizeof(ModelViewProjection);
      auto write_mvp = vkinit::writeDescriptorBuffer(
        descriptor_sets[i], 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &mvp_info
      );
      desc_writes.push_back(write_mvp);

      VkDescriptorBufferInfo primitive_info{};
      primitive_info.buffer = view.ubo_buffer;
      primitive_info.offset = i * size_ubo + size_mvp;
      primitive_info.range  = sizeof(PrimitiveParams);
      auto write_primitive = vkinit::writeDescriptorBuffer(
        descriptor_sets[i], 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &primitive_info
      );
      desc_writes.push_back(write_primitive);

      VkDescriptorBufferInfo extent_info{};
      extent_info.buffer = view.ubo_buffer;
      extent_info.offset = i * size_ubo + size_mvp + size_colors;
      extent_info.range  = sizeof(SceneParams);
      auto write_scene = vkinit::writeDescriptorBuffer(
        descriptor_sets[i], 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &extent_info
      );
      desc_writes.push_back(write_scene);
      
      if (view.params.resource_type == ResourceType::TextureLinear ||
          view.params.resource_type == ResourceType::Texture)
      {
        VkDescriptorImageInfo samp_img_info{};
        samp_img_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        samp_img_info.imageView   = view.vk_view;
        samp_img_info.sampler     = view.vk_sampler;

        auto write_samp_img = vkinit::writeDescriptorImage(descriptor_sets[i],
          5, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &samp_img_info
        );
        desc_writes.push_back(write_samp_img);

        VkDescriptorImageInfo samp_info{};
        samp_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        samp_info.imageView   = view.vk_view;
        samp_info.sampler     = view.vk_sampler;

        auto write_samp = vkinit::writeDescriptorImage(descriptor_sets[i],
          6, VK_DESCRIPTOR_TYPE_SAMPLER, &samp_info
        );
        desc_writes.push_back(write_samp);
      }
    }

    vkUpdateDescriptorSets(dev->logical_device,
      static_cast<uint32_t>(desc_writes.size()), desc_writes.data(), 0, nullptr
    );
  }
}

void VulkanEngine::drawGui()
{
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  //ImGui::ShowDemoWindow();
  ImGui::Render();
}

FrameBarrier& VulkanEngine::getCurrentFrame()
{
  return frames[current_frame % frames.size()];
}

void VulkanEngine::renderFrame()
{
  constexpr auto timeout = std::numeric_limits<uint64_t>::max();
  /*const uint64_t wait_value = 0;
  const uint64_t signal_value = 1;

  VkSemaphoreWaitInfo wait_info{};
  wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  wait_info.pSemaphores = &vk_timeline_semaphore;
  wait_info.semaphoreCount = 1;
  wait_info.pValues = &wait_value;
  vkWaitSemaphores(device, &wait_info, timeout);*/

  auto frame = getCurrentFrame();
  vkWaitForFences(dev->logical_device, 1, &frame.render_fence, VK_TRUE, timeout);

  // Acquire image from swap chain
  uint32_t image_idx;
  // TODO: vk_presentation_semaphore instead of frame.present_semaphore
  auto result = vkAcquireNextImageKHR(dev->logical_device, swap->swapchain,
    timeout, frame.present_semaphore, VK_NULL_HANDLE, &image_idx
  );
  if (result == VK_ERROR_OUT_OF_DATE_KHR)
  {
    recreateSwapchain();
  }
  else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
  {
    throw std::runtime_error("Failed to acquire swapchain image");
  }

  if (images_inflight[image_idx] != VK_NULL_HANDLE)
  {
    vkWaitForFences(dev->logical_device, 1, &images_inflight[image_idx], VK_TRUE, timeout);
  }
  images_inflight[image_idx] = frame.render_fence;

  auto cmd_flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  auto begin_info = vkinit::commandBufferBeginInfo(cmd_flags);

  auto cmd = command_buffers[image_idx];
  validation::checkVulkan(vkBeginCommandBuffer(cmd, &begin_info));

  auto render_pass_info = vkinit::renderPassBeginInfo(
    render_pass, fbs[image_idx].framebuffer, swap->swapchain_extent
  );
  std::array<VkClearValue, 2> clear_values{};
  color::setColor(clear_values[0].color.float32, bg_color);
  clear_values[1].depthStencil = {1.f, 0};
  render_pass_info.clearValueCount = clear_values.size();
  render_pass_info.pClearValues    = clear_values.data();

  vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

  drawObjects(image_idx);
  //ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

  vkCmdEndRenderPass(cmd);
  // Finalize command buffer recording, so it can be executed
  validation::checkVulkan(vkEndCommandBuffer(cmd));

  for (auto& view : views)
  {
    dev->updateUniformBuffers(view, 
      image_idx, camera->matrices.view, camera->matrices.perspective, bg_color
    );
  }

  std::vector<VkSemaphore> wait_semaphores;
  std::vector<VkPipelineStageFlags> wait_stages;
  wait_semaphores.push_back(frame.present_semaphore); //vk_timeline_semaphore
  wait_stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
  getWaitFrameSemaphores(wait_semaphores, wait_stages);

  std::vector<VkSemaphore> signal_semaphores;
  getSignalFrameSemaphores(signal_semaphores);
  signal_semaphores.push_back(frame.render_semaphore);//vk_timeline_semaphore

  auto submit_info = vkinit::submitInfo(&cmd);
  submit_info.waitSemaphoreCount   = wait_semaphores.size();
  submit_info.pWaitSemaphores      = wait_semaphores.data();
  submit_info.pWaitDstStageMask    = wait_stages.data();
  submit_info.signalSemaphoreCount = signal_semaphores.size();
  submit_info.pSignalSemaphores    = signal_semaphores.data();

  /*VkTimelineSemaphoreSubmitInfo timeline_info{};
  timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  timeline_info.waitSemaphoreValueCount = 1;
  timeline_info.pWaitSemaphoreValues = &wait_value;
  timeline_info.signalSemaphoreValueCount = 1;
  timeline_info.pSignalSemaphoreValues = &signal_value;
  submit_info.pNext = &timeline_info;*/

  vkResetFences(dev->logical_device, 1, &frame.render_fence);

  // Execute command buffer using image as attachment in framebuffer
  validation::checkVulkan(vkQueueSubmit(
    dev->graphics.queue, 1, &submit_info, frame.render_fence) //VK_NULL_HANDLE
  );

  // Return image result back to swapchain for presentation on screen
  auto present_info = vkinit::presentInfo();
  present_info.swapchainCount     = 1;
  present_info.pSwapchains        = &swap->swapchain;
  present_info.waitSemaphoreCount = 1;
  //present_info.pWaitSemaphores = &vk_presentation_semaphore;
  present_info.pWaitSemaphores    = &frame.render_semaphore;
  present_info.pImageIndices      = &image_idx;

  result = vkQueuePresentKHR(dev->present.queue, &present_info);
  // Resize should be done after presentation to ensure semaphore consistency
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR
    || should_resize)
  {
    recreateSwapchain();
    should_resize = false;
  }

  current_frame++;
}

void VulkanEngine::drawObjects(uint32_t image_idx)
{
  auto cmd = command_buffers[image_idx];
  for (const auto& view : views)
  {
    if (view.params.resource_type == ResourceType::TextureLinear ||
        view.params.resource_type == ResourceType::Texture)
    {
      if (view.params.primitive_type == PrimitiveType::Voxels)
      {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[7]);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
          pipeline_layout, 0, 1, &descriptor_sets[image_idx], 0, nullptr
        );
        VkBuffer vertex_buffers[] = { view.interop_buffer };
        VkDeviceSize offsets[] = { 0 };
        auto binding_count = sizeof(vertex_buffers) / sizeof(vertex_buffers[0]);
        vkCmdBindVertexBuffers(cmd, 0, binding_count, vertex_buffers, offsets);
        vkCmdDraw(cmd, view.params.element_count, 1, 0, 0);
      }
      else
      {
        if (view.params.data_domain == DataDomain::Domain2D)
        {
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[2]);
        }
        else if (view.params.data_domain == DataDomain::Domain3D)
        {
          vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[3]);
        }
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
          pipeline_layout, 0, 1, &descriptor_sets[image_idx], 0, nullptr
        );

        VkDeviceSize offsets[1] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, &view.vertex_buffer, offsets);
        vkCmdBindIndexBuffer(cmd, view.index_buffer, 0, VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
      }
    }
    else if (view.params.resource_type == ResourceType::StructuredBuffer)
    {
      if (view.params.primitive_type == PrimitiveType::Voxels)
      {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[6]);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
          pipeline_layout, 0, 1, &descriptor_sets[image_idx], 0, nullptr
        );
        VkBuffer vertex_buffers[] = { view.vertex_buffer, view.interop_buffer };
        VkDeviceSize offsets[] = { 0, 0 };
        auto binding_count = sizeof(vertex_buffers) / sizeof(vertex_buffers[0]);
        vkCmdBindVertexBuffers(cmd, 0, binding_count, vertex_buffers, offsets);
        vkCmdDraw(cmd, view.params.element_count, 1, 0, 0);
      }
    }
    else if (view.params.resource_type == ResourceType::UnstructuredBuffer)
    {
      switch (view.params.primitive_type)
      {
        case PrimitiveType::Points:
        {
          if (view.params.data_domain == DataDomain::Domain2D)
          {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[0]);
          }
          else if (view.params.data_domain == DataDomain::Domain3D)
          {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[1]);
          }
          // NOTE: Second parameter can be also used to bind a compute pipeline
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout, 0, 1, &descriptor_sets[image_idx], 0, nullptr
          );
          VkBuffer vertex_buffers[] = { view.interop_buffer };
          VkDeviceSize offsets[] = { 0 };
          auto binding_count = sizeof(vertex_buffers) / sizeof(vertex_buffers[0]);
          vkCmdBindVertexBuffers(cmd, 0, binding_count, vertex_buffers, offsets);
          vkCmdDraw(cmd, view.params.element_count, 1, 0, 0);
          break;
        }
        case PrimitiveType::Edges:
        {
          if (view.params.data_domain == DataDomain::Domain2D)
          {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[4]);
          }
          else if (view.params.data_domain == DataDomain::Domain3D)
          {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[5]);
          }
          vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout, 0, 1, &descriptor_sets[image_idx], 0, nullptr
          );
          VkBuffer vertexBuffers[] = { views[0].interop_buffer };
          VkDeviceSize offsets[] = {0};
          vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
          vkCmdBindIndexBuffer(cmd, view.interop_buffer, 0, VK_INDEX_TYPE_UINT32);
          vkCmdDrawIndexed(cmd, 3 * view.params.element_count, 1, 0, 0, 0);
          break;
        }
        case PrimitiveType::Voxels:
        {
          // TODO: Check what to do here
        }
      }
    }
  }
}

bool VulkanEngine::hasStencil(VkFormat format)
{
  return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

VkFormat VulkanEngine::findDepthFormat()
{
  return findSupportedFormat(
    {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
    VK_IMAGE_TILING_OPTIMAL,
    VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
  );
}

VkFormat VulkanEngine::findSupportedFormat(const std::vector<VkFormat>& candidates,
  VkImageTiling tiling, VkFormatFeatureFlags features)
{
  for (auto format : candidates)
  {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(dev->physical_device, format, &props);

    if (tiling == VK_IMAGE_TILING_LINEAR &&
      (props.linearTilingFeatures & features) == features)
    {
      return format;
    }
    else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
      (props.optimalTilingFeatures & features) == features)
    {
      return format;
    }
  }
  return VK_FORMAT_UNDEFINED;
}

VkRenderPass VulkanEngine::createRenderPass()
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
  auto dependency = vkinit::subpassDependency();
  dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass    = 0;
  dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                             VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

  std::array<VkAttachmentDescription, 2> attachments{ color, depth };
  auto pass_info = vkinit::renderPassCreateInfo();
  pass_info.attachmentCount = attachments.size();
  pass_info.pAttachments    = attachments.data();
  pass_info.subpassCount    = 1;
  pass_info.pSubpasses      = &subpass;
  pass_info.dependencyCount = 1;
  pass_info.pDependencies   = &dependency;

  VkRenderPass render_pass;
  validation::checkVulkan(
    vkCreateRenderPass(dev->logical_device, &pass_info, nullptr, &render_pass)
  );
  return render_pass;
}

// Read buffer with shader bytecode and create a shader module from it
VkShaderModule VulkanEngine::createShaderModule(const std::vector<char>& code)
{
  VkShaderModuleCreateInfo info{};
  info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  info.pNext    = nullptr;
  info.flags    = 0; // Unused
  info.codeSize = code.size();
  info.pCode    = reinterpret_cast<const uint32_t*>(code.data());

  VkShaderModule module;
  validation::checkVulkan(
    vkCreateShaderModule(dev->logical_device, &info, nullptr, &module)
  );
  // TODO: Move to vulkandevice deletor queue (likely a new one)
  deletors.pushFunction([=]{
    vkDestroyShaderModule(dev->logical_device, module, nullptr);
  });
  return module;
}

void diagnose(slang::IBlob *diag_blob)
{
  if (diag_blob != nullptr)
  {
    printf("%s", (const char*) diag_blob->getBufferPointer());
  }
}

VkShaderStageFlagBits getVulkanShaderFlag(SlangStage stage)
{
  switch (stage)
  {
    case SLANG_STAGE_VERTEX:   return VK_SHADER_STAGE_VERTEX_BIT;
    case SLANG_STAGE_GEOMETRY: return VK_SHADER_STAGE_GEOMETRY_BIT;
    case SLANG_STAGE_FRAGMENT: return VK_SHADER_STAGE_FRAGMENT_BIT;
    default:                   return VK_SHADER_STAGE_ALL_GRAPHICS;
  }
}

std::vector<VkPipelineShaderStageCreateInfo> VulkanEngine::compileSlang(
  Slang::ComPtr<slang::ISession> session, const std::string& module_path,
  const std::vector<std::string>& entry_names, const std::string& specialization)
{
  Slang::ComPtr<slang::IBlob> diag = nullptr;
  // Load code from [module_path].slang as a module
  auto module = session->loadModule(module_path.c_str(), diag.writeRef());
  diagnose(diag);

  std::vector<slang::IComponentType*> components;
  components.reserve(entry_names.size() + 1);
  components.push_back(module);
  // Lookup entry points by their names
  for (const auto& name : entry_names)
  {
    Slang::ComPtr<slang::IEntryPoint> entrypoint = nullptr;
    module->findEntryPointByName(name.c_str(), entrypoint.writeRef());
    if (entrypoint != nullptr) components.push_back(entrypoint);
  }
  Slang::ComPtr<slang::IComponentType> program = nullptr;
  validation::checkSlang(session->createCompositeComponentType(
    components.data(), components.size(), program.writeRef(), diag.writeRef())
  );
  diagnose(diag);

  if (!specialization.empty())
  {
    auto spec_type = module->getLayout()->findTypeByName(specialization.c_str());
    std::vector<slang::SpecializationArg> args;
    slang::SpecializationArg arg;
    arg.kind = slang::SpecializationArg::Kind::Type;
    arg.type = spec_type;
    args.push_back(arg);

    Slang::ComPtr<slang::IComponentType> spec_program;
    validation::checkSlang(program->specialize(
      args.data(), args.size(), spec_program.writeRef(), diag.writeRef())
    );
    diagnose(diag);
    program = spec_program;
  }

  auto layout = program->getLayout();
  std::vector<VkPipelineShaderStageCreateInfo> compiled_stages;
  compiled_stages.reserve(layout->getEntryPointCount());
  for (unsigned idx = 0; idx < layout->getEntryPointCount(); ++idx)
  {
    auto entrypoint = layout->getEntryPointByIndex(idx);
    auto stage = getVulkanShaderFlag(entrypoint->getStage());

    diag = nullptr;
    Slang::ComPtr<slang::IBlob> kernel = nullptr;
    validation::checkSlang(
      program->getEntryPointCode(idx, 0, kernel.writeRef(), diag.writeRef())
    );
    diagnose(diag);

    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.pNext    = nullptr;
    info.flags    = 0; // Unused by Vulkan API
    info.codeSize = kernel->getBufferSize();
    info.pCode    = static_cast<const uint32_t*>(kernel->getBufferPointer());
    VkShaderModule shader_module;
    validation::checkVulkan(
      vkCreateShaderModule(dev->logical_device, &info, nullptr, &shader_module)
    );
    deletors.pushFunction([=]{
      vkDestroyShaderModule(dev->logical_device, shader_module, nullptr);
    });
    auto shader_info = vkinit::pipelineShaderStageCreateInfo(stage, shader_module);
    compiled_stages.push_back(shader_info);
  }
  return compiled_stages;
}

void VulkanEngine::createGraphicsPipelines()
{
  auto start = std::chrono::steady_clock::now();
  auto orig_path = std::filesystem::current_path();
  std::filesystem::current_path(shader_path);

  // Create global session to work with the Slang API
  Slang::ComPtr<slang::IGlobalSession> global_session;
  validation::checkSlang(slang::createGlobalSession(global_session.writeRef()));

  slang::TargetDesc target_desc{};
  target_desc.format = SLANG_SPIRV;
  target_desc.profile = global_session->findProfile("sm_6_6");
  const char* search_paths[] = { "shaders/include" };
  slang::SessionDesc session_desc{};
  session_desc.targets = &target_desc;
  session_desc.targetCount = 1;
  session_desc.searchPaths = search_paths;
  session_desc.searchPathCount = 1;

  // Obtain a compilation session that scopes compilation and code loading
  Slang::ComPtr<slang::ISession> session;
  validation::checkSlang(
    global_session->createSession(session_desc, session.writeRef())
  );

  PipelineBuilder builder(pipeline_layout, swap->swapchain_extent);

  auto marker2d_slang = compileSlang(session, "shaders/marker.slang",
    {"vertex2dMain", "geometryMain", "fragmentMain"}
  );

  PipelineInfo points2d;
  points2d.shader_stages = marker2d_slang;
  points2d.vertex_input_info = getVertexDescriptions2d();
  points2d.input_assembly = vkinit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
  points2d.rasterizer = vkinit::rasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);
  points2d.multisampling = vkinit::multisampleStateCreateInfo();
  points2d.color_blend_attachment = vkinit::colorBlendAttachmentState();
  builder.addPipelineInfo(points2d);

  auto marker3d_slang = compileSlang(session, "shaders/marker.slang",
    {"vertex3dMain", "geometryMain", "fragmentMain"}
  );

  PipelineInfo points3d;
  points3d.shader_stages = marker3d_slang;
  points3d.vertex_input_info = getVertexDescriptions3d();
  points3d.input_assembly = vkinit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
  points3d.rasterizer = vkinit::rasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);
  points3d.multisampling = vkinit::multisampleStateCreateInfo();
  points3d.color_blend_attachment = vkinit::colorBlendAttachmentState();
  builder.addPipelineInfo(points3d);

  auto tex2d_slang = compileSlang(session, "shaders/texture.slang",
    {"vertex2dMain", "fragment2dMain"}, "RawColor"
  );

  PipelineInfo texture2d;
  texture2d.shader_stages = tex2d_slang;
  texture2d.vertex_input_info = getVertexDescriptionsVert();
  texture2d.input_assembly = vkinit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  texture2d.rasterizer = vkinit::rasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);
  texture2d.multisampling = vkinit::multisampleStateCreateInfo();
  texture2d.color_blend_attachment = vkinit::colorBlendAttachmentState();
  builder.addPipelineInfo(texture2d);

  auto tex3d_slang = compileSlang(session, "shaders/texture.slang",
    {"vertex3dMain", "fragment3dMain"}
  );

  PipelineInfo texture3d;
  texture3d.shader_stages = tex3d_slang;
  texture3d.vertex_input_info = getVertexDescriptionsVert();
  texture3d.input_assembly = vkinit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  texture3d.rasterizer = vkinit::rasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);
  texture3d.multisampling = vkinit::multisampleStateCreateInfo();
  texture3d.color_blend_attachment = vkinit::colorBlendAttachmentState();
  builder.addPipelineInfo(texture3d);

  auto mesh2d_slang = compileSlang(session, "shaders/mesh.slang",
    {"vertex2dMain", "fragmentMain"}
  );

  PipelineInfo mesh2d;
  mesh2d.shader_stages = mesh2d_slang;
  mesh2d.vertex_input_info = getVertexDescriptions2d();
  mesh2d.input_assembly = vkinit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  mesh2d.rasterizer = vkinit::rasterizationStateCreateInfo(VK_POLYGON_MODE_LINE);
  mesh2d.multisampling = vkinit::multisampleStateCreateInfo();
  mesh2d.color_blend_attachment = vkinit::colorBlendAttachmentState();
  builder.addPipelineInfo(mesh2d);

  auto mesh3d_slang = compileSlang(session, "shaders/mesh.slang",
    {"vertex3dMain", "fragmentMain"}
  );

  PipelineInfo mesh3d;
  mesh3d.shader_stages = mesh3d_slang;
  mesh3d.vertex_input_info = getVertexDescriptions3d();
  mesh3d.input_assembly = vkinit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  mesh3d.rasterizer = vkinit::rasterizationStateCreateInfo(VK_POLYGON_MODE_LINE);
  mesh3d.multisampling     = vkinit::multisampleStateCreateInfo();
  mesh3d.color_blend_attachment = vkinit::colorBlendAttachmentState();
  builder.addPipelineInfo(mesh3d);

  auto voxel_implicit_slang = compileSlang(session, "shaders/voxel.slang",
    {"vertexImplicitMain", "geometryMain", "fragmentMain"}
  );

  PipelineInfo voxel_implicit;
  voxel_implicit.shader_stages = voxel_implicit_slang;
  voxel_implicit.vertex_input_info = getVoxelDescriptions();
  voxel_implicit.input_assembly = vkinit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
  voxel_implicit.rasterizer = vkinit::rasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);
  voxel_implicit.multisampling = vkinit::multisampleStateCreateInfo();
  voxel_implicit.color_blend_attachment = vkinit::colorBlendAttachmentState();
  builder.addPipelineInfo(voxel_implicit);

  auto voxel_slang = compileSlang(session, "shaders/voxel.slang",
    {"vertexMain", "geometryMain", "fragmentMain"}
  );

  PipelineInfo voxel3d;
  voxel3d.shader_stages = voxel_slang;
  voxel3d.vertex_input_info = getVertexDescriptions3d();
  voxel3d.input_assembly = vkinit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
  voxel3d.rasterizer = vkinit::rasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);
  voxel3d.multisampling = vkinit::multisampleStateCreateInfo();
  voxel3d.color_blend_attachment = vkinit::colorBlendAttachmentState();
  builder.addPipelineInfo(voxel3d);

  pipelines = builder.createPipelines(dev->logical_device, render_pass);
  std::cout << pipelines.size() << " pipeline(s) created\n";

  // Restore original working directory
  std::filesystem::current_path(orig_path);
  auto end = std::chrono::steady_clock::now();
  std::cout << "Creation time for all pipelines: "
    << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    << " ms\n";
}
