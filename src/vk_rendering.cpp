#include "cudaview/vk_engine.hpp"
#include "validation.hpp"

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"

static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;

void VulkanEngine::mainLoop()
{
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    //ImGui::ShowDemoWindow();
    ImGui::Render();

    drawFrame();
  }
  vkDeviceWaitIdle(device);
}

void VulkanEngine::drawFrame()
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

  auto frame_idx = current_frame % MAX_FRAMES_IN_FLIGHT;
  vkWaitForFences(device, 1, &inflight_fences[frame_idx], VK_TRUE, timeout);

  // Acquire image from swap chain
  uint32_t image_idx;
  // TODO: vk_presentation_semaphore instead of image_available[frame_idx]
  auto result = vkAcquireNextImageKHR(device, swapchain, timeout,
    image_available[frame_idx], VK_NULL_HANDLE, &image_idx
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
    vkWaitForFences(device, 1, &images_inflight[image_idx], VK_TRUE, timeout);
  }
  images_inflight[image_idx] = inflight_fences[frame_idx];

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  begin_info.pInheritanceInfo = nullptr;

  validation::checkVulkan(vkBeginCommandBuffer(
    command_buffers[image_idx], &begin_info)
  );

  VkRenderPassBeginInfo render_pass_info{};
  render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  render_pass_info.renderPass = render_pass;
  render_pass_info.framebuffer = framebuffers[image_idx];
  render_pass_info.renderArea.offset = {0, 0};
  render_pass_info.renderArea.extent = swapchain_extent;

  VkClearValue clear_color = {{{.5f, .5f, .5f, 1.f}}};
  render_pass_info.clearValueCount = 1;
  render_pass_info.pClearValues = &clear_color;

  vkCmdBeginRenderPass(command_buffers[image_idx], &render_pass_info,
    VK_SUBPASS_CONTENTS_INLINE
  );

  /*if (rendering_modes["structured"])
  {
    vkCmdBindPipeline(command_buffers[image_idx],
      VK_PIPELINE_BIND_POINT_GRAPHICS, screen_pipeline
    );
    vkCmdBindDescriptorSets(command_buffers[image_idx],
      VK_PIPELINE_BIND_POINT_GRAPHICS, screen_layout, 0, 1,
      &descriptor_sets[image_idx], 0, nullptr
    );
    vkCmdDraw(command_buffers[image_idx], 3, 1, 0, 0);
  }*/

  if (rendering_modes["unstructured"])
  {
    // Note: Second parameter can be also used to bind a compute pipeline
    vkCmdBindPipeline(command_buffers[image_idx],
      VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline
    );
    vkCmdBindDescriptorSets(command_buffers[image_idx],
      VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1,
      &descriptor_sets[image_idx], 0, nullptr
    );
    setUnstructuredRendering(command_buffers[image_idx], element_count);
  }

  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffers[image_idx]);

  // End render pass and finish recording the command buffer
  vkCmdEndRenderPass(command_buffers[image_idx]);
  validation::checkVulkan(vkEndCommandBuffer(command_buffers[image_idx]));

  updateUniformBuffer(image_idx);

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

  std::vector<VkSemaphore> wait_semaphores;
  std::vector<VkPipelineStageFlags> wait_stages;
  wait_semaphores.push_back(image_available[frame_idx]); //vk_timeline_semaphore
  wait_stages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
  getWaitFrameSemaphores(wait_semaphores, wait_stages);

  submit_info.waitSemaphoreCount = (uint32_t)wait_semaphores.size();
  submit_info.pWaitSemaphores = wait_semaphores.data();
  submit_info.pWaitDstStageMask = wait_stages.data();

  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffers[image_idx];

  std::vector<VkSemaphore> signal_semaphores;
  getSignalFrameSemaphores(signal_semaphores);
  signal_semaphores.push_back(render_finished[frame_idx]); // vk_timeline_semaphore
  submit_info.signalSemaphoreCount = (uint32_t)signal_semaphores.size();
  submit_info.pSignalSemaphores = signal_semaphores.data();

  /*VkTimelineSemaphoreSubmitInfo timeline_info{};
  timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  timeline_info.waitSemaphoreValueCount = 1;
  timeline_info.pWaitSemaphoreValues = &wait_value;
  timeline_info.signalSemaphoreValueCount = 1;
  timeline_info.pSignalSemaphoreValues = &signal_value;
  submit_info.pNext = &timeline_info;*/

  vkResetFences(device, 1, &inflight_fences[frame_idx]);

  // Execute command buffer using image as attachment in framebuffer
  validation::checkVulkan(vkQueueSubmit(
    graphics_queue, 1, &submit_info, inflight_fences[frame_idx]) //VK_NULL_HANDLE
  );

  // Return image result back to swapchain for presentation on screen
  VkSwapchainKHR swapchains[] = { swapchain };
  VkPresentInfoKHR present_info{};
  present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  present_info.waitSemaphoreCount = 1;
  //present_info.pWaitSemaphores = &vk_presentation_semaphore;
  present_info.pWaitSemaphores = &render_finished[frame_idx];
  present_info.swapchainCount = 1;
  present_info.pSwapchains = swapchains;
  present_info.pImageIndices = &image_idx;

  result = vkQueuePresentKHR(present_queue, &present_info);
  // Resize should be done after presentation to ensure semaphore consistency
  if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || should_resize)
  {
    recreateSwapchain();
    should_resize = false;
  }

  current_frame++;
}


void VulkanEngine::createRenderPass()
{
  VkAttachmentDescription color_attachment{};
  color_attachment.format = swapchain_format;
  color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference color_attachment_ref{};
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_attachment_ref;

  // Specify memory and execution dependencies between subpasses
  VkSubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.srcAccessMask = 0;
  dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  VkRenderPassCreateInfo renderpass_info{};
  renderpass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderpass_info.attachmentCount = 1;
  renderpass_info.pAttachments = &color_attachment;
  renderpass_info.subpassCount = 1;
  renderpass_info.pSubpasses = &subpass;
  renderpass_info.dependencyCount = 1;
  renderpass_info.pDependencies = &dependency;

  validation::checkVulkan(vkCreateRenderPass(
    device, &renderpass_info, nullptr, &render_pass)
  );
}

void VulkanEngine::setUnstructuredRendering(VkCommandBuffer& cmd_buffer,
  uint32_t vertex_count)
{
  VkBuffer vertex_buffers[] = { vertex_buffer };
  VkDeviceSize offsets[] = { 0 };
  auto binding_count = sizeof(vertex_buffers) / sizeof(vertex_buffers[0]);
  vkCmdBindVertexBuffers(cmd_buffer, 0, binding_count, vertex_buffers, offsets);
  vkCmdDraw(cmd_buffer, vertex_count, 1, 0, 0);
  // NOTE: For indexed drawing, use the following:
  //vkCmdBindIndexBuffer(command_buffers[i], index_buffer, 0, VK_INDEX_TYPE_UINT16);
  //auto index_count = static_cast<uint32_t>(indices.size());
  //vkCmdDrawIndexed(command_buffers[i], index_count, 1, 0, 0, 0);*/
}

void VulkanEngine::getVertexDescriptions(
  std::vector<VkVertexInputBindingDescription>& bind_desc,
  std::vector<VkVertexInputAttributeDescription>& attr_desc)
{
  bind_desc.resize(1);
  bind_desc[0].binding = 0;
  bind_desc[0].stride = sizeof(Vertex);
  bind_desc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  attr_desc.resize(2);
  attr_desc[0].binding = 0;
  attr_desc[0].location = 0;
  attr_desc[0].format = VK_FORMAT_R32G32_SFLOAT;
  attr_desc[0].offset = offsetof(Vertex, pos);
  attr_desc[1].binding = 0;
  attr_desc[1].location = 1;
  attr_desc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attr_desc[1].offset = offsetof(Vertex, color);
}

void VulkanEngine::getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo& info)
{
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  //info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  info.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  info.primitiveRestartEnable = VK_FALSE;
}
