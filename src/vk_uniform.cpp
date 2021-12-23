#include "cudaview/vk_engine.hpp"
#include "cudaview/vk_types.hpp"
#include "vk_initializers.hpp"
#include "validation.hpp"

#include "glm/gtc/type_ptr.hpp"

#include <cstring> // memcpy

void VulkanEngine::createUniformBuffers()
{
  VkDeviceSize buffer_size = sizeof(ModelViewProjection) + 2 * 0x40;
  auto img_count = swapchain_images.size();
  uniform_buffers.resize(img_count);
  ubo_memory.resize(img_count);

  // TODO: Use a single buffer of "img_count * buffer_size" bytes
  for (size_t i = 0; i < img_count; ++i)
  {
    createBuffer(buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      uniform_buffers[i], ubo_memory[i]
    );
  }
}

void VulkanEngine::updateUniformBuffer(uint32_t image_index)
{
  ModelViewProjection ubo{};
  ubo.model = glm::mat4(1.f);
  ubo.view  = camera.matrices.view; // glm::mat4(1.f);
  ubo.proj  = camera.matrices.perspective; //glm::mat4(1.f);

  void *data = nullptr;
  vkMapMemory(device, ubo_memory[image_index], 0, sizeof(ubo), 0, &data);
  memcpy(data, &ubo, sizeof(ubo));
  vkUnmapMemory(device, ubo_memory[image_index]);

  ColorParams colors{};
  colors.point_color = setColor(point_color);
  colors.edge_color  = setColor(edge_color);
  data = nullptr;
  vkMapMemory(device, ubo_memory[image_index], sizeof(ubo),
    sizeof(ColorParams), 0, &data
  );
  memcpy(data, &colors, sizeof(colors));
  vkUnmapMemory(device, ubo_memory[image_index]);

  SceneParams params{};
  params.extent = glm::ivec3{data_extent.x, data_extent.y, data_extent.z};
  // TODO: Merge mappings
  data = nullptr;
  vkMapMemory(device, ubo_memory[image_index], sizeof(ubo) + 0x40,
    sizeof(SceneParams), 0, &data
  );
  memcpy(data, &params, sizeof(params));
  vkUnmapMemory(device, ubo_memory[image_index]);
}

void VulkanEngine::createDescriptorSets()
{
  auto img_count = swapchain_images.size();
  descriptor_sets.resize(img_count);

  std::vector<VkDescriptorSetLayout> layouts(img_count, descriptor_layout);
  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool     = descriptor_pool;
  alloc_info.descriptorSetCount = static_cast<uint32_t>(img_count);
  alloc_info.pSetLayouts        = layouts.data();

  validation::checkVulkan(
    vkAllocateDescriptorSets(device, &alloc_info, descriptor_sets.data())
  );
}

void VulkanEngine::createDescriptorSetLayout()
{
  auto ubo_layout = vkinit::descriptorLayoutBinding(0, // binding
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT
  );
  auto extent_layout = vkinit::descriptorLayoutBinding(1, // binding
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT
  );
  auto point_color_layout = vkinit::descriptorLayoutBinding(2, // binding
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT
  );
  auto sampler_layout = vkinit::descriptorLayoutBinding(3, // binding
    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT
  );

  std::array bindings{ubo_layout, extent_layout, point_color_layout, sampler_layout};

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = bindings.size();
  layout_info.pBindings    = bindings.data();

  validation::checkVulkan(vkCreateDescriptorSetLayout(
    device, &layout_info, nullptr, &descriptor_layout)
  );
}

void VulkanEngine::updateDescriptorSets()
{
  for (size_t i = 0; i < descriptor_sets.size(); ++i)
  {
    // Write MVP matrix, scene info and texture samplers
    std::vector<VkWriteDescriptorSet> desc_writes;
    desc_writes.reserve(3 + structured_buffers.size());

    VkDescriptorBufferInfo mvp_info{};
    mvp_info.buffer = uniform_buffers[i];
    mvp_info.offset = 0;
    mvp_info.range  = sizeof(ModelViewProjection);

    auto write_mvp = vkinit::writeDescriptorBuffer(
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptor_sets[i], &mvp_info, 0
    );
    desc_writes.push_back(write_mvp);

    VkDescriptorBufferInfo pcolor_info{};
    pcolor_info.buffer = uniform_buffers[i];
    pcolor_info.offset = sizeof(ModelViewProjection);
    pcolor_info.range  = sizeof(ColorParams);

    auto write_pcolor = vkinit::writeDescriptorBuffer(
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptor_sets[i], &pcolor_info, 2
    );
    desc_writes.push_back(write_pcolor);

    VkDescriptorBufferInfo extent_info{};
    extent_info.buffer = uniform_buffers[i];
    extent_info.offset = sizeof(ModelViewProjection) + 0x40;
    extent_info.range  = sizeof(SceneParams);

    auto write_scene = vkinit::writeDescriptorBuffer(
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptor_sets[i], &extent_info, 1
    );
    desc_writes.push_back(write_scene);

    for (const auto& buffer : structured_buffers)
    {
      VkDescriptorImageInfo img_info{};
      img_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      img_info.imageView   = buffer.vk_view;
      img_info.sampler     = texture_sampler;

      auto write_tex = vkinit::writeDescriptorImage(
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, descriptor_sets[i], &img_info, 3
      );
      desc_writes.push_back(write_tex);
    }

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(desc_writes.size()),
      desc_writes.data(), 0, nullptr
    );
  }
}
