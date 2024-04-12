#pragma once

#include <vulkan/vulkan.h>

#include <string> // std::string
#include <vector> // std::vector

#include <mimir/deletion_queue.hpp>
#include "physical_device.hpp"

namespace mimir
{

struct VulkanQueue
{
    uint32_t family_index = ~0u;
    VkQueue queue         = VK_NULL_HANDLE;
};

struct VulkanDevice
{
    // GPU used for Vulkan operations
    PhysicalDevice physical_device{};

    VkDevice logical_device = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;

    VulkanQueue graphics, present;
    DeletionQueue deletors;

    explicit VulkanDevice(PhysicalDevice dev);
    ~VulkanDevice();

    void initLogicalDevice(VkSurfaceKHR surface);
    VkCommandPool createCommandPool(uint32_t queue_idx, VkCommandPoolCreateFlags flags);
    VkDescriptorPool createDescriptorPool(const std::vector<VkDescriptorPoolSize>& sizes);

    std::vector<VkDescriptorSet> createDescriptorSets(
        VkDescriptorPool pool, VkDescriptorSetLayout layout, uint32_t set_count
    );
    VkDescriptorSetLayout createDescriptorSetLayout(
        const std::vector<VkDescriptorSetLayoutBinding>& layout_bindings
    );
    VkPipelineLayout createPipelineLayout(VkDescriptorSetLayout descriptor_layout);

    std::vector<VkCommandBuffer> createCommandBuffers(uint32_t buffer_count);
    void immediateSubmit(std::function<void(VkCommandBuffer cmd)>&& function);

    VkDeviceMemory allocateMemory(VkMemoryRequirements requirements,
        VkMemoryPropertyFlags properties, const void *export_info = nullptr
    );
    VkBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
        const void *extmem_info = nullptr
    );
    VkSampler createSampler(VkFilter filter, bool enable_anisotropy);
    VkFormat findSupportedImageFormat(const std::vector<VkFormat>& candidates,
        VkImageTiling tiling, VkFormatFeatureFlags features
    );
    void generateMipmaps(VkImage image, VkFormat img_format,
        int img_width, int img_height, int mip_levels
    );
    void transitionImageLayout(VkImage image,
        VkImageLayout old_layout, VkImageLayout new_layout
    );

    VkFence createFence(VkFenceCreateFlags flags);
    VkSemaphore createSemaphore(const void *extensions = nullptr);
    std::string readMemoryHeapFlags(VkMemoryHeapFlags flags);
    VkQueryPool createQueryPool(uint32_t query_count);
};

uint32_t getAlignedSize(size_t original_size, size_t min_alignment);

} // namespace mimir