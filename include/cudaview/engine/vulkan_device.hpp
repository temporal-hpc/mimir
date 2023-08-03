#pragma once

#include <vulkan/vulkan.h>

#include <string> // std::string
#include <vector> // std::vector

#include "cudaview/deletion_queue.hpp"

struct ConvertedMemory
{
    float data;
    std::string units;
};

struct DeviceMemoryProperties
{
    uint32_t heap_count = 0;
    VkDeviceSize gpu_usage = 0;
    VkDeviceSize gpu_budget = 0;
};

struct VulkanQueue
{
    uint32_t family_index = ~0u;
    VkQueue queue = VK_NULL_HANDLE;
};

struct VulkanDevice
{
    // GPU used for Vulkan operations
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties = {};
    VkPhysicalDeviceFeatures features = {};
    VkPhysicalDeviceMemoryProperties memory_properties = {};
    VkPhysicalDeviceMemoryProperties2 memory_properties2 = {};
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_properties = {};
    
    VkDevice logical_device = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;

    VulkanQueue graphics, present;
    DeviceMemoryProperties props;
    DeletionQueue deletors;

    explicit VulkanDevice(VkPhysicalDevice gpu);
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

    uint32_t findMemoryType(uint32_t type_filter, VkMemoryPropertyFlags mem_props);
    VkDeviceMemory allocateMemory(VkMemoryRequirements requirements,
        VkMemoryPropertyFlags properties, const void *export_info = nullptr
    );
    VkBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
        const void *extmem_info = nullptr
    );
    VkImage createImage(VkImageType type, VkFormat format, VkExtent3D extent,
        VkImageTiling tiling, VkImageUsageFlags usage, const void *extmem_info = nullptr
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
    ConvertedMemory formatMemory(uint64_t memsize) const; 
    std::string readMemoryHeapFlags(VkMemoryHeapFlags flags);
    void updateMemoryProperties();
    void listExtensions();
};

uint32_t getAlignedSize(size_t original_size, size_t min_alignment);
