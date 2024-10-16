#pragma once

#include "engine.hpp"

#include <span> // std::span

namespace mimir::gui
{

void init(VkInstance instance, VkPhysicalDevice ph_dev, VkDevice device, VkDescriptorPool pool,
    VkRenderPass pass, VulkanQueue queue, const GlfwContext& win_ctx
);
void shutdown();
void render(VkCommandBuffer cmd);
void draw(Camera& cam, ViewerOptions& opts, std::span<InteropView*> views,
    const std::function<void(void)>& callback
);
void handleResize(uint32_t image_count);
void addViewObjectGui(std::shared_ptr<InteropView> view, int uid);

} // namespace mimir::gui