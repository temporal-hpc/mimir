#pragma once

#include <mimir/view.hpp>
#include "engine.hpp"

#include <span> // std::span

namespace mimir::gui
{

void init(VkInstance instance, VkPhysicalDevice ph_dev, VkDevice device, VkDescriptorPool pool,
    VkRenderPass pass, VulkanQueue queue, const GlfwContext& win_ctx
);
void shutdown();
void render(VkCommandBuffer cmd);

// Live stats for the always-on path-tracing overlay (drawn only when path_tracing is true).
struct HudStats
{
    bool path_tracing = false;
    float fps = 0.f;
    float tlas_ms = 0.f;   // TLAS rebuild (instance writer + build), last frame
    float trace_ms = 0.f;  // vkCmdTraceRays, last frame
    unsigned int spp = 0;
    unsigned int bounces = 0;
};

void draw(Camera& cam, ViewerOptions& opts, std::span<View*> views,
    const std::function<void(void)>& callback, const HudStats& stats = {}
);
void handleResize(uint32_t image_count);
void addViewObjectGui(View *view_ptr, int uid);

} // namespace mimir::gui