#pragma once

#include <span> // std::span

#include <mimir/mimir.hpp>
#include "window.hpp"

namespace mimir::gui
{

void init(InteropDevice& dev, VkInstance instance, VkDescriptorPool pool, VkRenderPass pass, GlfwContext *win_ctx);
void shutdown();
void render(VkCommandBuffer cmd);
void draw(Camera* cam, ViewerOptions& opts, std::span<std::shared_ptr<InteropView>> views, const std::function<void(void)>& callback);
void addViewObjectGui(std::shared_ptr<InteropView> view, int uid);

} // namespace mimir::gui