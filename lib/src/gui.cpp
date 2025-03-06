#include <mimir/mimir.hpp>
#include "mimir/gui.hpp"
#include "mimir/api.hpp"
#include "mimir/framelimit.hpp"
#include "mimir/camera.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

#include <array> // std::array

namespace mimir::gui
{

std::string getExtent(ViewExtent ext, DomainType domain)
{
    switch (domain)
    {
        case DomainType::Domain2D: { return fmt::format("({},{})", ext.x, ext.y); }
        case DomainType::Domain3D: { return fmt::format("({},{},{})", ext.x, ext.y, ext.z); }
        default: return "unknown";
    }
}

void addTableRow(const std::string& key, const std::string& value)
{
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::AlignTextToFramePadding();
    ImGui::Text("%s", key.c_str());
    ImGui::TableSetColumnIndex(1);
    ImGui::Text("%s", value.c_str());
}

bool addTableRowCombo(const std::string& key, int* current_item,
    bool(*items_getter)(void* data, int idx, const char** out_text),
    void* data, int items_count)
{
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::AlignTextToFramePadding();
    ImGui::Text("%s", key.c_str());
    ImGui::TableSetColumnIndex(1);
    return ImGui::Combo(key.c_str(), current_item, items_getter, data, items_count);
}

void addViewObjectGui(View *view_ptr, int uid)
{
    ImGui::PushID(view_ptr);
    auto& desc = view_ptr->desc;
    //bool node_open = ImGui::TreeNode("Object", "%s_%u", "View", uid);
    bool node_open = ImGui::CollapsingHeader("", ImGuiTreeNodeFlags_AllowItemOverlap);
    ImGui::SameLine(); ImGui::Text("%s #%u", "View", uid);
    ImGui::SameLine(ImGui::GetWindowWidth()-60); ImGui::Checkbox("show", &desc.visible);
    if (node_open)
    {
        const float f32_zero = 0.f;
        const float f32_max  = 100.f;
        ImGui::DragScalar("Element size", ImGuiDataType_Float, &desc.default_size,
            0.005f, &f32_zero, &f32_max, "%f", ImGuiSliderFlags_Logarithmic
        );
        ImGui::ColorEdit4("Element color", &desc.default_color.x);
        // ImGui::SliderFloat("depth", &desc.options.depth, 0.f, 1.f);

        // if (desc.offsets.size() > 0)
        // {
        //     int min_scenario = 0;
        //     int max_scenario = desc.offsets.size() - 1;
        //     ImGui::SliderScalar("scenario", ImGuiDataType_S32, &desc.options.scenario_index, &min_scenario, &max_scenario);
        // }

        if (ImGui::InputFloat3("Position", &desc.position.x, "%.3f"))
        {
            translateView(view_ptr, desc.position);
        }
        if (ImGui::InputFloat3("Scale", &desc.scale.x, "%.3f"))
        {
            scaleView(view_ptr, desc.scale);
        }

        ImGuiTableFlags table_flags = ImGuiTableFlags_BordersOuter | ImGuiTableFlags_Resizable;
        if (ImGui::BeginTable("split", 2, table_flags))
        {
            addTableRow("View type",     getViewType(desc.view_type));
            addTableRow("Element count", std::to_string(desc.element_count));
            addTableRow("Domain type",   getDomainType(desc.domain_type));
            addTableRow("Domain extent", getExtent(desc.extent, desc.domain_type));
            ImGui::EndTable();
        }
        for (const auto &[type, attr] : desc.attributes)
        {
            if (ImGui::BeginTable("split", 2, table_flags))
            {
                addTableRow("Element count",  std::to_string(attr.size));
                addTableRow("Attribute type", getAttributeType(type));
                addTableRow("Data type",      getDataType(attr.format));
                addTableRow("Channel count",  std::to_string(attr.format.components));
                ImGui::EndTable();
            }
        }
        //ImGui::TreePop();
    }
    ImGui::PopID();
}

void draw(Camera& cam, ViewerOptions& opts, std::span<View*> views,
    const std::function<void(void)>& callback)
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (opts.show_demo_window) { ImGui::ShowDemoWindow(); }
    if (opts.show_metrics) { ImGui::ShowMetricsWindow(); }

    if (opts.show_panel)
    {
        ImGui::Begin("Scene parameters");
        //ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / framerate, framerate);
        ImGui::ColorEdit3("Clear color", (float*)&opts.background_color);
        if (ImGui::InputFloat3("Camera position", &cam.position.x, "%.3f"))
        {
            cam.setPosition(cam.position);
        }
        if (ImGui::InputFloat3("Camera rotation", &cam.rotation.x, "%.3f"))
        {
            cam.setRotation(cam.rotation);
        }

        // Use a separate flag for choosing whether to enable the FPS limit target value
        // This avoids the unpleasant feeling of going from 0 (no FPS limit)
        // to 1 (the lowest value) in a single step
        auto& op = opts.present;
        ImGui::Checkbox("Enable FPS limit", &op.enable_fps_limit);
        ImGui::BeginDisabled(!opts.present.enable_fps_limit);
        ImGuiSliderFlags slider_flags = ImGuiSliderFlags_AlwaysClamp;
        if (ImGui::SliderInt("FPS target", &op.target_fps, 1, op.max_fps, "%d%", slider_flags))
        {
            op.target_frame_time = getTargetFrameTime(op.enable_fps_limit, op.target_fps);
        }
        ImGui::EndDisabled();

        // Add tabs for showing view parameters
        for (size_t i = 0; i < views.size(); ++i) { addViewObjectGui(views[i], i); }
        ImGui::End();
        callback(); // Display user-provided addons
    }

    ImGui::Render();
}

void init(VkInstance instance, VkPhysicalDevice ph_dev, VkDevice device, VkDescriptorPool pool,
    VkRenderPass pass, VulkanQueue queue, const GlfwContext& win_ctx)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForVulkan(win_ctx.window, true);

    ImGui_ImplVulkan_InitInfo info{
        .Instance                    = instance,
        .PhysicalDevice              = ph_dev,
        .Device                      = device,
        .QueueFamily                 = queue.family_index,
        .Queue                       = queue.queue,
        .DescriptorPool              = pool,
        .RenderPass                  = pass,
        .MinImageCount               = 3, // TODO: Check if this is true
        .ImageCount                  = 3,
        .MSAASamples                 = VK_SAMPLE_COUNT_1_BIT,
        .PipelineCache               = nullptr,
        .Subpass                     = 0,
        .DescriptorPoolSize          = 0,
        .UseDynamicRendering         = false,
        .PipelineRenderingCreateInfo = {},
        .Allocator                   = nullptr,
        .CheckVkResultFn             = nullptr,
        .MinAllocationSize           = 0,
    };
    ImGui_ImplVulkan_Init(&info);
}

void render(VkCommandBuffer cmd)
{
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

void handleResize(uint32_t image_count)
{
    ImGui_ImplVulkan_SetMinImageCount(image_count);
}

void shutdown()
{
    ImGui_ImplVulkan_Shutdown();
}

} // namespace mimir::gui