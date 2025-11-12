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

#include <limits> // std::numeric_limits

namespace mimir::gui
{

std::string formatLayout(Layout layout)
{
    int dim_count = (layout.x > 1) + (layout.y > 1) + (layout.z > 1);
    switch (dim_count)
    {
        case 3: { return fmt::format("({},{},{})", layout.x, layout.y, layout.z); }
        case 2: { return fmt::format("({},{})", layout.x, layout.y); }
        case 1: default: { return fmt::format("{}", layout.x, layout.y); }
    }
}

// Helper for adding a GUI table row showing a combo box for setting values at runtime
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

// Helper for adding a GUI table row showing static info
void addTableRow(const std::string& key, const std::string& value)
{
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::AlignTextToFramePadding();
    ImGui::Text("%s", key.c_str());
    ImGui::TableSetColumnIndex(1);
    ImGui::Text("%s", value.c_str());
}

void addTableRowOptions(ViewType type, ViewOptions options)
{
    switch (type)
    {
        case ViewType::Markers:
        {
            auto marker = std::get<MarkerOptions>(options);
            addTableRow("Shape", getMarkerShape(marker.shape));
            break;
        }
        case ViewType::Edges:
        {
            auto mesh = std::get<MeshOptions>(options);
            addTableRow("Periodic", mesh.periodic? "yes" : "no");
            break;
        }
        // Don't know / don't care
        default: { return; }
    }
}

void addViewGUI(View *handle, int uid)
{
    ImGui::PushID(handle);
    auto& desc = handle->desc;
    bool node_open = ImGui::CollapsingHeader("", ImGuiTreeNodeFlags_AllowItemOverlap);
    ImGui::SameLine(); ImGui::Text("%s #%u", "View", uid);
    ImGui::SameLine(ImGui::GetWindowWidth()-60); ImGui::Checkbox("show", &desc.visible);
    if (node_open)
    {
        ImGui::ColorEdit4("Element color", &desc.default_color.x);
        const float f32_zero = 0.f;
        const float f32_max  = 10000.f;
        ImGui::DragScalar("Element size", ImGuiDataType_Float, &desc.default_size,
            0.005f, &f32_zero, &f32_max, "%f", ImGuiSliderFlags_Logarithmic
        );
        ImGui::DragScalar("Line width", ImGuiDataType_Float, &desc.linewidth,
            0.005f, &f32_zero, &f32_max, "%f", ImGuiSliderFlags_Logarithmic
        );
        ImGui::DragScalar("Antialias", ImGuiDataType_Float, &desc.antialias,
            0.005f, &f32_zero, &f32_max, "%f", ImGuiSliderFlags_Logarithmic
        );

        bool view_translated = ImGui::InputFloat3("Position", &desc.position.x, "%.3f");
        bool view_rotated    = ImGui::InputFloat3("Rotation", &desc.rotation.x, "%.3f");
        bool view_scaled     = ImGui::InputFloat3("Scale", &desc.scale.x, "%.3f");

        if (view_translated) { translateView(handle, desc.position); }
        if (view_rotated)    { rotateView(handle, desc.rotation); }
        if (view_scaled)     { scaleView(handle, desc.scale); }

        ImGuiTableFlags table_flags = ImGuiTableFlags_BordersOuter | ImGuiTableFlags_Resizable;
        if (ImGui::BeginTable("split", 2, table_flags))
        {
            addTableRow("Type",   getViewType(desc.type));
            addTableRow("Domain", getDomainType(desc.domain));
            addTableRow("Layout", formatLayout(desc.layout));
            addTableRow("Style",  getShapeStyle(desc.style));
            addTableRowOptions(desc.type, desc.options);
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

        ImGui::InputFloat3("Light position",  (float*)&opts.light_pos, "%.3f");
        ImGui::ColorEdit3("Light color",      (float*)&opts.light_color);
        ImGui::ColorEdit3("Specular color",   (float*)&opts.specular_color);
        ImGui::InputFloat("Specular power",   &opts.specular_power);
        ImGui::InputFloat("Ambient strength", &opts.ambient_strength);

        //ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / framerate, framerate);
        ImGui::ColorEdit3("Clear color", (float*)&opts.background_color);

        bool camera_moved = ImGui::InputFloat3("Camera position", &cam.position.x, "%.3f");
        if (camera_moved) { cam.setPosition(cam.position); }

        bool camera_rotated = ImGui::InputFloat3("Camera rotation", &cam.rotation.x, "%.3f");
        if (camera_rotated) { cam.setRotation(cam.rotation); }

        const float f32_zero = 0.f;
        const float f32_max  = 360.f;
        bool fov_changed = ImGui::DragScalar(
            "FOV", ImGuiDataType_Float, &cam.fov, 0.005f, &f32_zero, &f32_max, "%.3f"
        );
        bool znear_changed = ImGui::InputFloat("Near plane", &cam.near_clip);
        bool zfar_changed  = ImGui::InputFloat("Far plane", &cam.far_clip);
        if (fov_changed || znear_changed || zfar_changed)
        {
            float aspect = (float)opts.window.size.x / (float)opts.window.size.y;
            cam.setPerspective(cam.fov, aspect, cam.near_clip, cam.far_clip);
        }

        // Use a separate flag for choosing whether to enable the FPS limit target value
        // This avoids the unpleasant feeling of going from 0 (no FPS limit)
        // to 1 (the lowest value) in a single step
        auto& op = opts.present;
        ImGui::Checkbox("Enable FPS limit", &op.enable_fps_limit);
        ImGui::BeginDisabled(!opts.present.enable_fps_limit);
        ImGuiSliderFlags slider_flags = ImGuiSliderFlags_AlwaysClamp;
        if (ImGui::SliderInt("FPS target", &op.target_fps, 1, 240, "%d%", slider_flags))
        {
            op.target_frame_time = getTargetFrameTime(op.enable_fps_limit, op.target_fps);
        }
        ImGui::EndDisabled();

        // Add tabs for showing view parameters
        for (size_t i = 0; i < views.size(); ++i) { addViewGUI(views[i], i); }
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