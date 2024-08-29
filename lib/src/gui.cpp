#include "internal/gui.hpp"
#include "internal/camera.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <ImGuiFileDialog.h>

#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <array> // std::array

namespace mimir::gui
{

static std::array<ViewType, 4> kAllViewTypes = {
    ViewType::Markers,
    ViewType::Edges,
    ViewType::Voxels,
    ViewType::Image
};
/*
static std::array<ResourceType, 4> kAllResources = {
    ResourceType::Buffer,
    ResourceType::IndexBuffer,
    ResourceType::Texture,
    ResourceType::LinearTexture
};
static std::array<ComponentType, 7> kAllComponentTypes = {
    ComponentType::Int,
    ComponentType::Long,
    ComponentType::Short,
    ComponentType::Char,
    ComponentType::Float,
    ComponentType::Double,
    ComponentType::Half
};*/

struct AllResources
{
    static bool ItemGetter(void* data, int n, const char** out_str)
    {
        *out_str = getResourceType(((ResourceType*)data)[n]);
        return true;
    }
};
struct AllViewTypes
{
    static bool ItemGetter(void* data, int n, const char** out_str)
    {
        *out_str = getViewType(((ViewType*)data)[n]);
        return true;
    }
};
struct AllComponentTypes
{
    static bool ItemGetter(void* data, int n, const char** out_str)
    {
        *out_str = getComponentType(((ComponentType*)data)[n]);
        return true;
    }
};

std::string getExtent(uint3 extent, DataDomain domain)
{
    switch (domain)
    {
        case DataDomain::Domain2D:
        {
            return fmt::format("({},{})", extent.x, extent.y);
        }
        case DataDomain::Domain3D:
        {
            return fmt::format("({},{},{})", extent.x, extent.y, extent.z);
        }
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

void addViewObjectGui(std::shared_ptr<InteropView2> view_ptr, int uid)
{
    ImGui::PushID(view_ptr.get());
    auto& params = view_ptr->params;
    //bool node_open = ImGui::TreeNode("Object", "%s_%u", "View", uid);
    bool node_open = ImGui::CollapsingHeader("", ImGuiTreeNodeFlags_AllowItemOverlap);
    ImGui::SameLine(); ImGui::Text("%s #%u", "View", uid);
    ImGui::SameLine(ImGui::GetWindowWidth()-60); ImGui::Checkbox("show", &params.options.visible);
    if (node_open)
    {
        bool type_check = ImGui::Combo("View type", (int*)&params.view_type,
            &AllViewTypes::ItemGetter, kAllViewTypes.data(), kAllViewTypes.size()
        );
        if (type_check) spdlog::info("View {}: switched view type to {}", uid, getViewType(params.view_type));
        ImGui::SliderFloat("Element size (px)", &params.options.default_size, 1.f, 100.f);
        ImGui::ColorEdit4("Element color", (float*)&params.options.default_color);
        ImGui::SliderFloat("depth", &params.options.depth, 0.f, 1.f);

        int min_instance = 0;
        int max_instance = params.instance_count - 1;
        ImGui::SliderScalar("Instance index", ImGuiDataType_S32, &params.options.instance_index, &min_instance, &max_instance);

        if (ImGui::BeginTable("split", 2, ImGuiTableFlags_BordersOuter | ImGuiTableFlags_Resizable))
        {
            addTableRow("Data domain", getDataDomain(params.data_domain));
            addTableRow("Data extent", getExtent(params.extent, params.data_domain));
            ImGui::EndTable();
        }
        for (const auto &[type, attr] : params.attributes)
        {
            if (ImGui::BeginTable("split", 2, ImGuiTableFlags_BordersOuter | ImGuiTableFlags_Resizable))
            {
                //addTableRow("Element count", std::to_string(info.element_count));
                addTableRow("Attribute type", getAttributeType(type));
                //addTableRow("Resource type", getResourceType(info.resource_type));
                addTableRow("Data type", getDataType(attr.format.type));
                addTableRow("Channel count", std::to_string(attr.format.components));
                //addTableRow("Data layout", getDataLayout(attr.layout));

                /*bool res_check = addTableRowCombo("Resource type", (int*)&info.resource_type,
                    &AllResources::ItemGetter, kAllResources.data(), kAllResources.size()
                );
                if (res_check) printf("View %d: switched resource type to %s\n", uid, getResourceType(info.resource_type));
                bool data_check = addTableRowCombo("Data type", (int*)&info.component_type,
                    &AllComponentTypes::ItemGetter, kAllComponentTypes.data(), kAllComponentTypes.size()
                );
                if (data_check) printf("View %d: switched data type to %s\n", uid, getComponentType(info.component_type));*/

                ImGui::EndTable();
            }
        }
        //ImGui::TreePop();
    }
    ImGui::PopID();
}

void draw(Camera* cam, ViewerOptions& opts, std::span<std::shared_ptr<InteropView2>> views, const std::function<void(void)>& callback)
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (opts.show_demo_window) { ImGui::ShowDemoWindow(); }
    if (opts.show_metrics) { ImGui::ShowMetricsWindow(); }

    ImGui::Begin("Scene parameters");
    //ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / framerate, framerate);
    //ImGui::ColorEdit3("Clear color", (float*)&bg_color);
    ImGui::InputFloat3("Camera position", &cam->position.x, "%.3f");
    ImGui::InputFloat3("Camera rotation", &cam->rotation.x, "%.3f");

    // Use a separate flag for choosing whether to enable the FPS limit target value
    // This avoids the unpleasant feeling of going from 0 (no FPS limit)
    // to 1 (the lowest value) in a single step
    if (ImGui::Checkbox("Enable FPS limit", &opts.enable_fps_limit))
    {
        //target_frame_time = getTargetFrameTime(opts.enable_fps_limit, opts.target_fps);
    }
    if (!opts.enable_fps_limit) ImGui::BeginDisabled(true);
    if (ImGui::SliderInt("FPS target", &opts.target_fps, 1, opts.max_fps, "%d%", ImGuiSliderFlags_AlwaysClamp))
    {
        //target_frame_time = getTargetFrameTime(opts.enable_fps_limit, opts.target_fps);
    }
    if (!opts.enable_fps_limit) ImGui::EndDisabled();

    for (size_t i = 0; i < views.size(); ++i)
    {
        addViewObjectGui(views[i], i);
    }
    ImGui::End();

    callback(); // Display user-provided addons
    ImGui::Render();
}

void init(InteropDevice& dev, VkInstance instance, VkDescriptorPool pool, VkRenderPass pass, GlfwContext *win_ctx)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForVulkan(win_ctx->window, true);

    ImGui_ImplVulkan_InitInfo info{
        .Instance        = instance,
        .PhysicalDevice  = dev.physical_device.handle,
        .Device          = dev.logical_device,
        .QueueFamily     = dev.graphics.family_index,
        .Queue           = dev.graphics.queue,
        .DescriptorPool  = pool,
        .RenderPass      = pass,
        .MinImageCount   = 3, // TODO: Check if this is true
        .ImageCount      = 3,
        .MSAASamples     = VK_SAMPLE_COUNT_1_BIT,
        .PipelineCache   = nullptr,
        .Subpass         = 0,
        .UseDynamicRendering = false,
        .PipelineRenderingCreateInfo = {},
        .Allocator         = nullptr,
        .CheckVkResultFn   = nullptr,
        .MinAllocationSize = 0,
    };
    ImGui_ImplVulkan_Init(&info);
}

void render(VkCommandBuffer cmd)
{
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

void shutdown()
{
    ImGui_ImplVulkan_Shutdown();
}

} // namespace mimir::gui