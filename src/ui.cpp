#include <mimir/mimir.hpp>
#include "internal/camera.hpp"
#include "internal/framelimit.hpp" // getTargetFrametime

#include <imgui.h>
#include <ImGuiFileDialog.h>

#include <array> // std::array
#include <algorithm> // std::clamp
#include <cstdio> // std::sprintf
#include <stdexcept>// std::runtime_error

namespace mimir
{

struct AllResources
{
    static bool ItemGetter(void* data, int n, const char** out_str)
    {
        *out_str = getResourceType(((ResourceType*)data)[n]);
        return true;
    }
};
struct AllDomains
{
    static bool ItemGetter(void* data, int n, const char** out_str)
    {
        *out_str = getDomainType(((DomainType*)data)[n]);
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

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

std::string getExtent(uint3 extent, DataDomain domain)
{
    switch (domain)
    {
        case DataDomain::Domain2D:
        {
            return string_format("(%d,%d)", extent.x, extent.y);
        }
        case DataDomain::Domain3D:
        {
            return string_format("(%d,%d,%d)", extent.x, extent.y, extent.z);
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

void addViewObjectGui(InteropView *view_ptr, int uid)
{
    ImGui::PushID(view_ptr);
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
        if (type_check) printf("View %d: switched view type to %s\n", uid, getViewType(params.view_type));
        bool dom_check = ImGui::Combo("Domain type", (int*)&params.domain_type,
            &AllDomains::ItemGetter, kAllDomains.data(), kAllDomains.size()
        );
        if (dom_check) printf("View %d: switched domain type to %s\n", uid, getDomainType(params.domain_type));
        ImGui::SliderFloat("Element size (px)", &params.options.default_size, 1.f, 100.f);
        ImGui::ColorEdit4("Element color", (float*)&params.options.default_color);
        ImGui::SliderFloat("depth", &params.options.depth, 0.f, 1.f);
        if (ImGui::BeginTable("split", 2, ImGuiTableFlags_BordersOuter | ImGuiTableFlags_Resizable))
        {
            addTableRow("Data domain", getDataDomain(params.data_domain));
            addTableRow("Data extent", getExtent(params.extent, params.data_domain));
            ImGui::EndTable();
        }
        for (const auto &[attr, memory] : params.attributes)
        {
            if (ImGui::BeginTable("split", 2, ImGuiTableFlags_BordersOuter | ImGuiTableFlags_Resizable))
            {
                auto& info = memory.params;
                //addTableRow("Element count", std::to_string(info.element_count));
                addTableRow("Attribute type", getAttributeType(attr));
                addTableRow("Resource type", getResourceType(info.resource_type));
                addTableRow("Data type", getComponentType(info.component_type));
                addTableRow("Channel count", std::to_string(info.channel_count));
                addTableRow("Data layout", getDataLayout(info.layout));

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

void MimirEngine::displayEngineGUI()
{
    ImGui::Begin("Scene parameters");
    //ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / framerate, framerate);
    ImGui::ColorEdit3("Clear color", (float*)&bg_color);
    auto pos = camera->position;
    ImGui::Text("Camera position: %.3f %.3f %.3f", pos.x, pos.y, pos.z);
    auto rot = camera->rotation;
    ImGui::Text("Camera rotation: %.3f %.3f %.3f", rot.x, rot.y, rot.z);

    // Use a separate flag for choosing whether to enable the FPS limit target value
    // This avoids the unpleasant feeling of going from 0 (no FPS limit)
    // to 1 (the lowest value) in a single step
    if (ImGui::Checkbox("Enable FPS limit", &options.enable_fps_limit))
    {
        target_frame_time = getTargetFrameTime(options.enable_fps_limit, options.target_fps);
    }
    if (!options.enable_fps_limit) ImGui::BeginDisabled(true);
    if (ImGui::SliderInt("FPS target", &options.target_fps, 1, max_fps, "%d%", ImGuiSliderFlags_AlwaysClamp))
    {
        target_frame_time = getTargetFrameTime(options.enable_fps_limit, options.target_fps);
    }
    if (!options.enable_fps_limit) ImGui::EndDisabled();

    for (size_t i = 0; i < views.size(); ++i)
    {
        addViewObjectGui(views[i].get(), i);
    }
    ImGui::End();
}

MimirEngine *getHandler(GLFWwindow *window)
{
    return reinterpret_cast<MimirEngine*>(glfwGetWindowUserPointer(window));
}

void MimirEngine::setBackgroundColor(float4 color)
{
    bg_color = color;
}

// Translates GLFW mouse movement into Viewer flags for detecting camera movement
void MimirEngine::cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
{
    auto app = getHandler(window);
    auto dx = app->mouse_pos.x - static_cast<float>(xpos);
    auto dy = app->mouse_pos.y - static_cast<float>(ypos);

    if (app->mouse_buttons.left)
    {
        auto rot = app->camera->rotation_speed;
        app->camera->rotate(glm::vec3(dy * rot, -dx * rot, 0.f));
        app->view_updated = true;
    }
    if (app->mouse_buttons.right)
    {
        app->camera->translate(glm::vec3(0.f, 0.f, dy * .005f));
    }
    if (app->mouse_buttons.middle)
    {
        app->camera->translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.f));
    }
    app->mouse_pos = make_float2(xpos, ypos);    
}

// Translates GLFW mouse actions into Viewer flags for detecting camera actions
void MimirEngine::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    auto app = getHandler(window);
    // Perform action only if GUI does not want to use mouse input
    // (if not hovering over a menu item)
    if (!ImGui::GetIO().WantCaptureMouse)
    {
        if (action == GLFW_PRESS)
        {
            if (button == GLFW_MOUSE_BUTTON_MIDDLE) app->mouse_buttons.middle = true;
            else if (button == GLFW_MOUSE_BUTTON_LEFT) app->mouse_buttons.left = true;
            else if (button == GLFW_MOUSE_BUTTON_RIGHT) app->mouse_buttons.right = true;
        }
        else if (action == GLFW_RELEASE)
        {
            if (button == GLFW_MOUSE_BUTTON_MIDDLE) app->mouse_buttons.middle = false;
            else if (button == GLFW_MOUSE_BUTTON_LEFT) app->mouse_buttons.left = false;
            else if (button == GLFW_MOUSE_BUTTON_RIGHT) app->mouse_buttons.right = false;
        }
    }
}

void MimirEngine::framebufferResizeCallback(GLFWwindow *window,[[maybe_unused]] int width,[[maybe_unused]] int height)
{
    auto app = getHandler(window);
    app->should_resize = true;
}

void MimirEngine::keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    auto app = getHandler(window);
    // Toggle demo window
    if (key == GLFW_KEY_D && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL)
    {
        app->show_demo_window = !app->show_demo_window;
    }
    // Toggle metrics windows
    if (key == GLFW_KEY_M && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL)
    {
        app->options.show_metrics = !app->options.show_metrics;
    }
}

void MimirEngine::windowCloseCallback(GLFWwindow *window)
{
    //printf("Handling window close\n");
    auto engine = getHandler(window);
    engine->running = false;
    engine->signalKernelFinish();
}

void MimirEngine::showMetrics()
{
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);

    auto frame_sample_size = std::min(frame_times.size(), total_frame_count);
    float total_frame_time = 0;
    for (size_t i = 0; i < frame_sample_size; ++i) total_frame_time += frame_times[i];
    auto framerate = frame_times.size() / total_frame_time;
    //float min_fps = 1 / max_frame_time;
    //float max_fps = 1 / min_frame_time;

    dev->updateMemoryProperties();
    auto gpu_usage = dev->formatMemory(dev->props.gpu_usage);
    auto gpu_budget = dev->formatMemory(dev->props.gpu_budget);

    std::string label;
    if (w == 0 && h == 0) label = "None";
    else if (w == 1920 && h == 1080) label = "FHD";
    else if (w == 2560 && h == 1440) label = "QHD";
    else if (w == 3840 && h == 2160) label = "UHD";

    printf("%s,%d,%f,%f,%lf,%f,%f,%f,", label.c_str(), options.target_fps,
        framerate,perf.total_compute_time,total_pipeline_time,
        total_graphics_time,gpu_usage.data,gpu_budget.data
    );

    //auto fps = ImGui::GetIO().Framerate; printf("\nFPS %f\n", fps);
    //getTimeResults();
    /*printf("Framebuffer size: %dx%d\n", w, h);
    printf("Average frame rate over 120 frames: %.2f FPS\n", framerate);

    dev->updateMemoryProperties();
    auto gpu_usage = dev->formatMemory(dev->props.gpu_usage);
    printf("GPU memory usage: %.2f %s\n", gpu_usage.data, gpu_usage.units.c_str());
    auto gpu_budget = dev->formatMemory(dev->props.gpu_budget);
    printf("GPU memory budget: %.2f %s\n", gpu_budget.data, gpu_budget.units.c_str());
    //this->exit();

    auto props = dev->budget_properties;
    for (int i = 0; i < static_cast<int>(dev->props.heap_count); ++i)
    {
        auto heap_usage = dev->formatMemory(props.heapUsage[i]);
        printf("Heap %d usage: %.2f %s\n", i, heap_usage.data, heap_usage.units.c_str());
        auto heap_budget = dev->formatMemory(props.heapBudget[i]);
        printf("Heap %d budget: %.2f %s\n", i, heap_budget.data, heap_budget.units.c_str());
        auto heap_flags = dev->memory_properties2.memoryProperties.memoryHeaps[i].flags;
        printf("Heap %d flags: %s\n", i, dev->readMemoryHeapFlags(heap_flags).c_str());
    }*/
}

} // namespace mimir