#include <mimir/mimir.hpp>
#include "internal/camera.hpp"
#include "internal/framelimit.hpp"

#include <imgui.h>
#include <ImGuiFileDialog.h>

#include <array> // std::array
#include <algorithm> // std::clamp
#include <cstdio> // std::sprintf
#include <stdexcept>// std::runtime_error

namespace mimir
{

std::array<ResourceType, 3> kAllResources = {
    ResourceType::Buffer,
    ResourceType::IndexBuffer,
    ResourceType::Texture
};
std::array<DomainType, 2> kAllDomains = {
    DomainType::Structured,
    DomainType::Unstructured
};
std::array<ElementType, 4> kAllElements = {
    ElementType::Markers,
    ElementType::Edges,
    ElementType::Voxels,
    ElementType::Image
};
std::array<DataType, 6> kAllDataTypes = {
    DataType::Int,
    DataType::Long,
    DataType::Short,
    DataType::Char,
    DataType::Float,
    DataType::Double
};

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
struct AllElemTypes
{
    static bool ItemGetter(void* data, int n, const char** out_str)
    {
        *out_str = getElementType(((ElementType*)data)[n]);
        return true;
    }
};
struct AllDataTypes
{
    static bool ItemGetter(void* data, int n, const char** out_str)
    {
        *out_str = getDataType(((DataType*)data)[n]);
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

void addViewObjectGui(InteropView2 *view_ptr, int uid)
{
    ImGui::PushID(view_ptr);
    bool node_open = ImGui::TreeNode("Object", "%s_%u", "View", uid);
    if (node_open)
    {
        auto& params = view_ptr->params;
        ImGui::Checkbox("show", &params.options.visible);
        bool type_check = ImGui::Combo("View type", (int*)&params.view_type,
            &AllElemTypes::ItemGetter, kAllElements.data(), kAllElements.size()
        );
        if (type_check) printf("View %d: switched view type to %s\n", uid, getViewType(params.view_type));
        bool dom_check = ImGui::Combo("Domain type", (int*)&params.domain_type,
            &AllDomains::ItemGetter, kAllDomains.data(), kAllDomains.size()
        );       
        if (dom_check) printf("View %d: switched domain type to %s\n", uid, getDomainType(params.domain_type));
        ImGui::SliderFloat("Element size (px)", &params.options.default_size, 1.f, 100.f);
        ImGui::ColorEdit4("Element color", (float*)&params.options.default_color);
        //ImGui::SliderFloat("depth", &params.options.depth, 0.f, 1.f);
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
                addTableRow("Resource type", getResourceType(info.resource_type));
                addTableRow("Data type", getDataType(info.data_type));
                addTableRow("Channel count", std::to_string(info.channel_count));
                addTableRow("Data layout", getDataLayout(info.layout));
                
                /*bool res_check = addTableRowCombo("Resource type", (int*)&info.resource_type,
                    &AllResources::ItemGetter, kAllResources.data(), kAllResources.size()
                );
                if (res_check) printf("View %d: switched resource type to %s\n", uid, getResourceType(info.resource_type));
                bool data_check = addTableRowCombo("Data type", (int*)&info.data_type,
                    &AllDataTypes::ItemGetter, kAllDataTypes.data(), kAllDataTypes.size()
                );
                if (data_check) printf("View %d: switched data type to %s\n", uid, getDataType(info.data_type));*/

                ImGui::EndTable();
            }
        }
        ImGui::TreePop();
    }
    ImGui::PopID();
}

void CudaviewEngine::displayEngineGUI()
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

    for (size_t i = 0; i < views2.size(); ++i)
    {
        addViewObjectGui(views2[i].get(), i);
    }
    ImGui::End();
}

CudaviewEngine *getHandler(GLFWwindow *window)
{
    return reinterpret_cast<CudaviewEngine*>(glfwGetWindowUserPointer(window));
}

void CudaviewEngine::setBackgroundColor(float4 color)
{
    bg_color = color;
}

// Translates GLFW mouse movement into Viewer flags for detecting camera movement
void CudaviewEngine::handleMouseMove(float x, float y)
{
    auto dx = mouse_pos.x - x;
    auto dy = mouse_pos.y - y;

    if (mouse_buttons.left)
    {
        camera->rotate(glm::vec3(dy * camera->rotation_speed, -dx * camera->rotation_speed, 0.f));
        view_updated = true;
    }
    if (mouse_buttons.right)
    {
        camera->translate(glm::vec3(0.f, 0.f, dy * .005f));
    }
    if (mouse_buttons.middle)
    {
        camera->translate(glm::vec3(-dx * 0.01f, -dy * 0.01f, 0.f));
    }
    mouse_pos = float2{x, y};
}

void CudaviewEngine::cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
{
    auto app = getHandler(window);
    app->handleMouseMove(static_cast<float>(xpos), static_cast<float>(ypos));
}

// Translates GLFW mouse actions into Viewer flags for detecting camera actions 
void CudaviewEngine::handleMouseButton(int button, int action, [[maybe_unused]] int mods)
{
    // Perform action only if GUI does not want to use mouse input
    // (if not hovering over a menu item)
    if (!ImGui::GetIO().WantCaptureMouse)
    {
        if (action == GLFW_PRESS)
        {
            if (button == GLFW_MOUSE_BUTTON_MIDDLE) mouse_buttons.middle = true;
            else if (button == GLFW_MOUSE_BUTTON_LEFT) mouse_buttons.left = true;
            else if (button == GLFW_MOUSE_BUTTON_RIGHT) mouse_buttons.right = true;
        }
        else if (action == GLFW_RELEASE)
        {
            if (button == GLFW_MOUSE_BUTTON_MIDDLE) mouse_buttons.middle = false;
            else if (button == GLFW_MOUSE_BUTTON_LEFT) mouse_buttons.left = false;
            else if (button == GLFW_MOUSE_BUTTON_RIGHT) mouse_buttons.right = false;
        }
    }
}

// Translates GLFW mouse scroll into values for detecting camera zoom in/out 
void CudaviewEngine::handleScroll([[maybe_unused]] float xoffset, [[maybe_unused]] float yoffset)
{
    // depth = std::clamp(depth + yoffset / 10.f, 0.01f, 0.91f);
    // printf("depth= %f, offset= %f\n", depth, yoffset);
}

void CudaviewEngine::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    auto app = getHandler(window);
    app->handleMouseButton(button, action, mods);
}

void CudaviewEngine::framebufferResizeCallback(GLFWwindow *window,[[maybe_unused]] int width,[[maybe_unused]] int height)
{
    auto app = getHandler(window);
    app->should_resize = true;
}

void CudaviewEngine::scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    auto app = getHandler(window);
    app->handleScroll(static_cast<float>(xoffset), static_cast<float>(yoffset));
}

void CudaviewEngine::keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    auto app = getHandler(window);
    app->handleKey(key, scancode, action, mods);
}

void CudaviewEngine::handleKey(int key, [[maybe_unused]] int scancode, int action, int mods)
{
    // Toggle demo window
    if (key == GLFW_KEY_D && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL)
    {
        show_demo_window = !show_demo_window;
    }
    if (key == GLFW_KEY_M && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL)
    {
        options.show_metrics = !options.show_metrics;
    }
}

void CudaviewEngine::windowCloseCallback(GLFWwindow *window)
{
    //printf("Handling window close\n");
    auto engine = getHandler(window);
    engine->running = false;
    engine->signalKernelFinish();
}

void CudaviewEngine::showMetrics()
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