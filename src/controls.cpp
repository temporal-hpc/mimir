#include <cudaview/vk_engine.hpp>
#include "internal/camera.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm> // std::clamp

std::string to_string(DataDomain x)
{
    switch (x)
    {
        case DataDomain::Domain2D: return "2D";
        case DataDomain::Domain3D: return "3D";
        default: return "Unknown";
    }
}

std::string to_string(ResourceType x)
{
    switch (x)
    {
        case ResourceType::UnstructuredBuffer: return "Buffer (unstructured)";
        case ResourceType::StructuredBuffer: return "Buffer (structured)";
        case ResourceType::Texture: return "Texture";
        case ResourceType::TextureLinear: return "Texture (with linear buffer)";        
        default: return "Unknown";
    }
}

std::string to_string(PrimitiveType x)
{
    switch (x)
    {
        case PrimitiveType::Points: return "Markers";
        case PrimitiveType::Edges: return "Edges";
        case PrimitiveType::Voxels: return "Voxels";
        default: return "Unknown";
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

void VulkanEngine::addViewObjectGui(CudaView *view_ptr, int uid)
{
    ImGui::PushID(view_ptr);
    bool node_open = ImGui::TreeNode("Object", "%s_%u", "View", uid);

    if (node_open)
    {
        auto& info = view_ptr->params;
        if (ImGui::BeginTable("split", 2, ImGuiTableFlags_BordersOuter | ImGuiTableFlags_Resizable))
        {
            addTableRow("Data domain", to_string(info.data_domain));
            addTableRow("Resource type", to_string(info.resource_type));
            addTableRow("Primitive type", to_string(info.primitive_type));
            addTableRow("Element count", std::to_string(info.element_count));

            ImGui::EndTable();
        }
        ImGui::Checkbox("show", &info.options.visible);
        ImGui::SliderFloat("Primitive size (px)", &info.options.size, 1.f, 100.f);
        ImGui::ColorEdit4("Primitive color", (float*)&info.options.color);
        ImGui::SliderFloat("depth", &info.options.depth, 0.f, 1.f);
        ImGui::TreePop();
    }
    ImGui::PopID();
}

void VulkanEngine::drawGui()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    if (show_demo_window) { ImGui::ShowDemoWindow(); }
    if (show_metrics) { ImGui::ShowMetricsWindow(); }
    
    {
        ImGui::Begin("Scene parameters");
        //ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / framerate, framerate);
        ImGui::ColorEdit3("Clear color", (float*)&bg_color);
        auto pos = camera->position;
        ImGui::Text("Camera position: %.3f %.3f %.3f", pos.x, pos.y, pos.z);
        auto rot = camera->rotation;
        ImGui::Text("Camera rotation: %.3f %.3f %.3f", rot.x, rot.y, rot.z);
        for (size_t i = 0; i < views.size(); ++i)
        {
            addViewObjectGui(&views[i], i);
        }
        ImGui::End();
    }
    ImGui::Render();
}

VulkanEngine *getHandler(GLFWwindow *window)
{
    return reinterpret_cast<VulkanEngine*>(glfwGetWindowUserPointer(window));
}

void VulkanEngine::setBackgroundColor(float4 color)
{
    bg_color = color;
}

// Translates GLFW mouse movement into Viewer flags for detecting camera movement
void VulkanEngine::handleMouseMove(float x, float y)
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

void VulkanEngine::cursorPositionCallback(GLFWwindow *window, double xpos, double ypos)
{
    auto app = getHandler(window);
    app->handleMouseMove(static_cast<float>(xpos), static_cast<float>(ypos));
}

// Translates GLFW mouse actions into Viewer flags for detecting camera actions 
void VulkanEngine::handleMouseButton(int button, int action, [[maybe_unused]] int mods)
{
    // Perform action only if GUI does not want to use mouse input
    // (if not hovering over a menu item)
    auto& io = ImGui::GetIO();
    if (!io.WantCaptureMouse)
    {
        switch (action)
        {
            case GLFW_PRESS:
                switch (button)
                {
                    case GLFW_MOUSE_BUTTON_LEFT:
                        mouse_buttons.left = true;
                        break;
                    case GLFW_MOUSE_BUTTON_MIDDLE:
                        mouse_buttons.middle = true;
                        break;
                    case GLFW_MOUSE_BUTTON_RIGHT:
                        mouse_buttons.right = true;
                        break;
                    default:
                        break;
                }
                break;
            case GLFW_RELEASE:
                switch (button)
                {
                    case GLFW_MOUSE_BUTTON_LEFT:
                        mouse_buttons.left = false;
                        break;
                    case GLFW_MOUSE_BUTTON_MIDDLE:
                        mouse_buttons.middle = false;
                        break;
                    case GLFW_MOUSE_BUTTON_RIGHT:
                        mouse_buttons.right = false;
                        break;
                    default:
                        break;
                }
                break;
            default:
                break;
        }
    }
}

// Translates GLFW mouse scroll into values for detecting camera zoom in/out 
void VulkanEngine::handleScroll([[maybe_unused]] float xoffset, float yoffset)
{
    // depth = std::clamp(depth + yoffset / 10.f, 0.01f, 0.91f);
    // printf("depth= %f, offset= %f\n", depth, yoffset);
}

void VulkanEngine::mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{
    auto app = getHandler(window);
    app->handleMouseButton(button, action, mods);
}

void VulkanEngine::framebufferResizeCallback(GLFWwindow *window,[[maybe_unused]] int width,[[maybe_unused]] int height)
{
    auto app = getHandler(window);
    //app->should_resize = true;
}

void VulkanEngine::scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    auto app = getHandler(window);
    app->handleScroll(static_cast<float>(xoffset), static_cast<float>(yoffset));
}

void VulkanEngine::keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    auto app = getHandler(window);
    app->handleKey(key, scancode, action, mods);
}

void VulkanEngine::handleKey(int key, int scancode, int action, int mods)
{
    // Toggle demo window
    if (key == GLFW_KEY_D && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL)
    {
        show_demo_window = !show_demo_window;
    }
    if (key == GLFW_KEY_M && action == GLFW_PRESS && mods == GLFW_MOD_CONTROL)
    {
        show_metrics = !show_metrics;
    }
}