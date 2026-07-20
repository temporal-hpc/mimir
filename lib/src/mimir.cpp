#include <mimir/mimir.hpp>

#include "mimir/api.hpp"
#include "mimir/engine.hpp"

#include <atomic> // std::atomic_ref

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

namespace mimir
{

void createInstance(ViewerOptions opts, InstanceHandle *engine)
{
    *engine = new MimirInstance(MimirInstance::make(opts));
}

void createInstance(int width, int height, InstanceHandle *engine)
{
    *engine = new MimirInstance(MimirInstance::make(width, height));
}

void destroyInstance(InstanceHandle engine)
{
    engine->deinit();
    delete engine;
}

bool isRunning(InstanceHandle engine)
{
    return std::atomic_ref<bool>(engine->running).load(std::memory_order_acquire);
}

void allocLinear(InstanceHandle engine, void **dev_ptr, size_t size, AllocHandle *alloc)
{
    *alloc = engine->allocLinear(dev_ptr, size);
}

void allocMipmap(InstanceHandle engine, cudaMipmappedArray_t *dev_arr, const cudaChannelFormatDesc *desc,
    cudaExtent extent, unsigned int num_levels, AllocHandle *alloc)
{
    *alloc = engine->allocMipmap(dev_arr, desc, extent, num_levels);
}

void createView(InstanceHandle engine, ViewDescription *desc, ViewHandle *handle)
{
    *handle = engine->createView(desc);
}

bool toggleVisibility(ViewHandle view)
{
    auto& visibility = view->desc.visible;
    visibility = !visibility;
    return visibility;
}

void setViewDefaultColor(ViewHandle view, float4 color)
{
    view->desc.default_color = color;
}

void scaleView(ViewHandle view, float3 scale)
{
    glm::vec3 s{ scale.x, scale.y, scale.z };
    view->scale = glm::scale(glm::mat4x4(1.f), s);
}

void translateView(ViewHandle view, float3 pos)
{
    glm::vec3 t{ pos.x, pos.y, pos.z };
    view->translation = glm::translate(glm::mat4x4(1.f), t);
}

void rotateView(ViewHandle view, float3 rot)
{
    glm::vec3 euler_angles{ rot.x, rot.y, rot.z };
    glm::quat quat(glm::radians(euler_angles));
    view->rotation = glm::toMat4(quat);
}

size_t deviceLocalMemory(InstanceHandle handle)
{
    if (handle == nullptr || handle->physical_device.handle == VK_NULL_HANDLE) { return 0; }
    VkPhysicalDeviceMemoryProperties props{};
    vkGetPhysicalDeviceMemoryProperties(handle->physical_device.handle, &props);
    size_t total = 0;
    for (uint32_t i = 0; i < props.memoryHeapCount; ++i)
    {
        if (props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
        {
            total += props.memoryHeaps[i].size;
        }
    }
    return total;
}

DeviceBufferLimits deviceBufferLimits(InstanceHandle handle)
{
    DeviceBufferLimits out{ 0, 0, 0, 0, 0 };
    if (handle == nullptr || handle->physical_device.handle == VK_NULL_HANDLE) { return out; }

    // maxMemoryAllocationSize is maintenance3 (core since 1.1); maxBufferSize is maintenance4
    // (core since 1.3) and is left 0 if the runtime/driver does not report it. maxStorageBufferRange
    // is a core VkPhysicalDeviceLimits field (uint32_t, so it tops out at 4 GiB - 1). maxInstanceCount
    // comes from VK_KHR_acceleration_structure (0 if unsupported).
    VkPhysicalDeviceAccelerationStructurePropertiesKHR as_props{};
    as_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    VkPhysicalDeviceMaintenance3Properties m3{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES, .pNext = &as_props,
        .maxPerSetDescriptors = 0, .maxMemoryAllocationSize = 0 };
#ifdef VK_VERSION_1_3
    VkPhysicalDeviceMaintenance4Properties m4{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES,
        .pNext = &as_props, .maxBufferSize = 0 };
    m3.pNext = &m4;
#endif
    VkPhysicalDeviceProperties2 props{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m3, .properties = {} };
    vkGetPhysicalDeviceProperties2(handle->physical_device.handle, &props);

    out.max_storage_buffer_range   = props.properties.limits.maxStorageBufferRange;
    out.max_memory_allocation_size = m3.maxMemoryAllocationSize;
    out.max_instance_count         = as_props.maxInstanceCount;
    out.max_primitive_count        = as_props.maxPrimitiveCount;
#ifdef VK_VERSION_1_3
    out.max_buffer_size            = m4.maxBufferSize;
#endif
    return out;
}

uint32_t maxImageDimension2D(InstanceHandle handle)
{
    if (handle == nullptr || handle->physical_device.handle == VK_NULL_HANDLE) { return 0; }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(handle->physical_device.handle, &props);
    return props.limits.maxImageDimension2D;
}

uint32_t linearImageRowAlignment(InstanceHandle handle, FormatDescription format)
{
    if (handle == nullptr || handle->device == VK_NULL_HANDLE) { return 1; }
    const unsigned int texel = format.getSize();
    if (texel == 0) { return 1; }
    // A LINEAR-tiled image's row pitch is aligned by the driver; a buffer aliased to such an image
    // (as an interop Image view does) shears unless its row stride matches that pitch. Probe a
    // 1-texel-wide LINEAR image of this format: its rowPitch is the pitch granularity in bytes.
    VkImageCreateInfo ic{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext         = nullptr,
        .flags         = 0,
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = getVulkanFormat(format),
        .extent        = { 1, 1, 1 },
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_LINEAR,
        .usage                 = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .sharingMode           = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices   = nullptr,
        .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage img = VK_NULL_HANDLE;
    if (vkCreateImage(handle->device, &ic, nullptr, &img) != VK_SUCCESS) { return 1; }
    VkImageSubresource sub{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
    VkSubresourceLayout lay{};
    vkGetImageSubresourceLayout(handle->device, img, &sub, &lay);
    vkDestroyImage(handle->device, img, nullptr);
    uint32_t align = (uint32_t)(lay.rowPitch / texel);
    return align < 1 ? 1 : align;
}

void setCameraPosition(InstanceHandle handle, float3 pos)
{
    handle->camera.setPosition(glm::vec3(pos.x, pos.y, pos.z));
}

void setCameraRotation(InstanceHandle handle, float3 rot)
{
    handle->camera.setRotation(glm::vec3(rot.x, rot.y, rot.z));
}

void setCameraLookAt(InstanceHandle handle, float3 eye, float3 center, float3 up)
{
    const glm::vec3 e(eye.x, eye.y, eye.z);
    const glm::vec3 c(center.x, center.y, center.z);
    const glm::vec3 u(up.x, up.y, up.z);
    auto& cam = handle->camera;

    // matrices.view is interpreted differently per camera mode (see renderFrame): the fly camera and
    // the scripted auto-orbit read it as camera-to-world (eye/forward in the columns), while the
    // manual orbit trackball consumes it directly as a world-to-view matrix. Write whichever the
    // active mode expects so the framing is correct either way.
    if (handle->options.camera_control == CameraControl::Fly || handle->options.orbit_speed > 0.f)
    {
        cam.setLookAt(e, c, u); // camera-to-world; the render inverts / decodes the columns
    }
    else
    {
        // World-to-view for the trackball raster path (and the orbit PT eye decode = -R^T*pos).
        cam.matrices.view = glm::lookAt(e, c, u);
        // Keep the euler position roughly consistent so a later trackball drag (which rebuilds
        // matrices.view from position/rotation) starts from an equivalent view. Exact when framing
        // the world origin down an axis -- the trackball's natural pivot -- e.g. CA3D-voxels.
        cam.position = glm::vec3(cam.matrices.view[3]);
    }
}

void display(InstanceHandle engine, std::function<void(void)> func, size_t iter_count)
{
    engine->display(func, iter_count);
}

void setPaused(InstanceHandle engine, bool paused)
{
    std::atomic_ref<bool>(engine->paused).store(paused, std::memory_order_release);
}

bool isPaused(InstanceHandle engine)
{
    return std::atomic_ref<bool>(engine->paused).load(std::memory_order_acquire);
}

void requestStep(InstanceHandle engine)
{
    std::atomic_ref<uint64_t>(engine->pending_steps).fetch_add(1, std::memory_order_acq_rel);
}

bool shouldStep(InstanceHandle engine)
{
    return engine->consumeStep();
}

void setHudText(InstanceHandle engine, const char *text)
{
    if (engine->hud_panel == nullptr) { return; }
    std::lock_guard<std::mutex> lock(engine->hud_panel->mutex);
    engine->hud_panel->text = (text != nullptr) ? text : "";
}

void setScrollCallback(InstanceHandle engine, std::function<void(double, double)> callback)
{
    engine->setScrollCallback(std::move(callback));
}

bool isKeyDown(InstanceHandle engine, Key key)
{
    auto idx = static_cast<size_t>(key);
    if (idx >= static_cast<size_t>(Key::Count)) { return false; }
    return std::atomic_ref<uint8_t>(engine->key_down[idx]).load(std::memory_order_acquire) != 0;
}

bool isKeyPressed(InstanceHandle engine, Key key)
{
    auto idx = static_cast<size_t>(key);
    if (idx >= static_cast<size_t>(Key::Count)) { return false; }
    // Consume the latch so a press reports true exactly once.
    return std::atomic_ref<uint8_t>(engine->key_pressed[idx]).exchange(0, std::memory_order_acq_rel) != 0;
}

void displayAsync(InstanceHandle engine)
{
    engine->displayAsync();
}

void renderHeadless(InstanceHandle engine, std::function<void(void)> func, size_t iter_count)
{
    engine->renderHeadless(func, iter_count);
}

void saveFrame(InstanceHandle engine, const char *path)
{
    engine->saveFrameToPpm(path);
}

void serveRemote(InstanceHandle engine, unsigned short port,
    std::function<void(void)> func, size_t max_iters, bool use_h264,
    remote::TransportKind kind, const char *token, int bitrate_kbps, const char *stats_csv,
    int fps, int steps_per_frame)
{
    engine->serveRemote(port, func, max_iters, use_h264, kind, token ? token : "", bitrate_kbps,
        stats_csv ? stats_csv : "", fps, steps_per_frame);
}

void prepareViews(InstanceHandle engine)
{
    engine->prepareViews();
}

void updateViews(InstanceHandle engine)
{
    engine->updateViews();
}

void setGuiCallback(InstanceHandle engine, std::function<void(void)> callback)
{
    engine->setGuiCallback(callback);
}

AttributeDescription makeStructuredGrid(InstanceHandle engine, Layout extent, float3 start)
{
    return engine->makeStructuredGrid(extent, start);
}

AttributeDescription makeImageFrame(InstanceHandle engine)
{
    return engine->makeImageDomain();
}

void copyTextureData(InstanceHandle engine, TextureDescription tex_desc, void *data, size_t memsize)
{
    engine->loadTexture(tex_desc, data, memsize);
}

void exit(InstanceHandle engine)
{
    engine->exit();
}

PerformanceMetrics getMetrics(InstanceHandle engine)
{
    return engine->getMetrics();
}

} // namespace mimir