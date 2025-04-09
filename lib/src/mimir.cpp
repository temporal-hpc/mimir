#include <mimir/mimir.hpp>

#include "mimir/api.hpp"
#include "mimir/engine.hpp"

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

bool isRunning(InstanceHandle engine) { return engine->running; }

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

void setCameraPosition(InstanceHandle handle, float3 pos)
{
    handle->camera.setPosition(glm::vec3(pos.x, pos.y, pos.z));
}

void setCameraRotation(InstanceHandle handle, float3 rot)
{
    handle->camera.setRotation(glm::vec3(rot.x, rot.y, rot.z));
}

void display(InstanceHandle engine, std::function<void(void)> func, size_t iter_count)
{
    engine->display(func, iter_count);
}

void displayAsync(InstanceHandle engine)
{
    engine->displayAsync();
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