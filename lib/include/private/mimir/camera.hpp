#pragma once

#include "shader_types.hpp"

namespace mimir
{

struct Camera
{
    enum class CameraType { LookAt, FirstPerson };

    CameraType type;
    glm::vec3 position;
    glm::vec3 rotation;
    float rotation_speed, movement_speed;
    float fov, near_clip, far_clip;
    struct
    {
        glm::mat4 perspective;
        glm::mat4 view;
    } matrices;

    static Camera make();
    void updateViewMatrix();
    void setPerspective(float fov, float aspect, float znear, float zfar);
    void setPosition(glm::vec3 position);
    void setRotation(glm::vec3 rotation);
    void rotate(glm::vec3 delta);
    void translate(glm::vec3 delta);
    // Point the camera at `center` from `eye`, building the same camera-to-world view matrix
    // (columns = right/up/forward, translation = eye) that the LookAt updateViewMatrix produces.
    // Used by the scripted auto-orbit; leaves `rotation` untouched.
    void setLookAt(glm::vec3 eye, glm::vec3 center, glm::vec3 world_up);
    // Rebuild a roll-free FPS view from the stored yaw (rotation.y) / pitch (rotation.x), in
    // degrees, at the current position. Derives forward from yaw+pitch and routes through
    // setLookAt so `right` stays horizontal for any pitch/yaw -- unlike the euler updateViewMatrix
    // (Rx*Ry order), which rolls the horizon when pitch and yaw are combined. For the Fly camera.
    void setFlyLook();
};

static_assert(std::is_default_constructible_v<Camera>);
static_assert(std::is_nothrow_default_constructible_v<Camera>);
static_assert(std::is_trivially_default_constructible_v<Camera>);

} // namespace mimir