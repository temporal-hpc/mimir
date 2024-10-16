#pragma once

#include "shader_types.hpp"

namespace mimir
{

struct Camera
{
    enum class CameraType { LookAt, FirstPerson };
    CameraType type;

    glm::vec3 rotation;
    glm::vec3 position;
    glm::vec4 view_pos;

    float rotation_speed, movement_speed;
    float fov, near_clip, far_clip;
    bool updated, flip_y;

    struct
    {
        glm::mat4 perspective;
        glm::mat4 view;
    } matrices;

    static Camera make();
    void updateViewMatrix();
    float getNearClip();
    float getFarClip();
    void setPerspective(float fov, float aspect, float znear, float zfar);
    void updateAspectRatio(float aspect);
    void setPosition(glm::vec3 position);
    void setRotation(glm::vec3 rotation);
    void rotate(glm::vec3 delta);
    void setTranslation(glm::vec3 translation);
    void translate(glm::vec3 delta);
};

static_assert(std::is_default_constructible_v<Camera>);
static_assert(std::is_nothrow_default_constructible_v<Camera>);
static_assert(std::is_trivially_default_constructible_v<Camera>);

} // namespace mimir