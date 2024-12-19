#include "mimir/camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

namespace mimir
{

Camera Camera::make()
{
    return Camera{
        .type           = CameraType::LookAt,
        .rotation       = glm::vec3(),
        .position       = glm::vec3(),
        .view_pos       = glm::vec4(),
        .rotation_speed = 1.f,
        .movement_speed = 1.f,
        .fov            = 0.f,
        .near_clip      = 0.f,
        .far_clip       = 0.f,
        .updated        = false,
        .flip_y         = false,
        .matrices       = { .perspective = glm::mat4(), .view = glm::mat4() }
    };
}

void Camera::updateViewMatrix()
{
    glm::mat4 rotmat(1.f);
    auto flip = flip_y? -1.f : 1.f;
    rotmat = glm::rotate(rotmat, glm::radians(rotation.x * flip), glm::vec3(1.f, 0.f, 0.f));
    rotmat = glm::rotate(rotmat, glm::radians(rotation.y), glm::vec3(0.f, 1.f, 0.f));
    rotmat = glm::rotate(rotmat, glm::radians(rotation.z), glm::vec3(0.f, 0.f, 1.f));

    glm::vec3 translation = position;
    if (flip_y)
    {
        translation.y *= -1.f;
    }

    glm::mat4 transmat = glm::translate(glm::mat4(1.f), translation);
    matrices.view = (type == CameraType::FirstPerson)? rotmat * transmat : transmat * rotmat;

    view_pos = glm::vec4(position, 0.f) * glm::vec4(-1.f, 1.f, -1.f, 1.f);
    updated = true;
}

void Camera::setPerspective(float fov, float aspect, float znear, float zfar)
{
    this->fov       = fov;
    this->near_clip = znear;
    this->far_clip  = zfar;

    matrices.perspective = glm::perspective(glm::radians(fov), aspect, near_clip, far_clip);
    if (flip_y)
    {
        matrices.perspective[1][1] *= -1.f;
    }
}

void Camera::updateAspectRatio(float aspect)
{
    matrices.perspective = glm::perspective(glm::radians(fov), aspect, near_clip, far_clip);
    if (flip_y)
    {
        matrices.perspective[1][1] *= -1.f;
    }
}

void Camera::setPosition(glm::vec3 position)
{
    this->position = position;
    updateViewMatrix();
}

void Camera::setRotation(glm::vec3 rotation)
{
    this->rotation = rotation;
    updateViewMatrix();
}

void Camera::rotate(glm::vec3 delta)
{
    this->rotation += delta;
    updateViewMatrix();
}

void Camera::setTranslation(glm::vec3 translation)
{
    this->position = translation;
    updateViewMatrix();
}

void Camera::translate(glm::vec3 delta)
{
    this->position += delta;
    updateViewMatrix();
}

} // namespace mimir