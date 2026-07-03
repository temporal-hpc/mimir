#include "mimir/camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath> // std::sin, std::cos

namespace mimir
{

Camera Camera::make()
{
    return Camera{
        .type           = CameraType::LookAt,
        .position       = glm::vec3(),
        .rotation       = glm::vec3(),
        .rotation_speed = 1.f,
        .movement_speed = 1.f,
        .fov            = 0.f,
        .near_clip      = 0.f,
        .far_clip       = 0.f,
        .matrices       = { .perspective = glm::mat4(), .view = glm::mat4() }
    };
}

void Camera::updateViewMatrix()
{
    glm::mat4 rotmat(1.f);
    rotmat = glm::rotate(rotmat, glm::radians(rotation.x), glm::vec3(1.f, 0.f, 0.f));
    rotmat = glm::rotate(rotmat, glm::radians(rotation.y), glm::vec3(0.f, 1.f, 0.f));
    rotmat = glm::rotate(rotmat, glm::radians(rotation.z), glm::vec3(0.f, 0.f, 1.f));

    glm::vec3 translation = position;
    glm::mat4 transmat = glm::translate(glm::mat4(1.f), translation);
    matrices.view = (type == CameraType::FirstPerson)? rotmat * transmat : transmat * rotmat;
}

glm::mat4x4 perspective(float vertical_fov, float aspect_ratio, float near, float far)
{
    float fov_rad = vertical_fov * glm::pi<float>() / 180.f;
    float focal_length = 1.f / std::tan(fov_rad / 2.f);

    float x = focal_length / aspect_ratio;
    float y = -focal_length;
    float A = near / (far - near);
    float B = far * A;

    glm::mat4x4 projection({
        x,   0.f,  0.f, 0.f,
        0.f,   y,  0.f, 0.f,
        0.f, 0.f,    A,   B,
        0.f, 0.f, -1.f, 0.f,
    });
    return glm::transpose(projection);
}

void Camera::setPerspective(float fov, float aspect, float znear, float zfar)
{
    this->fov       = fov;
    this->near_clip = znear;
    this->far_clip  = zfar;

    matrices.perspective = perspective(fov, aspect, znear, zfar);
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

void Camera::translate(glm::vec3 delta)
{
    this->position += delta;
    updateViewMatrix();
}

void Camera::setFlyLook()
{
    // Forward from yaw/pitch matching the rotation-0 convention (yaw=pitch=0 -> +z, pitch>0 looks
    // down). setLookAt then rebuilds a horizontal `right`, so the horizon never rolls.
    float pitch = glm::radians(rotation.x);
    float yaw   = glm::radians(rotation.y);
    float cp = std::cos(pitch);
    glm::vec3 forward = {
        std::sin(yaw) * cp,
        -std::sin(pitch),
        std::cos(yaw) * cp,
    };
    setLookAt(position, position + forward, glm::vec3(0.f, 1.f, 0.f));
}

void Camera::setLookAt(glm::vec3 eye, glm::vec3 center, glm::vec3 world_up)
{
    // Build camera-to-world directly (right-handed, matching the euler LookAt: at rotation 0,
    // right=+x, up=+y, forward=+z). right = up x forward keeps the +x handedness.
    glm::vec3 fwd   = glm::normalize(center - eye);
    glm::vec3 right = glm::normalize(glm::cross(world_up, fwd));
    glm::vec3 up    = glm::cross(fwd, right);

    glm::mat4 m(1.f);
    m[0] = glm::vec4(right, 0.f);
    m[1] = glm::vec4(up,    0.f);
    m[2] = glm::vec4(fwd,   0.f);
    m[3] = glm::vec4(eye,   1.f);
    matrices.view = m;
    this->position = eye;
}

} // namespace mimir