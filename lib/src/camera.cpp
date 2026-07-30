#include "mimir/camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <cmath> // std::sin, std::cos

namespace mimir
{

// The euler rotation matrix (Rx*Ry*Rz, applied to world axes) for a given (rotation.x/y/z) in
// degrees. Shared by updateViewMatrix() and rotate() so both agree on exactly the same matrix for
// a given `rotation` value -- rotate()'s pivot compensation depends on that agreement.
static glm::mat4 buildEulerRotation(glm::vec3 rot)
{
    glm::mat4 r(1.f);
    r = glm::rotate(r, glm::radians(rot.x), glm::vec3(1.f, 0.f, 0.f));
    r = glm::rotate(r, glm::radians(rot.y), glm::vec3(0.f, 1.f, 0.f));
    r = glm::rotate(r, glm::radians(rot.z), glm::vec3(0.f, 0.f, 1.f));
    return r;
}

Camera Camera::make()
{
    return Camera{
        .type           = CameraType::LookAt,
        .position       = glm::vec3(),
        .rotation       = glm::vec3(),
        .pivot          = glm::vec3(),
        .base_rotation  = glm::mat4(1.f),
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
    // LookAt/trackball total rotation = the drag delta accumulated since the last setCameraLookAt
    // bind, applied on top of that bind's actual orientation (base_rotation) -- NOT `rotation`
    // alone, which would silently discard base_rotation and snap to whatever `rotation` alone
    // represents (identity, for any caller that has not dragged yet). base_rotation defaults to
    // identity, so this is exactly the old formula for every caller that never binds via
    // setCameraLookAt (setPosition/setRotation/FirstPerson are unaffected).
    glm::mat4 rotmat = buildEulerRotation(rotation) * base_rotation;

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
    // Orbit around `pivot` (world-space) instead of the world origin -- the world origin was the
    // ONLY fixed point of the old transmat(position)*rotmat(rotation) formula (p_view = R*p_world +
    // position; at p_world=0, p_view=position regardless of R), so dragging always spun the scene
    // around (0,0,0) no matter what setCameraLookAt's `center` was. Here, compute how far pivot's
    // view-space position shifts under this incremental rotation and cancel it by adjusting
    // `position`, so pivot stays fixed on screen through the drag. This ADDS to the existing
    // position rather than recomputing it from scratch, so any pan/zoom already baked into it
    // (Camera::translate, unrelated to rotation) survives untouched. pivot defaults to the origin,
    // so callers that never set it (setCameraPosition, most samples) keep today's exact behavior.
    // Must use the SAME total rotation as updateViewMatrix (delta * base_rotation, not delta
    // alone), or this compensation is computed against a baseline the render never actually used.
    glm::mat4 R_old = buildEulerRotation(rotation) * base_rotation;
    glm::vec3 pivot_before = glm::vec3(R_old * glm::vec4(pivot, 1.f));
    this->rotation += delta;
    glm::mat4 R_new = buildEulerRotation(rotation) * base_rotation;
    glm::vec3 pivot_after = glm::vec3(R_new * glm::vec4(pivot, 1.f));
    position += (pivot_before - pivot_after);
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

void Camera::freeLook(float yaw_deg, float pitch_deg)
{
    // Decode eye + world look direction from the current world-to-view matrix (the trackball
    // form translate(pos)*rotmat, and the same form glm::lookAt produces): R = mat3(view), the
    // eye is -R^T * translation, and the view looks down -z so world forward = -(R^T)[2]. This
    // mirrors the manual-orbit decode in the path tracer (engine.cpp), so the gaze turn renders
    // on the same branch as an orbit -- no camera-to-world / Fly inversion.
    const glm::mat3 rt = glm::transpose(glm::mat3(matrices.view)); // R^T: columns are world axes
    const glm::vec3 eye = -(rt * glm::vec3(matrices.view[3]));
    glm::vec3 fwd = -rt[2];

    // Yaw about world-up, pitch about the camera's (horizontal) right: locking right to the
    // world-up plane keeps the horizon level for any yaw/pitch (no roll), as glm::lookAt below
    // rebuilds up = right x forward from world-up anyway.
    const glm::vec3 world_up(0.f, 1.f, 0.f);
    const glm::vec3 right = glm::normalize(glm::cross(fwd, world_up));
    glm::mat4 turn = glm::rotate(glm::mat4(1.f), glm::radians(yaw_deg), world_up);
    turn = glm::rotate(turn, glm::radians(pitch_deg), right);
    fwd = glm::normalize(glm::vec3(turn * glm::vec4(fwd, 0.f)));

    // Only matrices.view is updated: position/rotation stay the trackball (pos, rot) they were,
    // so a later orbit/zoom/pan still composes from a valid trackball pose (a look that returns
    // to center leaves that pose showing the same view, so there is no jump).
    matrices.view = glm::lookAt(eye, eye + fwd, world_up);
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