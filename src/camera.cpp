#include "internal/camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

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

  if (type == CameraType::FirstPerson)
  {
    matrices.view = rotmat * transmat;
  }
  else
  {
    matrices.view = transmat * rotmat;
  }
  view_pos = glm::vec4(position, 0.f) * glm::vec4(-1.f, 1.f, -1.f, 1.f);
  updated = true;
}

bool Camera::moving()
{
  return keys.left || keys.right || keys.up || keys.down;
}

float Camera::getNearClip()
{
  return z_near;
}

float Camera::getFarClip()
{
  return z_far;
}

void Camera::setPerspective(float fov, float aspect, float znear, float zfar)
{
  this->fov    = fov;
  this->z_near = znear;
  this->z_far  = zfar;

  matrices.perspective = glm::perspective(glm::radians(fov), aspect, z_near, z_far);
  if (flip_y)
  {
    matrices.perspective[1][1] *= -1.f;
  }
}

void Camera::updateAspectRatio(float aspect)
{
  matrices.perspective = glm::perspective(glm::radians(fov), aspect, z_near, z_far);
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

void Camera::setRotationSpeed(float rot_speed)
{
  this->rotation_speed = rot_speed;
}

void Camera::setMovementSpeed(float mov_speed)
{
  this->movement_speed = mov_speed;
}
