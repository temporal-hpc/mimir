#pragma once

#include "cudaview/vk_types.hpp"

class Camera
{
private:
  float fov;
  float z_near, z_far;

  void updateViewMatrix();

public:
  enum class CameraType
  {
    LookAt,
    FirstPerson
  };
  CameraType type = CameraType::LookAt;

  glm::vec3 rotation = glm::vec3();
  glm::vec3 position = glm::vec3();
  glm::vec4 view_pos = glm::vec4();

  float rotation_speed = 1.f;
  float movement_speed = 1.f;

  bool updated = false;
  bool flip_y  = false;

  struct
  {
    glm::mat4 perspective;
    glm::mat4 view;
  } matrices;

  struct
  {
    bool left  = false;
    bool right = false;
    bool up    = false;
    bool down  = false;
  } keys;

  bool moving();
  float getNearClip();
  float getFarClip();
  void setPerspective(float fov, float aspect, float znear, float zfar);
  void updateAspectRatio(float aspect);
  void setPosition(glm::vec3 position);
  void setRotation(glm::vec3 rotation);
  void rotate(glm::vec3 delta);
  void setTranslation(glm::vec3 translation);
  void translate(glm::vec3 delta);
  void setRotationSpeed(float rot_speed);
  void setMovementSpeed(float mov_speed);
};
