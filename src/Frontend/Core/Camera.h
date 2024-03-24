#pragma once
#include <string>
#include <memory>
#include <glm/mat4x4.hpp> 

#include "minipbrt.h"

namespace EasyRender
{

enum class CameraType
{
    Pinhole
};

const std::string CameraTypeStr[] = { "pinhole" };

class Camera
{
public:
    Camera() = default;
    virtual ~Camera() = default;
    virtual CameraType type() const = 0;
    virtual std::string ToString() const = 0;
};

using CameraPtr = std::unique_ptr<Camera>;

class PinholeCamera : public Camera
{
public:
    PinholeCamera(float fov_, minipbrt::Transform cameraToWorld);
    PinholeCamera(float fov_, glm::vec3 pos_, glm::vec3 lookAt_, glm::vec3 up_);
    CameraType type() const { return CameraType::Pinhole; }
    std::string ToString() const;

public:
    float fov;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 up;
    glm::vec3 right;
};

} // namespace EasyRender
