#pragma once
#include <string>
#include <memory>
#include <glm/mat4x4.hpp> 

#include "minipbrt.h"

using namespace std;
using glm::mat4, glm::vec3;

namespace EasyRender
{

enum class CameraType
{
    Pinhole
};

const string CameraTypeStr[] = { "pinhole" };

class Camera
{
public:
    Camera() = default;
    virtual ~Camera() = default;
    virtual CameraType type() const = 0;
    virtual string ToString() const = 0;
};

using CameraPtr = unique_ptr<Camera>;

class PinholeCamera : public Camera
{
public:
    PinholeCamera(float fov_, minipbrt::Transform cameraToWorld);
    PinholeCamera(float fov_, vec3 pos_, vec3 lookAt_,
        vec3 up_);
    CameraType type() const { return CameraType::Pinhole; }
    string ToString() const;

public:
    float fov;
    vec3 position;
    vec3 lookAt;
    vec3 up;
    vec3 right;
};

} // namespace EasyRender
