#pragma once
#include <string>
#include <memory>
#include <glm/mat4x4.hpp> 

#include "minipbrt.h"

using namespace std;
using glm::mat4, glm::vec3;

namespace Wayland
{

enum class CameraType
{
    Pinhole
};

const string CameraTypeStr[] = { "pinhole" };

class Camera
{
public:
    Camera() {};
	virtual ~Camera() {}
    virtual CameraType type() const = 0;
    virtual string toString() const = 0;
};

using CameraPtr = unique_ptr<Camera>;

class PinholeCamera : public Camera
{
public:
    PinholeCamera(float fov_, minipbrt::Transform cameraToWorld);
    PinholeCamera(float fov_, vec3 pos_, vec3 lookAt_,
        vec3 up_);
    CameraType type() const { return CameraType::Pinhole; }
    string toString() const;

private:
    float fov;
    vec3 position;
    vec3 lookAt;
    vec3 up;
};


} // namespace Wayland
