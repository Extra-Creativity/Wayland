#pragma once

#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include "spdlog/spdlog.h"

#include "DeviceUtils/Camera.h"

namespace Wayland::Example
{

class Camera
{
public:
    Camera(const glm::vec3 &init_pos, const glm::vec3 &init_up,
           const glm::vec3 &init_front) noexcept
        : position_(init_pos), gaze_(glm::normalize(init_front)),
          up_(glm::normalize(init_up - gaze_ * glm::dot(gaze_, init_up)))
    {
        if (init_up == glm::zero<glm::vec3>() ||
            init_front == glm::zero<glm::vec3>()) [[unlikely]]
        {
            SPDLOG_WARN(
                "Detect zero vec in up or front for camera initialization,"
                "making up = (0, 1, 0) and gaze = (0, 0, -1)\n");
            up_ = { 0, 1, 0 };
            gaze_ = { 0, 0, -1 };
        }
    };

    glm::vec3 Front() const noexcept { return gaze_; }
    glm::vec3 Back() const noexcept { return -gaze_; }
    glm::vec3 Up() const noexcept { return up_; }
    glm::vec3 Down() const noexcept { return -up_; }
    glm::vec3 Left() const noexcept { return glm::cross(up_, gaze_); };
    glm::vec3 Right() const noexcept { return glm::cross(gaze_, up_); };

    glm::mat4 GetViewMatrix() const noexcept
    {
        return glm::lookAt(position_, position_ + gaze_, up_);
    };

    void Rotate(glm::vec3 eulerAngles) noexcept
    {
        glm::quat rotation = glm::quat(eulerAngles);
        Rotate(rotation);
        return;
    };

    inline void Rotate(glm::quat rotation) noexcept
    {
        gaze_ = rotation * gaze_;
        up_ = rotation * up_;
    }

    void Rotate(float angle, glm::vec3 axis) noexcept
    {
        glm::quat rotation =
            glm::angleAxis(glm::radians(angle), glm::normalize(axis));
        Rotate(rotation);
    }

    void Translate(glm::vec3 vec) noexcept
    {
        position_ += vec;
        return;
    }

    void RotateAroundCenter(float angle, glm::vec3 axis,
                            const glm::vec3 &center) noexcept
    {
        glm::vec3 distanceVec = position_ - center;
        glm::quat rotation =
            glm::angleAxis(glm::radians(angle), glm::normalize(axis));
        Rotate(rotation);
        position_ = rotation * distanceVec + center;
        return;
    }

    const glm::vec3 &GetPosition() const noexcept { return position_; };
    const glm::vec3 &GetGaze() const noexcept { return gaze_; }
    const glm::vec3 &GetUp() const noexcept { return up_; }

    Camera &SetFoV(float fov) noexcept
    {
        fov_ = fov;
        return *this;
    }

    Camera &SetDepth(float depth) noexcept
    {
        depth_ = depth;
        return *this;
    }

    float GetVerticalRatio() const noexcept
    {
        return depth_ * std::tanf(fov_ / 2);
    }

    DeviceUtils::Camera ToDeviceCamera(float aspect) const noexcept
    {
        float vRatio = GetVerticalRatio();
        return DeviceUtils::Camera{
            .horizontalRatio = aspect * vRatio,
            .verticalRatio = vRatio,
            .position = position_,
            .gaze = gaze_,
            .up = up_,
        };
    }

private:
    float fov_ = 45.0f;
    float depth_ = 1.0f;
    // Note that this sequence is deliberate, so that up will be initialized
    // after front.
    glm::vec3 position_;
    glm::vec3 gaze_;
    glm::vec3 up_;
};

} // namespace Wayland::Example