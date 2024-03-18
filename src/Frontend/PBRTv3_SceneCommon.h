#pragma once
#include <string>
using namespace std;

namespace Wayland::PBRTv3
{

enum class CameraType
{
    Perspective,
    Orthographic,
    Environment,
    Realistic,
    CameraTypeMax
};
const string CameraTypeStr[] = { "Perspective", "Orthographic", "Environment",
                                 "Realistic", "Unknown" };

} // namespace Wayland::PBRTv3