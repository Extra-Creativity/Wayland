#pragma once
#include <string>
using namespace std;

namespace EasyRender::PBRTv3
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

enum class ShapeType
{
    Cone,
    Curve,
    Cylinder,
    Disk,
    Hyperboloid,
    Paraboloid,
    Sphere,
    TriangleMesh,
    HeightField,
    LoopSubdiv,
    Nurbs,
    PLYMesh,
    ShapeTypeMax
};
const string ShapeTypeStr[] = { "Cone", "Curve", "Cylinder", "Disk", "Hyperboloid",
								"Paraboloid", "Sphere", "TriangleMesh", "HeightField",
								"LoopSubdiv", "Nurbs", "PLYMesh", "Unknown" };

} // namespace EasyRender::PBRTv3