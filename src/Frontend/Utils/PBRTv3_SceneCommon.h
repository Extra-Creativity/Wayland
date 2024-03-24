#pragma once
#include <string>

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

const std::string CameraTypeStr[] = { "Perspective", "Orthographic", "Environment",
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
const std::string ShapeTypeStr[] = { "Cone", "Curve", "Cylinder", "Disk", "Hyperboloid",
								"Paraboloid", "Sphere", "TriangleMesh", "HeightField",
								"LoopSubdiv", "Nurbs", "PLYMesh", "Unknown" };

  enum class MaterialType
{
    Disney,
    Fourier,
    Glass,
    Hair,
    KdSubsurface,
    Matte,
    Metal,
    Mirror,
    Mix,
    None,
    Plastic,
    Substrate,
    Subsurface,
    Translucent,
    Uber,
    MaterialTypeMax
};

const std::string MaterialTypeStr[] = { "Disney", "Fourier", "Glass", "Hair", "KdSubsurface",
								   "Matte", "Metal", "Mirror", "Mix", "None", "Plastic",
								   "Substrate", "Subsurface", "Translucent", "Uber", "Unknown" };

} // namespace EasyRender::PBRTv3