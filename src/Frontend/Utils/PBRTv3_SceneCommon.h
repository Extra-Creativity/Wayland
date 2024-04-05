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

const std::string CameraTypeStr[] = { "Perspective", "Orthographic",
                                      "Environment", "Realistic", "Unknown" };

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

const std::string ShapeTypeStr[] = {
    "Cone",       "Curve",   "Cylinder",     "Disk",        "Hyperboloid",
    "Paraboloid", "Sphere",  "TriangleMesh", "HeightField", "LoopSubdiv",
    "Nurbs",      "PLYMesh", "Unknown"
};

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

const std::string MaterialTypeStr[] = {
    "Disney",     "Fourier",     "Glass", "Hair",   "KdSubsurface", "Matte",
    "Metal",      "Mirror",      "Mix",   "None",   "Plastic",      "Substrate",
    "Subsurface", "Translucent", "Uber",  "Unknown"
};

enum class TextureType
{
    Bilerp,
    Checkerboard2D,
    Checkerboard3D,
    Constant,
    Dots,
    FBM,
    ImageMap,
    Marble,
    Mix,
    Scale,
    UV,
    Windy,
    Wrinkled,
    PTex,
    TextureTypeMax
};

const std::string TextureTypeStr[] = {
    "Bilerp", "Checkerboard2D", "Checkerboard3D", "Constant", "Dots",
    "FBM",    "ImageMap",       "Marble",         "Mix",      "Scale",
    "UV",     "Windy",          "Wrinkled",       "PTex",     "TextureTypeMax"
};

} // namespace EasyRender::PBRTv3