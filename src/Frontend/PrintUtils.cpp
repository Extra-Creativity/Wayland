#pragma once
#include <glm/mat4x4.hpp>
#include <iostream>
#include <sstream>
#include <string>

#include "PrintUtils.h"
#include "minipbrt.h"
#include "PBRTv3_SceneCommon.h"

using namespace std;

namespace Wayland::printUtils
{

    inline string checkIndex(uint32_t idx)
{
    return idx == minipbrt::kInvalidIndex ? "null" : to_string(idx);
}

string toString(glm::vec3 v)
{
    ostringstream oss;
    oss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return oss.str();
}

string toString(minipbrt::Scene* scene)
{
    ostringstream oss;
    oss << "  Shape: " << scene->shapes.size() << "\n";
    oss << "  Object: " << scene->objects.size() << "\n";
    oss << "  Instance: " << scene->instances.size() << "\n";
    oss << "  Light: " << scene->lights.size() << "\n";
    oss << "  AreaLight: " << scene->areaLights.size() << "\n";
    oss << "  Material: " << scene->materials.size() << "\n";
    oss << "  Medium: " << scene->mediums.size() << "\n";
    oss << "\n----- shapes -----\n";
    for (int i = 0; i < scene->shapes.size(); ++i)
        oss << "  shape " << i << "\n" << toString(scene->shapes[i]) << "\n";

    return oss.str();
}

string toString(minipbrt::Shape *shape)
{
    ostringstream oss;
    oss << "  type: " << PBRTv3::ShapeTypeStr[int(shape->type())] << "\n";
    oss << "  areaLight: " << checkIndex(shape->areaLight) << "\n";
    oss << "  material:  " << checkIndex(shape->material) << "\n"; 
    if (shape->type() == minipbrt::ShapeType::TriangleMesh)
    {
        auto mesh = dynamic_cast<minipbrt::TriangleMesh*>(shape);
        oss << "  vertices: " << mesh->num_vertices << "\n";
        oss << "  indices: " << mesh->num_indices << "\n";
    }
    return oss.str();
}


} // namespace Wayland