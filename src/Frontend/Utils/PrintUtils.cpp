#pragma once
#include <iostream>
#include <sstream>
#include <string>

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "minipbrt.h"
#include "Utils/PBRTv3_SceneCommon.h"
#include "Utils/PrintUtils.h"

using namespace std;

namespace EasyRender::PrintUtils
{

inline string checkIndex(uint32_t idx)
{
    return idx == minipbrt::kInvalidIndex ? "null" : to_string(idx);
}

string ToString(glm::vec3 v)
{
    ostringstream oss;
    oss << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return oss.str();
}

string ToString(minipbrt::Scene *scene)
{
    ostringstream oss;
    oss << "  Shape: " << scene->shapes.size() << "\n";
    oss << "  Object: " << scene->objects.size() << "\n";
    oss << "  Instance: " << scene->instances.size() << "\n";
    oss << "  Light: " << scene->lights.size() << "\n";
    oss << "  AreaLight: " << scene->areaLights.size() << "\n";
    oss << "  Material: " << scene->materials.size() << "\n";
    oss << "  Medium: " << scene->mediums.size() << "\n";
    oss << "\n----- materials -----\n";
    for (int i = 0; i < scene->materials.size(); ++i)
        oss << "  material " << i << "\n" << ToString(scene->materials[i]) << "\n";
    oss << "\n----- shapes -----\n";
    for (int i = 0; i < scene->shapes.size(); ++i)
        oss << "  shape " << i << "\n" << ToString(scene->shapes[i]) << "\n";

    return oss.str();
}

string ToString(minipbrt::Material *material)
{
	ostringstream oss;
	oss << "  type: " << PBRTv3::MaterialTypeStr[int(material->type())] << "\n";
    oss << "  name: " << material->name << "\n";
    if (material->type() == minipbrt::MaterialType::Matte)
    {
        auto mat = dynamic_cast<minipbrt::MatteMaterial *>(material);
        oss << "  Kd: " << ToString(glm::make_vec3(mat->Kd.value)) << "\n";
    }
	return oss.str();
}

string ToString(minipbrt::Shape *shape)
{
    ostringstream oss;
    oss << "  type: " << PBRTv3::ShapeTypeStr[int(shape->type())] << "\n";
    oss << "  areaLight: " << checkIndex(shape->areaLight) << "\n";
    oss << "  material:  " << checkIndex(shape->material) << "\n";
    if (shape->type() == minipbrt::ShapeType::TriangleMesh)
    {
        auto mesh = dynamic_cast<minipbrt::TriangleMesh *>(shape);
        oss << "  vertices: " << mesh->num_vertices << "\n";
        oss << "  indices: " << mesh->num_indices << "\n";
    }
    return oss.str();
}

} // namespace EasyRender::PrintUtils