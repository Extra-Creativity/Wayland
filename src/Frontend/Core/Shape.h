#pragma once
#include <vector>
#include <memory>
#include <iterator>

#include "glm/mat4x4.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "minipbrt.h"

using std::vector, std::unique_ptr, glm::vec2, glm::vec3, glm::ivec3;

namespace EasyRender
{
/*! a simple indexed triangle mesh that our sample renderer will
    render */
class TriangleMesh
{
public:
    TriangleMesh() = default;
    TriangleMesh(minipbrt::TriangleMesh* miniMesh);
    ~TriangleMesh() = default;


public:
    vector<vec3> vertex;
    vector<vec3> normal;
    vector<vec2> uv;
    vector<ivec3> index;

};

using TriangleMeshPtr = unique_ptr<TriangleMesh>;

} // namespace EasyRender
