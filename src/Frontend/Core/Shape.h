#pragma once
#include <vector>
#include <memory>
#include <iterator>

#include "glm/mat4x4.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "minipbrt.h"

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

    std::vector<glm::vec3> vertex;
    std::vector<glm::vec3> normal;
    std::vector<glm::vec2> uv;
    std::vector<glm::ivec3> triangle;
    uint32_t material;
    uint32_t areaLight;
    bool hasNormal;
    bool hasUV;
};

using TriangleMeshPtr = std::unique_ptr<TriangleMesh>;

} // namespace EasyRender
