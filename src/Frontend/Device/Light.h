#pragma once
#include "glm/glm.hpp"

namespace EasyRender::Device
{
/* Support general mesh-based areaLight */
/* TBD: Importance sampling */
struct DeviceAreaLight
{
    int triangleNum;
    int twoSided;
    glm::vec3 L;
    glm::vec3 *vertices;
    glm::vec3 *normals;
    glm::ivec3 *indices;

    __host__ __device__ __forceinline__ void print()
    {
        printf("triangle: %d\n", triangleNum);
        printf("twoSided: %d\n", twoSided);
        printf("L: %f %f %f\n", L.x, L.y, L.z);
        printf("vertices[0]: %f %f %f\n", vertices[0].x, vertices[0].y,
               vertices[0].z);
        printf("normals[0]: %f %f %f\n", normals[0].x, normals[0].y,
               normals[0].z);
        printf("indices[0]: %d %d %d\n", indices[0].x, indices[0].y,
               indices[0].z);
    }
};

} // namespace EasyRender::Device