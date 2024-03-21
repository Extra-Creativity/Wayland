#pragma once
#include "Core/Shape.h"

namespace Wayland
{

TriangleMesh::TriangleMesh(minipbrt::TriangleMesh *miniMesh)
{
    assert(miniMesh);
    assert(miniMesh->num_indices % 3 == 0);
    /* Must have position, normal, indices */
    assert(miniMesh->P);
    assert(miniMesh->N);
    assert(miniMesh->indices);

    int nVertices = miniMesh->num_vertices;
    int nIndices = miniMesh->num_indices/3;
    for (int i = 0; i < nVertices; ++i)
    {
        vertex.push_back(glm::make_vec3(&miniMesh->P[3 * i]));
        normal.push_back(glm::make_vec3(&miniMesh->N[3 * i]));
    }
    for (int i = 0; i < nIndices; ++i)
    {
		index.push_back(glm::make_vec3(&miniMesh->indices[3 * i]));
	}

    /* uv is optional, so we need to check first */
    if (miniMesh->uv)
        for (int i = 0; i < nVertices; ++i)
            uv.push_back(glm::make_vec2(&miniMesh->uv[2 * i]));
}

} // namespace Wayland
