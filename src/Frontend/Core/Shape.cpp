#include "Core/Shape.h"
#include "Utils/Common.h"

namespace EasyRender
{

TriangleMesh::TriangleMesh(minipbrt::TriangleMesh *miniMesh)
{
    assert(miniMesh);
    assert(miniMesh->num_indices % 3 == 0);
    /* Must have position, normal, indices */
    assert(miniMesh->P);
    assert(miniMesh->N);
    assert(miniMesh->indices);

    using namespace glm;
    glm::mat4 t = transpose(make_mat4(&miniMesh->shapeToWorld.start[0][0]));

    int nVertices = miniMesh->num_vertices;
    int nIndices = miniMesh->num_indices/3;

    for (int i = 0; i < nVertices; ++i)
    {
        /* Simple and slow shape transform, should optimize */
        vertex.push_back(TransformVec3(t, &miniMesh->P[3 * i], 1));
        normal.push_back(TransformVec3(t, &miniMesh->N[3 * i], 1));
    }
    for (int i = 0; i < nIndices; ++i)
    {
		index.push_back(glm::make_vec3(&miniMesh->indices[3 * i]));
	}

    /* uv is optional, so we need to check first */
    if (miniMesh->uv)
        for (int i = 0; i < nVertices; ++i)
            uv.push_back(glm::make_vec2(&miniMesh->uv[2 * i]));

    /* Bind material*/
    material = miniMesh->material;
    /* Bind areaLight */
    areaLight = miniMesh->areaLight;

}

} // namespace EasyRender
