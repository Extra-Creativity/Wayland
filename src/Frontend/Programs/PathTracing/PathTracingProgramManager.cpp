#pragma once
#include "PathTracingProgramManager.h"
#include <assert.h>

using namespace EasyRender::Device;
using namespace EasyRender::Programs::PathTracing;

namespace EasyRender
{

void PathTracingProgramManager::Setup()
{
    param.frameID = 0;
    param.fbSize = renderer->window.size;

    auto &scene = renderer->scene;
    auto &device = renderer->device;
    param.colorBuffer = (glm::u8vec4 *)device.d_FrameBuffer;
    param.traversable = device.GetTraversableHandle();

    Camera *cam = scene.camera.get();
    assert(cam);
    assert(cam->type() == CameraType::Pinhole);
    PinholeCamera *pCam = reinterpret_cast<PinholeCamera *>(cam);
    param.camera.pos = pCam->position;
    param.camera.lookAt = pCam->lookAt;
    param.camera.up = pCam->up;
    param.camera.right = pCam->right * (float(param.fbSize.x) / param.fbSize.y);

    int bSize = param.fbSize.x * param.fbSize.y * sizeof(glm::vec4);
    cudaMalloc(&param.radianceBuffer, bSize);
    cudaMemset(param.radianceBuffer, 0, bSize);

    /* Prepare areaLightBuffer */
    auto &lights = scene.lights;
    param.areaLights = renderer->device.d_AreaLightBuffer;
    param.areaLightCount = lights.size();
    /* Use a little trick to malloc void buffer in Cpp */
    void *hostBuf = ::operator new(device.areaLightBufferSize);
    auto *lightBuf = reinterpret_cast<DeviceAreaLight *>(hostBuf);
    auto *vertexBuf = reinterpret_cast<glm::vec3 *>(lightBuf + lights.size());

    uint32_t v_offset = 0;
    for (int i = 0; i < lights.size(); ++i)
    {
        lightBuf[i].L = lights[i]->L;
        lightBuf[i].twoSided = lights[i]->twoSided;
        uint32_t meshIdx = lights[i]->mesh;
        /* AreaLight must have a corresponding mesh */
        assert(meshIdx < INVALID_INDEX);
        lightBuf[i].normals = normalBuffer + scene.vertexOffset[meshIdx];
        lightBuf[i].indices = indexBuffer + scene.triangleOffset[meshIdx];
        lightBuf[i].vertices = device.d_AreaLightVertexBuffer + v_offset;
        lightBuf[i].triangleNum = scene.meshes[meshIdx]->triangle.size();

        auto &mVertex = scene.meshes[meshIdx]->vertex;
        memcpy(vertexBuf + v_offset, mVertex.data(),
               mVertex.size() * sizeof(glm::vec3));
        v_offset += mVertex.size();
    }
    cudaMemcpy(device.d_AreaLightBuffer, hostBuf, device.areaLightBufferSize,
               cudaMemcpyHostToDevice);
    ::operator delete(hostBuf);
}

void PathTracingProgramManager::Update()
{
    // std::cout << "frame " << param.frameID << "\n";
    param.frameID += 1;
    param.fbSize = renderer->window.size;
    param.colorBuffer = (glm::u8vec4 *)renderer->device.d_FrameBuffer;
    param.traversable = renderer->device.GetTraversableHandle();
}

void PathTracingProgramManager::End()
{
    /* I believe this part should be put to DeviceManager/SceneManager. This is
     * just a temporary solution. */
    assert(param.radianceBuffer);
    cudaFree(param.radianceBuffer);
}

Optix::ShaderBindingTable PathTracingProgramManager::GenerateSBT(
    const Optix::ProgramGroupArray &pg)
{
    using namespace Optix;
    SBTData<void> raygenData{};

    uint32_t rayTypeCount = static_cast<uint32_t>(RAY_TYPE_COUNT);

    std::vector<SBTData<MissData>> missData{};
    std::vector<std::size_t> missIdx{ 1, 2 };
    missData.resize(rayTypeCount);
    for (auto &m : missData)
        m.data.bg_color = { 0, 0, 0 };

    auto &scene = renderer->scene;

    /* I believe this part should be put in DeviceManager/SceneManager. This is
     * just a temporary solution. */
    std::vector<glm::vec3> normalAggregate;
    std::vector<glm::ivec3> indexAggregate;

    uint32_t nSize = scene.vertexNum, iSize = scene.triangleNum;

    normalAggregate.resize(nSize);
    indexAggregate.resize(iSize);
    for (int i = 0; i < scene.meshes.size(); ++i)
    {
        auto *mesh = scene.meshes[i].get();
        int n_s = mesh->normal.size();
        int i_s = mesh->triangle.size();
        memcpy(normalAggregate.data() + scene.vertexOffset[i],
               mesh->normal.data(), mesh->normal.size() * sizeof(glm::vec3));
        memcpy(indexAggregate.data() + scene.triangleOffset[i],
               mesh->triangle.data(),
               mesh->triangle.size() * sizeof(glm::ivec3));
    }
    normalBuffer = renderer->device.d_NormalBuffer;
    indexBuffer = renderer->device.d_IndexBuffer;

    cudaMemcpy(normalBuffer, normalAggregate.data(), nSize * sizeof(glm::vec3),
               cudaMemcpyHostToDevice);
    cudaMemcpy(indexBuffer, indexAggregate.data(), iSize * sizeof(glm::ivec3),
               cudaMemcpyHostToDevice);

    
    std::vector<SBTData<HitData>> hitDatas;
    std::vector<std::size_t> hitIdx(scene.meshes.size() * rayTypeCount);
    hitDatas.resize(scene.meshes.size() * rayTypeCount);

    for (size_t meshID = 0; meshID < scene.meshes.size(); ++meshID)
    {
        for (uint32_t rayID = 0; rayID < rayTypeCount; ++rayID)
        {
            uint32_t idx = static_cast<uint32_t>(meshID * rayTypeCount + rayID);
            if (rayID == RADIANCE_TYPE)
            {
                hitIdx[idx] = 3;
            }
            else if (rayID == SHADOW_TYPE)
            {
                hitIdx[idx] = 4;
                continue;
            }

            uint32_t matIdx = scene.meshes[meshID]->material;
            if (matIdx < INVALID_INDEX &&
                scene.materials[matIdx]->type() == MaterialType::Diffuse)
            {
                hitDatas[idx].data.Kd =
                    static_cast<Diffuse *>(scene.materials[matIdx].get())->Kd;
            }
            else
            {
                hitDatas[idx].data.Kd = { 0.0, 0.0, 0.0 };
            }
            hitDatas[idx].data.areaLightID = scene.meshes[meshID]->areaLight;
            hitDatas[idx].data.materialID = scene.meshes[meshID]->material;
            hitDatas[idx].data.meshID = meshID;
            hitDatas[idx].data.normals = normalBuffer + scene.vertexOffset[meshID];
            hitDatas[idx].data.indices = indexBuffer + scene.triangleOffset[meshID];
        }
    }

    return ShaderBindingTable{ raygenData,
                               0,
                               std::span(missData),
                               missIdx.data(),
                               std::span(hitDatas),
                               hitIdx.data(),
                               pg };
}

} // namespace EasyRender
