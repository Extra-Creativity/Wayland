#pragma once
#include "PathTracingProgramManager.h"


using namespace EasyRender::Programs::PathTracing;

namespace EasyRender
{

void PathTracingProgramManager::Setup()
{
    param.frameID = 0;
    param.fbSize = renderer->window.size;
    param.colorBuffer = (glm::u8vec4 *)renderer->device.deviceFrameBuffer;
    param.traversable = renderer->device.GetTraversableHandle();

    Camera *cam = renderer->scene.camera.get();
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
}

void PathTracingProgramManager::Update()
{
    //std::cout << "frame " << param.frameID << "\n";
    param.frameID += 1;
    param.fbSize = renderer->window.size;
    param.colorBuffer = (glm::u8vec4 *)renderer->device.deviceFrameBuffer;
    param.traversable = renderer->device.GetTraversableHandle();
}


void PathTracingProgramManager::End()
{
    /* I believe this part should be put to DeviceManager/SceneManager. This is
     * just a temporary solution. */
    assert(normalBuffer);
    assert(indexBuffer);
    assert(param.radianceBuffer);
    cudaFree(normalBuffer);
    cudaFree(indexBuffer);
    cudaFree(param.radianceBuffer);
}

Optix::ShaderBindingTable PathTracingProgramManager::GenerateSBT(
    const Optix::ProgramGroupArray &pg)
{
    using namespace Optix;
    SBTData<void> raygenData{};
    SBTData<MissData> missData{};
    missData.data.bg_color = { 0, 0, 0 };
    std::vector<SBTData<HitData>> hitData;
    auto &scene = renderer->scene;
    std::vector<std::size_t> hitIdx(scene.meshes.size(), 2);
    hitData.resize(scene.meshes.size());

    /* I believe this part should be put in DeviceManager/SceneManager. This is
     * just a temporary solution. */
    std::vector<glm::vec3> normalAggregate;
    std::vector<int> normalOffset{ 0 };
    std::vector<glm::ivec3> indexAggregate;
    std::vector<int> indexOffset{ 0 };
    int nSize = 0, iSize = 0;
    for (int i = 0; i < scene.meshes.size(); ++i)
    {
        auto *mesh = scene.meshes[i].get();
        int n_s = mesh->normal.size();
        int i_s = mesh->index.size();
        nSize += n_s;
        iSize += i_s;
        normalOffset.push_back(nSize);
        indexOffset.push_back(iSize);
    }
    normalAggregate.resize(nSize);
    indexAggregate.resize(iSize);
    for (int i = 0; i < scene.meshes.size(); ++i)
    {
        auto *mesh = scene.meshes[i].get();
        int n_s = mesh->normal.size();
        int i_s = mesh->index.size();
        memcpy(normalAggregate.data() + normalOffset[i], mesh->normal.data(),
               mesh->normal.size() * sizeof(glm::vec3));
        memcpy(indexAggregate.data() + indexOffset[i], mesh->index.data(),
               mesh->index.size() * sizeof(glm::ivec3));
    }
    cudaMalloc(&normalBuffer, nSize * sizeof(glm::vec3));
    cudaMalloc(&indexBuffer, iSize * sizeof(glm::ivec3));
    cudaMemcpy(normalBuffer, normalAggregate.data(), nSize * sizeof(glm::vec3),
               cudaMemcpyHostToDevice);
    cudaMemcpy(indexBuffer, indexAggregate.data(), iSize * sizeof(glm::ivec3),
               cudaMemcpyHostToDevice);

    for (int i = 0; i < hitData.size(); ++i)
    {
        uint32_t matIdx = scene.meshes[i]->material;
        if (matIdx < INVALID_INDEX &&
            scene.materials[matIdx]->type() == MaterialType::Diffuse)
        {
            hitData[i].data.Kd =
                static_cast<Diffuse *>(scene.materials[matIdx].get())->Kd;
        }
        else
        {
            hitData[i].data.Kd = { 0.0, 0.0, 0.0 };
        }
        uint32_t lightIdx = scene.meshes[i]->areaLight;
        if (lightIdx < INVALID_INDEX)
        {
            hitData[i].data.L = scene.lights[lightIdx].get()->L;
        }
        else
        {
            hitData[i].data.L = { -1, -1, -1 };
        }
        hitData[i].data.meshID = i;
        hitData[i].data.normals = normalBuffer + normalOffset[i];
        hitData[i].data.indices = indexBuffer + indexOffset[i];
    }

    return ShaderBindingTable{
        raygenData, 0, missData, 1, std::span(hitData), hitIdx.data(), pg
    };
}

} // namespace EasyRender
