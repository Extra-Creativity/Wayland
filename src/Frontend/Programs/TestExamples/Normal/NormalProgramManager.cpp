#pragma once
#include "NormalProgramManager.h"

using namespace EasyRender::Device;
using namespace EasyRender::Programs::Normal;

namespace EasyRender
{

void NormalProgramManager::Setup()
{
    param.frameID = 0;
    param.fbSize = renderer->window.size;
    param.colorBuffer = (glm::u8vec4 *)renderer->device.d_FrameBuffer;
    param.traversable = renderer->device.GetTraversableHandle();

    Camera *cam = renderer->scene.camera.get();
    assert(cam);
    assert(cam->type() == CameraType::Pinhole);
    PinholeCamera *pCam = reinterpret_cast<PinholeCamera *>(cam);
    param.camera.pos = pCam->position;
    param.camera.lookAt = pCam->lookAt;
    param.camera.up = pCam->up;
    param.camera.right = pCam->right * (float(param.fbSize.x) / param.fbSize.y);
}

void NormalProgramManager::Update()
{
    param.frameID += 1;
    param.fbSize = renderer->window.size;
    param.colorBuffer = (glm::u8vec4 *)renderer->device.d_FrameBuffer;
    param.traversable = renderer->device.GetTraversableHandle();
}

void NormalProgramManager::End()
{

}

Optix::ShaderBindingTable NormalProgramManager::GenerateSBT(
    const Optix::ProgramGroupArray &pg)
{
    using namespace Optix;
    SBTData<void> raygenData{};
    SBTData<MissData> missData{};
    missData.data.bg_color = { 0, 0, 0 };
    std::vector<SBTData<HitData>> hitData;
    auto &s = renderer->scene;
    std::vector<std::size_t> hitIdx(s.meshes.size(), 2);
    hitData.resize(s.meshes.size());

    /* I believe this part should be put in DeviceManager/SceneManager. This is
     * just a temporary solution. */
    std::vector<glm::vec3> normalAggregate;
    std::vector<glm::ivec3> indexAggregate;

    int nSize = s.vertexNum, iSize = s.triangleNum;

    normalAggregate.resize(nSize);
    indexAggregate.resize(iSize);
    for (int i = 0; i < s.meshes.size(); ++i)
    {
        auto *mesh = s.meshes[i].get();
        int n_s = mesh->normal.size();
        int i_s = mesh->triangle.size();
        memcpy(normalAggregate.data() + s.vertexOffset[i], mesh->normal.data(),
               mesh->normal.size() * sizeof(glm::vec3));
        memcpy(indexAggregate.data() + s.triangleOffset[i], mesh->triangle.data(),
               mesh->triangle.size() * sizeof(glm::ivec3));
    }
    normalBuffer = renderer->device.d_NormalBuffer;
    indexBuffer = renderer->device.d_IndexBuffer;
    cudaMemcpy(normalBuffer, normalAggregate.data(), nSize * sizeof(glm::vec3),
               cudaMemcpyHostToDevice);
    cudaMemcpy(indexBuffer, indexAggregate.data(), iSize * sizeof(glm::ivec3),
               cudaMemcpyHostToDevice);

    for (int i = 0; i < hitData.size(); ++i)
    {
        uint32_t matIdx = s.meshes[i]->material;
        if (matIdx < INVALID_INDEX &&
            s.materials[matIdx]->type() == MaterialType::Diffuse)
        {
            hitData[i].data.Kd =
                static_cast<Diffuse *>(s.materials[matIdx].get())->Kd;
        }
        else
        {
            hitData[i].data.Kd = { 0.8, 0.8, 0.8 };
        }
        hitData[i].data.meshID = i;
        hitData[i].data.normals = normalBuffer + s.vertexOffset[i];
        hitData[i].data.indices = indexBuffer + s.triangleOffset[i];
    }

    return ShaderBindingTable{
        raygenData, 0, missData, 1, std::span(hitData), hitIdx.data(), pg
    };
}

} // namespace EasyRender
