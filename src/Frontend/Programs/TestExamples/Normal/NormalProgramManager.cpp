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

void NormalProgramManager::End() {}

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
        hitData[i].data.hasNormal = s.meshes[i]->hasNormal;
        hitData[i].data.vertices = 
            renderer->device.d_VertexBuffer + s.vertexOffset[i];
        hitData[i].data.normals =
            renderer->device.d_NormalBuffer + s.vertexOffset[i];
        hitData[i].data.indices =
            renderer->device.d_IndexBuffer + s.triangleOffset[i];
    }

    return ShaderBindingTable{
        raygenData, 0, missData, 1, hitData, hitIdx.data(), pg
    };
}

} // namespace EasyRender
