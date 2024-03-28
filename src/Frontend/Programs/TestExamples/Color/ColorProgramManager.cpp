#pragma once
#include "ColorProgramManager.h"

using namespace EasyRender::Device;
using namespace EasyRender::Programs::Color;

namespace EasyRender
{

void ColorProgramManager::Setup()
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

void ColorProgramManager::Update()
{
    param.frameID += 1;
    param.fbSize = renderer->window.size;
    param.colorBuffer = (glm::u8vec4 *)renderer->device.d_FrameBuffer;
    param.traversable = renderer->device.GetTraversableHandle();
}

Optix::ShaderBindingTable ColorProgramManager::GenerateSBT(
    const Optix::ProgramGroupArray &pg)
{
    using namespace Optix;
    SBTData<void> raygenData{};
    SBTData<MissData> missData{};
    missData.data.bg_color = { 1, 1, 1, 1 };
    std::vector<SBTData<HitData>> hitData;
    auto &scene = renderer->scene;
    std::vector<std::size_t> hitIdx(scene.meshes.size(), 2);
    hitData.resize(scene.meshes.size());

    for (int i = 0; i < hitData.size(); ++i)
    {
        uint32_t matIdx = scene.meshes[i]->material;
        if (matIdx < INVALID_INDEX &&
            scene.materials[matIdx]->type() == MaterialType::Diffuse)
        {
            hitData[i].data.Kd =
                static_cast<Diffuse*>( scene.materials[matIdx].get())->Kd;
        }
        else
        {
            hitData[i].data.Kd = { 0.8, 0.8, 0.8 };
        }
    }
    return ShaderBindingTable{
        raygenData, 0, missData, 1, std::span(hitData), hitIdx.data(), pg
    };
}

} // namespace EasyRender
