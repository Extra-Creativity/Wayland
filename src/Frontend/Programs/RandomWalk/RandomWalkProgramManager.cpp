#pragma once
#include "RandomWalkProgramManager.h"
#include <assert.h>

using namespace EasyRender::Device;
using namespace EasyRender::Programs::RandomWalk;

namespace EasyRender
{

void RandomWalkProgramManager::Setup()
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
}

void RandomWalkProgramManager::Update()
{
    // std::cout << "frame " << param.frameID << "\n";
    param.frameID += 1;
    param.fbSize = renderer->window.size;
    param.colorBuffer = (glm::u8vec4 *)renderer->device.d_FrameBuffer;
    param.traversable = renderer->device.GetTraversableHandle();
}

void RandomWalkProgramManager::End()
{
    /* I believe this part should be put to DeviceManager/SceneManager. This is
     * just a temporary solution. */
    assert(param.radianceBuffer);
    cudaFree(param.radianceBuffer);
}

Optix::ShaderBindingTable RandomWalkProgramManager::GenerateSBT(
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
    auto &device = renderer->device;
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
                scene.materials[matIdx]->type() == MaterialType::Disney)
            {
                auto m = static_cast<Disney *>(scene.materials[matIdx].get());
                m->ToDevice(hitDatas[idx].data.disneyMat);
                if (m->HasTexture())
                {
                    hitDatas[idx].data.hasTexture = true;
                    hitDatas[idx].data.texture =
                        renderer->device.textureObjects[m->textureId];
                }
                else
                {
                    hitDatas[idx].data.hasTexture = false;
                }
            }

            hitDatas[idx].data.areaLightID = scene.meshes[meshID]->areaLight;
            hitDatas[idx].data.materialID = scene.meshes[meshID]->material;
            hitDatas[idx].data.meshID = meshID;
            hitDatas[idx].data.hasNormal = scene.meshes[meshID]->hasNormal;
            hitDatas[idx].data.vertices =
                renderer->device.d_VertexBuffer + scene.vertexOffset[meshID];
            hitDatas[idx].data.normals =
                device.d_NormalBuffer + scene.vertexOffset[meshID];
            hitDatas[idx].data.texcoords =
                renderer->device.d_TexcoordBuffer + scene.vertexOffset[meshID];
            hitDatas[idx].data.indices =
                device.d_IndexBuffer + scene.triangleOffset[meshID];
        }
    }

    return ShaderBindingTable{ raygenData,
                               0,
                               missData,
                               missIdx.data(),
                               hitDatas,
                               hitIdx.data(),
                               pg };
}

} // namespace EasyRender
