#pragma once
#include "Core/ProgramManager.h"
#include "Core/Renderer.h"
#include "WireFrameLaunchParams.h"

namespace EasyRender
{

class WireFrameProgramManager : public ProgramManager
{
public:
    WireFrameProgramManager(Renderer *r_) : renderer(r_) {}

    void Setup()
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
        param.camera.right =
            pCam->right * (float(param.fbSize.x) / param.fbSize.y);

        int bSize = param.fbSize.x * param.fbSize.y * sizeof(glm::vec4);
        cudaMalloc(&param.radianceBuffer, bSize);
        cudaMemset(param.radianceBuffer, 0, bSize);
    }

    void Update()
    {
        param.frameID += 1;
        param.fbSize = renderer->window.size;
        param.colorBuffer = (glm::u8vec4 *)renderer->device.d_FrameBuffer;
        param.traversable = renderer->device.GetTraversableHandle();
    }

    void End() {}

    void *GetParamPtr() { return &param; }
    size_t GetParamSize() { return sizeof(param); };

    Optix::ShaderBindingTable GenerateSBT(const Optix::ProgramGroupArray &pg)
    {
        using namespace Optix;
        SBTData<void> raygenData{};
        SBTData<int> missData{};
        std::vector<SBTData<int>> hitData;
        hitData.resize(renderer->scene.meshes.size());
        std::vector<std::size_t> hitIdx(renderer->scene.meshes.size(), 2);
        return ShaderBindingTable{
            raygenData, 0, missData, 1, hitData, hitIdx.data(), pg
        };
    }

private:
    Renderer *renderer;
    Programs::WireFrame::LaunchParams param;
};

} // namespace EasyRender
