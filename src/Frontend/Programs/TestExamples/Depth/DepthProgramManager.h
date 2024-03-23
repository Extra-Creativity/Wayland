#pragma once
#include "Core/ProgramManager.h"
#include "Core/Renderer.h"
#include "DepthLaunchParams.h"

namespace Wayland
{

class DepthProgramManager : public ProgramManager
{
public:
    DepthProgramManager(Renderer *r_) : renderer(r_) {}

    void Setup()
    {
        param.frameID = 0;
        param.fbSize.x = renderer->window.size.w;
        param.fbSize.y = renderer->window.size.h;
        param.colorBuffer = (glm::u8vec4 *)renderer->device.deviceFrameBuffer;
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

        /* !!! Malloc but not free currently, don't forget` !!!*/
        cudaMalloc(&param.depthBuffer,
                   param.fbSize.x * param.fbSize.y * sizeof(float));
    }

    void Update()
    {
        param.frameID += 1;
        param.fbSize.x = renderer->window.size.w;
        param.fbSize.y = renderer->window.size.h;
        param.colorBuffer = (glm::u8vec4 *)renderer->device.deviceFrameBuffer;
        param.traversable = renderer->device.GetTraversableHandle();

        /* Get max depth at first frame */
        if (param.frameID == 1)
        {
            int size = param.fbSize.x * param.fbSize.y;
            float *hostDepthBuffer = new float[size];
            cudaMemcpy(hostDepthBuffer, param.depthBuffer,
                       size * sizeof(float),
                       cudaMemcpyDeviceToHost);
            param.maxDepth = 0;
            param.minDepth = 1e30;
            for (int i = 0; i < size; ++i)
            {
                param.maxDepth = max(param.maxDepth, hostDepthBuffer[i]);
                param.minDepth = min(param.minDepth, hostDepthBuffer[i] > 0
                                                         ? hostDepthBuffer[i]
                                                         : param.minDepth);
            }
            delete []hostDepthBuffer;
        }
    }

    void *GetParamPtr() { return &param; }
    size_t GetParamSize() { return sizeof(param); };

    Optix::ShaderBindingTable GenerateSBT(const Optix::ProgramGroupArray &pg)
    {
        using namespace Optix;
        SBTData<void> raygenData{};
        SBTData<int> missData{};
        vector<SBTData<int>> hitData;
        hitData.resize(renderer->scene.meshes.size());
        vector<std::size_t> hitIdx(renderer->scene.meshes.size(), 2);
        return ShaderBindingTable{
            raygenData, 0, missData, 1, std::span(hitData), hitIdx.data(), pg
        };
    }

private:
    Renderer *renderer;
    DepthLaunchParams param;
};

} // namespace Wayland
