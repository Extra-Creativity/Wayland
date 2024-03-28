#pragma once
#include <memory>
#include <string>

#include "Core/MainWindow.h"
#include "Core/Optix-All.h"
#include "Core/ProgramManager.h"
#include "Core/SceneManager.h"
#include "Device/Light.h"
#include "glm/glm.hpp"

namespace EasyRender
{

using TriangleBuildInputArrayPtr = std::unique_ptr<Optix::TriangleBuildInputArray>;
using StaticAccelStructurePtr = std::unique_ptr<Optix::StaticAccelStructure>;
using ProgramGroupArrayPtr = std::unique_ptr<Optix::ProgramGroupArray>;
using ShaderBindingTablePtr = std::unique_ptr<Optix::ShaderBindingTable>;
using ModulePtr = std::unique_ptr<Optix::Module>;
using PipelinePtr = std::unique_ptr<Optix::Pipeline>;

class DeviceManager
{
public:
    DeviceManager();
    void SetupOptix(SceneManager &scene, MainWindow &window,
                    std::string_view programSrc, ProgramManager *program);
    void Launch(ProgramManager *program, glm::ivec2 wSize);
    void DownloadFrameBuffer(MainWindow &window) const;
    OptixTraversableHandle GetTraversableHandle();

private:
    void SetEnvironment();
    void BuildAccelStructure(SceneManager &scene);
    void BuildPipeline(std::string_view programSrc);
    void BuildSBT(ProgramManager *program);
    void AllocDeviceBuffer(SceneManager &scene, glm::ivec2 wSize);
    //void FreeDeviceBuffer();


public:
    /* Device frame buffer */
    glm::u8vec4 *d_FrameBuffer;
    uint32_t frameBufferSize;
    /* Device mesh index buffer, pass to program via SBT */
    glm::ivec3 *d_IndexBuffer;
    uint32_t indexBufferSize;
    /* Device mesh normal buffer, pass to program via SBT */
    glm::vec3 *d_NormalBuffer;
    uint32_t normalBufferSize;
    /* Device areaLight buffer, pass to program via LaunchParams */
    /* This two buffers are consecutive, and malloced together */
    Device::DeviceAreaLight *d_AreaLightBuffer; 
    glm::vec3 *d_AreaLightVertexBuffer;
    uint32_t areaLightBufferSize;   

private:
    Optix::ContextManager contextManager;
    /* AS require memory in buildInput */
    TriangleBuildInputArrayPtr asBuildInput;
    StaticAccelStructurePtr as;
    ProgramGroupArrayPtr pg;
    ModulePtr module;
    PipelinePtr pipeline;
    ShaderBindingTablePtr sbt;
};

} // namespace EasyRender