#pragma once
#include <memory>
#include <string>

#include "Core/MainWindow.h"
#include "Core/Optix-All.h"
#include "Core/ProgramManager.h"
#include "Core/SceneManager.h"
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

public:
    void *deviceFrameBuffer;

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