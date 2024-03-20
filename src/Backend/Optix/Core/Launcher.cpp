#include "Launcher.h"
#include "HostUtils/CommonHeaders.h"
#include "HostUtils/ErrorCheck.h"
#include "Pipeline.h"
#include "ShaderBindingTable.h"

using namespace Wayland;

namespace Wayland::Optix
{

void Launcher::Launch(const Pipeline &pipeline, CUstream stream,
                      const ShaderBindingTable &sbt, unsigned int width,
                      unsigned int height, unsigned int depth)
{
    HostUtils::CheckOptixError(optixLaunch(
        pipeline.GetHandle(), stream, HostUtils::ToDriverPointer(buffer_.get()),
        size_, &sbt.GetHandle(), width, height, depth));
    return;
}

} // namespace Wayland::Optix