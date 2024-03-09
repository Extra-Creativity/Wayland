#include "Common.h"

#include "cuda_runtime.h"
#include "stb_image/stb_image_write.h"

#include "Optix/Core/Module.h"

namespace Wayland::Example
{

Wayland::Optix::ContextManager SetupEnvironment()
{
    Wayland::Optix::ContextManager manager;
    manager.SetCachePath(".");
    Wayland::Optix::Module::SetOptixSDKPath(OPTIX_DIR);
    Wayland::Optix::Module::AddArgs("-I\"" GLM_DIR "\"");
    Wayland::Optix::Module::AddArgs("-I\"" UTILS_INC_PATH "\"");
    Wayland::Optix::Module::AddArgs("-l\"" DEVICE_UTILS_LIB_PATH "\"");
    Wayland::Optix::Module::AddArgs("-diag-suppress 20012 -diag-suppress 3012");
    return manager;
}

void SaveImageImpl(const std::filesystem::path &path, int width, int height,
                   std::size_t elemSize, void *gpuBuffer)
{
    auto bufferSize = width * height * elemSize;
    auto cpuBuffer = std::make_unique_for_overwrite<std::byte[]>(bufferSize);
    cudaMemcpy(cpuBuffer.get(), gpuBuffer, bufferSize, cudaMemcpyDeviceToHost);

    [[maybe_unused]] std::string pathBuffer;
    const char *rawPath;
    if constexpr (std::is_same_v<std::filesystem::path::value_type, char>)
    { // cast is just to disable stupid compiler error.
        rawPath = (const char *)path.c_str();
    }
    else
    {
        pathBuffer = path.string();
        rawPath = pathBuffer.c_str();
    }

    HostUtils::CheckError(
        stbi_write_jpg(rawPath, width, height, 4, cpuBuffer.get(), 90) != 0,
        ("Cannot write image to " + path.string()).c_str());
}

} // namespace Wayland::Example