#pragma once

#include "Optix/Core/ContextManager.h"
#include <filesystem>
#include <string_view>

namespace Wayland::Example
{

Wayland::Optix::ContextManager SetupEnvironment();

void SaveImageImpl(const std::filesystem::path &path, int width, int height,
                   std::size_t elemSize, void *gpuBuffer);

template<typename T>
void SaveImage(const std::filesystem::path &path, int width, int height,
               T *gpuBuffer)
{
    SaveImageImpl(path, width, height, sizeof(T), gpuBuffer);
}

struct StringHasher
{
    using is_transparent = void;
    std::size_t operator()(std::string_view view) const noexcept
    {
        return std::hash<std::string_view>{}(view);
    }
};

} // namespace Wayland::Example