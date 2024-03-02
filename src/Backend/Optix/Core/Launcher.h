#pragma once

#include "HostUtils/DeviceAllocators.h"

namespace Wayland::Optix
{

class Pipeline;
class ShaderBindingTable;

/// @brief The final procedure to launch a ray by optix.
class Launcher
{
public:
    /// @brief Construct a launcher with data that can be safely copied to
    /// device in bytes.
    template<typename T>
        requires std::is_trivially_copyable_v<std::remove_reference_t<T>>
    Launcher(T &&data)
        : buffer_{ Wayland::HostUtils::DeviceMakeUnique<std::byte[]>(
              { reinterpret_cast<std::byte *>(std::addressof(data)),
                sizeof(T) }) },
          size_{ sizeof(T) }
    {
    }

    /// @brief Launch a ray by optix, with data designated by either ctor or
    /// SetData. \ref{void SetData(T&&)}
    void Launch(const Pipeline &pipeline, CUstream stream,
                const ShaderBindingTable &sbt, unsigned int width,
                unsigned int height, unsigned int depth = 1);

    /// @brief Change data of launcher; if the data have different size from
    /// before, the stored buffer will be reallocated.
    /// @param data new data, required to be safely copiable to devices.
    template<typename T>
        requires std::is_trivially_copyable_v<std::remove_reference_t<T>>
    void SetData(T &&data)
    {
        auto ptr = reinterpret_cast<std::byte *>(std::addressof(data));
        auto newSize = sizeof(T);
        if (newSize != size_)
        {
            buffer_ = Wayland::HostUtils::DeviceMakeUnique<std::byte[]>(
                { ptr, newSize });
            size_ = newSize;
        }
        else
            thrust::copy_n(ptr, newSize, buffer_.get());
    }

private:
    Wayland::HostUtils::DeviceUniquePtr<std::byte[]> buffer_;
    std::size_t size_;
};

} // namespace Wayland::Optix