#pragma once

#include "HostUtils/DeviceAllocators.h"

class Pipeline;
class ShaderBindingTable;

class Launcher
{
public:
    template<typename T>
        requires std::is_trivially_copyable_v<std::remove_reference_t<T>>
    Launcher(T &&data)
        : buffer_{ HostUtils::DeviceMakeUnique<std::byte[]>(
              { reinterpret_cast<std::byte *>(std::addressof(data)),
                sizeof(T) }) },
          size_{ sizeof(T) }
    {
    }

    void Launch(const Pipeline &pipeline, CUstream stream,
                const ShaderBindingTable &sbt, unsigned int width,
                unsigned int height, unsigned int depth = 1);

    template<typename T>
        requires std::is_trivially_copyable_v<std::remove_reference_t<T>>
    void SetData(T &&data)
    {
        auto ptr = reinterpret_cast<std::byte *>(std::addressof(data));
        auto newSize = sizeof(T);
        if (newSize != size_)
        {
            buffer_ =
                HostUtils::DeviceMakeUnique<std::byte[]>({ ptr, newSize });
            size_ = newSize;
        }
        else
            thrust::copy_n(ptr, newSize, buffer_.get());
    }

private:
    HostUtils::DeviceUniquePtr<std::byte[]> buffer_;
    std::size_t size_;
};