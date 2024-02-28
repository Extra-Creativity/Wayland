#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace HostUtils
{

template<typename T>
auto MakeAlignedByteBuffer(std::size_t size)
{
    constexpr auto deleter = [](void *ptr) {
        ::operator delete[](ptr, std::align_val_t{ alignof(T) });
    };
    auto result = std::unique_ptr<std::byte[], decltype(deleter)>{
        static_cast<std::byte *>(
            ::operator new[](size, std::align_val_t{ alignof(T) }))
    };
    assert(reinterpret_cast<std::uintptr_t>(result.get()) % alignof(T) == 0);
    return result;
}

template<typename T>
using AlignedBufferType = decltype(MakeAlignedByteBuffer<T>(1));

} // namespace HostUtils