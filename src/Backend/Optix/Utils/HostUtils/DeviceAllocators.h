#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "thrust/addressof.h"
#include "thrust/copy.h"
#include "thrust/device_ptr.h"
#include "thrust/universal_ptr.h"

#include "ErrorCheck.h"

#include <cstddef>
#include <memory>
#include <span>
#include <type_traits>

namespace HostUtils
{
using AllocType = cudaError_t (*)(void **, std::size_t);
using DeallocType = cudaError_t (*)(void *);

template<typename T, AllocType Alloc, DeallocType Free,
         typename DevicePointerType>
class DeviceAllocatorBase
{
public:
    static_assert(!std::is_const_v<T>,
                  "The C++ Standard forbids containers of const elements "
                  "because allocator<const T> is ill-formed.");
    static_assert(!std::is_function_v<T>,
                  "The C++ Standard forbids allocators for function elements "
                  "because of [allocator.requirements].");
    static_assert(!std::is_reference_v<T>,
                  "The C++ Standard forbids allocators for reference elements "
                  "because of [allocator.requirements].");

    using value_type = T;

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using propagate_on_container_move_assignment = std::true_type;

    constexpr DeviceAllocatorBase() noexcept = default;
    constexpr DeviceAllocatorBase(const DeviceAllocatorBase &) noexcept =
        default;
    template<class Other>
    constexpr DeviceAllocatorBase(
        const DeviceAllocatorBase<Other, Alloc, Free, DevicePointerType>
            &) noexcept
    {
    }
    constexpr ~DeviceAllocatorBase() = default;
    constexpr DeviceAllocatorBase &operator=(const DeviceAllocatorBase &) =
        default;

    constexpr void deallocate(
        T *const ptr, [[maybe_unused]] const size_t _Count) const noexcept
    {
        CheckCUDAError<OnlyLog>(Free(ptr));
    }

    [[nodiscard("Allocated memory should be freed before discarded")]]
    constexpr T *allocate(const size_t count) const
    {
        void *ptr;
        Alloc(&ptr, count * sizeof(T));
        if (!CheckLastCUDAError<OnlyLog>())
            throw std::bad_alloc{};
        return static_cast<T *>(ptr);
    }

    struct Deleter
    {
        using pointer = DevicePointerType;
        void operator()(pointer ptr) const noexcept
        {
            CheckCUDAError<OnlyLog>(Free(ptr.get()));
        }
    };
};

template<typename T>
using DeviceAllocator =
    DeviceAllocatorBase<T, cudaMalloc, cudaFree, thrust::device_ptr<T>>;

template<unsigned int Flags>
cudaError_t ManagedMalloc(void **ptr, std::size_t size)
{
    return cudaMallocManaged(ptr, size, Flags);
}

template<typename T, unsigned int Flags = cudaMemAttachGlobal>
using DeviceManagedAllocator =
    DeviceAllocatorBase<T, ManagedMalloc<Flags>, cudaFree,
                        thrust::universal_ptr<T>>;

template<typename T>
class PinnedPointer
{
public:
    PinnedPointer(T *init_hostPtr) noexcept : hostPtr_{ init_hostPtr }
    {
        // It would be very troublesome to throw, so only log here.
        CheckCUDAError<OnlyLog>(
            cudaHostGetDevicePointer(&devicePtr_, hostPtr_, 0));
    }
    auto &operator*() const noexcept(noexcept(*std::declval<const T *>()))
    {
        return *hostPtr_;
    }
    T *operator->() const noexcept { return hostPtr_; }

    T *get() const noexcept { return devicePtr_; }
    T *get_host() const noexcept { return hostPtr_; }

    operator T *() const noexcept { return hostPtr_; }

private:
    T *hostPtr_;
    T *devicePtr_ = nullptr; // access 0x0 if cudaHostGetDevicePointer fails.
};

template<unsigned int Flags>
cudaError_t PinnedMalloc(void **ptr, std::size_t size)
{
    return cudaMallocHost(ptr, size, Flags);
}

// TODO: it seems incorrect to use universal_ptr, since it needs
// cudaHostGetDevicePointer to get device pointer; wrap another class.
template<typename T, unsigned int Flags = cudaHostAllocDefault>
using DevicePinnedAllocator =
    DeviceAllocatorBase<T, PinnedMalloc<Flags>, cudaFreeHost, PinnedPointer<T>>;

template<typename V, typename AllocatorType = DeviceAllocator<V>>
[[nodiscard("Allocation does nothing if return value is discarded.")]]
auto DeviceMakeUninitializedUnique()
{
    using Deleter = typename AllocatorType::Deleter;
    using Pointer = typename Deleter::pointer;
    return std::unique_ptr<V, Deleter>{ Pointer{
        AllocatorType{}.allocate(1) } };
}

template<typename V, typename AllocatorType = DeviceAllocator<V>>
    requires std::is_trivially_copyable_v<std::remove_reference_t<V>>
[[nodiscard("Allocation does nothing if return value is discarded.")]]
auto DeviceMakeUnique(const V &hostValue)
{
    auto result = DeviceMakeUninitializedUnique<V, AllocatorType>();
    thrust::copy_n(&hostValue, 1, result.get());
    return result;
}

template<typename V,
         typename AllocatorType = DeviceAllocator<std::remove_extent_t<V>>>
    requires std::is_unbounded_array_v<V>
[[nodiscard("Allocation does nothing if return value is discarded.")]]
auto DeviceMakeUninitializedUnique(std::size_t size)
{
    using Deleter = typename AllocatorType::Deleter;
    using Pointer = typename Deleter::pointer;
    return std::unique_ptr<V, Deleter>{ Pointer{
        AllocatorType{}.allocate(size) } };
}

template<typename V,
         typename AllocatorType = DeviceAllocator<std::remove_extent_t<V>>>
    requires std::is_unbounded_array_v<V> &&
             std::is_trivially_copyable_v<std::remove_extent_t<V>>
[[nodiscard("Allocation does nothing if return value is discarded.")]]
auto DeviceMakeUnique(const std::span<const std::remove_extent_t<V>> hostBuffer)
{
    auto result =
        DeviceMakeUninitializedUnique<V, AllocatorType>(hostBuffer.size());
    thrust::copy_n(hostBuffer.data(), hostBuffer.size(), result.get());
    return result;
}

template<typename T>
    requires requires(T ptr) { std::is_pointer_v<decltype(ptr.get())>; }
[[nodiscard]] auto ToDriverPointer(T ptr)
{
    return static_cast<CUdeviceptr>(
        reinterpret_cast<std::uintptr_t>(ptr.get()));
}

template<typename T>
    requires std::is_pointer_v<T>
[[nodiscard]] auto ToDriverPointer(T ptr)
{
    return static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(ptr));
}

inline void CheckValidDevicePointer(const void *ptr)
{
#ifdef NEED_VALID_DEVICE_POINTER_CHECK
    cudaPointerAttributes attribute;
    CheckCUDAError(cudaPointerGetAttributes(&attribute, ptr));
    CheckError(attribute.type == cudaMemoryType::cudaMemoryTypeDevice ||
                   attribute.type == cudaMemoryType::cudaMemoryTypeManaged,
               "Access Non-cuda memory", "Invalid access");
#else
    return;
#endif
}

namespace Details
{

template<typename T>
struct DeviceUniquePtrHelper
{
    using type = std::unique_ptr<T, typename DeviceAllocator<T>::Deleter>;
};

template<typename T>
struct DeviceUniquePtrHelper<T[]>
{
    using type = std::unique_ptr<T[], typename DeviceAllocator<T>::Deleter>;
};

} // namespace Details

template<typename T>
using DeviceUniquePtr = Details::DeviceUniquePtrHelper<T>::type;

} // namespace HostUtils