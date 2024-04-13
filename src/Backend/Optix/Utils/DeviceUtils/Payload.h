#pragma once
#include "cuda_runtime.h"
#include "optix_device.h"
#include <cuda/std/array>
#include <thrust/addressof.h>

namespace EasyRender::DeviceUtils
{

template<std::size_t Size, typename... Args>
__device__ __forceinline__ void optixTraceUnpack(
    cuda::std::array<unsigned char, Size> &arr, Args &&...args)
{
    Details::optixTraceUnpackImpl(arr, std::make_index_sequence<Size / 4>(),
                                  args...);
}

template<std::size_t Size, typename... Args>
__device__ __forceinline__ void optixTraverseUnpack(
    cuda::std::array<unsigned char, Size> &arr, Args &&...args)
{
    Details::optixTraverseUnpackImpl(arr, std::make_index_sequence<Size / 4>(),
                                     args...);
}

template<std::size_t Size, typename... Args>
__device__ __forceinline__ void optixInvokeUnpack(
    cuda::std::array<unsigned char, Size> &arr, Args &&...args)
{
    Details::optixInvokeUnpackImpl(arr, std::make_index_sequence<Size / 4>(),
                                   args...);
}

template<std::size_t>
struct OptixGetPayload
{
};

template<std::size_t>
struct OptixSetPayload
{
};

template<typename T>
__device__ __forceinline__ auto PackPayloads(const T &arg)
    -> cuda::std::array<unsigned char, sizeof(T)>
{
    static_assert(sizeof(T) / 4 <= 32 && sizeof(T) % 4 == 0);
    alignas(T) cuda::std::array<unsigned char, sizeof(T)> arr;
    ::memcpy(arr.data(), thrust::addressof(arg), sizeof(T));
    return arr;
}

// It's toleratable for optix payloads to be uninitialized.
#pragma nv_diag_suppress 549
template<typename T>
__device__ __forceinline__ auto PackPayloads()
    -> cuda::std::array<unsigned char, sizeof(T)>
{
    static_assert(sizeof(T) / 4 <= 32 && sizeof(T) % 4 == 0);
    alignas(T) cuda::std::array<unsigned char, sizeof(T)> arr;
    return arr;
}
#pragma nv_diag_default 549

template<typename T>
__device__ __forceinline__ T &UnpackPayloads(
    cuda::std::array<unsigned char, sizeof(T)> &arr)
{
    return *reinterpret_cast<T *>(arr.data());
}

template<typename T, std::size_t BeginIdx>
__device__ __forceinline__ T GetFromPayload()
{
    static_assert(sizeof(T) % 4 == 0);
    constexpr std::size_t payloadNum = sizeof(T) / 4;
    return Details::GetFromPayloadImpl<T, BeginIdx>(
        std::make_index_sequence<payloadNum>());
}

template<std::size_t BeginIdx, typename T>
__device__ __forceinline__ void SetToPayload(const T &arg)
{
    static_assert(sizeof(T) % 4 == 0);
    constexpr std::size_t payloadNum = sizeof(T) / 4;
    Details::SetToPayloadImpl<T, BeginIdx>(
        arg, std::make_index_sequence<payloadNum>());
}

namespace Details
{

template<std::size_t Size, std::size_t... Indices, typename... Args>
__device__ __forceinline__ void optixTraceUnpackImpl(
    cuda::std::array<unsigned char, Size> &arr, std::index_sequence<Indices...>,
    Args &&...args)
{
    optixTrace(args..., (*reinterpret_cast<std::uint32_t *>(arr.data() +
                                                            Indices * 4))...);
}

template<std::size_t Size, std::size_t... Indices, typename... Args>
__device__ __forceinline__ void optixTraverseUnpackImpl(
    cuda::std::array<unsigned char, Size> &arr, std::index_sequence<Indices...>,
    Args &&...args)
{
    optixTraverse(args..., (*reinterpret_cast<std::uint32_t *>(
                               arr.data() + Indices * 4))...);
}

template<std::size_t Size, std::size_t... Indices, typename... Args>
__device__ __forceinline__ void optixInvokeUnpackImpl(
    cuda::std::array<unsigned char, Size> &arr, std::index_sequence<Indices...>,
    Args &&...args)
{
    optixInvoke(args..., (*reinterpret_cast<std::uint32_t *>(arr.data() +
                                                             Indices * 4))...);
}

template<typename T, std::size_t BeginIdx, std::size_t... Indices>
__device__ __forceinline__ T GetFromPayloadImpl(std::index_sequence<Indices...>)
{
    cuda::std::array<std::uint32_t, sizeof(T) / 4> arr{
        OptixGetPayload<BeginIdx + Indices>{}()...
    };
    T t;
    ::memcpy(thrust::addressof(t), arr.data(), sizeof(T));
    return t;
}

template<typename T, std::size_t BeginIdx, std::size_t... Indices>
__device__ __forceinline__ void SetToPayloadImpl(
    const T &arg, std::index_sequence<Indices...>)
{
    cuda::std::array<std::uint32_t, sizeof(T) / 4> arr;
    ::memcpy(arr.data(), thrust::addressof(arg), sizeof(T));
    (OptixSetPayload<BeginIdx + Indices>{}(arr[Indices]), ...);
    return;
}

} // namespace Details

} // namespace EasyRender::DeviceUtils

#define SPECIALIZE_OPTIX_GET_PAYLOAD(Index)             \
    template<>                                          \
    struct EasyRender::DeviceUtils::OptixGetPayload<Index> \
    {                                                   \
        __device__ __forceinline__ auto operator()()    \
        {                                               \
            return optixGetPayload_##Index##();         \
        }                                               \
    };

#define SPECIALIZE_OPTIX_SET_PAYLOAD(Index)                               \
    template<>                                                            \
    struct EasyRender::DeviceUtils::OptixSetPayload<Index>                   \
    {                                                                     \
        __device__ __forceinline__ auto operator()(std::uint32_t payload) \
        {                                                                 \
            return optixSetPayload_##Index##(payload);                    \
        }                                                                 \
    };

SPECIALIZE_OPTIX_GET_PAYLOAD(0);
SPECIALIZE_OPTIX_GET_PAYLOAD(1);
SPECIALIZE_OPTIX_GET_PAYLOAD(2);
SPECIALIZE_OPTIX_GET_PAYLOAD(3);
SPECIALIZE_OPTIX_GET_PAYLOAD(4);
SPECIALIZE_OPTIX_GET_PAYLOAD(5);
SPECIALIZE_OPTIX_GET_PAYLOAD(6);
SPECIALIZE_OPTIX_GET_PAYLOAD(7);
SPECIALIZE_OPTIX_GET_PAYLOAD(8);
SPECIALIZE_OPTIX_GET_PAYLOAD(9);
SPECIALIZE_OPTIX_GET_PAYLOAD(10);
SPECIALIZE_OPTIX_GET_PAYLOAD(11);
SPECIALIZE_OPTIX_GET_PAYLOAD(12);
SPECIALIZE_OPTIX_GET_PAYLOAD(13);
SPECIALIZE_OPTIX_GET_PAYLOAD(14);
SPECIALIZE_OPTIX_GET_PAYLOAD(15);
SPECIALIZE_OPTIX_GET_PAYLOAD(16);
SPECIALIZE_OPTIX_GET_PAYLOAD(17);
SPECIALIZE_OPTIX_GET_PAYLOAD(18);
SPECIALIZE_OPTIX_GET_PAYLOAD(19);
SPECIALIZE_OPTIX_GET_PAYLOAD(20);
SPECIALIZE_OPTIX_GET_PAYLOAD(21);
SPECIALIZE_OPTIX_GET_PAYLOAD(22);
SPECIALIZE_OPTIX_GET_PAYLOAD(23);
SPECIALIZE_OPTIX_GET_PAYLOAD(24);
SPECIALIZE_OPTIX_GET_PAYLOAD(25);
SPECIALIZE_OPTIX_GET_PAYLOAD(26);
SPECIALIZE_OPTIX_GET_PAYLOAD(27);
SPECIALIZE_OPTIX_GET_PAYLOAD(28);
SPECIALIZE_OPTIX_GET_PAYLOAD(29);
SPECIALIZE_OPTIX_GET_PAYLOAD(30);
SPECIALIZE_OPTIX_GET_PAYLOAD(31);

SPECIALIZE_OPTIX_SET_PAYLOAD(0);
SPECIALIZE_OPTIX_SET_PAYLOAD(1);
SPECIALIZE_OPTIX_SET_PAYLOAD(2);
SPECIALIZE_OPTIX_SET_PAYLOAD(3);
SPECIALIZE_OPTIX_SET_PAYLOAD(4);
SPECIALIZE_OPTIX_SET_PAYLOAD(5);
SPECIALIZE_OPTIX_SET_PAYLOAD(6);
SPECIALIZE_OPTIX_SET_PAYLOAD(7);
SPECIALIZE_OPTIX_SET_PAYLOAD(8);
SPECIALIZE_OPTIX_SET_PAYLOAD(9);
SPECIALIZE_OPTIX_SET_PAYLOAD(10);
SPECIALIZE_OPTIX_SET_PAYLOAD(11);
SPECIALIZE_OPTIX_SET_PAYLOAD(12);
SPECIALIZE_OPTIX_SET_PAYLOAD(13);
SPECIALIZE_OPTIX_SET_PAYLOAD(14);
SPECIALIZE_OPTIX_SET_PAYLOAD(15);
SPECIALIZE_OPTIX_SET_PAYLOAD(16);
SPECIALIZE_OPTIX_SET_PAYLOAD(17);
SPECIALIZE_OPTIX_SET_PAYLOAD(18);
SPECIALIZE_OPTIX_SET_PAYLOAD(19);
SPECIALIZE_OPTIX_SET_PAYLOAD(20);
SPECIALIZE_OPTIX_SET_PAYLOAD(21);
SPECIALIZE_OPTIX_SET_PAYLOAD(22);
SPECIALIZE_OPTIX_SET_PAYLOAD(23);
SPECIALIZE_OPTIX_SET_PAYLOAD(24);
SPECIALIZE_OPTIX_SET_PAYLOAD(25);
SPECIALIZE_OPTIX_SET_PAYLOAD(26);
SPECIALIZE_OPTIX_SET_PAYLOAD(27);
SPECIALIZE_OPTIX_SET_PAYLOAD(28);
SPECIALIZE_OPTIX_SET_PAYLOAD(29);
SPECIALIZE_OPTIX_SET_PAYLOAD(30);
SPECIALIZE_OPTIX_SET_PAYLOAD(31);
