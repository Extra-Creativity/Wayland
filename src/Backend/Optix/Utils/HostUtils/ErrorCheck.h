#pragma once
#include "CommonHeaders.h"
#include <source_location>

namespace EasyRender::HostUtils
{
constexpr bool OnlyLog = true; // used by users to make it self-explainable.

#ifdef ERROR_DEBUG

template<bool LogOnly = false>
void CheckCUDAError(CUresult err, const std::source_location & =
                                      std::source_location::current());

template<bool LogOnly = false>
void CheckCUDAError(cudaError_t err, const std::source_location & =
                                         std::source_location::current());

template<bool LogOnly = false>
bool CheckLastCUDAError(
    const std::source_location & = std::source_location::current());

template<bool LogOnly = false>
void CheckOptixError(OptixResult err, const std::source_location & =
                                          std::source_location::current());

template<bool LogOnly = false>
void CheckError(bool success, const char *log, const char *exceptionInfo,
                const std::source_location & = std::source_location::current());

template<bool LogOnly = false>
void CheckError(bool success, const char *log,
                const std::source_location & = std::source_location::current());

#else
template<bool = false>
inline void CheckCUDAError(CUresult err)
{
}
template<bool = false>
inline void CheckCUDAError(cudaError_t err)
{
}
template<bool = false>
inline bool CheckLastCUDAError()
{
    return true;
}
template<bool = false>
inline void CheckOptixError(OptixResult err)
{
}
template<bool = false>
inline void CheckError(bool, const char *log,
                       const char *exceptionInfo = nullptr)
{
}

#endif
} // namespace EasyRender::HostUtils