#include "ErrorCheck.h"

#include "spdlog/spdlog.h"

#include <cassert>
#include <stacktrace>

namespace HostUtils
{

#ifdef ERROR_DEBUG

template<bool LogOnly>
inline static void ReportError(const char *log, const char *exceptionInfo,
                               const std::source_location &location)
{
    SPDLOG_ERROR("Error: {}\nAt line {}, function {}, file {}, traceback: \n{}",
                 log, location.line(), location.function_name(),
                 location.file_name(), std::stacktrace::current());
    if constexpr (!LogOnly)
        throw std::runtime_error{ exceptionInfo };
}

template<bool LogOnly>
void CheckCUDAError(CUresult err, const std::source_location &loc)
{
    if (err != CUresult::CUDA_SUCCESS)
    {
        const char *log, *exceptionInfo;
        CUresult res = cuGetErrorString(err, &log);
        assert(res == CUresult::CUDA_SUCCESS);
        cuGetErrorName(err, &exceptionInfo);

        ReportError<LogOnly>(log, exceptionInfo, loc);
    }
}
template void CheckCUDAError<false>(CUresult, const std::source_location &);
template void CheckCUDAError<true>(CUresult, const std::source_location &);

template<bool LogOnly>
void CheckCUDAError(cudaError_t err, const std::source_location &loc)
{
    if (err != cudaError::cudaSuccess)
        ReportError<LogOnly>(cudaGetErrorString(err), cudaGetErrorName(err),
                             loc);
}
template void CheckCUDAError<false>(cudaError_t, const std::source_location &);
template void CheckCUDAError<true>(cudaError_t, const std::source_location &);

template<bool LogOnly>
bool CheckLastCUDAError(const std::source_location &loc)
{
    auto err = cudaGetLastError();
    CheckCUDAError<LogOnly>(err, loc);
    return err == cudaError::cudaSuccess;
}
template bool CheckLastCUDAError<false>(const std::source_location &);
template bool CheckLastCUDAError<true>(const std::source_location &);

template<bool LogOnly>
void CheckOptixError(OptixResult err, const std::source_location &loc)
{
    if (err != OptixResult::OPTIX_SUCCESS)
        ReportError<LogOnly>(optixGetErrorString(err), optixGetErrorName(err),
                             loc);
}
template void CheckOptixError<false>(OptixResult, const std::source_location &);
template void CheckOptixError<true>(OptixResult, const std::source_location &);

template<bool LogOnly>
void CheckError(bool success, const char *log, const char *exceptionInfo,
                const std::source_location &loc)
{
    if (!success)
        ReportError<LogOnly>(log, exceptionInfo, loc);
}
template void CheckError<false>(bool success, const char *log,
                                const char *exceptionInfo,
                                const std::source_location &loc);
template void CheckError<true>(bool success, const char *log,
                               const char *exceptionInfo,
                               const std::source_location &loc);

template<bool LogOnly>
void CheckError(bool success, const char *log, const std::source_location &loc)
{
    CheckError<LogOnly>(success, log, log, loc);
}
template void CheckError<false>(bool success, const char *log,
                                const std::source_location &loc);
template void CheckError<true>(bool success, const char *log,
                               const std::source_location &loc);

#endif
} // namespace HostUtils