#pragma once
#include "HostUtils/CommonHeaders.h"
#include "HostUtils/DebugUtils.h"
#include "HostUtils/ErrorCheck.h"
#include "HostUtils/SpecialMacros.h"

#include <cassert>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace Wayland::OptiX
{

/// @brief Where the optix context is created.
class ContextManager
{
    friend class LocalContextSetter;
    struct OptixDeviceContextWrapper
    {
        OptixDeviceContextWrapper() = default;
        OptixDeviceContextWrapper(bool &) noexcept;
        MOVE_ONLY_SINGLE_MEMBER_SPECIAL_METHOD(OptixDeviceContextWrapper,
                                               context, nullptr);
        ~OptixDeviceContextWrapper();
        OptixDeviceContext context = nullptr;
    };

    struct CUDAStreamWrapper
    {
        CUDAStreamWrapper() = default;
        CUDAStreamWrapper(bool &) noexcept;
        MOVE_ONLY_SINGLE_MEMBER_SPECIAL_METHOD(CUDAStreamWrapper, stream,
                                               nullptr);
        ~CUDAStreamWrapper();
        cudaStream_t stream = nullptr;
    };

public:
    /// @brief All visible devices will be used to initialize cuda context and
    /// optix contexts. If any device fails to do so, it will be skipped.
    ContextManager();
    /// @brief Provided visible device IDs, contexts are initialized on the
    /// regulated devices. If the id exceeds available devices, it will be
    /// skipped.
    /// @note int is to be same as CUDA APIs to represent devices.
    ContextManager(std::span<const int> visibleDevices);

    struct ContextInfo
    {
        int deviceID;
        OptixDeviceContextWrapper context;
        CUDAStreamWrapper stream;
        /// @brief Used to check whether managed memory can be used without
        /// synchonizing.
        int canAsyncAccessUnifiedMemory;
    };

    // CUDA ensures that device number is within int.
    int GetContextNum() const noexcept
    {
        return static_cast<int>(contexts_.size());
    }
    int GetDeviceID(int idx) const
    {
        return Wayland::HostUtils::Access(contexts_, idx).deviceID;
    }

    /// @param path set cache paths of all devices to path.
    /// @note optix has a default cache path, so it's optional to call this
    /// method to make the whole procedures work.
    void SetCachePath(const std::string &path);

private:
    void CheckContextAvailable_();
    std::vector<ContextInfo> contexts_;
};

/// @brief almost all necessary APIs of optix needs an OptixContext, so
/// LocalContextSetter is used to set the current active OptixContext to save
/// troubles of passing them over and over again.
/// @note This setter is thread-safe; each thread has its own active setter.
class LocalContextSetter
{
public:
    /// @brief Set the current active OptixContext.
    /// @param manager the related context manager
    /// @param idx device index.
    LocalContextSetter(ContextManager &manager, int idx)
    {
        auto deviceID = manager.GetDeviceID(idx);
        Wayland::HostUtils::CheckCUDAError(cudaSetDevice(deviceID));
        currentContext_ = &(manager.contexts_[idx]);
    }

    static auto GetCurrentOptixContext() noexcept
    {
        assert(currentContext_);
        return currentContext_->context.context;
    }

    static auto GetCurrentCUDAStream() noexcept
    {
        assert(currentContext_);
        return currentContext_->stream.stream;
    }

    static auto GetCurrentDeviceID() noexcept
    {
        assert(currentContext_);
        return currentContext_->deviceID;
    }

    static auto CurrentCanAsyncAccessUnifiedMemory() noexcept
    {
        assert(currentContext_);
        return currentContext_->canAsyncAccessUnifiedMemory;
    }

private:
    static thread_local ContextManager::ContextInfo *currentContext_;
};

} // namespace Wayland::OptiX