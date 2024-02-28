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
    ContextManager();
    // int is to be same as CUDA APIs to represent devices.
    ContextManager(std::span<const int> visibleDevices);

    struct ContextInfo
    {
        int deviceID;
        OptixDeviceContextWrapper context;
        CUDAStreamWrapper stream;
        int canAsyncAccessUnifiedMemory;
    };

    // CUDA ensures that device number is within int.
    int GetContextNum() const noexcept
    {
        return static_cast<int>(contexts_.size());
    }
    int GetDeviceID(int idx) const
    {
        return HostUtils::Access(contexts_, idx).deviceID;
    }

    void SetCachePath(const std::string &path);

private:
    void CheckContextAvailable_();
    std::vector<ContextInfo> contexts_;
};

class LocalContextSetter
{
public:
    LocalContextSetter(ContextManager &manager, int idx)
    {
        auto deviceID = manager.GetDeviceID(idx);
        HostUtils::CheckCUDAError(cudaSetDevice(deviceID));
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
