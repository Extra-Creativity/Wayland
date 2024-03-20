#pragma once
#include "HostUtils/CommonHeaders.h"
#include <string>

namespace Wayland::Optix
{

/// @brief ABC for all traversables, like AS and transforms.
class Traversable
{
public:
    // unsigned int because optix needs it.
    virtual std::string DisplayInfo() const = 0;
    virtual unsigned int GetDepth() const noexcept = 0;
    virtual ~Traversable() = default;

    auto GetHandle() const noexcept { return handle_; }

protected:
    static unsigned int UncheckedGetDepthForSingleChild_(
        const Traversable *ptr) noexcept
    {
        return ptr->GetDepth() + 1;
    }
    static unsigned int GetDepthForSingleChild_(const Traversable *ptr) noexcept
    {
        return ptr ? ptr->GetDepth() + 1 : 0;
    }

    OptixTraversableHandle handle_;
};

} // namespace Wayland::Optix