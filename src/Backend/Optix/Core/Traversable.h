#pragma once
#include "HostUtils/CommonHeaders.h"

#include <any>
#include <string>

#include "SBTData.h"

namespace Wayland::Optix
{

/// @brief ABC for all traversables, like AS and transforms.
class Traversable
{
public:
    // unsigned int because optix needs it.
    virtual std::string DisplayInfo() const = 0;
    virtual unsigned int GetDepth() const noexcept = 0;
    virtual void FillSBT(unsigned int rayTypeNum, std::any &buffer) const
    {
        throw std::runtime_error{ std::string{
                                      "SBT cannot be filled by type " } +
                                  typeid(*this).name() };
    }
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

template<typename T>
std::vector<SBTData<T>> GetSBTHitRecordBuffer(unsigned int rayTypeNum,
                                              const Traversable &root)
{
    using ContainerType = std::vector<SBTData<T>>;
    std::any buffer = ContainerType{};
    root.FillSBT(rayTypeNum, buffer);
    return std::any_cast<ContainerType &&>(std::move(buffer));
}

} // namespace Wayland::Optix