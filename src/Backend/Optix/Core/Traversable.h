#pragma once
#include "HostUtils/CommonHeaders.h"

#include <any>
#include <string>

#include "SBTData.h"

namespace Wayland::Optix
{

class SBTHitRecordBufferProxy
{
public:
    template<typename T>
    SBTHitRecordBufferProxy(std::vector<SBTData<T>> cont)
        : hiddenBuffer_{ std::move(cont) },
          offsetChecker_{ [](std::any &object, std::size_t currOffset) {
              auto &buffer = std::any_cast<std::vector<SBTData<T>> &>(object);
              if (auto size = buffer.size(); size != currOffset)
              {
                  buffer.resize(currOffset);
                  return size;
              }
              return currOffset;
          } }
    {
    }

    auto CheckCurrOffset(std::size_t expectOffset)
    {
        return offsetChecker_(hiddenBuffer_, expectOffset);
    }

    auto &&GetBuffer(this auto &&self)
    {
        return std::forward<decltype(self)>(self).hiddenBuffer_;
    }

private:
    std::any hiddenBuffer_;
    std::size_t (*offsetChecker_)(std::any &, std::size_t);
};

/// @brief ABC for all traversables, like AS and transforms.
class Traversable
{
public:
    // unsigned int because optix needs it.
    virtual std::string DisplayInfo() const = 0;
    virtual unsigned int GetDepth() const noexcept = 0;
    virtual void FillSBT(unsigned int rayTypeNum,
                         SBTHitRecordBufferProxy &buffer) const
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

    SBTHitRecordBufferProxy proxy{ ContainerType{} };
    root.FillSBT(rayTypeNum, proxy);
    return std::any_cast<ContainerType &&>(std::move(proxy).GetBuffer());
}

} // namespace Wayland::Optix