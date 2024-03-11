#pragma once
#include "HostUtils/CommonHeaders.h"

#include <any>
#include <string>

#include "SBTData.h"

namespace Wayland::Optix
{

template<typename T>
struct SBTHitRecordInfo
{
    std::vector<SBTData<T>> hitRecords;
    std::vector<std::size_t> groupIndices;
};

class SBTHitRecordBufferProxy
{
public:
    template<typename T>
    SBTHitRecordBufferProxy(SBTHitRecordInfo<T> cont)
        : hiddenBuffer_{ std::move(cont) },
          offsetChecker_{ [](std::any &object, std::size_t currOffset) {
              auto &[hitRecords, groupIndices] =
                  std::any_cast<SBTHitRecordInfo<T> &>(object);
              if (auto size = hitRecords.size(); size != currOffset)
              {
                  hitRecords.resize(currOffset);
                  groupIndices.resize(currOffset);
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
    Traversable() = default;

    Traversable(const Traversable &) = delete;
    Traversable &operator=(const Traversable &) = delete;

    // TODO: temporarily disable it; we may add relocation in the future.
    Traversable(Traversable &&) = delete;
    Traversable &operator=(Traversable &&) = delete;

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

    OptixTraversableHandle handle_ = 0;
};

template<typename T>
SBTHitRecordInfo<T> GetSBTHitRecordBuffer(unsigned int rayTypeNum,
                                          const Traversable &root)
{
    SBTHitRecordBufferProxy proxy{ SBTHitRecordInfo<T>{} };
    root.FillSBT(rayTypeNum, proxy);
    return std::any_cast<SBTHitRecordInfo<T> &&>(std::move(proxy).GetBuffer());
}

} // namespace Wayland::Optix