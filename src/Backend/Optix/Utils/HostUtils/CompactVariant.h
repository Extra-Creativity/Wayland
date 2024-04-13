#pragma once
#include <memory>
#include <span>
#include <type_traits>

namespace EasyRender::HostUtils
{

// @brief Store either an element or an array of elements; when size is 1, use
// the single element; otherwise allocate dynamically.
// @note Why not std::variant: we want to use "index" of variant to store the
// number (which is also unique!), to reduce the space needed.
template<typename T>
    requires std::is_default_constructible_v<T> &&
             std::is_move_assignable_v<T> && std::is_move_constructible_v<T>
class CompactVariant
{
    std::size_t size_;
    union Data {
        Data() {}
        T singleElem;
        std::unique_ptr<T[]> elems;
        ~Data() {}
    } data_;
    bool IsSingle_() const noexcept { return size_ == 1; }
    void Clear_() const noexcept
    {
        if (!IsSingle_())
            data_.elems.~unique_ptr();
        else
            data_.singleElem.~T();
    }

public:
    auto GetSize() const noexcept { return size_; }
    auto GetPtr() const noexcept
    {
        return IsSingle_() ? &data_.singleElem : data_.elems.get();
    }

    CompactVariant(T elem) : size_{ 1 } { new (&data_) T{ std::move(elem) }; }

    CompactVariant(std::span<const T> elems)
    {
        if (auto size = elems.size(); size == 0)
            size_ = 1, new (&data_.singleElem) T{};
        else if (size == 1)
            size_ = 1, new (&data_.singleElem) T{ elems[0] };
        else [[likely]]
        {
            size_ = size;
            new (&data_) decltype(data_.elems){
                std::make_unique_for_overwrite<T[]>(size)
            };
            try
            {
                std::ranges::copy(elems, data_.elems.get());
            }
            catch (...)
            {
                data_.elems.~unique_ptr();
                throw;
            }
        }
    }

    // Don't allow any exception because we don't allow valueless status.
    CompactVariant(CompactVariant &&another) noexcept : size_{ another.size_ }
    {
        if (IsSingle_())
            data_.singleElem = std::move(another.data_.singleElem);
        else
            new (
                &data_) decltype(data_.elems){ std::move(another.data_.elems) };
    }

    // Don't allow any exception because we don't allow valueless status.
    CompactVariant &operator=(CompactVariant &&another) noexcept
    {
        Clear_();
        size_ = another.size_;
        if (IsSingle_())
            data_.singleElem = std::move(another.data_.singleElem);
        else
            data_.elems = std::move(another.data_.elems);

        return *this;
    }

    ~CompactVariant() { Clear_(); }
};

} // namespace EasyRender::HostUtils