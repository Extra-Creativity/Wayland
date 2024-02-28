#pragma once
// This header should be small, since macros are bad practice in modern C++.

#define MOVE_ONLY_SINGLE_MEMBER_SPECIAL_METHOD(Type, member, emptyVal) \
    Type(const Type &) = delete;                                       \
    Type &operator=(const Type &) = delete;                            \
    Type(Type &&another) noexcept(                                     \
        noexcept(std::exchange(another.member, emptyVal)))             \
        : member{ std::exchange(another.member, emptyVal) }            \
    {                                                                  \
    }                                                                  \
    Type &operator=(Type &&another) noexcept(                          \
        noexcept(std::ranges::swap(member, another.member)))           \
    {                                                                  \
        std::ranges::swap(member, another.member);                     \
        return *this;                                                  \
    }

#define DISABLE_ALL_SPECIAL_METHODS(Type)   \
    Type(const Type &) = delete;            \
    Type &operator=(const Type &) = delete; \
    Type(Type &&another) = delete;          \
    Type &operator=(Type &&another) = delete;

#define DEFINE_SIMPLE_CHAINED_SETTER(name, member)   \
    auto &Set##name##(decltype(member) val) noexcept \
    {                                                \
        member = val;                                \
        return *this;                                \
    }
