#pragma once
#include <type_traits>
#include <utility>

namespace HostUtils
{

template<typename T>
    requires std::is_scoped_enum_v<T>
struct NeedBinaryOp
{
    static const inline bool value = false;
};

template<typename EnumType>
    requires HostUtils::NeedBinaryOp<EnumType>::value
bool TestEnum(EnumType a, EnumType b) noexcept
{
    return bool(a & b);
}

template<typename EnumType>
    requires HostUtils::NeedBinaryOp<EnumType>::value
bool TestEnum(std::underlying_type_t<EnumType> a, EnumType b) noexcept
{
    return bool(EnumType{ a } & b);
}

template<typename EnumType>
    requires HostUtils::NeedBinaryOp<EnumType>::value
bool TestEnum(EnumType a, std::underlying_type_t<EnumType> b) noexcept
{
    return bool(a & EnumType{ b });
}

} // namespace HostUtils

template<typename EnumType>
    requires HostUtils::NeedBinaryOp<EnumType>::value
EnumType operator|(EnumType a, EnumType b) noexcept
{
    return EnumType{ std::to_underlying(a) | std::to_underlying(b) };
}

template<typename EnumType>
    requires HostUtils::NeedBinaryOp<EnumType>::value
EnumType operator&(EnumType a, EnumType b) noexcept
{
    return EnumType{ std::to_underlying(a) & std::to_underlying(b) };
}

#define ENABLE_BINARY_OP_FOR_SCOPED_ENUM(Type) \
    template<>                                 \
    struct HostUtils::NeedBinaryOp<Type>       \
    {                                          \
        static const inline bool value = true; \
    };