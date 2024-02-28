// This is required by Clang due to a unresolved core issue of C++
// -- instantiation can be delayed to happen until the whole TU
// is resolved. In our file, this has problem in:
//      Allocator<Alloc>
//          ManagedMalloc<Flags> -> This is a template
//      Allocator<ManagedAlloc<...>> // We use it, but the TU hasn't ended.
//                                   // so it could be undefined function.
// Notice that Make ManagedAlloc declared first won't help.
// See https://github.com/llvm/llvm-project/issues/73232 for more details.

#include "DeviceAllocators.h"

template<unsigned int... Macros>
auto ForceClangInstantiation(
    std::integer_sequence<unsigned int, Macros...> int_seq)
{
    static auto funcs = std::make_tuple(&HostUtils::ManagedMalloc<Macros>...);
    return &funcs;
}

static auto force1 =
    ForceClangInstantiation(std::make_integer_sequence<unsigned int, 4>());

template<unsigned int... Macros>
auto ForceClangInstantiation2(
    std::integer_sequence<unsigned int, Macros...> int_seq)
{
    static auto funcs = std::make_tuple(&HostUtils::PinnedMalloc<Macros>...);
    return &funcs;
}

static auto force2 =
    ForceClangInstantiation2(std::make_integer_sequence<unsigned int, 7>());