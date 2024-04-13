#pragma once
#include <vector>

// Return CPU unsigned char buffer from a GPU float buffer.
std::vector<unsigned char> FromFloatToChar(float *, std::size_t size);