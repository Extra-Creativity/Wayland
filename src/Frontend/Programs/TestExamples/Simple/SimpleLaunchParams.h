#pragma once
#include "glm/glm.hpp"
#include "optix.h"

namespace EasyRender
{

struct SimpleLaunchParams
{
    int frameID;
    struct
    {
        int x;
        int y;
    } fbSize;

    uint32_t *colorBuffer;
};

} // namespace EasyRender
