#pragma once
#include <memory>
#include "glm/glm.hpp"
#include "minipbrt.h"

namespace EasyRender
{
	class AreaLight {
    public:
        AreaLight(minipbrt::DiffuseAreaLight * miniLight);
		bool twoSided;
        glm::vec3 L;
        uint32_t mesh;
	};

	using LightPtr = std::unique_ptr<AreaLight>;

}
