#pragma once
#include <memory>
#include "glm/glm.hpp"
#include "minipbrt.h"

namespace EasyRender
{
	class AreaLight {
    public:
        AreaLight(minipbrt::DiffuseAreaLight * miniLight);
		glm::vec3 L;
	};

	using LightPtr = std::unique_ptr<AreaLight>;

}
