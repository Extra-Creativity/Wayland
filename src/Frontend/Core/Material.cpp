#include <sstream>

#include "Material.h"
#include "SceneManager.h"
#include "Utils/Common.h"
#include "Utils/PrintUtils.h"
#include "glm/gtc/type_ptr.hpp"

using namespace std;

namespace EasyRender
{

Diffuse::Diffuse(minipbrt::MatteMaterial *miniMatte)
{
    Kd = glm::make_vec3(miniMatte->Kd.value);
    textureId = miniMatte->Kd.texture;
}

string Diffuse::ToString() const
{
    ostringstream oss;
    oss << "  Type: Diffuse\n";
    oss << "  Kd:   " << PrintUtils::ToString(Kd) << "\n";
    return oss.str();
}

bool Diffuse::HasTexture()
{
    return textureId < INVALID_INDEX;
}

} // namespace EasyRender
