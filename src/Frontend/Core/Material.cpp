#include <sstream>

#include "glm/gtc/type_ptr.hpp"
#include "Material.h"
#include "Utils/Common.h"
#include "Utils/PrintUtils.h"

using namespace std;

namespace EasyRender
{


Diffuse::Diffuse(minipbrt::MatteMaterial *miniMatte)
{
    Kd = glm::make_vec3(miniMatte->Kd.value);
}

string Diffuse::ToString() const
{
    ostringstream oss;
    oss << "  Type: Diffuse\n";
    oss << "  Kd:   " << PrintUtils::ToString(Kd) << "\n";
    return oss.str();
}

} // namespace EasyRender
