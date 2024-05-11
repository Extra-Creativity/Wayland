#include "glm/glm.hpp"

namespace EasyRender::Programs::BDPT
{

struct BDPTVertex
{
    glm::vec3 pos;
    glm::vec3 Ns;
    glm::vec3 Ng;
    glm::vec3 Wi;
    glm::vec3 tput;
    glm::vec3 texcolor;
    Device::DisneyMaterial *mat;
    Device::DeviceAreaLight *light;
    /* The local pdf of sampling this vertex */
    float pdf;
    /* The local pdf of sampling last vertex inversely */
    float pdfInverse;
    /* The pdf of sampling this vertex's position on light */
    float pdfLight;
    /* Trace septh */
    int depth;
};

}