#pragma once
#include <minipbrt.h>
#include <string>
#include <memory>

#include "Core/Camera.h"
#include "Core/Shape.h"
#include "Core/Material.h"
#include "Core/Light.h"

namespace EasyRender
{

constexpr uint32_t INVALID_INDEX = 0xFFFFFFFFu;

class SceneManager
{
public:
    SceneManager(std::string_view sceneSrc_);
    SceneManager() = default;
    ~SceneManager() = default;

    void LoadScene(){ /* TBD */ };

    void PrintScene() const;
    void PrintCamera() const;
    void PrintMeshes() const;

public:
    CameraPtr camera;
    std::vector<MaterialPtr> materials;
    std::vector<LightPtr> lights;

    std::vector<TriangleMeshPtr> meshes;
    std::vector<uint32_t> vertexOffset;
    std::vector<uint32_t> triangleOffset;

public:
    uint32_t vertexNum;
    uint32_t triangleNum;
    /* For areaLight */
    uint32_t areaLightVertexNum;
    

private:
    void TransformScene(minipbrt::Scene *miniScene);
    void TransformCamera(minipbrt::Scene *miniScene);
    void TransformMaterial(minipbrt::Scene *miniScene);
    void TransformLight(minipbrt::Scene *miniScene);
    void TransformMeshes(minipbrt::Scene *miniScene);
    /* Bind AreaLight and mesh */
    void BindAreaLight();

};

} // namespace EasyRender
