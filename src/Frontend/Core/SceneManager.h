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

private:
    void TransformScene(minipbrt::Scene *miniScene);
    void TransformCamera(minipbrt::Scene *miniScene);
    void TransformMaterial(minipbrt::Scene *miniScene);
    void TransformLight(minipbrt::Scene *miniScene);
    void TransformMeshes(minipbrt::Scene *miniScene);

};

} // namespace EasyRender
