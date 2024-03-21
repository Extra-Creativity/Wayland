#pragma once
#include <minipbrt.h>
#include <string>
#include <memory>

#include "Core/Camera.h"
#include "Core/Shape.h"

namespace Wayland
{

class SceneManager
{
public:
    SceneManager(string_view sceneSrc_);
    SceneManager() = default;
    ~SceneManager() = default;

    void LoadScene(){ /* TBD */ };

    void PrintScene() const;
    void PrintCamera() const;
    void PrintMeshes() const;

public:
    CameraPtr camera;
    vector<TriangleMeshPtr> meshes;

private:
    void TransformScene(minipbrt::Scene *miniScene);
    void TransformCamera(minipbrt::Scene *miniScene);
    void TransformMeshes(minipbrt::Scene *miniScene);

private:
};

} // namespace Wayland
