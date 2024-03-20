#pragma once
#include <minipbrt.h>
#include <string>
#include "Camera.h"
#include "Shape.h"


namespace Wayland
{

class SceneManager
{
public:
    SceneManager(string sceneSrc);
    SceneManager() = default;
    ~SceneManager() = default;

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
