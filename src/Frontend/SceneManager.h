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

    void printScene() const;
    void printCamera() const;
    void printMeshes() const;

public:
    CameraPtr camera;
    vector<TriangleMeshPtr> meshes;

private:
    void transformScene(minipbrt::Scene *miniScene);
    void transformCamera(minipbrt::Scene *miniScene);
    void transformMeshes(minipbrt::Scene *miniScene);


private:
};

} // namespace Wayland
