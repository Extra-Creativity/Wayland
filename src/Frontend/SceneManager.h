#pragma once
#include <minipbrt.h>
#include <string>
#include "Camera.h"

using namespace std;

namespace Wayland
{

class SceneManager
{
public:
    SceneManager(string sceneSrc);
    ~SceneManager() {}

    void printScene() const;
    void printCamera() const;

public:
    CameraPtr camera;

private:
    void transformScene();
    void transformCamera();


private:
    minipbrt::Scene* miniScene;
};

} // namespace Wayland
