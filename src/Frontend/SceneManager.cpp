#include "SceneManager.h"
#include "HostUtils/ErrorCheck.h"
#include "PBRTv3_SceneCommon.h"
#include "spdlog/spdlog.h"

#include <iostream>

using namespace std;

namespace Wayland
{

SceneManager::SceneManager(string sceneSrc) : miniScene(nullptr)
{
    minipbrt::Loader loader;
    bool loadResult = loader.load(sceneSrc.c_str());

    if (loadResult == false)
    {
        // If parsing failed, the parser will have an error object.
        const minipbrt::Error *err = loader.error();

        spdlog::error("Fail to load {}", err->filename());
        spdlog::error("[{}, line {}, column {}] {}", err->filename(),
                      err->line(), err->column(), err->message());
        // Don't delete err, it's still owned by the parser.
        return;
    }

    try
    {
        spdlog::info("Successfully load {}", sceneSrc);
        miniScene = loader.take_scene();
        /* Transform minipbrt scene to ours */
        transformScene();
    }
    catch (...)
    {
        delete miniScene;
        miniScene = nullptr;
        throw;
    }

    delete miniScene;
    miniScene = nullptr;
}

// Transform minipbrt::Scene to Wayland::SceneManager
void SceneManager::transformScene()
{
    assert(miniScene);
    /* A legal scene should have a camera */
    transformCamera();
    return;
}

void SceneManager::transformCamera()
{
    /* Check the scene do have a camera */
    HostUtils::CheckError(miniScene->camera,
                          "Loaded scene does not have a camera.");
    Camera *mCamera = nullptr;

    switch (miniScene->camera->type())
    {
    case minipbrt::CameraType::Perspective: {
        /* pinhole */
        auto miniCamera =
            dynamic_cast<minipbrt::PerspectiveCamera *>(miniScene->camera);
        mCamera = new PinholeCamera(miniCamera->fov, miniCamera->cameraToWorld);
    }
    break;

    default:
        /* We only handle pinhole(perspective), currently */
        int t = int(miniScene->camera->type());
        /* This should not happen, check for safety */
        if (t < 0 || t > int(PBRTv3::CameraType::CameraTypeMax))
            t = int(PBRTv3::CameraType::CameraTypeMax);
        string errMsg = "Unsupported camera type: " + PBRTv3::CameraTypeStr[t];
        /* Report error and throw an exception */
        HostUtils::CheckError(false, errMsg.c_str());
    }

    assert(mCamera);
    camera = CameraPtr(mCamera);
    spdlog::info("Successfully transform camera");
    return;
}

void SceneManager::printScene() const
{
    cout << "\n----- Scene -----\n";
    printCamera();
    cout << "\n----- End of Scene -----\n";
    return;
}

void SceneManager::printCamera() const
{
    assert(camera);
    cout << "Camera:\n";
    cout << camera->toString();
    return;
}

} // namespace Wayland
