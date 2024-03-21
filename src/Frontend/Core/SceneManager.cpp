#include "spdlog/spdlog.h"
#include "HostUtils/ErrorCheck.h"
#include "Core/SceneManager.h"
#include "Utils/PrintUtils.h"

#include <iostream>

using namespace std;

namespace Wayland
{

SceneManager::SceneManager(string_view sceneSrc_)
{
    minipbrt::Loader loader;
    string sceneSrc (sceneSrc_);
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

    minipbrt::Scene *miniScene = nullptr;
    try
    {
        spdlog::info("Successfully load {}", sceneSrc);
        miniScene = loader.take_scene();
        /* Transform minipbrt scene to ours */
        TransformScene(miniScene);
        // cout << "\nminiScene:\n" << printUtils::toString(miniScene) << endl;
        delete miniScene;
    }
    catch (...)
    {
        delete miniScene;
        throw;
    }

    return;
}

// Transform minipbrt::Scene to Wayland::SceneManager
void SceneManager::TransformScene(minipbrt::Scene *miniScene)
{
    assert(miniScene);
    /* A legal scene should have a camera */
    TransformCamera(miniScene);
    TransformMeshes(miniScene);
    return;
}

void SceneManager::TransformCamera(minipbrt::Scene *miniScene)
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


void SceneManager::TransformMeshes(minipbrt::Scene* miniScene) {

    /* Only handle triangle mesh currently */
    for (int i = 0; i < miniScene->shapes.size(); ++i)
    {
		auto miniShape = miniScene->shapes[i];
        if (miniShape->type() == minipbrt::ShapeType::TriangleMesh)
        {
			auto miniMesh = dynamic_cast<minipbrt::TriangleMesh*>(miniShape);
            meshes.push_back(TriangleMeshPtr(new TriangleMesh(miniMesh)));
		}   
    }
    spdlog::info("Successfully transform meshes");
    return;
}



void SceneManager::PrintScene() const
{
    cout << "\n----- Scene -----\n";
    PrintCamera();
    PrintMeshes();
    cout << "\n----- End of Scene -----\n";
    return;
}

void SceneManager::PrintCamera() const
{
    assert(camera);
    cout << "\nCamera:\n";
    cout << camera->ToString();
    return;
}

void SceneManager::PrintMeshes() const
{
    cout << "\nMeshes:\n";
    cout << "  size: " << meshes.size() << "\n";
    int nV = 0, nT = 0;
    for (int i = 0; i < meshes.size(); ++i)
    {
        cout << "  <- mesh " << i << " ->\n";
        cout << "    " << meshes[i]->vertex.size()*3 << " vertices, ";
        cout << meshes[i]->index.size()<< " triangles\n";
        nV += meshes[i]->vertex.size();
        nT += meshes[i]->index.size();
    }
    cout << "  " << nV*3 << " vertices, " << nT << " triangles in total\n";
}

} // namespace Wayland
