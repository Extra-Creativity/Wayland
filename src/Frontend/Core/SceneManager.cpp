#include "Core/SceneManager.h"
#include "HostUtils/ErrorCheck.h"
#include "Utils/PBRTv3_SceneCommon.h"
#include "Utils/PrintUtils.h"
#include "spdlog/spdlog.h"

#include <iostream>

using namespace std;

namespace EasyRender
{

SceneManager::SceneManager(string_view sceneSrc_)
{
    minipbrt::Loader loader;
    string sceneSrc(sceneSrc_);
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
        delete miniScene;
    }
    catch (...)
    {
        delete miniScene;
        throw;
    }

    return;
}

// Transform minipbrt::Scene to EasyRender::SceneManager
void SceneManager::TransformScene(minipbrt::Scene *miniScene)
{
    assert(miniScene);
    /* A legal scene should have a camera */
    TransformCamera(miniScene);
    TransformMaterial(miniScene);
    TransformLight(miniScene);
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

void SceneManager::TransformMaterial(minipbrt::Scene *miniScene)
{
    /* Only handle diffuse material currently */
    for (int i = 0; i < miniScene->materials.size(); ++i)
    {
        auto miniMat = miniScene->materials[i];
        if (miniMat->type() == minipbrt::MaterialType::Matte)
        {
            auto miniMatte = dynamic_cast<minipbrt::MatteMaterial *>(miniMat);
            materials.push_back(make_unique<Diffuse>(miniMatte));
        }
        else
        {
            assert(int(miniMat->type()) <
                   int(PBRTv3::MaterialType::MaterialTypeMax));
            spdlog::warn("Unsupported shape type: {}",
                         PBRTv3::MaterialTypeStr[(int)miniMat->type()]);
            /* Break currently, for safety */
            throw std::exception("");
        }
    }
    spdlog::info("Successfully transform materials");
    return;
}

void SceneManager::TransformLight(minipbrt::Scene *miniScene)
{
    /* Only handle diffuse areaLight currently */
    for (int i = 0; i < miniScene->areaLights.size(); ++i)
    {
        auto miniLight = miniScene->areaLights[i];
        assert(miniLight->type() == minipbrt::AreaLightType::Diffuse);
        auto l = dynamic_cast<minipbrt::DiffuseAreaLight *>(miniLight);
        lights.push_back(make_unique<AreaLight>(l));
    }
    spdlog::info("Successfully transform lights");
    return;
}

void SceneManager::TransformMeshes(minipbrt::Scene *miniScene)
{
    /* Only handle triangle mesh currently */
    for (int i = 0; i < miniScene->shapes.size(); ++i)
    {
        auto miniShape = miniScene->shapes[i];
        if (miniShape->type() == minipbrt::ShapeType::TriangleMesh)
        {
            auto miniMesh = dynamic_cast<minipbrt::TriangleMesh *>(miniShape);
            meshes.push_back(make_unique<TriangleMesh>(miniMesh));
        }
        else
        {
            assert(int(miniShape->type()) <
                   int(PBRTv3::ShapeType::ShapeTypeMax));
            spdlog::warn("Unsupported shape type: {}",
                         PBRTv3::ShapeTypeStr[(int)miniShape->type()]);
            /* Break currently, for safety */
            throw std::exception("");
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
        cout << "    " << meshes[i]->vertex.size() * 3 << " vertices, ";
        cout << meshes[i]->index.size() << " triangles\n";
        nV += meshes[i]->vertex.size();
        nT += meshes[i]->index.size();
    }
    cout << "  " << nV * 3 << " vertices, " << nT << " triangles in total\n";
}

} // namespace EasyRender
