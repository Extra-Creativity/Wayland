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
        miniScene->load_all_ply_meshes();
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
    TransformTexture(miniScene);
    TransformMaterial(miniScene);
    TransformLight(miniScene);
    TransformMesh(miniScene);
    BindAreaLight();
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
            /* Break for safety */
            throw std::exception("");
        }
    }
    spdlog::info("Successfully transform materials");
    return;
}

void SceneManager::TransformTexture(minipbrt::Scene *miniScene)
{
    for (int i = 0; i < miniScene->textures.size(); ++i)
    {
        auto *miniTex = miniScene->textures[i];
        switch (miniTex->type())
        {
        case minipbrt::TextureType::ImageMap: {
            auto *t = dynamic_cast<minipbrt::ImageMapTexture *>(miniTex);
            textures.push_back(make_unique<Texture>(t));
        }
        break;
        case minipbrt::TextureType::Scale:
            /* TBD */
            break;
        default:
            assert(int(miniTex->type()) <
                   int(PBRTv3::TextureType::TextureTypeMax));
            spdlog::warn("Unsupported texture type: {}",
                         PBRTv3::TextureTypeStr[(int)miniTex->type()]);
            /* Break currently, for safety */
            throw std::exception("");
        }
    }
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

void SceneManager::TransformMesh(minipbrt::Scene *miniScene)
{
    /* Set for safety */
    vertexNum = triangleNum = areaLightVertexNum = 0;
    /* Only handle triangle mesh currently */
    for (int i = 0; i < miniScene->shapes.size(); ++i)
    {
        auto miniShape = miniScene->shapes[i];

        switch (miniShape->type())
        {
        case minipbrt::ShapeType::TriangleMesh: {
            auto miniMesh = dynamic_cast<minipbrt::TriangleMesh *>(miniShape);
            meshes.push_back(make_unique<TriangleMesh>(miniMesh));
            vertexOffset.push_back(vertexNum);
            triangleOffset.push_back(triangleNum);
            triangleNum += miniMesh->num_indices / 3;
            vertexNum += miniMesh->num_vertices;
            if (miniMesh->areaLight < minipbrt::kInvalidIndex)
                areaLightVertexNum += miniMesh->num_vertices;
            break;
        }
        //case minipbrt::ShapeType::PLYMesh: {

        //    auto miniMesh = dynamic_cast<minipbrt::PLYMesh *>(miniShape);
        //    auto triangleMesh = miniMesh->triangle_mesh();
        //    /* Same as TriangleMesh */
        //    meshes.push_back(make_unique<TriangleMesh>(triangleMesh));
        //    vertexOffset.push_back(vertexNum);
        //    triangleOffset.push_back(triangleNum);
        //    triangleNum += triangleMesh->num_indices / 3;
        //    vertexNum += triangleMesh->num_vertices;
        //    if (triangleMesh->areaLight < minipbrt::kInvalidIndex)
        //        areaLightVertexNum += triangleMesh->num_vertices;
        //    delete triangleMesh;
        //    break;
        //}
        default: {
            assert(int(miniShape->type()) <
                   int(PBRTv3::ShapeType::ShapeTypeMax));
            spdlog::warn("Unsupported shape type: {}",
                         PBRTv3::ShapeTypeStr[(int)miniShape->type()]);
            /* Break currently, for safety */
            throw std::exception("");
        }
        }
    }
    spdlog::info("Successfully transform meshes");
    return;
}

void SceneManager::BindAreaLight()
{
    /* This is ONLY valid after transforming mesh and areaLight */
    for (int i = 0; i < meshes.size(); ++i)
    {
        if (meshes[i]->areaLight < INVALID_INDEX)
        {
            lights[meshes[i]->areaLight]->mesh = i;
        }
    }
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
    cout << "  size: " << meshes.size() << "  vertex: " << vertexNum
         << "  triangle: " << triangleNum << "\n";
    int nV = 0, nT = 0;
    for (int i = 0; i < meshes.size(); ++i)
    {
        cout << "  <- mesh " << i << " ->\n";
        cout << "    " << meshes[i]->vertex.size() * 3 << " vertices, ";
        cout << meshes[i]->triangle.size() << " triangles\n";
    }
}

} // namespace EasyRender
