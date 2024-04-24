#include "Core/DeviceManager.h"
#include "Device/Light.h"
#include "HostUtils/DeviceAllocators.h"
#include "Utils/Common.h"
#include "Utils/PrintUtils.h"
#include "spdlog/spdlog.h"

using namespace EasyRender::Optix;
using namespace std;

namespace EasyRender
{

DeviceManager::DeviceManager()
    : frameBufferSize(0), indexBufferSize(0), normalBufferSize(0)
{
    SetEnvironment();
}

void DeviceManager::SetEnvironment()
{
    contextManager.SetCachePath(".");
    Module::SetOptixSDKPath(OPTIX_DIR);
    Module::AddArgs("-I\"" GLM_DIR "\"");
    Module::AddArgs("-I\"" UTILS_INC_PATH "\"");
    Module::AddArgs("-I\"" FRONT_PATH "\"");
    Module::AddArgs("-diag-suppress 20012 -diag-suppress 3012");
}

void DeviceManager::SetupOptix(SceneManager &scene, MainWindow &window,
                               std::string_view programSrc,
                               ProgramManager *program)
{
    AllocDeviceBuffer(scene, window.size);
    BuildAccelStructure(scene);
    BuildPipeline(programSrc);
    UploadTextures(scene);
    UploadDeviceBuffer(scene);
    BuildSBT(program);
}

void DeviceManager::BuildAccelStructure(SceneManager &scene)
{
    asBuildInput =
        std::make_unique<TriangleBuildInputArray>(scene.meshes.size());
    for (int i = 0; i < scene.meshes.size(); ++i)
    {
        asBuildInput->AddBuildInput(
            { VecToSpan<const float, 3>(scene.meshes[i]->vertex) },
            VecToSpan<const unsigned int, 3>(scene.meshes[i]->triangle),
            GeometryFlags::DiableAnyHit);
    }
    as = std::make_unique<StaticAccelStructure>(*asBuildInput);
    cudaDeviceSynchronize();
}

void DeviceManager::BuildPipeline(std::string_view programSrc)
{
    assert(as);
    auto pipelineOption = PipelineConfig::GetDefault().SetNumPayloadValues(2);
    module = std::make_unique<Module>(programSrc, ModuleConfig::GetDefaultRef(),
                                      pipelineOption);
    pg = std::make_unique<ProgramGroupArray>();
    module->IdentifyPrograms({ string(programSrc) }, *pg);
    pipeline = std::make_unique<Pipeline>(*pg, 1, as->GetDepth(), 0, 0,
                                          pipelineOption);
}

void DeviceManager::BuildSBT(ProgramManager *program)
{
    assert(pg);
    assert(frameBufferSize > 0);
    sbt =
        std::make_unique<Optix::ShaderBindingTable>(program->GenerateSBT(*pg));
}

void DeviceManager::AllocDeviceBuffer(SceneManager &scene, glm::ivec2 wSize)
{
    /* TBD: Optimize by mallocing a large block at once */
    frameBufferSize = wSize.x * wSize.y * sizeof(glm::u8vec4);
    cudaMalloc(&d_FrameBuffer, frameBufferSize);
    spdlog::info("Malloc {} bytes at device -> frameBuffer", frameBufferSize);

    assert(scene.triangleNum % 3 == 0);
    indexBufferSize = (uint32_t)scene.triangleNum * sizeof(glm::ivec3);
    cudaMalloc(&d_IndexBuffer, indexBufferSize);
    spdlog::info("Malloc {} bytes at device -> indexBuffer", indexBufferSize);

    normalBufferSize = (uint32_t)scene.vertexNum * sizeof(glm::vec3);
    cudaMalloc(&d_NormalBuffer, normalBufferSize);
    spdlog::info("Malloc {} bytes at device -> vertexBuffer", normalBufferSize);

    texcoordBufferSize = (uint32_t)scene.vertexNum * sizeof(glm::vec2);
    cudaMalloc(&d_TexcoordBuffer, texcoordBufferSize);
    spdlog::info("Malloc {} bytes at device -> vertexBuffer", normalBufferSize);

    /* We only have areaLight, for now */
    areaLightBufferSize =
        (uint32_t)scene.lights.size() * sizeof(Device::DeviceAreaLight);
    areaLightBufferSize += scene.areaLightVertexNum * sizeof(glm::vec3);
    cudaMalloc(&d_AreaLightBuffer, areaLightBufferSize);
    d_AreaLightVertexBuffer =
        reinterpret_cast<glm::vec3 *>(d_AreaLightBuffer + scene.lights.size());
    spdlog::info("Malloc {} bytes at device -> areaLightBuffer",
                 areaLightBufferSize);
}

void DeviceManager::UploadDeviceBuffer(SceneManager &scene)
{

    /* Upload normal, index, texcoord */
    std::vector<glm::vec3> normalAggregate;
    std::vector<glm::ivec3> indexAggregate;
    std::vector<glm::vec2> texcoordAggregate;

    uint32_t nSize = scene.vertexNum, iSize = scene.triangleNum;

    normalAggregate.resize(nSize);
    indexAggregate.resize(iSize);
    texcoordAggregate.resize(nSize);
    for (int i = 0; i < scene.meshes.size(); ++i)
    {
        auto *mesh = scene.meshes[i].get();
        int n_s = mesh->normal.size();
        int i_s = mesh->triangle.size();
        int t_s = mesh->uv.size();
        memcpy(normalAggregate.data() + scene.vertexOffset[i],
               mesh->normal.data(), mesh->normal.size() * sizeof(glm::vec3));
        memcpy(indexAggregate.data() + scene.triangleOffset[i],
               mesh->triangle.data(),
               mesh->triangle.size() * sizeof(glm::ivec3));
        memcpy(texcoordAggregate.data() + scene.vertexOffset[i],
               mesh->uv.data(), mesh->uv.size() * sizeof(glm::vec2));
    }

    cudaMemcpy(d_NormalBuffer, normalAggregate.data(),
               nSize * sizeof(glm::vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_IndexBuffer, indexAggregate.data(), iSize * sizeof(glm::ivec3),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_TexcoordBuffer, texcoordAggregate.data(),
               nSize * sizeof(glm::vec2), cudaMemcpyHostToDevice);

    /* Upload areaLight */
    auto &lights = scene.lights;
    /* Use a little trick to malloc void buffer in Cpp */
    void *hostBuf = ::operator new(areaLightBufferSize);
    auto *lightBuf = reinterpret_cast<Device::DeviceAreaLight *>(hostBuf);
    auto *vertexBuf = reinterpret_cast<glm::vec3 *>(lightBuf + lights.size());

    uint32_t v_offset = 0;
    for (int i = 0; i < lights.size(); ++i)
    {
        lightBuf[i].L = lights[i]->L;
        lightBuf[i].twoSided = lights[i]->twoSided;
        uint32_t meshIdx = lights[i]->mesh;
        /* AreaLight must have a corresponding mesh */
        assert(meshIdx < INVALID_INDEX);
        lightBuf[i].normals = d_NormalBuffer + scene.vertexOffset[meshIdx];
        lightBuf[i].indices = d_IndexBuffer + scene.triangleOffset[meshIdx];
        lightBuf[i].vertices = d_AreaLightVertexBuffer + v_offset;
        lightBuf[i].triangleNum = static_cast<uint32_t>(scene.meshes[meshIdx]->triangle.size());

        auto &mVertex = scene.meshes[meshIdx]->vertex;
        memcpy(vertexBuf + v_offset, mVertex.data(),
               mVertex.size() * sizeof(glm::vec3));
        v_offset += mVertex.size();
    }
    cudaMemcpy(d_AreaLightBuffer, hostBuf, areaLightBufferSize,
               cudaMemcpyHostToDevice);
    ::operator delete(hostBuf);
}

void DeviceManager::UploadTextures(SceneManager &scene)
{
    int32_t numTextures = scene.textures.size();

    textureArrays.resize(numTextures);
    textureObjects.resize(numTextures);

    for (int32_t textureID = 0; textureID < numTextures; textureID++)
    {
        auto &tex = scene.textures[textureID];
        cudaResourceDesc res_desc = {};
        cudaChannelFormatDesc channel_desc;
        int32_t width = tex->size.x;
        int32_t height = tex->size.y;
        int32_t numComponents = 4;
        int32_t pitch = width * numComponents * sizeof(uint8_t);
        channel_desc = cudaCreateChannelDesc<uchar4>();

        cudaArray_t &pixelArray = textureArrays[textureID];
        cudaMallocArray(&pixelArray, &channel_desc, width, height);

        cudaMemcpy2DToArray(pixelArray,
                            /* offset */ 0, 0, tex->data, pitch, pitch, height,
                            cudaMemcpyHostToDevice);

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = pixelArray;

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode = cudaFilterModePoint;
        tex_desc.borderColor[0] = 1.0f;
        tex_desc.sRGB = 0;

        // Create texture object
        cudaTextureObject_t cuda_tex = 0;
        cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr);
        textureObjects[textureID] = cuda_tex;
    }
}

void DeviceManager::Launch(ProgramManager *program, glm::ivec2 wSize)
{
    assert(as);
    assert(pipeline);
    assert(sbt);

    Launcher launcher{ program->GetParamPtr(), program->GetParamSize() };
    launcher.Launch(*pipeline, LocalContextSetter::GetCurrentCUDAStream(), *sbt,
                    wSize.x, wSize.y);
    cudaDeviceSynchronize();
}

void DeviceManager::DownloadFrameBuffer(MainWindow &window) const
{
    int size = window.size.x * window.size.y * 4;
    cudaMemcpy(window.frameBuffer.data(), d_FrameBuffer, size,
               cudaMemcpyDeviceToHost);
}

OptixTraversableHandle DeviceManager::GetTraversableHandle()
{
    assert(as);
    return as->GetHandle();
}

} // namespace EasyRender