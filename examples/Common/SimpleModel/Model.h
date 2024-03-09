#pragma once

#include "Optix/Core/AccelStructure.h"

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "glm/glm.hpp"

#include <filesystem>
#include <unordered_map>
#include <vector>

#include "Common.h"

namespace Wayland::Example
{

class MeshCollection
{
    friend class StaticModel;
    void LoadScene_(const aiScene *, Wayland::Optix::GeometryFlags);

public:
    template<typename T>
        requires std::invocable<T, const StaticModel &, unsigned int,
                                unsigned int, unsigned int>
    MeshCollection(const StaticModel &model, const aiScene *init_scene,
                   Wayland::Optix::GeometryFlags flag, T &&init_setter)
        : buildInputs_{ init_scene->mNumMeshes }
    {
        LoadScene_(init_scene, flag);
        buildInputs_.SetSBTSetter(
            [setter = std::move(init_setter), &model](unsigned int buildInputID,
                                                      unsigned int sbtRecordID,
                                                      unsigned int rayType) {
                return setter(model, buildInputID, sbtRecordID, rayType);
            });
    }

    auto GetMeshNum() const noexcept { return buildInputs_.GetBuildInputNum(); }

private:
    using DeviceTexCoordBuffer =
        Wayland::HostUtils::DeviceUniquePtr<glm::vec2[]>;
    using DeviceNormalBuffer = Wayland::HostUtils::DeviceUniquePtr<glm::vec3[]>;

    Wayland::Optix::TriangleBuildInputArray buildInputs_;
    std::unique_ptr<DeviceNormalBuffer[]> normals_;
    std::unique_ptr<DeviceTexCoordBuffer[]> texCoords_;
};

using CUArrayUniquePtr =
    std::unique_ptr<cudaArray, decltype([](cudaArray_t arr) {
                        HostUtils::CheckCUDAError<HostUtils::OnlyLog>(
                            cudaFreeArray(arr));
                    })>;

class GPUTextureData
{
    template<typename T>
    void LoadResources_(const std::filesystem::path &);

public:
    enum class Format
    {
        UByte,
        UShort,
        Float
    };

    GPUTextureData(const std::filesystem::path &path, Format format);
    GPUTextureData(const GPUTextureData &) = delete;
    GPUTextureData &operator=(const GPUTextureData &) = delete;
    GPUTextureData(GPUTextureData &&) noexcept;
    GPUTextureData &operator=(GPUTextureData &&) noexcept;
    ~GPUTextureData();

    auto GetHandle() const noexcept { return texture_; }

private:
    CUArrayUniquePtr buffer_;
    cudaTextureObject_t texture_;
};

class StaticModel
{
    void LoadTextures_(const std::filesystem::path &, GPUTextureData::Format);

public:
    template<typename T>
    StaticModel(const std::filesystem::path &path, T &&setter,
                Wayland::Optix::GeometryFlags flag,
                GPUTextureData::Format format = GPUTextureData::Format::UByte,
                unsigned int postProcess = aiProcess_Triangulate |
                                           aiProcess_GenSmoothNormals |
                                           aiProcess_FlipUVs |
                                           aiProcess_JoinIdenticalVertices)
        : model_{},
          meshCollection_{ *this, model_.ReadFile(path.string(), postProcess),
                           flag, std::forward<T>(setter) },
          as_{ meshCollection_.buildInputs_ }, memoryResource_{},
          texturePool_{ std::pmr::polymorphic_allocator<
              decltype(texturePool_)::value_type>{ &memoryResource_ } }
    {
        LoadTextures_(path.parent_path(), format);
    }

    const aiScene *GetScene() const noexcept { return model_.GetScene(); }

    const GPUTextureData *GetTexture(const aiString &path) const noexcept
    {
        auto it =
            texturePool_.find(std::string_view{ path.C_Str(), path.length });
        return it == texturePool_.end() ? nullptr : &it->second;
    }

    glm::vec3 *GetDeviceNormalBuffer(std::size_t i) const
    {
        HostUtils::CheckError(i < meshCollection_.GetMeshNum(),
                              "Normal buffer access out of range");
        return meshCollection_.normals_[i].get().get();
    }

    glm::vec2 *GetDeviceTextureBuffer(std::size_t i) const
    {
        HostUtils::CheckError(i < meshCollection_.GetMeshNum(),
                              "Texture buffer access out of range");
        return meshCollection_.texCoords_[i].get().get();
    }

    glm::ivec3 *GetIndices(std::size_t i) const
    {
        return static_cast<glm::ivec3 *>(
            meshCollection_.buildInputs_.GetTriangleIndicesBuffer(i));
    }

    const auto &GetAS() const noexcept { return as_; }

private:
    Assimp::Importer model_;
    MeshCollection meshCollection_;
    Wayland::Optix::StaticGeometryAccelStructure as_;
    std::pmr::monotonic_buffer_resource memoryResource_;
    std::pmr::unordered_map<std::pmr::string, GPUTextureData, StringHasher,
                            std::equal_to<>>
        texturePool_;
};

} // namespace Wayland::Example