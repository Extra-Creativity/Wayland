#include "Model.h"

#include "spdlog/spdlog.h"
#include "stb_image/stb_image.h"

#include <format>
#include <fstream>

namespace Wayland::Example
{

template<typename T>
using STBUniquePtr =
    std::unique_ptr<T, decltype([](T *img) { stbi_image_free(img); })>;

template<typename T>
static inline STBUniquePtr<T> STBLoadImage(const std::filesystem::path &path,
                                           int &width, int &height,
                                           int &channels)
{
    std::ifstream fin{ path, std::ios::binary };
    std::ostringstream image;
    image << fin.rdbuf();
    const unsigned char *imagePtr;
    int size;
    {
        auto imageBuffer = image.view();
        imagePtr = reinterpret_cast<const unsigned char *>(imageBuffer.data());
        HostUtils::CheckError(
            HostUtils::CheckInRangeAndSet(imageBuffer.size(), size),
            "Image over 2G cannot be loaded.");
    }
    HostUtils::CheckError(
        stbi_info_from_memory(imagePtr, size, &width, &height, &channels) != 0,
        "Unable to load image.");

    int expectChannels = 0;
    if (channels == 3)
    {
        SPDLOG_WARN(
            "CUDA texture only supports 1, 2, 4 channels; image {} rgb will be "
            "turned into rgba.",
            path.string());
        channels = expectChannels = 4;
    }
    int discardInfo;

    STBUniquePtr<T> ptr;
    if constexpr (std::is_same_v<T, unsigned char>)
        ptr = STBUniquePtr<T>{ stbi_load_from_memory(
            imagePtr, size, &width, &height, &discardInfo, expectChannels) };
    else if constexpr (std::is_same_v<T, unsigned short>)
        ptr = STBUniquePtr<T>{ stbi_load_16_from_memory(
            imagePtr, size, &width, &height, &discardInfo, expectChannels) };
    else if constexpr (std::is_same_v<T, float>)
        ptr = STBUniquePtr<T>{ stbi_loadf_from_memory(
            imagePtr, size, &width, &height, &discardInfo, expectChannels) };

    HostUtils::CheckError(ptr != nullptr,
                          ("Cannot load image " + path.string()).c_str(),
                          stbi_failure_reason());

    return ptr;
}

template<typename T>
static inline cudaChannelFormatDesc CreateChannelFormatDesc(int channelNum)
{
    constexpr auto size = sizeof(T) * 8;
    constexpr cudaChannelFormatKind kind = std::is_integral_v<T>
                                               ? cudaChannelFormatKindUnsigned
                                               : cudaChannelFormatKindFloat;
    switch (channelNum)
    {
    case 1:
        return cudaCreateChannelDesc(size, 0, 0, 0, kind);
    case 2:
        return cudaCreateChannelDesc(size, size, 0, 0, kind);
    case 4:
        return cudaCreateChannelDesc(size, size, size, size, kind);
    [[unlikely]] default:
        throw std::runtime_error{ "Unrecognized texture format." };
    }
    std::unreachable();
    return cudaChannelFormatDesc{};
}

template<typename T>
void GPUTextureData::LoadResources_(const std::filesystem::path &path)
{
    int width, height, channels;
    auto ptr = STBLoadImage<T>(path, width, height, channels);
    auto channelDesc = CreateChannelFormatDesc<unsigned char>(channels);

    buffer_ = [&]() {
        cudaArray_t buffer;
        HostUtils::CheckCUDAError(
            cudaMallocArray(&buffer, &channelDesc, width, height));
        return CUArrayUniquePtr{ buffer };
    }();

    // TODO: float are not related to channels?
    auto bytesPerElem = channels * sizeof(T);
    // stb_image doesn't have any padding, so width == spitch.
    HostUtils::CheckCUDAError(cudaMemcpy2DToArray(
        buffer_.get(), 0, 0, ptr.get(), width * bytesPerElem,
        width * bytesPerElem, height, cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = buffer_.get();

    cudaTextureDesc texDesc{};
    // NOTICE: currently, we do not let users to specify wrap mode themselves.
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    // NOTICE: currently, we don't specify cudaResourceViewDesc.
    HostUtils::CheckCUDAError(
        cudaCreateTextureObject(&texture_, &resDesc, &texDesc, NULL));
}

GPUTextureData::GPUTextureData(const std::filesystem::path &path, Format format)
{
    switch (format)
    {
    case Format::UByte:
        LoadResources_<unsigned char>(path);
        return;
    case Format::UShort:
        LoadResources_<unsigned short>(path);
        return;
    case Format::Float:
        LoadResources_<float>(path);
        return;
    default:
        throw std::runtime_error{ "Unrecognized texture format." };
    }
    std::unreachable();
    return;
}

GPUTextureData::GPUTextureData(GPUTextureData &&another) noexcept
    : buffer_{ std::move(another.buffer_) },
      texture_{ std::exchange(another.texture_, 0) }
{
}

GPUTextureData &GPUTextureData::operator=(GPUTextureData &&another) noexcept
{
    buffer_ = std::move(another.buffer_);
    texture_ = std::exchange(another.texture_, 0);
    return *this;
}

GPUTextureData::~GPUTextureData()
{
    HostUtils::CheckCUDAError<HostUtils::OnlyLog>(
        cudaDestroyTextureObject(texture_));
}

void MeshCollection::LoadScene_(const aiScene *scene,
                                Wayland::Optix::GeometryFlags flag)
{
    // TODO: importer.GetErrorString() ?
    HostUtils::CheckError(
        (scene != nullptr) &
            ((scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) == 0) &
            (scene->mRootNode != nullptr),
        "Unable to load model.");

    texCoords_ = std::make_unique<DeviceTexCoordBuffer[]>(scene->mNumMeshes);
    normals_ = std::make_unique<DeviceNormalBuffer[]>(scene->mNumMeshes);
    std::vector<unsigned int> triangles;
    std::vector<glm::vec2> texCoords;
    for (unsigned int i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh *mesh = scene->mMeshes[i];

        triangles.reserve(static_cast<std::size_t>(mesh->mNumFaces) * 3);
        for (unsigned int i = 0; i < mesh->mNumFaces; i++)
        {
            auto &currFace = mesh->mFaces[i];
            assert(currFace.mNumIndices == 3);
            triangles.append_range(std::span{ currFace.mIndices, 3 });
        }

        auto vertNum = mesh->mNumVertices;
        if (mesh->mTextureCoords[0] != nullptr)
        {
            texCoords.resize(vertNum);
            std::ranges::transform(
                std::span{ mesh->mTextureCoords[0], vertNum },
                texCoords.begin(), [](const aiVector3D vec) {
                    return glm::vec2{ vec.x, vec.y };
                });
            texCoords_[i] = HostUtils::DeviceMakeUnique<glm::vec2[]>(texCoords);
        }

        normals_[i] =
            HostUtils::DeviceMakeUninitializedUnique<glm::vec3[]>(vertNum);
        HostUtils::CheckCUDAError(
            cudaMemcpy(normals_[i].get().get(), mesh->mNormals,
                       vertNum * sizeof(glm::vec3), cudaMemcpyHostToDevice));

        buildInputs_.AddBuildInput(
            { std::span{ reinterpret_cast<const float *>(mesh->mVertices),
                         static_cast<std::size_t>(mesh->mNumVertices) * 3 } },
            triangles, flag);
        triangles.clear();
    }
}

void StaticModel::LoadTextures_(const std::filesystem::path &rootPath,
                                GPUTextureData::Format format)
{
    stbi_set_flip_vertically_on_load(true);
    auto scene = model_.GetScene();
    aiString relPath;
    // TODO : test speed up.
    std::pmr::polymorphic_allocator<char> allocator{ &memoryResource_ };

    for (unsigned int i = 0; i < scene->mNumMaterials; i++)
    {
        aiMaterial *material = scene->mMaterials[i];
        for (std::underlying_type_t<aiTextureType> type = aiTextureType_NONE;
             type < AI_TEXTURE_TYPE_MAX; type++)
        {
            auto texType = static_cast<aiTextureType>(type);
            auto textureNum = material->GetTextureCount(texType);
            for (unsigned int textureID = 0; textureID < textureNum;
                 textureID++)
            {
                HostUtils::CheckError(
                    material->GetTexture(texType, textureID, &relPath) ==
                        aiReturn_SUCCESS,
                    "Cannot get texture from model.");
                // reason: to make filesystem recognize it.
                std::u8string_view u8RelPath{ reinterpret_cast<const char8_t *>(
                                                  relPath.C_Str()),
                                              relPath.length };
                texturePool_.try_emplace(std::pmr::string{ relPath.C_Str(),
                                                           relPath.length,
                                                           allocator },
                                         rootPath / u8RelPath, format);
            }
        }
    }
}

} // namespace Wayland::Example
