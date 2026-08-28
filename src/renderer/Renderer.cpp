#include "renderer/renderer/Renderer.hpp"

#include "renderer/assets/GltfImporter.hpp"
#include "renderer/assets/ImageDecoder.hpp"
#include "renderer/assets/ImageWriter.hpp"
#include "renderer/core/Log.hpp"
#include "renderer/gpu_scene/GpuScene.hpp"
#include "renderer/lighting/IblBaker.hpp"
#include "renderer/platform/Window.hpp"
#include "renderer/ray_tracing/AccelerationStructurePlan.hpp"
#include "renderer/temporal/TemporalSampling.hpp"
#include "renderer/vulkan/AccelerationStructure.hpp"
#include "renderer/vulkan/Buffer.hpp"
#include "renderer/vulkan/Descriptors.hpp"
#include "renderer/vulkan/Device.hpp"
#include "renderer/vulkan/Image.hpp"
#include "renderer/vulkan/Instance.hpp"
#include "renderer/vulkan/Pipeline.hpp"
#include "renderer/vulkan/Sampler.hpp"
#include "renderer/vulkan/Shader.hpp"
#include "renderer/vulkan/UploadContext.hpp"
#include "renderer/vulkan/VulkanError.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/matrix.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace renderer {
namespace {

constexpr VkFormat iblColorFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
constexpr VkFormat brdfLutFormat = VK_FORMAT_R32G32_SFLOAT;
constexpr VkFormat hiZFormat = VK_FORMAT_R32_SFLOAT;
constexpr bool rayQueryShadowBackendImplemented = false;

// Adapter from the public, implementation-independent perceptual radius to the
// current single-level filter. A future pyramid is free to replace this mapping.
float currentBloomFilterScale(float perceptualRadius) { return perceptualRadius; }

VkDeviceSize alignDeviceSize(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment == 0 || value > std::numeric_limits<VkDeviceSize>::max() - (alignment - 1U)) {
        throw std::overflow_error("Vulkan device-size alignment overflow");
    }
    return ((value + alignment - 1U) / alignment) * alignment;
}

VkDeviceSize addDeviceSize(VkDeviceSize left, VkDeviceSize right) {
    if (left > std::numeric_limits<VkDeviceSize>::max() - right) {
        throw std::overflow_error("Vulkan device-size addition overflow");
    }
    return left + right;
}

vulkan::Swapchain createInitialSwapchain(const vulkan::Instance& instance,
                                         const vulkan::Device& device, platform::Window& window,
                                         bool transferSource) {
    const auto [width, height] = window.waitForDrawableSize();
    return vulkan::Swapchain(instance, device, width, height, transferSource);
}

bool supportsOptimalFormat(const vulkan::Device& device, VkFormat format,
                           VkFormatFeatureFlags requiredFeatures) {
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(device.physicalHandle(), format, &properties);
    return (properties.optimalTilingFeatures & requiredFeatures) == requiredFeatures;
}

VkFormat selectDepthFormat(const vulkan::Device& device) {
    constexpr std::array candidates{VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM};
    for (const VkFormat format : candidates) {
        if (supportsOptimalFormat(device, format,
                                  VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
            return format;
        }
    }
    throw std::runtime_error("No supported sampled depth-only attachment format");
}

struct SphereMesh {
    std::vector<gpu_scene::VertexRecord> vertices;
    std::vector<std::uint32_t> indices;
};

SphereMesh createSphereMesh(std::uint32_t latitudeSegments, std::uint32_t longitudeSegments) {
    if (latitudeSegments < 2U || longitudeSegments < 3U) {
        throw std::invalid_argument("Sphere mesh requires at least 2x3 segments");
    }
    SphereMesh mesh;
    mesh.vertices.reserve(static_cast<std::size_t>(latitudeSegments + 1) * (longitudeSegments + 1));
    mesh.indices.reserve(static_cast<std::size_t>(latitudeSegments) * longitudeSegments * 6);

    for (std::uint32_t latitude = 0; latitude <= latitudeSegments; ++latitude) {
        const float v = static_cast<float>(latitude) / static_cast<float>(latitudeSegments);
        const float phi = v * std::numbers::pi_v<float>;
        const float ringRadius = std::sin(phi);
        const float y = std::cos(phi);
        for (std::uint32_t longitude = 0; longitude <= longitudeSegments; ++longitude) {
            const float u = static_cast<float>(longitude) / static_cast<float>(longitudeSegments);
            const float theta = u * 2.0F * std::numbers::pi_v<float>;
            const glm::vec3 position{ringRadius * std::cos(theta), y, ringRadius * std::sin(theta)};
            const glm::vec3 tangent{-std::sin(theta), 0.0F, std::cos(theta)};
            mesh.vertices.push_back({.position = position,
                                     .normal = position,
                                     .uv0 = {u, v},
                                     .tangent = {tangent, -1.0F},
                                     .uv1 = {u, v}});
        }
    }

    const std::uint32_t rowWidth = longitudeSegments + 1;
    for (std::uint32_t latitude = 0; latitude < latitudeSegments; ++latitude) {
        for (std::uint32_t longitude = 0; longitude < longitudeSegments; ++longitude) {
            const std::uint32_t first = latitude * rowWidth + longitude;
            const std::uint32_t second = first + rowWidth;
            mesh.indices.insert(mesh.indices.end(),
                                {first, first + 1, second, first + 1, second + 1, second});
        }
    }
    return mesh;
}

struct CpuScenePayload {
    std::vector<gpu_scene::VertexRecord> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<gpu_scene::InstanceRecord> instances;
    std::vector<gpu_scene::MaterialRecord> materials;
    std::vector<gpu_scene::TransformRecord> transforms;
    std::vector<gpu_scene::PrimitiveRecord> primitives;
    std::vector<gpu_scene::PrimitiveLodRange> primitiveLods;
    std::vector<gpu_scene::LodSourceRecord> lodSources;
    struct Texture {
        assets::DecodedMipChain mips;
        scene::Sampler sampler;
        std::string label;
    };
    std::vector<Texture> textures;
    std::vector<gpu_scene::MeshletWorkRecord> meshletWork;
    std::string label;
};

CpuScenePayload makeAlphaShadowScenePayload() {
    constexpr float quadRadius = 1.41421356237F;
    std::vector<gpu_scene::VertexRecord> vertices{
        {.position = {-1.0F, -1.0F, 0.0F}, .uv0 = {0.0F, 0.0F}, .uv1 = {0.0F, 0.0F}},
        {.position = {1.0F, -1.0F, 0.0F}, .uv0 = {1.0F, 0.0F}, .uv1 = {1.0F, 0.0F}},
        {.position = {1.0F, 1.0F, 0.0F}, .uv0 = {1.0F, 1.0F}, .uv1 = {1.0F, 1.0F}},
        {.position = {-1.0F, 1.0F, 0.0F}, .uv0 = {0.0F, 1.0F}, .uv1 = {0.0F, 1.0F}},
    };
    std::vector<std::uint32_t> indices{0U, 1U, 2U, 0U, 2U, 3U};
    std::vector<gpu_scene::PrimitiveRecord> primitives{
        {.indexCount = static_cast<std::uint32_t>(indices.size()),
         .firstIndex = 0U,
         .vertexOffset = 0,
         .materialIndex = 0U,
         .centerRadius = {0.0F, 0.0F, 0.0F, quadRadius}},
    };
    std::vector<gpu_scene::PrimitiveLodRange> primitiveLods{{.firstLod = 0U, .lodCount = 1U}};
    std::vector<gpu_scene::LodSourceRecord> lodSources{
        {.primitiveIndex = 0U, .minimumProjectedDiameter = 0.0F}};

    constexpr glm::vec3 casterCenter{-0.9F, 1.0F, 0.0F};
    constexpr glm::vec3 casterScale{1.15F, 1.15F, 1.0F};
    constexpr glm::vec3 receiverCenter{0.0F, 0.0F, -1.5F};
    constexpr glm::vec3 receiverScale{4.0F, 3.5F, 1.0F};
    const glm::mat4 casterWorld =
        glm::translate(glm::mat4(1.0F), casterCenter) * glm::scale(glm::mat4(1.0F), casterScale);
    const glm::mat4 receiverWorld = glm::translate(glm::mat4(1.0F), receiverCenter) *
                                    glm::scale(glm::mat4(1.0F), receiverScale);
    std::vector<gpu_scene::TransformRecord> transforms{
        {.world = casterWorld,
         .normalToWorld = glm::transpose(glm::inverse(casterWorld)),
         .previousWorld = casterWorld},
        {.world = receiverWorld,
         .normalToWorld = glm::transpose(glm::inverse(receiverWorld)),
         .previousWorld = receiverWorld},
    };
    std::vector<gpu_scene::InstanceRecord> instances{
        {.centerRadius = {casterCenter, quadRadius * casterScale.x},
         .meshMaterial = {0U, 0U, 0U, 0U}},
        {.centerRadius = {receiverCenter, std::sqrt(receiverScale.x * receiverScale.x +
                                                    receiverScale.y * receiverScale.y)},
         .meshMaterial = {0U, 1U, 1U, 0U}},
    };
    std::vector<gpu_scene::MaterialRecord> materials{
        {.baseColorFactor = {0.10F, 0.72F, 0.24F, 1.0F},
         .emissiveAlphaCutoff = {0.0F, 0.0F, 0.0F, 0.5F},
         .metallicRoughnessOcclusionNormal = {0.0F, 0.55F, 1.0F, 1.0F},
         .textureIndices = {1U, 0U, 0U, 0U},
         .textureState = {0U, gpu_scene::baseColorTextureBit, gpu_scene::alphaMaskMaterial, 0U}},
        {.baseColorFactor = {0.18F, 0.20F, 0.24F, 1.0F},
         .emissiveAlphaCutoff = {0.0F, 0.0F, 0.0F, 0.5F},
         .metallicRoughnessOcclusionNormal = {0.0F, 0.85F, 1.0F, 1.0F}},
    };

    constexpr std::uint32_t textureExtent = 128U;
    assets::DecodedImage cutoutTexture{
        .width = textureExtent,
        .height = textureExtent,
        .componentType = assets::ImageComponentType::unorm8,
        .colorSpace = assets::ImageColorSpace::srgb,
    };
    cutoutTexture.unorm8.resize(static_cast<std::size_t>(textureExtent) * textureExtent * 4U);
    for (std::uint32_t y = 0; y < textureExtent; ++y) {
        for (std::uint32_t x = 0; x < textureExtent; ++x) {
            const bool opaque = ((x / 16U) + (y / 16U)) % 2U == 0U;
            const std::size_t offset = (static_cast<std::size_t>(y) * textureExtent + x) * 4U;
            cutoutTexture.unorm8[offset + 0U] = 255U;
            cutoutTexture.unorm8[offset + 1U] = 255U;
            cutoutTexture.unorm8[offset + 2U] = 255U;
            cutoutTexture.unorm8[offset + 3U] = opaque ? 255U : 0U;
        }
    }

    gpu_scene::MeshletHierarchy hierarchy =
        gpu_scene::buildMeshletHierarchy(vertices, indices, primitives, primitiveLods, lodSources);
    CpuScenePayload payload{
        .vertices = std::move(vertices),
        .indices = std::move(indices),
        .instances = std::move(instances),
        .materials = std::move(materials),
        .transforms = std::move(transforms),
        .primitives = std::move(primitives),
        .primitiveLods = std::move(primitiveLods),
        .lodSources = std::move(lodSources),
        .label = "procedural alpha-cutout shadow demo",
    };
    payload.textures.push_back({
        .mips = assets::generateMipChain(std::move(cutoutTexture)),
        .sampler = scene::Sampler{},
        .label = "procedural checker cutout",
    });
    payload.meshletWork = gpu_scene::makeMeshletWorkRecords(hierarchy.primitiveLods, hierarchy.lods,
                                                            hierarchy.meshlets, payload.instances);
    return payload;
}

CpuScenePayload makeProceduralScenePayload(bool emissiveSphereDemo, bool directionalShadowDemo,
                                           bool alphaShadowDemo) {
    if (alphaShadowDemo) {
        return makeAlphaShadowScenePayload();
    }
    SphereMesh fineSphere = createSphereMesh(24U, 48U);
    SphereMesh coarseSphere = createSphereMesh(12U, 24U);
    gpu_scene::SceneRecords scene = gpu_scene::makePbrSphereGrid();
    if (emissiveSphereDemo) {
        scene.instances.resize(1);
        scene.materials.resize(1);
        scene.transforms.resize(1);
        scene.instances.front() = {
            .centerRadius = {0.0F, 0.0F, 0.0F, 1.0F},
            .meshMaterial = {0U, 0U, 0U, 0U},
        };
        scene.materials.front() = {
            .baseColorFactor = {0.005F, 0.01F, 0.02F, 1.0F},
            .emissiveAlphaCutoff = {1.5F, 7.0F, 16.0F, 0.5F},
            .metallicRoughnessOcclusionNormal = {0.0F, 0.35F, 1.0F, 1.0F},
        };
        const glm::mat4 world = glm::scale(glm::mat4(1.0F), glm::vec3(0.42F));
        scene.transforms.front() = {
            .world = world,
            .normalToWorld = glm::transpose(glm::inverse(world)),
            .previousWorld = world,
        };
    }
    if (fineSphere.indices.size() > std::numeric_limits<std::uint32_t>::max() ||
        coarseSphere.indices.size() > std::numeric_limits<std::uint32_t>::max() ||
        fineSphere.vertices.size() >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("PBR sphere index count exceeds Vulkan draw limits");
    }
    const std::uint32_t coarseFirstIndex = static_cast<std::uint32_t>(fineSphere.indices.size());
    const std::int32_t coarseVertexOffset = static_cast<std::int32_t>(fineSphere.vertices.size());
    std::vector<gpu_scene::VertexRecord> vertices = std::move(fineSphere.vertices);
    vertices.insert(vertices.end(), std::make_move_iterator(coarseSphere.vertices.begin()),
                    std::make_move_iterator(coarseSphere.vertices.end()));
    std::vector<std::uint32_t> indices = std::move(fineSphere.indices);
    indices.insert(indices.end(), coarseSphere.indices.begin(), coarseSphere.indices.end());
    std::vector<gpu_scene::PrimitiveRecord> primitives{
        gpu_scene::PrimitiveRecord{
            .indexCount = coarseFirstIndex,
            .firstIndex = 0,
            .vertexOffset = 0,
            .materialIndex = 0,
            .centerRadius = {0.0F, 0.0F, 0.0F, 1.0F},
        },
        gpu_scene::PrimitiveRecord{
            .indexCount = static_cast<std::uint32_t>(coarseSphere.indices.size()),
            .firstIndex = coarseFirstIndex,
            .vertexOffset = coarseVertexOffset,
            .materialIndex = 0,
            .centerRadius = {0.0F, 0.0F, 0.0F, 1.0F},
        },
    };
    std::vector<gpu_scene::PrimitiveLodRange> primitiveLods{
        gpu_scene::PrimitiveLodRange{.firstLod = 0U, .lodCount = 2U}};
    std::vector<gpu_scene::LodSourceRecord> lodSources{
        gpu_scene::LodSourceRecord{.primitiveIndex = 0U, .minimumProjectedDiameter = 96.0F},
        gpu_scene::LodSourceRecord{.primitiveIndex = 1U, .minimumProjectedDiameter = 0.0F},
    };
    if (directionalShadowDemo && !emissiveSphereDemo) {
        const std::uint32_t planeFirstIndex = static_cast<std::uint32_t>(indices.size());
        const std::int32_t planeVertexOffset = static_cast<std::int32_t>(vertices.size());
        constexpr std::array<glm::vec3, 4> planePositions{
            glm::vec3{-4.2F, -3.2F, 0.0F}, glm::vec3{4.2F, -3.2F, 0.0F},
            glm::vec3{4.2F, 3.2F, 0.0F}, glm::vec3{-4.2F, 3.2F, 0.0F}};
        constexpr std::array<glm::vec2, 4> planeUvs{glm::vec2{0.0F, 0.0F}, glm::vec2{1.0F, 0.0F},
                                                    glm::vec2{1.0F, 1.0F}, glm::vec2{0.0F, 1.0F}};
        for (std::size_t vertex = 0; vertex < planePositions.size(); ++vertex) {
            vertices.push_back({.position = planePositions[vertex],
                                .normal = {0.0F, 0.0F, 1.0F},
                                .uv0 = planeUvs[vertex],
                                .tangent = {1.0F, 0.0F, 0.0F, 1.0F},
                                .uv1 = planeUvs[vertex]});
        }
        indices.insert(indices.end(), {0U, 1U, 2U, 0U, 2U, 3U});
        primitives.push_back({.indexCount = 6U,
                              .firstIndex = planeFirstIndex,
                              .vertexOffset = planeVertexOffset,
                              .materialIndex = static_cast<std::uint32_t>(scene.materials.size()),
                              .centerRadius = {0.0F, 0.0F, 0.0F, 5.3F}});
        primitiveLods.push_back({.firstLod = 2U, .lodCount = 1U});
        lodSources.push_back({.primitiveIndex = 2U, .minimumProjectedDiameter = 0.0F});

        const std::uint32_t materialIndex = static_cast<std::uint32_t>(scene.materials.size());
        const std::uint32_t transformIndex = static_cast<std::uint32_t>(scene.transforms.size());
        scene.materials.push_back({
            .baseColorFactor = {0.16F, 0.18F, 0.22F, 1.0F},
            .emissiveAlphaCutoff = {0.0F, 0.0F, 0.0F, 0.5F},
            .metallicRoughnessOcclusionNormal = {0.0F, 0.82F, 1.0F, 1.0F},
        });
        const glm::mat4 planeWorld = glm::translate(glm::mat4(1.0F), glm::vec3(0.0F, 0.0F, -0.55F));
        scene.transforms.push_back({.world = planeWorld,
                                    .normalToWorld = glm::transpose(glm::inverse(planeWorld)),
                                    .previousWorld = planeWorld});
        scene.instances.push_back({.centerRadius = {0.0F, 0.0F, -0.55F, 5.3F},
                                   .meshMaterial = {1U, materialIndex, transformIndex, 0U}});
    }
    gpu_scene::MeshletHierarchy hierarchy =
        gpu_scene::buildMeshletHierarchy(vertices, indices, primitives, primitiveLods, lodSources);
    CpuScenePayload payload{
        .vertices = std::move(vertices),
        .indices = std::move(indices),
        .instances = std::move(scene.instances),
        .materials = std::move(scene.materials),
        .transforms = std::move(scene.transforms),
        .primitives = std::move(primitives),
        .primitiveLods = std::move(primitiveLods),
        .lodSources = std::move(lodSources),
        .label = emissiveSphereDemo
                     ? "procedural HDR emissive sphere"
                     : (directionalShadowDemo ? "procedural directional-shadow demo"
                                              : "procedural PBRT-checked sphere grid"),
    };
    payload.meshletWork = gpu_scene::makeMeshletWorkRecords(hierarchy.primitiveLods, hierarchy.lods,
                                                            hierarchy.meshlets, payload.instances);
    return payload;
}

CpuScenePayload loadGltfScenePayload(const std::filesystem::path& scenePath) {
    scene::Scene source = assets::importGltf(scenePath);
    enum class TextureUsage : std::uint8_t { unused, linearData, srgbColor, normal };
    std::vector<TextureUsage> usages(source.textures().size(), TextureUsage::unused);
    const auto markTexture = [&](const scene::TextureBinding& binding, TextureUsage usage) {
        if (!binding.texture.valid()) {
            return;
        }
        const std::size_t index = binding.texture.index();
        if (index >= usages.size()) {
            throw std::runtime_error("glTF material references an invalid texture");
        }
        if (usages[index] != TextureUsage::unused && usages[index] != usage) {
            throw std::runtime_error(
                "One glTF texture is shared by incompatible color, data, or normal slots");
        }
        usages[index] = usage;
    };
    for (const scene::Material& material : source.materials()) {
        markTexture(material.baseColorTexture, TextureUsage::srgbColor);
        markTexture(material.emissiveTexture, TextureUsage::srgbColor);
        markTexture(material.metallicRoughnessTexture, TextureUsage::linearData);
        markTexture(material.normalTexture, TextureUsage::normal);
        markTexture(material.occlusionTexture, TextureUsage::linearData);
    }

    std::vector<std::uint32_t> textureDescriptors(source.textures().size(),
                                                  gpu_scene::invalidTextureDescriptor);
    std::vector<CpuScenePayload::Texture> textures;
    textures.reserve(source.textures().size());
    for (std::size_t textureIndex = 0; textureIndex < source.textures().size(); ++textureIndex) {
        if (usages[textureIndex] == TextureUsage::unused) {
            continue;
        }
        const scene::Texture& texture = source.textures()[textureIndex];
        const scene::Image& image = source.image(texture.image);
        if (image.encodedData.empty()) {
            throw std::runtime_error("glTF importer did not materialize texture image data");
        }
        if (textures.size() + 1U >= 64U) {
            throw std::runtime_error("glTF scene exceeds the 63 resident material-texture limit");
        }
        textureDescriptors[textureIndex] = static_cast<std::uint32_t>(textures.size() + 1U);
        assets::DecodedImage decoded =
            assets::decodeImage(image.encodedData, usages[textureIndex] == TextureUsage::srgbColor
                                                       ? assets::ImageColorSpace::srgb
                                                       : assets::ImageColorSpace::linear);
        textures.push_back({
            .mips = assets::generateMipChain(std::move(decoded),
                                             usages[textureIndex] == TextureUsage::normal
                                                 ? assets::ImageMipSemantic::tangentSpaceNormal
                                                 : assets::ImageMipSemantic::colorOrData),
            .sampler = texture.sampler.valid() ? source.sampler(texture.sampler) : scene::Sampler{},
            .label = !texture.name.empty() ? texture.name
                                           : (!image.name.empty() ? image.name : image.sourceUri),
        });
    }

    gpu_scene::PackedScene packed = gpu_scene::packScene(source, textureDescriptors);
    CpuScenePayload payload{
        .vertices = std::move(packed.vertices),
        .indices = std::move(packed.indices),
        .instances = std::move(packed.instances),
        .materials = std::move(packed.materials),
        .transforms = std::move(packed.transforms),
        .primitives = std::move(packed.primitives),
        .primitiveLods = std::move(packed.primitiveLods),
        .lodSources = std::move(packed.lodSources),
        .textures = std::move(textures),
        .label = scenePath.string(),
    };
    payload.meshletWork = gpu_scene::makeMeshletWorkRecords(payload.primitiveLods, packed.lods,
                                                            packed.meshlets, payload.instances);
    return payload;
}

glm::vec3 cameraForward(float yaw, float pitch) {
    return glm::normalize(glm::vec3(std::cos(yaw) * std::cos(pitch), std::sin(pitch),
                                    std::sin(yaw) * std::cos(pitch)));
}

VkFormat decodedTextureFormat(const assets::DecodedImage& image) {
    if (image.componentType == assets::ImageComponentType::float32) {
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    return image.colorSpace == assets::ImageColorSpace::srgb ? VK_FORMAT_R8G8B8A8_SRGB
                                                             : VK_FORMAT_R8G8B8A8_UNORM;
}

std::span<const std::byte> decodedTextureBytes(const assets::DecodedImage& image) {
    return image.componentType == assets::ImageComponentType::float32
               ? std::as_bytes(std::span(image.float32))
               : std::as_bytes(std::span(image.unorm8));
}

VkSamplerAddressMode samplerAddressMode(scene::WrapMode mode) {
    switch (mode) {
    case scene::WrapMode::clampToEdge:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case scene::WrapMode::mirroredRepeat:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case scene::WrapMode::repeat:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
    return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

VkSamplerCreateInfo materialSamplerInfo(const scene::Sampler& sampler, std::uint32_t mipLevels) {
    if (mipLevels == 0) {
        throw std::runtime_error("Material sampler requires at least one mip level");
    }
    return {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter =
            sampler.magFilter == scene::Filter::nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
        .minFilter =
            sampler.minFilter == scene::Filter::nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR,
        .mipmapMode = sampler.mipFilter == scene::MipFilter::nearest
                          ? VK_SAMPLER_MIPMAP_MODE_NEAREST
                          : VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = samplerAddressMode(sampler.wrapU),
        .addressModeV = samplerAddressMode(sampler.wrapV),
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .mipLodBias = 0.0F,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0F,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0.0F,
        .maxLod = static_cast<float>(mipLevels - 1U),
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };
}

} // namespace

Renderer::Renderer(platform::Window& window, const vulkan::Instance& instance,
                   const vulkan::Device& device, RendererConfig config)
    : window_(window), instance_(instance), device_(device), config_(std::move(config)),
      mainView_(config_.view),
      swapchain_(createInitialSwapchain(instance, device, window, !config_.capturePath.empty())),
      memoryAllocator_(device), pipelineCache_(device, RENDERER_PIPELINE_CACHE_PATH),
      uploadContext_(device, memoryAllocator_),
      environmentRotationRadians_(config_.environmentRotationRadians) {
    try {
        if (!config_.capturePath.empty() && config_.captureFrame == 0) {
            throw std::runtime_error("Capture frame must be greater than zero");
        }
        if (!std::isfinite(config_.bloomIntensity) || config_.bloomIntensity < 0.0F ||
            config_.bloomIntensity > 16.0F || !std::isfinite(config_.bloomRadius) ||
            config_.bloomRadius < 0.25F || config_.bloomRadius > 16.0F) {
            throw std::runtime_error("Bloom intensity or radius is outside the supported range");
        }
        const vulkan::OptionalCapabilities& optional = device_.optionalCapabilities();
        const ray_tracing::ShadowBackendSelection rayTracingSelection =
            ray_tracing::selectShadowBackend(
                config_.rayTracingRequest,
                {.accelerationStructure = optional.accelerationStructure,
                 .rayQuery = optional.rayQuery,
                 .rayTracingPipeline = optional.rayTracingPipeline},
                rayQueryShadowBackendImplemented);
        if (!rayTracingSelection.accepted) {
            throw std::runtime_error(
                "--ray-tracing required cannot be satisfied: " +
                std::string(ray_tracing::selectionReason(rayTracingSelection.reason)));
        }
        rayTracingShadowBackend_ = rayTracingSelection.backend;
        rayTracingFoundationEnabled_ =
            config_.rayTracingRequest != ray_tracing::Request::disabled &&
            optional.accelerationStructure && (optional.rayQuery || optional.rayTracingPipeline);
        core::log(core::LogLevel::info,
                  "Shadow backend: ", ray_tracing::backendName(rayTracingShadowBackend_), " (",
                  ray_tracing::selectionReason(rayTracingSelection.reason), ')');
        core::log(core::LogLevel::info, "Ray-tracing acceleration-structure foundation: ",
                  rayTracingFoundationEnabled_ ? "enabled" : "skipped");
        depthFormat_ = selectDepthFormat(device_);
        if (!supportsOptimalFormat(device_, hdrFormat_,
                                   VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                       VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
            throw std::runtime_error("RGBA16F is unavailable as a sampled color attachment");
        }
        if (!supportsOptimalFormat(device_, motionFormat_,
                                   VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                       VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
            throw std::runtime_error(
                "RGBA16F is unavailable for temporal motion and normal attachments");
        }
        if (!supportsOptimalFormat(device_, temporalDepthFormat_,
                                   VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                       VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
            throw std::runtime_error("R32F is unavailable for persistent temporal depth");
        }
        if (!supportsOptimalFormat(device_, hiZFormat,
                                   VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                       VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
            throw std::runtime_error("R32F is unavailable for sampled/storage Hi-Z images");
        }
        constexpr VkFormatFeatureFlags iblFeatures =
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if (!supportsOptimalFormat(device_, iblColorFormat, iblFeatures) ||
            !supportsOptimalFormat(device_, brdfLutFormat, iblFeatures)) {
            throw std::runtime_error("Required floating-point IBL texture formats are unavailable");
        }
        createFrameResources();
        createCaptureBuffer();
        createSceneResources();
        createTemporalResources();
        createGraphicsPipelines();
        createRenderGraph();
        updateShadowDescriptor();
        updateToneMappingDescriptor();
        updateHiZDescriptors();
        imageInitialized_.assign(swapchain_.imageCount(), false);
    } catch (...) {
        destroyFrameResources();
        throw;
    }
}

Renderer::~Renderer() {
    if (device_.handle() != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_.handle());
        for (std::size_t index = 0; index < frames_.size(); ++index) {
            collectGpuTimings(index);
        }
        writeGpuTimingReport();
    }
    destroyFrameResources();
}

void Renderer::drawFrame() {
    updateCamera();
    Frame& frame = frames_[frameIndex_];
    if (frame.completionValue != 0) {
        const VkSemaphoreWaitInfo waitInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .semaphoreCount = 1,
            .pSemaphores = &frameTimeline_,
            .pValues = &frame.completionValue,
        };
        vulkan::check(vkWaitSemaphores(device_.handle(), &waitInfo,
                                       std::numeric_limits<std::uint64_t>::max()),
                      "vkWaitSemaphores(frame slot)");
        collectGpuTimings(frameIndex_);
    }

    std::uint32_t imageIndex = 0;
    const VkResult acquireResult = swapchain_.acquire(frame.imageAvailable, imageIndex);
    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        vulkan::check(acquireResult, "vkAcquireNextImageKHR");
    }

    vulkan::check(vkResetCommandPool(device_.handle(), frame.commandPool, 0), "vkResetCommandPool");
    updateFrameUniform(frameIndex_);
    const bool capture = !captureComplete_ && !config_.capturePath.empty() &&
                         renderedFrameCount_ + 1U == config_.captureFrame;
    recordFrame(frame.commandBuffer, imageIndex, frameIndex_, capture);

    std::array<VkSemaphoreSubmitInfo, 2> waitSemaphores{};
    waitSemaphores[0] = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = frame.imageAvailable,
        .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
    };
    std::uint32_t waitCount = 1;
    if (sceneUploadPending_) {
        waitSemaphores[waitCount++] = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = sceneUpload_.semaphore,
            .value = sceneUpload_.value,
            .stageMask =
                VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
                VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        };
    }
    const std::uint64_t completionValue = nextTimelineValue_++;
    const VkSemaphore renderFinished = renderFinishedSemaphores_[imageIndex];
    const std::array signalSemaphores{
        VkSemaphoreSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = renderFinished,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        },
        VkSemaphoreSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = frameTimeline_,
            .value = completionValue,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        },
    };
    const VkCommandBufferSubmitInfo commandInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = frame.commandBuffer,
    };
    const VkSubmitInfo2 submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = waitCount,
        .pWaitSemaphoreInfos = waitSemaphores.data(),
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandInfo,
        .signalSemaphoreInfoCount = static_cast<std::uint32_t>(signalSemaphores.size()),
        .pSignalSemaphoreInfos = signalSemaphores.data(),
    };
    vulkan::check(vkQueueSubmit2(device_.graphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE),
                  "vkQueueSubmit2");
    frame.completionValue = completionValue;
    frame.timingPending = timestampQueryPool_ != VK_NULL_HANDLE;
    imageInitialized_[imageIndex] = true;
    sceneUploadPending_ = false;
    temporalState_.commitFrame();
    temporalImageLayoutsInitialized_ = config_.taaEnabled;

    if (capture) {
        const VkSemaphoreWaitInfo captureWait{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .semaphoreCount = 1,
            .pSemaphores = &frameTimeline_,
            .pValues = &completionValue,
        };
        vulkan::check(vkWaitSemaphores(device_.handle(), &captureWait,
                                       std::numeric_limits<std::uint64_t>::max()),
                      "vkWaitSemaphores(capture)");
        writeCapture();
        captureComplete_ = true;
    }

    const VkSwapchainKHR swapchain = swapchain_.handle();
    const VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &renderFinished,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &imageIndex,
    };
    const VkResult presentResult = vkQueuePresentKHR(device_.presentQueue(), &presentInfo);
    const bool resized = window_.consumeFramebufferResize();
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR ||
        resized || acquireResult == VK_SUBOPTIMAL_KHR) {
        recreateSwapchain();
    } else {
        vulkan::check(presentResult, "vkQueuePresentKHR");
    }

    frameIndex_ = (frameIndex_ + 1) % framesInFlight;
    ++renderedFrameCount_;
}

void Renderer::waitIdle() {
    vulkan::check(vkDeviceWaitIdle(device_.handle()), "vkDeviceWaitIdle(shutdown)");
    for (std::size_t index = 0; index < frames_.size(); ++index) {
        collectGpuTimings(index);
    }
    writeGpuTimingReport();
}

void Renderer::writeGpuTimestamp(VkCommandBuffer commandBuffer, std::size_t frameIndex,
                                 GpuTimingPass pass, bool end) const {
    if (timestampQueryPool_ == VK_NULL_HANDLE) {
        return;
    }
    constexpr std::uint32_t queriesPerFrame = static_cast<std::uint32_t>(gpuTimingPassCount * 2U);
    const std::uint32_t query = static_cast<std::uint32_t>(frameIndex) * queriesPerFrame +
                                static_cast<std::uint32_t>(pass) * 2U + (end ? 1U : 0U);
    vkCmdWriteTimestamp2(commandBuffer,
                         end ? VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT
                             : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                         timestampQueryPool_, query);
}

void Renderer::collectGpuTimings(std::size_t frameIndex) {
    if (timestampQueryPool_ == VK_NULL_HANDLE || frameIndex >= frames_.size() ||
        !frames_[frameIndex].timingPending) {
        return;
    }
    constexpr std::uint32_t queriesPerFrame = static_cast<std::uint32_t>(gpuTimingPassCount * 2U);
    for (std::size_t passIndex = 0; passIndex < gpuTimingTotalMilliseconds_.size(); ++passIndex) {
        std::array<std::uint64_t, 2> timestamps{};
        const std::uint32_t firstQuery = static_cast<std::uint32_t>(frameIndex) * queriesPerFrame +
                                         static_cast<std::uint32_t>(passIndex) * 2U;
        const VkResult result = vkGetQueryPoolResults(
            device_.handle(), timestampQueryPool_, firstQuery, 2, sizeof(timestamps),
            timestamps.data(), sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);
        if (result == VK_NOT_READY) {
            continue;
        }
        vulkan::check(result, "vkGetQueryPoolResults(GPU timestamps)");
        const std::uint64_t mask = timestampValidBits_ >= 64U
                                       ? std::numeric_limits<std::uint64_t>::max()
                                       : (std::uint64_t{1} << timestampValidBits_) - 1U;
        const std::uint64_t ticks = (timestamps[1] - timestamps[0]) & mask;
        gpuTimingTotalMilliseconds_[passIndex] +=
            static_cast<double>(ticks) * static_cast<double>(timestampPeriodNanoseconds_) / 1.0e6;
        ++gpuTimingSampleCounts_[passIndex];
    }
    frames_[frameIndex].timingPending = false;
}

void Renderer::writeGpuTimingReport() {
    if (timingReportWritten_) {
        return;
    }
    timingReportWritten_ = true;
    constexpr std::array names{"directional shadow culling",  "directional shadow map",
                               "bloom extraction/downsample", "bloom horizontal blur",
                               "bloom vertical blur",         "composite/exposure/tone mapping"};
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device_.physicalHandle(), &properties);
    if (timestampQueryPool_ == VK_NULL_HANDLE) {
        if (!config_.timingOutputPath.empty()) {
            std::ofstream output(config_.timingOutputPath);
            output << "GPU timings unavailable: graphics queue has no timestamp support\n";
        }
        return;
    }
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (gpuTimingSampleCounts_[index] == 0U) {
            core::log(core::LogLevel::info, "GPU timing: ", names[index], " skipped");
            continue;
        }
        const double average =
            gpuTimingTotalMilliseconds_[index] / static_cast<double>(gpuTimingSampleCounts_[index]);
        core::log(core::LogLevel::info, "GPU timing: ", names[index], " ", average,
                  " ms average over ", gpuTimingSampleCounts_[index], " frame(s)");
    }
    if (config_.timingOutputPath.empty()) {
        return;
    }
    const std::filesystem::path parent = config_.timingOutputPath.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream output(config_.timingOutputPath);
    if (!output) {
        throw std::runtime_error("Could not write GPU timing report: " +
                                 config_.timingOutputPath.string());
    }
    const VkExtent2D extent = swapchain_.extent();
    output << "GPU: " << properties.deviceName << '\n'
           << "Build: " << RENDERER_BUILD_CONFIGURATION << '\n'
           << "Resolution: " << extent.width << 'x' << extent.height << '\n'
           << "Directional shadows: " << (config_.shadowsEnabled ? "on" : "off") << '\n'
           << "Directional shadow cascades: "
           << (config_.shadowsEnabled ? directionalShadowCascadeCount : 0U) << '\n'
           << "Directional shadow resolution: "
           << (config_.shadowsEnabled ? directionalShadowResolution : 0U) << '\n'
           << "Bloom: " << (config_.bloomEnabled ? "on" : "off") << '\n'
           << "Bloom intensity: " << config_.bloomIntensity << '\n'
           << "Bloom radius: " << config_.bloomRadius << '\n';
    for (std::size_t index = 0; index < names.size(); ++index) {
        output << names[index] << ": ";
        if (gpuTimingSampleCounts_[index] == 0U) {
            output << "skipped\n";
        } else {
            output << gpuTimingTotalMilliseconds_[index] /
                          static_cast<double>(gpuTimingSampleCounts_[index])
                   << " ms (" << gpuTimingSampleCounts_[index] << " samples)\n";
        }
    }
}

void Renderer::createFrameResources() {
    std::uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device_.physicalHandle(), &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueProperties(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device_.physicalHandle(), &queueFamilyCount,
                                             queueProperties.data());
    if (device_.queueFamilies().graphics < queueProperties.size() &&
        queueProperties[device_.queueFamilies().graphics].timestampValidBits > 0U) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device_.physicalHandle(), &properties);
        timestampPeriodNanoseconds_ = properties.limits.timestampPeriod;
        timestampValidBits_ = queueProperties[device_.queueFamilies().graphics].timestampValidBits;
        const VkQueryPoolCreateInfo queryInfo{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = static_cast<std::uint32_t>(framesInFlight * gpuTimingPassCount * 2U),
        };
        vulkan::check(
            vkCreateQueryPool(device_.handle(), &queryInfo, nullptr, &timestampQueryPool_),
            "vkCreateQueryPool(GPU timestamps)");
    } else {
        core::log(core::LogLevel::warning,
                  "Graphics queue does not expose timestamp queries; GPU timings unavailable");
    }

    const VkSemaphoreTypeCreateInfo timelineType{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0,
    };
    const VkSemaphoreCreateInfo timelineInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &timelineType,
    };
    vulkan::check(vkCreateSemaphore(device_.handle(), &timelineInfo, nullptr, &frameTimeline_),
                  "vkCreateSemaphore(timeline)");

    const VkSemaphoreCreateInfo binaryInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    for (Frame& frame : frames_) {
        const VkCommandPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = device_.queueFamilies().graphics,
        };
        vulkan::check(vkCreateCommandPool(device_.handle(), &poolInfo, nullptr, &frame.commandPool),
                      "vkCreateCommandPool");
        const VkCommandBufferAllocateInfo allocateInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = frame.commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        vulkan::check(
            vkAllocateCommandBuffers(device_.handle(), &allocateInfo, &frame.commandBuffer),
            "vkAllocateCommandBuffers");
        vulkan::check(
            vkCreateSemaphore(device_.handle(), &binaryInfo, nullptr, &frame.imageAvailable),
            "vkCreateSemaphore(image available)");
    }
    renderFinishedSemaphores_.reserve(swapchain_.imageCount());
    for (std::uint32_t index = 0; index < swapchain_.imageCount(); ++index) {
        VkSemaphore semaphore = VK_NULL_HANDLE;
        vulkan::check(vkCreateSemaphore(device_.handle(), &binaryInfo, nullptr, &semaphore),
                      "vkCreateSemaphore(render finished)");
        renderFinishedSemaphores_.push_back(semaphore);
    }
}

void Renderer::createCaptureBuffer() {
    captureReadbackBuffer_.reset();
    if (config_.capturePath.empty()) {
        return;
    }
    const VkExtent2D extent = swapchain_.extent();
    const VkDeviceSize byteSize = static_cast<VkDeviceSize>(extent.width) * extent.height * 4U;
    captureReadbackBuffer_ = std::make_unique<vulkan::Buffer>(
        device_, memoryAllocator_,
        vulkan::BufferCreateInfo{
            .size = byteSize,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .requiredMemory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            .preferredMemory =
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            .randomHostAccess = true,
        });
}

void Renderer::createTemporalResources() {
    for (auto& view : historyColorViews_) {
        view.reset();
    }
    for (auto& image : historyColorImages_) {
        image.reset();
    }
    for (auto& view : historyDepthViews_) {
        view.reset();
    }
    for (auto& image : historyDepthImages_) {
        image.reset();
    }
    for (auto& view : historyNormalViews_) {
        view.reset();
    }
    for (auto& image : historyNormalImages_) {
        image.reset();
    }
    temporalImageLayoutsInitialized_ = false;
    temporalState_.reset();
    if (!config_.taaEnabled) {
        return;
    }

    const VkExtent2D extent = swapchain_.extent();
    for (std::size_t index = 0; index < historyColorImages_.size(); ++index) {
        historyColorImages_[index] = std::make_unique<vulkan::Image>(
            device_, memoryAllocator_,
            vulkan::ImageCreateInfo{
                .extent = {extent.width, extent.height, 1},
                .format = hdrFormat_,
                .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            });
        historyColorViews_[index] =
            std::make_unique<vulkan::ImageView>(device_, *historyColorImages_[index]);
        historyDepthImages_[index] = std::make_unique<vulkan::Image>(
            device_, memoryAllocator_,
            vulkan::ImageCreateInfo{
                .extent = {extent.width, extent.height, 1},
                .format = temporalDepthFormat_,
                .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            });
        historyDepthViews_[index] =
            std::make_unique<vulkan::ImageView>(device_, *historyDepthImages_[index]);
        historyNormalImages_[index] = std::make_unique<vulkan::Image>(
            device_, memoryAllocator_,
            vulkan::ImageCreateInfo{
                .extent = {extent.width, extent.height, 1},
                .format = normalFormat_,
                .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            });
        historyNormalViews_[index] =
            std::make_unique<vulkan::ImageView>(device_, *historyNormalImages_[index]);
    }

    const std::uint64_t pixelCount = static_cast<std::uint64_t>(extent.width) * extent.height;
    const std::uint64_t motionBytes = pixelCount * 8U;
    const std::uint64_t normalBytes = pixelCount * 8U;
    const std::uint64_t colorHistoryBytes = pixelCount * 8U * 2U;
    const std::uint64_t depthHistoryBytes = pixelCount * 4U * 2U;
    const std::uint64_t normalHistoryBytes = pixelCount * 8U * 2U;
    constexpr double bytesPerMiB = 1024.0 * 1024.0;
    core::log(core::LogLevel::info, "Temporal buffers at ", extent.width, "x", extent.height,
              ": motion ", static_cast<double>(motionBytes) / bytesPerMiB, " MiB and normals ",
              static_cast<double>(normalBytes) / bytesPerMiB, " MiB transient, HDR history ",
              static_cast<double>(colorHistoryBytes) / bytesPerMiB, " MiB, depth history ",
              static_cast<double>(depthHistoryBytes) / bytesPerMiB, " MiB, normal history ",
              static_cast<double>(normalHistoryBytes) / bytesPerMiB, " MiB; persistent total ",
              static_cast<double>(colorHistoryBytes + depthHistoryBytes + normalHistoryBytes) /
                  bytesPerMiB,
              " MiB");
}

void Renderer::destroyFrameResources() {
    if (timestampQueryPool_ != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device_.handle(), timestampQueryPool_, nullptr);
        timestampQueryPool_ = VK_NULL_HANDLE;
    }
    for (VkSemaphore semaphore : renderFinishedSemaphores_) {
        vkDestroySemaphore(device_.handle(), semaphore, nullptr);
    }
    renderFinishedSemaphores_.clear();
    for (Frame& frame : frames_) {
        if (frame.imageAvailable != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_.handle(), frame.imageAvailable, nullptr);
            frame.imageAvailable = VK_NULL_HANDLE;
        }
        if (frame.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_.handle(), frame.commandPool, nullptr);
            frame.commandPool = VK_NULL_HANDLE;
            frame.commandBuffer = VK_NULL_HANDLE;
        }
    }
    if (frameTimeline_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(device_.handle(), frameTimeline_, nullptr);
        frameTimeline_ = VK_NULL_HANDLE;
    }
}

void Renderer::createSceneResources() {
    CpuScenePayload scene =
        config_.scenePath.empty()
            ? makeProceduralScenePayload(config_.emissiveSphereDemo, config_.directionalShadowDemo,
                                         config_.alphaShadowDemo)
            : loadGltfScenePayload(config_.scenePath);
    const auto& vertices = scene.vertices;
    const auto& indices = scene.indices;
    const auto& instances = scene.instances;
    const auto& materials = scene.materials;
    const auto& transforms = scene.transforms;
    const auto& workItems = scene.meshletWork;
    if (vertices.empty() || indices.empty() || instances.empty() || materials.empty() ||
        transforms.empty() || workItems.empty()) {
        throw std::runtime_error(
            "GPU scene requires geometry and matching instance, material, transform, and "
            "meshlet-work records");
    }
    if (instances.size() > std::numeric_limits<std::uint32_t>::max() ||
        workItems.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("GPU scene instance or meshlet-work count exceeds draw limits");
    }
    sceneWorkItemCount_ = static_cast<std::uint32_t>(workItems.size());
    const std::size_t layerEligibleCount =
        gpu_scene::countLayerVisibleInstances(instances, mainView_.visibilityMask);
    core::log(core::LogLevel::info, "View visibility mask 0x", std::hex, mainView_.visibilityMask,
              std::dec, " admits ", layerEligibleCount, " of ", instances.size(),
              " GPU scene instances before frustum/Hi-Z culling");
    glm::vec3 minimum(std::numeric_limits<float>::infinity());
    glm::vec3 maximum(-std::numeric_limits<float>::infinity());
    for (const gpu_scene::InstanceRecord& instance : instances) {
        const glm::vec3 center(instance.centerRadius);
        const float radius = instance.centerRadius.w;
        if (!std::isfinite(radius) || radius <= 0.0F) {
            throw std::runtime_error("GPU scene has an invalid instance bounding sphere");
        }
        minimum = glm::min(minimum, center - glm::vec3(radius));
        maximum = glm::max(maximum, center + glm::vec3(radius));
    }
    sceneCenter_ = (minimum + maximum) * 0.5F;
    sceneRadius_ = 0.0F;
    for (const gpu_scene::InstanceRecord& instance : instances) {
        sceneRadius_ =
            std::max(sceneRadius_, glm::length(glm::vec3(instance.centerRadius) - sceneCenter_) +
                                       instance.centerRadius.w);
    }
    sceneRadius_ = std::max(sceneRadius_, 1.0e-4F);
    resetCamera();
    static_assert(sizeof(gpu_scene::DrawIndexedIndirectRecord) ==
                  sizeof(VkDrawIndexedIndirectCommand));
    constexpr std::array<std::uint8_t, 16> texturePixels{
        255, 255, 255, 255, 230, 240, 255, 255, 255, 238, 224, 255, 235, 255, 230, 255,
    };

    std::array<std::uint32_t, 2> sharingFamilies{device_.queueFamilies().graphics,
                                                 device_.queueFamilies().transfer};
    const std::size_t sharingFamilyCount = sharingFamilies[0] == sharingFamilies[1] ? 1 : 2;
    const std::span<const std::uint32_t> sharing{sharingFamilies.data(), sharingFamilyCount};
    const VkBufferUsageFlags rayTracingGeometryUsage =
        rayTracingFoundationEnabled_
            ? static_cast<VkBufferUsageFlags>(
                  VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR)
            : VkBufferUsageFlags{0U};
    vertexBuffer_ = std::make_unique<vulkan::Buffer>(
        device_, memoryAllocator_,
        vulkan::BufferCreateInfo{
            .size = vertices.size() * sizeof(gpu_scene::VertexRecord),
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | rayTracingGeometryUsage,
            .requiredMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .queueFamilies = sharing,
        });
    indexBuffer_ = std::make_unique<vulkan::Buffer>(
        device_, memoryAllocator_,
        vulkan::BufferCreateInfo{
            .size = indices.size() * sizeof(std::uint32_t),
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | rayTracingGeometryUsage,
            .requiredMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .queueFamilies = sharing,
        });
    instanceBuffer_ = std::make_unique<vulkan::Buffer>(
        device_, memoryAllocator_,
        vulkan::BufferCreateInfo{
            .size = instances.size() * sizeof(gpu_scene::InstanceRecord),
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .requiredMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .queueFamilies = sharing,
        });
    materialBuffer_ = std::make_unique<vulkan::Buffer>(
        device_, memoryAllocator_,
        vulkan::BufferCreateInfo{
            .size = materials.size() * sizeof(gpu_scene::MaterialRecord),
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .requiredMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .queueFamilies = sharing,
        });
    transformBuffer_ = std::make_unique<vulkan::Buffer>(
        device_, memoryAllocator_,
        vulkan::BufferCreateInfo{
            .size = transforms.size() * sizeof(gpu_scene::TransformRecord),
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .requiredMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .queueFamilies = sharing,
        });
    meshletWorkBuffer_ = std::make_unique<vulkan::Buffer>(
        device_, memoryAllocator_,
        vulkan::BufferCreateInfo{
            .size = workItems.size() * sizeof(gpu_scene::MeshletWorkRecord),
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .requiredMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .queueFamilies = sharing,
        });
    const auto createVisibleInstanceBuffer = [&]() {
        return std::make_unique<vulkan::Buffer>(
            device_, memoryAllocator_,
            vulkan::BufferCreateInfo{
                .size = workItems.size() * sizeof(std::uint32_t),
                .usage =
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                .requiredMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                .queueFamilies = sharing,
            });
    };
    const auto createIndirectDrawBuffer = [&]() {
        return std::make_unique<vulkan::Buffer>(
            device_, memoryAllocator_,
            vulkan::BufferCreateInfo{
                .size = workItems.size() * sizeof(gpu_scene::DrawIndexedIndirectRecord),
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                .requiredMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                .queueFamilies = sharing,
            });
    };
    const auto createDrawCountBuffer = [&]() {
        return std::make_unique<vulkan::Buffer>(
            device_, memoryAllocator_,
            vulkan::BufferCreateInfo{
                .size = sizeof(std::uint32_t),
                .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                .requiredMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                .queueFamilies = sharing,
            });
    };
    frustumVisibleInstanceBuffer_ = createVisibleInstanceBuffer();
    frustumIndirectDrawBuffer_ = createIndirectDrawBuffer();
    frustumDrawCountBuffer_ = createDrawCountBuffer();
    visibleInstanceBuffer_ = createVisibleInstanceBuffer();
    indirectDrawBuffer_ = createIndirectDrawBuffer();
    drawCountBuffer_ = createDrawCountBuffer();
    for (std::size_t cascadeIndex = 0; cascadeIndex < directionalShadowCascadeCount;
         ++cascadeIndex) {
        shadowVisibleInstanceBuffers_[cascadeIndex] = createVisibleInstanceBuffer();
        shadowIndirectDrawBuffers_[cascadeIndex] = createIndirectDrawBuffer();
        shadowDrawCountBuffers_[cascadeIndex] = createDrawCountBuffer();
    }
    for (auto& uniformBuffer : uniformBuffers_) {
        uniformBuffer = std::make_unique<vulkan::Buffer>(
            device_, memoryAllocator_,
            vulkan::BufferCreateInfo{
                .size = sizeof(FrameUniform),
                .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                .requiredMemory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                .preferredMemory =
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            });
    }
    bloomControlsBuffer_ = std::make_unique<vulkan::Buffer>(
        device_, memoryAllocator_,
        vulkan::BufferCreateInfo{
            .size = sizeof(BloomControls),
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .requiredMemory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            .preferredMemory =
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        });
    const BloomControls bloomControls{{config_.bloomIntensity,
                                       currentBloomFilterScale(config_.bloomRadius),
                                       post::bloomThreshold, post::bloomKnee}};
    bloomControlsBuffer_->write(std::as_bytes(std::span(&bloomControls, 1)));
    const VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .mipLodBias = 0.0F,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0F,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0.0F,
        .maxLod = 0.0F,
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };
    textureImages_.clear();
    textureViews_.clear();
    textureSamplers_.clear();
    textureImages_.reserve(scene.textures.size() + 1U);
    textureViews_.reserve(scene.textures.size() + 1U);
    textureSamplers_.reserve(scene.textures.size() + 1U);
    textureImages_.push_back(std::make_unique<vulkan::Image>(
        device_, memoryAllocator_,
        vulkan::ImageCreateInfo{
            .extent = {2, 2, 1},
            .format = VK_FORMAT_R8G8B8A8_SRGB,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .queueFamilies = sharing,
        }));
    textureViews_.push_back(std::make_unique<vulkan::ImageView>(device_, *textureImages_.back()));
    textureSamplers_.push_back(std::make_unique<vulkan::Sampler>(device_, samplerInfo));
    for (const CpuScenePayload::Texture& texture : scene.textures) {
        if (!texture.mips.valid()) {
            throw std::runtime_error("Imported glTF texture has an invalid mip chain");
        }
        const assets::DecodedImage& baseLevel = texture.mips.levels.front();
        const VkFormat format = decodedTextureFormat(baseLevel);
        if (!supportsOptimalFormat(device_, format,
                                   VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                                       VK_FORMAT_FEATURE_TRANSFER_DST_BIT)) {
            throw std::runtime_error("GPU cannot sample an imported glTF texture format");
        }
        textureImages_.push_back(std::make_unique<vulkan::Image>(
            device_, memoryAllocator_,
            vulkan::ImageCreateInfo{
                .extent = {baseLevel.width, baseLevel.height, 1},
                .format = format,
                .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                .mipLevels = static_cast<std::uint32_t>(texture.mips.levels.size()),
                .queueFamilies = sharing,
            }));
        textureViews_.push_back(
            std::make_unique<vulkan::ImageView>(device_, *textureImages_.back()));
        textureSamplers_.push_back(std::make_unique<vulkan::Sampler>(
            device_, materialSamplerInfo(texture.sampler,
                                         static_cast<std::uint32_t>(texture.mips.levels.size()))));
    }

    const auto bakeStart = std::chrono::steady_clock::now();
    lighting::FloatImage environment =
        config_.environmentPath.empty()
            ? lighting::makeProceduralStudioEnvironment()
            : lighting::toLinearFloatImage(
                  assets::loadDecodedImage(config_.environmentPath, assets::ImageColorSpace::srgb));
    lighting::IblAssets ibl = lighting::bakeImageBasedLighting(std::move(environment));
    const auto bakeMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                                      std::chrono::steady_clock::now() - bakeStart)
                                      .count();
    const std::uint32_t specularMipCount =
        static_cast<std::uint32_t>(ibl.prefilteredSpecular.levels.size());
    environmentImage_ = std::make_unique<vulkan::Image>(
        device_, memoryAllocator_,
        vulkan::ImageCreateInfo{
            .extent = {ibl.environment.width, ibl.environment.height, 1},
            .format = iblColorFormat,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .queueFamilies = sharing,
        });
    environmentView_ = std::make_unique<vulkan::ImageView>(device_, *environmentImage_);
    diffuseIrradianceImage_ = std::make_unique<vulkan::Image>(
        device_, memoryAllocator_,
        vulkan::ImageCreateInfo{
            .extent = {ibl.diffuseIrradiance.width, ibl.diffuseIrradiance.height, 1},
            .format = iblColorFormat,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .queueFamilies = sharing,
        });
    diffuseIrradianceView_ = std::make_unique<vulkan::ImageView>(device_, *diffuseIrradianceImage_);
    prefilteredSpecularImage_ = std::make_unique<vulkan::Image>(
        device_, memoryAllocator_,
        vulkan::ImageCreateInfo{
            .extent = {ibl.prefilteredSpecular.levels.front().width,
                       ibl.prefilteredSpecular.levels.front().height, 1},
            .format = iblColorFormat,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .mipLevels = specularMipCount,
            .queueFamilies = sharing,
        });
    prefilteredSpecularView_ =
        std::make_unique<vulkan::ImageView>(device_, *prefilteredSpecularImage_);
    brdfLutImage_ = std::make_unique<vulkan::Image>(
        device_, memoryAllocator_,
        vulkan::ImageCreateInfo{
            .extent = {ibl.brdfLut.width, ibl.brdfLut.height, 1},
            .format = brdfLutFormat,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .queueFamilies = sharing,
        });
    brdfLutView_ = std::make_unique<vulkan::ImageView>(device_, *brdfLutImage_);

    VkSamplerCreateInfo iblSamplerInfo = samplerInfo;
    iblSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    iblSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    iblSamplerInfo.maxLod = static_cast<float>(specularMipCount - 1);
    iblSampler_ = std::make_unique<vulkan::Sampler>(device_, iblSamplerInfo);
    VkSamplerCreateInfo brdfSamplerInfo = iblSamplerInfo;
    brdfSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    brdfSamplerInfo.maxLod = 0.0F;
    brdfLutSampler_ = std::make_unique<vulkan::Sampler>(device_, brdfSamplerInfo);
    VkSamplerCreateInfo hiZSamplerInfo = brdfSamplerInfo;
    hiZSamplerInfo.magFilter = VK_FILTER_NEAREST;
    hiZSamplerInfo.minFilter = VK_FILTER_NEAREST;
    hiZSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    hiZSampler_ = std::make_unique<vulkan::Sampler>(device_, hiZSamplerInfo);
    VkSamplerCreateInfo shadowSamplerInfo = brdfSamplerInfo;
    shadowSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadowSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadowSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shadowSamplerInfo.compareEnable = VK_TRUE;
    shadowSamplerInfo.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    shadowSamplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    shadowSampler_ = std::make_unique<vulkan::Sampler>(device_, shadowSamplerInfo);

    const std::array bufferUploads{
        vulkan::BufferUpload{
            .destination = vertexBuffer_.get(),
            .data = std::as_bytes(std::span(vertices)),
        },
        vulkan::BufferUpload{
            .destination = indexBuffer_.get(),
            .data = std::as_bytes(std::span(indices)),
        },
        vulkan::BufferUpload{
            .destination = instanceBuffer_.get(),
            .data = std::as_bytes(std::span(instances)),
        },
        vulkan::BufferUpload{
            .destination = materialBuffer_.get(),
            .data = std::as_bytes(std::span(materials)),
        },
        vulkan::BufferUpload{
            .destination = transformBuffer_.get(),
            .data = std::as_bytes(std::span(transforms)),
        },
        vulkan::BufferUpload{
            .destination = meshletWorkBuffer_.get(),
            .data = std::as_bytes(std::span(workItems)),
        },
    };
    std::vector<vulkan::ImageUpload> imageUploads;
    std::size_t materialMipCount = 0;
    for (const CpuScenePayload::Texture& texture : scene.textures) {
        materialMipCount += texture.mips.levels.size();
    }
    imageUploads.reserve(static_cast<std::size_t>(specularMipCount) + materialMipCount + 4U);
    imageUploads.push_back({.destination = textureImages_.front().get(),
                            .data = std::as_bytes(std::span(texturePixels))});
    for (std::size_t textureIndex = 0; textureIndex < scene.textures.size(); ++textureIndex) {
        const CpuScenePayload::Texture& texture = scene.textures[textureIndex];
        for (std::uint32_t mipLevel = 0; mipLevel < texture.mips.levels.size(); ++mipLevel) {
            imageUploads.push_back({
                .destination = textureImages_[textureIndex + 1U].get(),
                .data = decodedTextureBytes(texture.mips.levels[mipLevel]),
                .mipLevel = mipLevel,
            });
        }
    }
    imageUploads.push_back({.destination = environmentImage_.get(),
                            .data = std::as_bytes(std::span(ibl.environment.texels))});
    imageUploads.push_back({.destination = diffuseIrradianceImage_.get(),
                            .data = std::as_bytes(std::span(ibl.diffuseIrradiance.texels))});
    for (std::uint32_t level = 0; level < specularMipCount; ++level) {
        const lighting::FloatImage& mip = ibl.prefilteredSpecular.levels[level];
        imageUploads.push_back({.destination = prefilteredSpecularImage_.get(),
                                .data = std::as_bytes(std::span(mip.texels)),
                                .mipLevel = level});
    }
    imageUploads.push_back(
        {.destination = brdfLutImage_.get(), .data = std::as_bytes(std::span(ibl.brdfLut.texels))});
    sceneUpload_ = uploadContext_.upload(bufferUploads, imageUploads);
    sceneUploadPending_ = true;

    if (config_.rayTracingRequest != ray_tracing::Request::disabled) {
        rayTracingBuildPlan_ = ray_tracing::makeSceneBuildPlan(
            vertices.size(), indices, scene.primitives, scene.primitiveLods, scene.lodSources,
            instances, materials, transforms.size());
        core::log(core::LogLevel::info, "RT scene plan: ", rayTracingBuildPlan_.bottomLevels.size(),
                  " reusable mesh structure(s), ", rayTracingBuildPlan_.topLevelInstances.size(),
                  " object instance(s), ", rayTracingBuildPlan_.uniqueTriangleCount,
                  " unique triangles, ", rayTracingBuildPlan_.instancedTriangleCount,
                  " triangles after instancing");
        if (rayTracingFoundationEnabled_) {
            createRayTracingResources(transforms);
        } else {
            core::log(core::LogLevel::info,
                      "RT GPU structures skipped; persistent RT memory: 0 MiB");
        }
    }

    const std::filesystem::path shaderDirectory(RENDERER_SHADER_DIR);
    vertexShader_ =
        std::make_unique<vulkan::ShaderModule>(device_, shaderDirectory / "mesh.vert.spv");
    fragmentShader_ =
        std::make_unique<vulkan::ShaderModule>(device_, shaderDirectory / "mesh.frag.spv");
    shadowVertexShader_ =
        std::make_unique<vulkan::ShaderModule>(device_, shaderDirectory / "shadow.vert.spv");
    shadowFragmentShader_ =
        std::make_unique<vulkan::ShaderModule>(device_, shaderDirectory / "shadow.frag.spv");
    shadowCullShader_ =
        std::make_unique<vulkan::ShaderModule>(device_, shaderDirectory / "shadow_cull.comp.spv");
    frustumCullShader_ =
        std::make_unique<vulkan::ShaderModule>(device_, shaderDirectory / "frustum_cull.comp.spv");
    occlusionCullShader_ = std::make_unique<vulkan::ShaderModule>(
        device_, shaderDirectory / "occlusion_cull.comp.spv");
    hiZCopyShader_ =
        std::make_unique<vulkan::ShaderModule>(device_, shaderDirectory / "hiz_copy.comp.spv");
    hiZReduceShader_ =
        std::make_unique<vulkan::ShaderModule>(device_, shaderDirectory / "hiz_reduce.comp.spv");
    skyFragmentShader_ =
        std::make_unique<vulkan::ShaderModule>(device_, shaderDirectory / "sky.frag.spv");
    postTestFragmentShader_ =
        std::make_unique<vulkan::ShaderModule>(device_, shaderDirectory / "post_test.frag.spv");
    bloomDownsampleShader_ = std::make_unique<vulkan::ShaderModule>(
        device_, shaderDirectory / "bloom_downsample.frag.spv");
    bloomBlurHorizontalShader_ =
        std::make_unique<vulkan::ShaderModule>(device_, shaderDirectory / "bloom_blur_h.frag.spv");
    bloomBlurVerticalShader_ =
        std::make_unique<vulkan::ShaderModule>(device_, shaderDirectory / "bloom_blur_v.frag.spv");
    toneMappingVertexShader_ =
        std::make_unique<vulkan::ShaderModule>(device_, shaderDirectory / "tonemap.vert.spv");
    toneMappingFragmentShader_ =
        std::make_unique<vulkan::ShaderModule>(device_, shaderDirectory / "tonemap.frag.spv");
    taaFragmentShader_ =
        std::make_unique<vulkan::ShaderModule>(device_, shaderDirectory / "taa.frag.spv");
    if (vertexShader_->reflection().stage != VK_SHADER_STAGE_VERTEX_BIT ||
        fragmentShader_->reflection().stage != VK_SHADER_STAGE_FRAGMENT_BIT ||
        skyFragmentShader_->reflection().stage != VK_SHADER_STAGE_FRAGMENT_BIT ||
        postTestFragmentShader_->reflection().stage != VK_SHADER_STAGE_FRAGMENT_BIT ||
        postTestFragmentShader_->reflection().descriptorBindings.size() != 1U ||
        postTestFragmentShader_->reflection().descriptorBindings.front().set != 0U ||
        postTestFragmentShader_->reflection().descriptorBindings.front().binding != 0U ||
        postTestFragmentShader_->reflection().descriptorBindings.front().type !=
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
        vertexShader_->reflection().inputLocations != std::vector<std::uint32_t>{0, 1, 2, 3, 4}) {
        throw std::runtime_error("Shader reflection does not match the PBR mesh pipeline");
    }
    const std::array<const vulkan::ShaderReflection*, 3> reflections{
        &vertexShader_->reflection(), &fragmentShader_->reflection(),
        &skyFragmentShader_->reflection()};
    const auto bindings = vulkan::mergeDescriptorBindings(reflections);
    const auto hasBinding = [&bindings](std::uint32_t set, std::uint32_t binding,
                                        VkDescriptorType type, std::uint32_t count) {
        return std::ranges::any_of(bindings, [&](const vulkan::DescriptorBinding& candidate) {
            return candidate.set == set && candidate.binding == binding && candidate.type == type &&
                   candidate.count == count;
        });
    };
    bool validPbrBindings = bindings.size() == 11 &&
                            hasBinding(0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1) &&
                            hasBinding(0, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1) &&
                            hasBinding(0, 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1) &&
                            hasBinding(0, 4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1) &&
                            hasBinding(0, 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1) &&
                            hasBinding(0, 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) &&
                            hasBinding(0, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) &&
                            hasBinding(0, 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) &&
                            hasBinding(0, 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) &&
                            hasBinding(0, 10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       directionalShadowCascadeCount) &&
                            hasBinding(1, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0);
    if (!validPbrBindings) {
        throw std::runtime_error("Shader reflection found an unexpected descriptor interface");
    }
    descriptorSetLayout_ = std::make_unique<vulkan::DescriptorSetLayout>(device_, 0, bindings);
    descriptorPool_ = std::make_unique<vulkan::DescriptorPool>(
        device_, *descriptorSetLayout_, static_cast<std::uint32_t>(framesInFlight));
    bindlessDescriptorSetLayout_ =
        std::make_unique<vulkan::DescriptorSetLayout>(device_, 1, bindings, maxBindlessTextures);
    bindlessDescriptorPool_ =
        std::make_unique<vulkan::DescriptorPool>(device_, *bindlessDescriptorSetLayout_, 1);

    const std::array<const vulkan::ShaderReflection*, 2> shadowReflections{
        &shadowVertexShader_->reflection(), &shadowFragmentShader_->reflection()};
    const auto shadowBindings = vulkan::mergeDescriptorBindings(shadowReflections);
    const auto hasShadowBinding = [&shadowBindings](std::uint32_t set, std::uint32_t binding,
                                                    VkDescriptorType type, std::uint32_t count) {
        return std::ranges::any_of(shadowBindings, [&](const vulkan::DescriptorBinding& candidate) {
            return candidate.set == set && candidate.binding == binding && candidate.type == type &&
                   candidate.count == count;
        });
    };
    if (shadowVertexShader_->reflection().stage != VK_SHADER_STAGE_VERTEX_BIT ||
        shadowVertexShader_->reflection().inputLocations != std::vector<std::uint32_t>{0, 2, 4} ||
        shadowFragmentShader_->reflection().stage != VK_SHADER_STAGE_FRAGMENT_BIT ||
        shadowFragmentShader_->reflection().inputLocations != std::vector<std::uint32_t>{0, 1, 2} ||
        shadowBindings.size() != 6 ||
        !hasShadowBinding(0, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1) ||
        !hasShadowBinding(0, 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) ||
        !hasShadowBinding(0, 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) ||
        !hasShadowBinding(0, 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) ||
        !hasShadowBinding(0, 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1) ||
        !hasShadowBinding(1, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0)) {
        throw std::runtime_error("Shader reflection does not match directional shadow drawing");
    }
    shadowDescriptorSetLayout_ =
        std::make_unique<vulkan::DescriptorSetLayout>(device_, 0, shadowBindings);
    shadowDescriptorPool_ = std::make_unique<vulkan::DescriptorPool>(
        device_, *shadowDescriptorSetLayout_,
        static_cast<std::uint32_t>(framesInFlight * directionalShadowCascadeCount));

    const std::array<const vulkan::ShaderReflection*, 1> cullingReflections{
        &frustumCullShader_->reflection()};
    const auto cullingBindings = vulkan::mergeDescriptorBindings(cullingReflections);
    bool validCullingBindings =
        frustumCullShader_->reflection().stage == VK_SHADER_STAGE_COMPUTE_BIT &&
        cullingBindings.size() == 6 && cullingBindings[0].set == 0 &&
        cullingBindings[0].binding == 0 &&
        cullingBindings[0].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
        cullingBindings[0].count == 1;
    for (std::size_t bindingIndex = 1; bindingIndex < cullingBindings.size(); ++bindingIndex) {
        validCullingBindings =
            validCullingBindings && cullingBindings[bindingIndex].binding == bindingIndex &&
            cullingBindings[bindingIndex].set == 0 &&
            cullingBindings[bindingIndex].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER &&
            cullingBindings[bindingIndex].count == 1;
    }
    bool validShadowCullingBindings =
        shadowCullShader_->reflection().stage == VK_SHADER_STAGE_COMPUTE_BIT &&
        shadowCullShader_->reflection().descriptorBindings.size() == cullingBindings.size();
    for (std::size_t bindingIndex = 0;
         validShadowCullingBindings && bindingIndex < cullingBindings.size(); ++bindingIndex) {
        const vulkan::DescriptorBinding& candidate =
            shadowCullShader_->reflection().descriptorBindings[bindingIndex];
        validShadowCullingBindings = candidate.set == cullingBindings[bindingIndex].set &&
                                     candidate.binding == cullingBindings[bindingIndex].binding &&
                                     candidate.type == cullingBindings[bindingIndex].type &&
                                     candidate.count == cullingBindings[bindingIndex].count;
    }
    if (!validCullingBindings || !validShadowCullingBindings) {
        throw std::runtime_error("Shader reflection does not match GPU camera/shadow culling");
    }
    cullingDescriptorSetLayout_ =
        std::make_unique<vulkan::DescriptorSetLayout>(device_, 0, cullingBindings);
    cullingDescriptorPool_ = std::make_unique<vulkan::DescriptorPool>(
        device_, *cullingDescriptorSetLayout_,
        static_cast<std::uint32_t>(framesInFlight * (1U + directionalShadowCascadeCount)));

    const std::array<const vulkan::ShaderReflection*, 1> occlusionReflections{
        &occlusionCullShader_->reflection()};
    const auto occlusionBindings = vulkan::mergeDescriptorBindings(occlusionReflections);
    const std::array expectedOcclusionTypes{
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    bool validOcclusionBindings =
        occlusionCullShader_->reflection().stage == VK_SHADER_STAGE_COMPUTE_BIT &&
        occlusionBindings.size() == expectedOcclusionTypes.size();
    for (std::size_t bindingIndex = 0;
         validOcclusionBindings && bindingIndex < occlusionBindings.size(); ++bindingIndex) {
        const std::uint32_t expectedCount = bindingIndex == 4 ? maxHiZLevels : 1U;
        validOcclusionBindings =
            occlusionBindings[bindingIndex].set == 0 &&
            occlusionBindings[bindingIndex].binding == bindingIndex &&
            occlusionBindings[bindingIndex].type == expectedOcclusionTypes[bindingIndex] &&
            occlusionBindings[bindingIndex].count == expectedCount;
    }
    if (!validOcclusionBindings) {
        throw std::runtime_error("Shader reflection does not match GPU occlusion culling");
    }
    occlusionDescriptorSetLayout_ =
        std::make_unique<vulkan::DescriptorSetLayout>(device_, 0, occlusionBindings);
    occlusionDescriptorPool_ = std::make_unique<vulkan::DescriptorPool>(
        device_, *occlusionDescriptorSetLayout_, static_cast<std::uint32_t>(framesInFlight));

    const std::array<const vulkan::ShaderReflection*, 2> hiZReflections{
        &hiZCopyShader_->reflection(), &hiZReduceShader_->reflection()};
    const auto hiZBindings = vulkan::mergeDescriptorBindings(hiZReflections);
    if (hiZCopyShader_->reflection().stage != VK_SHADER_STAGE_COMPUTE_BIT ||
        hiZReduceShader_->reflection().stage != VK_SHADER_STAGE_COMPUTE_BIT ||
        hiZBindings.size() != 2 || hiZBindings[0].set != 0 || hiZBindings[0].binding != 0 ||
        hiZBindings[0].type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
        hiZBindings[0].count != 1 || hiZBindings[1].set != 0 || hiZBindings[1].binding != 1 ||
        hiZBindings[1].type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || hiZBindings[1].count != 1) {
        throw std::runtime_error("Shader reflection does not match Hi-Z construction");
    }
    hiZDescriptorSetLayout_ =
        std::make_unique<vulkan::DescriptorSetLayout>(device_, 0, hiZBindings);
    hiZDescriptorPool_ =
        std::make_unique<vulkan::DescriptorPool>(device_, *hiZDescriptorSetLayout_, maxHiZLevels);

    const std::array<const vulkan::ShaderReflection*, 3> bloomReflections{
        &bloomDownsampleShader_->reflection(), &bloomBlurHorizontalShader_->reflection(),
        &bloomBlurVerticalShader_->reflection()};
    const auto bloomBindings = vulkan::mergeDescriptorBindings(bloomReflections);
    if (bloomDownsampleShader_->reflection().stage != VK_SHADER_STAGE_FRAGMENT_BIT ||
        bloomBlurHorizontalShader_->reflection().stage != VK_SHADER_STAGE_FRAGMENT_BIT ||
        bloomBlurVerticalShader_->reflection().stage != VK_SHADER_STAGE_FRAGMENT_BIT ||
        bloomBindings.size() != 2 || bloomBindings[0].set != 0 || bloomBindings[0].binding != 0 ||
        bloomBindings[0].type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
        bloomBindings[0].count != 1 || bloomBindings[1].set != 0 || bloomBindings[1].binding != 1 ||
        bloomBindings[1].type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || bloomBindings[1].count != 1) {
        throw std::runtime_error("Shader reflection does not match the bloom pipelines");
    }
    bloomDescriptorSetLayout_ =
        std::make_unique<vulkan::DescriptorSetLayout>(device_, 0, bloomBindings);
    bloomDescriptorPool_ =
        std::make_unique<vulkan::DescriptorPool>(device_, *bloomDescriptorSetLayout_, 4);

    const std::array<const vulkan::ShaderReflection*, 2> taaReflections{
        &toneMappingVertexShader_->reflection(), &taaFragmentShader_->reflection()};
    const auto taaBindings = vulkan::mergeDescriptorBindings(taaReflections);
    bool validTaaBindings =
        taaFragmentShader_->reflection().stage == VK_SHADER_STAGE_FRAGMENT_BIT &&
        taaBindings.size() == 8;
    for (std::size_t bindingIndex = 0; validTaaBindings && bindingIndex < taaBindings.size();
         ++bindingIndex) {
        const VkDescriptorType expectedType = bindingIndex == 7
                                                  ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                  : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        validTaaBindings = taaBindings[bindingIndex].set == 0 &&
                           taaBindings[bindingIndex].binding == bindingIndex &&
                           taaBindings[bindingIndex].type == expectedType &&
                           taaBindings[bindingIndex].count == 1;
    }
    if (!validTaaBindings) {
        throw std::runtime_error("Shader reflection does not match the temporal resolve");
    }
    taaDescriptorSetLayout_ =
        std::make_unique<vulkan::DescriptorSetLayout>(device_, 0, taaBindings);
    taaDescriptorPool_ = std::make_unique<vulkan::DescriptorPool>(
        device_, *taaDescriptorSetLayout_, static_cast<std::uint32_t>(framesInFlight));

    const std::array<const vulkan::ShaderReflection*, 2> toneMappingReflections{
        &toneMappingVertexShader_->reflection(), &toneMappingFragmentShader_->reflection()};
    const auto toneMappingBindings = vulkan::mergeDescriptorBindings(toneMappingReflections);
    if (toneMappingVertexShader_->reflection().stage != VK_SHADER_STAGE_VERTEX_BIT ||
        toneMappingFragmentShader_->reflection().stage != VK_SHADER_STAGE_FRAGMENT_BIT ||
        !toneMappingVertexShader_->reflection().inputLocations.empty() ||
        toneMappingBindings.size() != 4 || toneMappingBindings[0].set != 0 ||
        toneMappingBindings[0].binding != 0 ||
        toneMappingBindings[0].type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
        toneMappingBindings[0].count != 1 || toneMappingBindings[1].set != 0 ||
        toneMappingBindings[1].binding != 1 ||
        toneMappingBindings[1].type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
        toneMappingBindings[1].count != 1 || toneMappingBindings[2].set != 0 ||
        toneMappingBindings[2].binding != 2 ||
        toneMappingBindings[2].type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
        toneMappingBindings[2].count != 1 || toneMappingBindings[3].set != 0 ||
        toneMappingBindings[3].binding != 3 ||
        toneMappingBindings[3].type != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
        toneMappingBindings[3].count != 1) {
        throw std::runtime_error("Shader reflection does not match the tone-mapping pipeline");
    }
    toneMappingDescriptorSetLayout_ =
        std::make_unique<vulkan::DescriptorSetLayout>(device_, 0, toneMappingBindings);
    toneMappingDescriptorPool_ = std::make_unique<vulkan::DescriptorPool>(
        device_, *toneMappingDescriptorSetLayout_, static_cast<std::uint32_t>(framesInFlight));

    for (std::size_t index = 0; index < framesInFlight; ++index) {
        const VkDescriptorBufferInfo bufferInfo{
            .buffer = uniformBuffers_[index]->handle(),
            .offset = 0,
            .range = sizeof(FrameUniform),
        };
        const VkDescriptorBufferInfo instanceInfo{
            .buffer = instanceBuffer_->handle(),
            .offset = 0,
            .range = instanceBuffer_->size(),
        };
        const VkDescriptorBufferInfo meshletWorkInfo{
            .buffer = meshletWorkBuffer_->handle(),
            .offset = 0,
            .range = meshletWorkBuffer_->size(),
        };
        const VkDescriptorBufferInfo visibleInfo{
            .buffer = visibleInstanceBuffer_->handle(),
            .offset = 0,
            .range = visibleInstanceBuffer_->size(),
        };
        const VkDescriptorBufferInfo indirectInfo{
            .buffer = indirectDrawBuffer_->handle(),
            .offset = 0,
            .range = indirectDrawBuffer_->size(),
        };
        const VkDescriptorBufferInfo frustumVisibleInfo{
            .buffer = frustumVisibleInstanceBuffer_->handle(),
            .offset = 0,
            .range = frustumVisibleInstanceBuffer_->size(),
        };
        const VkDescriptorBufferInfo frustumIndirectInfo{
            .buffer = frustumIndirectDrawBuffer_->handle(),
            .offset = 0,
            .range = frustumIndirectDrawBuffer_->size(),
        };
        const VkDescriptorBufferInfo frustumDrawCountInfo{
            .buffer = frustumDrawCountBuffer_->handle(),
            .offset = 0,
            .range = sizeof(std::uint32_t),
        };
        const VkDescriptorBufferInfo drawCountInfo{
            .buffer = drawCountBuffer_->handle(),
            .offset = 0,
            .range = sizeof(std::uint32_t),
        };
        const VkDescriptorBufferInfo materialInfo{
            .buffer = materialBuffer_->handle(),
            .offset = 0,
            .range = materialBuffer_->size(),
        };
        const VkDescriptorBufferInfo transformInfo{
            .buffer = transformBuffer_->handle(),
            .offset = 0,
            .range = transformBuffer_->size(),
        };
        const VkDescriptorImageInfo irradianceInfo{
            .sampler = iblSampler_->handle(),
            .imageView = diffuseIrradianceView_->handle(),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkDescriptorImageInfo specularInfo{
            .sampler = iblSampler_->handle(),
            .imageView = prefilteredSpecularView_->handle(),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkDescriptorImageInfo brdfInfo{
            .sampler = brdfLutSampler_->handle(),
            .imageView = brdfLutView_->handle(),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkDescriptorImageInfo environmentInfo{
            .sampler = iblSampler_->handle(),
            .imageView = environmentView_->handle(),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const std::array writes{
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorPool_->set(index),
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &bufferInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorPool_->set(index),
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &irradianceInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorPool_->set(index),
                .dstBinding = 3,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &specularInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorPool_->set(index),
                .dstBinding = 4,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &brdfInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorPool_->set(index),
                .dstBinding = 5,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &environmentInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorPool_->set(index),
                .dstBinding = 6,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &instanceInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorPool_->set(index),
                .dstBinding = 7,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &visibleInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorPool_->set(index),
                .dstBinding = 8,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &materialInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorPool_->set(index),
                .dstBinding = 9,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &transformInfo,
            },
        };
        vkUpdateDescriptorSets(device_.handle(), static_cast<std::uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
        const std::array cullingWrites{
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = cullingDescriptorPool_->set(index),
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &bufferInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = cullingDescriptorPool_->set(index),
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &instanceInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = cullingDescriptorPool_->set(index),
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &meshletWorkInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = cullingDescriptorPool_->set(index),
                .dstBinding = 3,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &frustumVisibleInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = cullingDescriptorPool_->set(index),
                .dstBinding = 4,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &frustumIndirectInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = cullingDescriptorPool_->set(index),
                .dstBinding = 5,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &frustumDrawCountInfo,
            },
        };
        vkUpdateDescriptorSets(device_.handle(), static_cast<std::uint32_t>(cullingWrites.size()),
                               cullingWrites.data(), 0, nullptr);
        for (std::size_t cascadeIndex = 0; cascadeIndex < directionalShadowCascadeCount;
             ++cascadeIndex) {
            const std::size_t shadowSetIndex = cascadeIndex * framesInFlight + index;
            const std::size_t shadowCullingSetIndex = framesInFlight + shadowSetIndex;
            const VkDescriptorBufferInfo shadowVisibleInfo{
                .buffer = shadowVisibleInstanceBuffers_[cascadeIndex]->handle(),
                .offset = 0,
                .range = shadowVisibleInstanceBuffers_[cascadeIndex]->size(),
            };
            const VkDescriptorBufferInfo shadowIndirectInfo{
                .buffer = shadowIndirectDrawBuffers_[cascadeIndex]->handle(),
                .offset = 0,
                .range = shadowIndirectDrawBuffers_[cascadeIndex]->size(),
            };
            const VkDescriptorBufferInfo shadowDrawCountInfo{
                .buffer = shadowDrawCountBuffers_[cascadeIndex]->handle(),
                .offset = 0,
                .range = sizeof(std::uint32_t),
            };
            const std::array shadowCullingWrites{
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = cullingDescriptorPool_->set(shadowCullingSetIndex),
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &bufferInfo,
                },
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = cullingDescriptorPool_->set(shadowCullingSetIndex),
                    .dstBinding = 1,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &instanceInfo,
                },
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = cullingDescriptorPool_->set(shadowCullingSetIndex),
                    .dstBinding = 2,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &meshletWorkInfo,
                },
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = cullingDescriptorPool_->set(shadowCullingSetIndex),
                    .dstBinding = 3,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &shadowVisibleInfo,
                },
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = shadowDescriptorPool_->set(shadowSetIndex),
                    .dstBinding = 8,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &materialInfo,
                },
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = cullingDescriptorPool_->set(shadowCullingSetIndex),
                    .dstBinding = 4,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &shadowIndirectInfo,
                },
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = cullingDescriptorPool_->set(shadowCullingSetIndex),
                    .dstBinding = 5,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &shadowDrawCountInfo,
                },
            };
            vkUpdateDescriptorSets(device_.handle(),
                                   static_cast<std::uint32_t>(shadowCullingWrites.size()),
                                   shadowCullingWrites.data(), 0, nullptr);
            const std::array shadowDrawWrites{
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = shadowDescriptorPool_->set(shadowSetIndex),
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo = &bufferInfo,
                },
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = shadowDescriptorPool_->set(shadowSetIndex),
                    .dstBinding = 6,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &instanceInfo,
                },
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = shadowDescriptorPool_->set(shadowSetIndex),
                    .dstBinding = 7,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &shadowVisibleInfo,
                },
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = shadowDescriptorPool_->set(shadowSetIndex),
                    .dstBinding = 9,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .pBufferInfo = &transformInfo,
                },
            };
            vkUpdateDescriptorSets(device_.handle(),
                                   static_cast<std::uint32_t>(shadowDrawWrites.size()),
                                   shadowDrawWrites.data(), 0, nullptr);
        }
        const std::array occlusionWrites{
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = occlusionDescriptorPool_->set(index),
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &bufferInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = occlusionDescriptorPool_->set(index),
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &instanceInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = occlusionDescriptorPool_->set(index),
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &frustumVisibleInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = occlusionDescriptorPool_->set(index),
                .dstBinding = 3,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &frustumIndirectInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = occlusionDescriptorPool_->set(index),
                .dstBinding = 5,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &visibleInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = occlusionDescriptorPool_->set(index),
                .dstBinding = 6,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &indirectInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = occlusionDescriptorPool_->set(index),
                .dstBinding = 7,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &frustumDrawCountInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = occlusionDescriptorPool_->set(index),
                .dstBinding = 8,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &drawCountInfo,
            },
        };
        vkUpdateDescriptorSets(device_.handle(), static_cast<std::uint32_t>(occlusionWrites.size()),
                               occlusionWrites.data(), 0, nullptr);
        const VkWriteDescriptorSet toneUniformWrite{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = toneMappingDescriptorPool_->set(index),
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &bufferInfo,
        };
        vkUpdateDescriptorSets(device_.handle(), 1, &toneUniformWrite, 0, nullptr);
        const VkWriteDescriptorSet taaUniformWrite{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = taaDescriptorPool_->set(index),
            .dstBinding = 7,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &bufferInfo,
        };
        vkUpdateDescriptorSets(device_.handle(), 1, &taaUniformWrite, 0, nullptr);
    }

    std::vector<VkDescriptorImageInfo> bindlessTextureInfos;
    bindlessTextureInfos.reserve(textureImages_.size());
    for (std::size_t index = 0; index < textureImages_.size(); ++index) {
        bindlessTextureInfos.push_back({
            .sampler = textureSamplers_[index]->handle(),
            .imageView = textureViews_[index]->handle(),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        });
    }
    const VkWriteDescriptorSet bindlessTextureWrite{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = bindlessDescriptorPool_->set(0),
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = static_cast<std::uint32_t>(bindlessTextureInfos.size()),
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = bindlessTextureInfos.data(),
    };
    vkUpdateDescriptorSets(device_.handle(), 1, &bindlessTextureWrite, 0, nullptr);

    core::log(core::LogLevel::info, "Scene uploaded asynchronously: ", scene.label, "; ",
              vertices.size(), " vertices, ", indices.size(), " indices, ", workItems.size(),
              " meshlet/LOD work candidates; vertex BDA=0x", std::hex,
              vertexBuffer_->deviceAddress(), ", index BDA=0x", indexBuffer_->deviceAddress(),
              std::dec);
    core::log(core::LogLevel::info, "IBL reference bake: ", ibl.environment.width, "x",
              ibl.environment.height, " environment, ", specularMipCount, " specular mips, ",
              bakeMilliseconds, " ms");
    core::log(core::LogLevel::info, config_.environmentPath.empty()
                                        ? "IBL source: procedural HDR studio"
                                        : "IBL source: " + config_.environmentPath.string());
    core::log(core::LogLevel::info, "GPU scene: ", instances.size(), " instances, ",
              materials.size(), " materials, ", textureImages_.size(), "/", maxBindlessTextures,
              " bindless texture descriptors resident");
    core::log(core::LogLevel::info,
              "Controls: WASD move, Q/E down/up, Shift boost, hold right mouse to look, "
              "Left/Right rotate environment, Escape releases mouse, R reframes scene");
}

void Renderer::createRayTracingResources(std::span<const gpu_scene::TransformRecord> transforms) {
    const vulkan::OptionalCapabilities& capabilities = device_.optionalCapabilities();
    if (!rayTracingFoundationEnabled_ || !capabilities.accelerationStructure ||
        rayTracingBuildPlan_.bottomLevels.empty() ||
        rayTracingBuildPlan_.topLevelInstances.empty()) {
        throw std::runtime_error("RT acceleration-structure build was requested without a plan");
    }
    if (vkGetAccelerationStructureBuildSizesKHR == nullptr ||
        vkCmdBuildAccelerationStructuresKHR == nullptr) {
        throw std::runtime_error("Enabled RT build commands were not loaded by Vulkan");
    }
    if (capabilities.maxAccelerationStructureGeometryCount < 1U ||
        rayTracingBuildPlan_.topLevelInstances.size() >
            capabilities.maxAccelerationStructureInstanceCount) {
        throw std::runtime_error("RT scene plan exceeds this GPU's acceleration-structure limits");
    }

    constexpr VkBuildAccelerationStructureFlagsKHR bottomLevelFlags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    constexpr VkBuildAccelerationStructureFlagsKHR topLevelFlags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
        VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    std::vector<VkAccelerationStructureBuildSizesInfoKHR> bottomLevelSizes(
        rayTracingBuildPlan_.bottomLevels.size());
    std::vector<std::uint32_t> bottomLevelPrimitiveCounts(rayTracingBuildPlan_.bottomLevels.size());

    const VkDeviceAddress vertexAddress = vertexBuffer_->deviceAddress();
    const VkDeviceAddress indexAddress = indexBuffer_->deviceAddress();
    for (std::size_t index = 0; index < rayTracingBuildPlan_.bottomLevels.size(); ++index) {
        const ray_tracing::BlasGeometryPlan& plan = rayTracingBuildPlan_.bottomLevels[index];
        const std::uint32_t primitiveCount = plan.indexCount / 3U;
        if (primitiveCount == 0U ||
            primitiveCount > capabilities.maxAccelerationStructurePrimitiveCount) {
            throw std::runtime_error("RT mesh exceeds this GPU's BLAS primitive limit");
        }
        bottomLevelPrimitiveCounts[index] = primitiveCount;
        const VkAccelerationStructureGeometryKHR geometry{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            .geometry =
                {
                    .triangles =
                        {
                            .sType =
                                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                            .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
                            .vertexData =
                                {
                                    .deviceAddress = vertexAddress +
                                                     static_cast<VkDeviceSize>(plan.firstVertex) *
                                                         sizeof(gpu_scene::VertexRecord),
                                },
                            .vertexStride = sizeof(gpu_scene::VertexRecord),
                            .maxVertex = plan.maximumVertex,
                            .indexType = VK_INDEX_TYPE_UINT32,
                            .indexData =
                                {
                                    .deviceAddress =
                                        indexAddress + static_cast<VkDeviceSize>(plan.firstIndex) *
                                                           sizeof(std::uint32_t),
                                },
                        },
                },
            .flags = plan.opaque ? static_cast<VkGeometryFlagsKHR>(VK_GEOMETRY_OPAQUE_BIT_KHR)
                                 : VkGeometryFlagsKHR{0U},
        };
        const VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
            .flags = bottomLevelFlags,
            .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .geometryCount = 1U,
            .pGeometries = &geometry,
        };
        bottomLevelSizes[index] = {
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
        };
        vkGetAccelerationStructureBuildSizesKHR(
            device_.handle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo,
            &primitiveCount, &bottomLevelSizes[index]);
        if (bottomLevelSizes[index].accelerationStructureSize == 0U) {
            throw std::runtime_error("Vulkan returned an empty BLAS allocation size");
        }
    }

    bottomLevelStructures_.clear();
    bottomLevelStructures_.reserve(bottomLevelSizes.size());
    rayTracingPersistentBytes_ = 0U;
    for (const VkAccelerationStructureBuildSizesInfoKHR& sizes : bottomLevelSizes) {
        bottomLevelStructures_.push_back(std::make_unique<vulkan::AccelerationStructure>(
            device_, memoryAllocator_,
            vulkan::AccelerationStructureCreateInfo{
                .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                .size = sizes.accelerationStructureSize,
            }));
        rayTracingPersistentBytes_ =
            addDeviceSize(rayTracingPersistentBytes_, sizes.accelerationStructureSize);
    }

    std::vector<VkAccelerationStructureInstanceKHR> accelerationInstances;
    accelerationInstances.reserve(rayTracingBuildPlan_.topLevelInstances.size());
    for (const ray_tracing::TlasInstancePlan& plan : rayTracingBuildPlan_.topLevelInstances) {
        if (plan.transformIndex >= transforms.size() ||
            plan.blasIndex >= bottomLevelStructures_.size()) {
            throw std::runtime_error("RT instance plan became inconsistent during GPU creation");
        }
        VkTransformMatrixKHR transform{};
        const glm::mat4& world = transforms[plan.transformIndex].world;
        const std::array<float, 12> rowMajorTransform =
            ray_tracing::makeInstanceTransform3x4(world);
        for (std::size_t row = 0; row < 3U; ++row) {
            for (std::size_t column = 0; column < 4U; ++column) {
                transform.matrix[row][column] = rowMajorTransform[row * 4U + column];
            }
        }
        VkGeometryInstanceFlagsKHR flags = 0U;
        if (plan.doubleSided) {
            flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        }
        if (plan.mirrored) {
            flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FLIP_FACING_BIT_KHR;
        }
        VkAccelerationStructureInstanceKHR instance{
            .transform = transform,
            .accelerationStructureReference =
                bottomLevelStructures_[plan.blasIndex]->deviceAddress(),
        };
        // Vulkan defines these as 24/8-bit fields. The planner bounds the
        // custom index and mask before this intentionally narrowing assignment.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#endif
        instance.instanceCustomIndex = plan.sourceInstanceIndex;
        instance.mask = plan.visibilityMask;
        instance.instanceShaderBindingTableRecordOffset = 0U;
        instance.flags = flags;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
        accelerationInstances.push_back(instance);
    }

    rayTracingInstanceBuffer_ = std::make_unique<vulkan::Buffer>(
        device_, memoryAllocator_,
        vulkan::BufferCreateInfo{
            .size = accelerationInstances.size() * sizeof(VkAccelerationStructureInstanceKHR),
            .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .requiredMemory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            .preferredMemory =
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        });
    rayTracingInstanceBuffer_->write(std::as_bytes(std::span(accelerationInstances)));
    rayTracingPersistentBytes_ =
        addDeviceSize(rayTracingPersistentBytes_, rayTracingInstanceBuffer_->size());

    const VkAccelerationStructureGeometryKHR topLevelGeometry{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        .geometry =
            {
                .instances =
                    {
                        .sType =
                            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                        .arrayOfPointers = VK_FALSE,
                        .data = {.deviceAddress = rayTracingInstanceBuffer_->deviceAddress()},
                    },
            },
    };
    const VkAccelerationStructureBuildGeometryInfoKHR topLevelQuery{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        .flags = topLevelFlags,
        .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .geometryCount = 1U,
        .pGeometries = &topLevelGeometry,
    };
    const std::uint32_t topLevelPrimitiveCount =
        static_cast<std::uint32_t>(accelerationInstances.size());
    VkAccelerationStructureBuildSizesInfoKHR topLevelSizes{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    };
    vkGetAccelerationStructureBuildSizesKHR(
        device_.handle(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &topLevelQuery,
        &topLevelPrimitiveCount, &topLevelSizes);
    if (topLevelSizes.accelerationStructureSize == 0U) {
        throw std::runtime_error("Vulkan returned an empty TLAS allocation size");
    }
    topLevelStructure_ = std::make_unique<vulkan::AccelerationStructure>(
        device_, memoryAllocator_,
        vulkan::AccelerationStructureCreateInfo{
            .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            .size = topLevelSizes.accelerationStructureSize,
        });
    rayTracingPersistentBytes_ =
        addDeviceSize(rayTracingPersistentBytes_, topLevelSizes.accelerationStructureSize);

    const VkDeviceSize scratchAlignment =
        std::max<VkDeviceSize>(1U, capabilities.minAccelerationStructureScratchAlignment);
    std::vector<VkDeviceSize> bottomLevelScratchOffsets(bottomLevelSizes.size());
    VkDeviceSize scratchPayloadSize = 0U;
    for (std::size_t index = 0; index < bottomLevelSizes.size(); ++index) {
        scratchPayloadSize = alignDeviceSize(scratchPayloadSize, scratchAlignment);
        bottomLevelScratchOffsets[index] = scratchPayloadSize;
        scratchPayloadSize = addDeviceSize(
            scratchPayloadSize,
            alignDeviceSize(std::max<VkDeviceSize>(1U, bottomLevelSizes[index].buildScratchSize),
                            scratchAlignment));
    }
    scratchPayloadSize = alignDeviceSize(scratchPayloadSize, scratchAlignment);
    const VkDeviceSize topLevelScratchOffset = scratchPayloadSize;
    scratchPayloadSize =
        addDeviceSize(scratchPayloadSize,
                      alignDeviceSize(std::max<VkDeviceSize>(1U, topLevelSizes.buildScratchSize),
                                      scratchAlignment));
    if (scratchPayloadSize > std::numeric_limits<VkDeviceSize>::max() - (scratchAlignment - 1U)) {
        throw std::overflow_error("RT scratch allocation size overflow");
    }
    auto scratchBuffer = std::make_unique<vulkan::Buffer>(
        device_, memoryAllocator_,
        vulkan::BufferCreateInfo{
            .size = scratchPayloadSize + scratchAlignment - 1U,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            .requiredMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        });
    const VkDeviceAddress scratchBufferAddress = scratchBuffer->deviceAddress();
    const VkDeviceAddress scratchBaseAddress =
        alignDeviceSize(scratchBufferAddress, scratchAlignment);
    if (scratchBaseAddress - scratchBufferAddress + scratchPayloadSize > scratchBuffer->size()) {
        throw std::runtime_error("Aligned RT scratch ranges exceed their allocation");
    }

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkFence completionFence = VK_NULL_HANDLE;
    bool submitted = false;
    try {
        const VkCommandPoolCreateInfo poolInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = device_.queueFamilies().graphics,
        };
        vulkan::check(vkCreateCommandPool(device_.handle(), &poolInfo, nullptr, &commandPool),
                      "vkCreateCommandPool(RT build)");
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        const VkCommandBufferAllocateInfo allocateInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1U,
        };
        vulkan::check(vkAllocateCommandBuffers(device_.handle(), &allocateInfo, &commandBuffer),
                      "vkAllocateCommandBuffers(RT build)");
        const VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vulkan::check(vkBeginCommandBuffer(commandBuffer, &beginInfo),
                      "vkBeginCommandBuffer(RT build)");

        for (std::size_t index = 0; index < rayTracingBuildPlan_.bottomLevels.size(); ++index) {
            const ray_tracing::BlasGeometryPlan& plan = rayTracingBuildPlan_.bottomLevels[index];
            const VkAccelerationStructureGeometryKHR geometry{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
                .geometry =
                    {
                        .triangles =
                            {
                                .sType =
                                    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                                .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
                                .vertexData =
                                    {
                                        .deviceAddress =
                                            vertexAddress +
                                            static_cast<VkDeviceSize>(plan.firstVertex) *
                                                sizeof(gpu_scene::VertexRecord),
                                    },
                                .vertexStride = sizeof(gpu_scene::VertexRecord),
                                .maxVertex = plan.maximumVertex,
                                .indexType = VK_INDEX_TYPE_UINT32,
                                .indexData =
                                    {
                                        .deviceAddress = indexAddress + static_cast<VkDeviceSize>(
                                                                            plan.firstIndex) *
                                                                            sizeof(std::uint32_t),
                                    },
                            },
                    },
                .flags = plan.opaque ? static_cast<VkGeometryFlagsKHR>(VK_GEOMETRY_OPAQUE_BIT_KHR)
                                     : VkGeometryFlagsKHR{0U},
            };
            const VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
                .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                .flags = bottomLevelFlags,
                .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
                .dstAccelerationStructure = bottomLevelStructures_[index]->handle(),
                .geometryCount = 1U,
                .pGeometries = &geometry,
                .scratchData =
                    {
                        .deviceAddress = scratchBaseAddress + bottomLevelScratchOffsets[index],
                    },
            };
            const VkAccelerationStructureBuildRangeInfoKHR range{
                .primitiveCount = bottomLevelPrimitiveCounts[index],
            };
            const VkAccelerationStructureBuildRangeInfoKHR* rangePointer = &range;
            vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1U, &buildInfo, &rangePointer);
        }

        const VkMemoryBarrier2 bottomToTopBarrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        };
        const VkDependencyInfo dependency{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1U,
            .pMemoryBarriers = &bottomToTopBarrier,
        };
        vkCmdPipelineBarrier2(commandBuffer, &dependency);

        const VkAccelerationStructureBuildGeometryInfoKHR topLevelBuild{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            .flags = topLevelFlags,
            .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            .dstAccelerationStructure = topLevelStructure_->handle(),
            .geometryCount = 1U,
            .pGeometries = &topLevelGeometry,
            .scratchData = {.deviceAddress = scratchBaseAddress + topLevelScratchOffset},
        };
        const VkAccelerationStructureBuildRangeInfoKHR topLevelRange{
            .primitiveCount = topLevelPrimitiveCount,
        };
        const VkAccelerationStructureBuildRangeInfoKHR* topLevelRangePointer = &topLevelRange;
        vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1U, &topLevelBuild,
                                            &topLevelRangePointer);
        vulkan::check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer(RT build)");

        const VkFenceCreateInfo fenceInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vulkan::check(vkCreateFence(device_.handle(), &fenceInfo, nullptr, &completionFence),
                      "vkCreateFence(RT build)");
        const VkSemaphoreSubmitInfo uploadWait{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = sceneUpload_.semaphore,
            .value = sceneUpload_.value,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        };
        const VkCommandBufferSubmitInfo commandInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = commandBuffer,
        };
        const VkSubmitInfo2 submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .waitSemaphoreInfoCount = 1U,
            .pWaitSemaphoreInfos = &uploadWait,
            .commandBufferInfoCount = 1U,
            .pCommandBufferInfos = &commandInfo,
        };
        vulkan::check(vkQueueSubmit2(device_.graphicsQueue(), 1U, &submitInfo, completionFence),
                      "vkQueueSubmit2(RT build)");
        submitted = true;
        vulkan::check(vkWaitForFences(device_.handle(), 1U, &completionFence, VK_TRUE,
                                      std::numeric_limits<std::uint64_t>::max()),
                      "vkWaitForFences(RT build)");
    } catch (...) {
        if (submitted && completionFence != VK_NULL_HANDLE) {
            static_cast<void>(vkWaitForFences(device_.handle(), 1U, &completionFence, VK_TRUE,
                                              std::numeric_limits<std::uint64_t>::max()));
        }
        if (completionFence != VK_NULL_HANDLE) {
            vkDestroyFence(device_.handle(), completionFence, nullptr);
        }
        if (commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_.handle(), commandPool, nullptr);
        }
        throw;
    }
    vkDestroyFence(device_.handle(), completionFence, nullptr);
    vkDestroyCommandPool(device_.handle(), commandPool, nullptr);
    sceneUploadPending_ = false;
    scratchBuffer.reset();

    constexpr double bytesPerMiB = 1024.0 * 1024.0;
    core::log(
        core::LogLevel::info, "Built ", bottomLevelStructures_.size(),
        " BLAS and one TLAS; persistent RT resource bytes ", rayTracingPersistentBytes_, " (",
        static_cast<double>(rayTracingPersistentBytes_) / bytesPerMiB, " MiB), temporary scratch ",
        static_cast<double>(scratchPayloadSize + scratchAlignment - 1U) / bytesPerMiB, " MiB");
}

void Renderer::createGraphicsPipelines() {
    const std::array pbrSetLayouts{descriptorSetLayout_->handle(),
                                   bindlessDescriptorSetLayout_->handle()};
    const std::array temporalAttachmentFormats{motionFormat_, normalFormat_};
    const VkDescriptorSetLayout cullingSetLayout = cullingDescriptorSetLayout_->handle();
    constexpr VkPushConstantRange shadowComputePush{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(std::uint32_t),
    };
    constexpr VkPushConstantRange shadowVertexPush{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = sizeof(std::uint32_t),
    };
    frustumCullPipeline_ = std::make_unique<vulkan::ComputePipeline>(
        device_, vulkan::ComputePipelineCreateInfo{
                     .shader = frustumCullShader_.get(),
                     .cache = pipelineCache_.handle(),
                     .descriptorSetLayouts = std::span(&cullingSetLayout, 1),
                 });
    shadowCullPipeline_ = std::make_unique<vulkan::ComputePipeline>(
        device_, vulkan::ComputePipelineCreateInfo{
                     .shader = shadowCullShader_.get(),
                     .cache = pipelineCache_.handle(),
                     .descriptorSetLayouts = std::span(&cullingSetLayout, 1),
                     .pushConstantRanges = std::span(&shadowComputePush, 1),
                 });
    const VkDescriptorSetLayout occlusionSetLayout = occlusionDescriptorSetLayout_->handle();
    occlusionCullPipeline_ = std::make_unique<vulkan::ComputePipeline>(
        device_, vulkan::ComputePipelineCreateInfo{
                     .shader = occlusionCullShader_.get(),
                     .cache = pipelineCache_.handle(),
                     .descriptorSetLayouts = std::span(&occlusionSetLayout, 1),
                 });
    const VkDescriptorSetLayout hiZSetLayout = hiZDescriptorSetLayout_->handle();
    hiZCopyPipeline_ = std::make_unique<vulkan::ComputePipeline>(
        device_, vulkan::ComputePipelineCreateInfo{
                     .shader = hiZCopyShader_.get(),
                     .cache = pipelineCache_.handle(),
                     .descriptorSetLayouts = std::span(&hiZSetLayout, 1),
                 });
    hiZReducePipeline_ = std::make_unique<vulkan::ComputePipeline>(
        device_, vulkan::ComputePipelineCreateInfo{
                     .shader = hiZReduceShader_.get(),
                     .cache = pipelineCache_.handle(),
                     .descriptorSetLayouts = std::span(&hiZSetLayout, 1),
                 });
    constexpr VkVertexInputBindingDescription binding{
        .binding = 0,
        .stride = sizeof(gpu_scene::VertexRecord),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    constexpr std::array attributes{
        VkVertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(gpu_scene::VertexRecord, position),
        },
        VkVertexInputAttributeDescription{
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(gpu_scene::VertexRecord, normal),
        },
        VkVertexInputAttributeDescription{
            .location = 2,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = offsetof(gpu_scene::VertexRecord, uv0),
        },
        VkVertexInputAttributeDescription{
            .location = 3,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
            .offset = offsetof(gpu_scene::VertexRecord, tangent),
        },
        VkVertexInputAttributeDescription{
            .location = 4,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = offsetof(gpu_scene::VertexRecord, uv1),
        },
    };
    constexpr std::array shadowAttributes{attributes[0], attributes[2], attributes[4]};
    graphicsPipeline_ = std::make_unique<vulkan::GraphicsPipeline>(
        device_, vulkan::GraphicsPipelineCreateInfo{
                     .vertexShader = vertexShader_.get(),
                     .fragmentShader = fragmentShader_.get(),
                     .colorFormat = hdrFormat_,
                     .additionalColorFormats = temporalAttachmentFormats,
                     .depthFormat = depthFormat_,
                     .depthTestEnable = true,
                     .depthWriteEnable = true,
                     .depthCompareOp = VK_COMPARE_OP_LESS,
                     .cache = pipelineCache_.handle(),
                     .descriptorSetLayouts = pbrSetLayouts,
                     .vertexBindings = std::span(&binding, 1),
                     .vertexAttributes = attributes,
                 });

    const std::array shadowSetLayouts{shadowDescriptorSetLayout_->handle(),
                                      bindlessDescriptorSetLayout_->handle()};
    shadowPipeline_ = std::make_unique<vulkan::GraphicsPipeline>(
        device_, vulkan::GraphicsPipelineCreateInfo{
                     .vertexShader = shadowVertexShader_.get(),
                     .fragmentShader = shadowFragmentShader_.get(),
                     .depthFormat = depthFormat_,
                     .depthTestEnable = true,
                     .depthWriteEnable = true,
                     .depthCompareOp = VK_COMPARE_OP_LESS,
                     .depthBiasEnable = true,
                     .depthBiasConstantFactor = 2.0F,
                     .depthBiasSlopeFactor = 1.5F,
                     .cache = pipelineCache_.handle(),
                     .descriptorSetLayouts = shadowSetLayouts,
                     .pushConstantRanges = std::span(&shadowVertexPush, 1),
                     .vertexBindings = std::span(&binding, 1),
                     .vertexAttributes = shadowAttributes,
                 });

    skyPipeline_ = std::make_unique<vulkan::GraphicsPipeline>(
        device_, vulkan::GraphicsPipelineCreateInfo{
                     .vertexShader = toneMappingVertexShader_.get(),
                     .fragmentShader = skyFragmentShader_.get(),
                     .colorFormat = hdrFormat_,
                     .additionalColorFormats = temporalAttachmentFormats,
                     .depthFormat = depthFormat_,
                     .cullMode = VK_CULL_MODE_NONE,
                     .cache = pipelineCache_.handle(),
                     .descriptorSetLayouts = pbrSetLayouts,
                 });

    postTestPipeline_.reset();
    if (config_.postTestPattern != post::TestPattern::none) {
        const VkDescriptorSetLayout postTestSetLayout = descriptorSetLayout_->handle();
        postTestPipeline_ = std::make_unique<vulkan::GraphicsPipeline>(
            device_, vulkan::GraphicsPipelineCreateInfo{
                         .vertexShader = toneMappingVertexShader_.get(),
                         .fragmentShader = postTestFragmentShader_.get(),
                         .colorFormat = hdrFormat_,
                         .additionalColorFormats = temporalAttachmentFormats,
                         .depthFormat = depthFormat_,
                         .depthTestEnable = false,
                         .depthWriteEnable = false,
                         .cullMode = VK_CULL_MODE_NONE,
                         .cache = pipelineCache_.handle(),
                         .descriptorSetLayouts = std::span(&postTestSetLayout, 1),
                     });
    }

    const VkDescriptorSetLayout bloomSetLayout = bloomDescriptorSetLayout_->handle();
    const auto createBloomPipeline = [&](const vulkan::ShaderModule& fragmentShader) {
        return std::make_unique<vulkan::GraphicsPipeline>(
            device_, vulkan::GraphicsPipelineCreateInfo{
                         .vertexShader = toneMappingVertexShader_.get(),
                         .fragmentShader = &fragmentShader,
                         .colorFormat = hdrFormat_,
                         .cullMode = VK_CULL_MODE_NONE,
                         .cache = pipelineCache_.handle(),
                         .descriptorSetLayouts = std::span(&bloomSetLayout, 1),
                     });
    };
    bloomDownsamplePipeline_ = createBloomPipeline(*bloomDownsampleShader_);
    bloomBlurHorizontalPipeline_ = createBloomPipeline(*bloomBlurHorizontalShader_);
    bloomBlurVerticalPipeline_ = createBloomPipeline(*bloomBlurVerticalShader_);

    const VkDescriptorSetLayout taaSetLayout = taaDescriptorSetLayout_->handle();
    const std::array taaAttachmentFormats{temporalDepthFormat_, normalFormat_};
    taaPipeline_ = std::make_unique<vulkan::GraphicsPipeline>(
        device_, vulkan::GraphicsPipelineCreateInfo{
                     .vertexShader = toneMappingVertexShader_.get(),
                     .fragmentShader = taaFragmentShader_.get(),
                     .colorFormat = hdrFormat_,
                     .additionalColorFormats = taaAttachmentFormats,
                     .cullMode = VK_CULL_MODE_NONE,
                     .cache = pipelineCache_.handle(),
                     .descriptorSetLayouts = std::span(&taaSetLayout, 1),
                 });

    const VkDescriptorSetLayout toneMappingSetLayout = toneMappingDescriptorSetLayout_->handle();
    toneMappingPipeline_ = std::make_unique<vulkan::GraphicsPipeline>(
        device_, vulkan::GraphicsPipelineCreateInfo{
                     .vertexShader = toneMappingVertexShader_.get(),
                     .fragmentShader = toneMappingFragmentShader_.get(),
                     .colorFormat = swapchain_.format(),
                     .cullMode = VK_CULL_MODE_NONE,
                     .cache = pipelineCache_.handle(),
                     .descriptorSetLayouts = std::span(&toneMappingSetLayout, 1),
                 });
}

void Renderer::createRenderGraph() {
    using render_graph::BufferState;
    using render_graph::ImageState;
    using render_graph::QueueType;

    renderGraph_ = std::make_unique<render_graph::RenderGraph>(device_, memoryAllocator_);
    const bool concurrentUpload =
        device_.queueFamilies().graphics != device_.queueFamilies().transfer;
    constexpr VkImageSubresourceRange colorSubresource{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    const VkImageSubresourceRange specularSubresource{
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0,
        .levelCount = prefilteredSpecularImage_->mipLevels(),
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
    constexpr ImageState sampledImageState{
        .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    constexpr ImageState historySampledState{
        .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        .layout = VK_IMAGE_LAYOUT_GENERAL,
    };
    graphSwapchainImage_ = renderGraph_->importImage("Swapchain", VK_NULL_HANDLE, colorSubresource,
                                                     {}, QueueType::graphics);
    graphTextureImages_.clear();
    graphTextureImages_.reserve(textureImages_.size());
    for (std::size_t index = 0; index < textureImages_.size(); ++index) {
        const VkImageSubresourceRange textureSubresource{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = textureImages_[index]->mipLevels(),
            .baseArrayLayer = 0,
            .layerCount = 1,
        };
        graphTextureImages_.push_back(renderGraph_->importImage(
            "Scene texture " + std::to_string(index), textureImages_[index]->handle(),
            textureSubresource, sampledImageState, QueueType::graphics, concurrentUpload));
    }
    graphEnvironmentImage_ =
        renderGraph_->importImage("HDR environment", environmentImage_->handle(), colorSubresource,
                                  sampledImageState, QueueType::graphics, concurrentUpload);
    graphDiffuseIrradianceImage_ = renderGraph_->importImage(
        "Diffuse irradiance", diffuseIrradianceImage_->handle(), colorSubresource,
        sampledImageState, QueueType::graphics, concurrentUpload);
    graphPrefilteredSpecularImage_ = renderGraph_->importImage(
        "Prefiltered specular", prefilteredSpecularImage_->handle(), specularSubresource,
        sampledImageState, QueueType::graphics, concurrentUpload);
    graphBrdfLutImage_ =
        renderGraph_->importImage("BRDF integration LUT", brdfLutImage_->handle(), colorSubresource,
                                  sampledImageState, QueueType::graphics, concurrentUpload);
    graphHistoryColorReadImage_ = {};
    graphHistoryColorWriteImage_ = {};
    graphHistoryDepthReadImage_ = {};
    graphHistoryDepthWriteImage_ = {};
    graphHistoryNormalReadImage_ = {};
    graphHistoryNormalWriteImage_ = {};
    if (config_.taaEnabled) {
        graphHistoryColorReadImage_ =
            renderGraph_->importImage("Temporal color history read", VK_NULL_HANDLE,
                                      colorSubresource, {}, QueueType::graphics);
        graphHistoryColorWriteImage_ =
            renderGraph_->importImage("Temporal color history write", VK_NULL_HANDLE,
                                      colorSubresource, {}, QueueType::graphics);
        graphHistoryDepthReadImage_ =
            renderGraph_->importImage("Temporal depth history read", VK_NULL_HANDLE,
                                      colorSubresource, {}, QueueType::graphics);
        graphHistoryDepthWriteImage_ =
            renderGraph_->importImage("Temporal depth history write", VK_NULL_HANDLE,
                                      colorSubresource, {}, QueueType::graphics);
        graphHistoryNormalReadImage_ =
            renderGraph_->importImage("Temporal normal history read", VK_NULL_HANDLE,
                                      colorSubresource, {}, QueueType::graphics);
        graphHistoryNormalWriteImage_ =
            renderGraph_->importImage("Temporal normal history write", VK_NULL_HANDLE,
                                      colorSubresource, {}, QueueType::graphics);
    }
    const VkExtent2D targetExtent = swapchain_.extent();
    const std::uint32_t shadowExtent = config_.shadowsEnabled ? directionalShadowResolution : 1U;
    for (std::size_t cascadeIndex = 0; cascadeIndex < directionalShadowCascadeCount;
         ++cascadeIndex) {
        graphDirectionalShadowImages_[cascadeIndex] = renderGraph_->createTransientImage(
            "Directional shadow cascade " + std::to_string(cascadeIndex),
            vulkan::ImageCreateInfo{
                .extent = {shadowExtent, shadowExtent, 1},
                .format = depthFormat_,
                .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            },
            vulkan::ImageViewCreateInfo{.aspect = VK_IMAGE_ASPECT_DEPTH_BIT});
    }
    graphHdrImage_ = renderGraph_->createTransientImage(
        "HDR color",
        vulkan::ImageCreateInfo{
            .extent = {targetExtent.width, targetExtent.height, 1},
            .format = hdrFormat_,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        },
        vulkan::ImageViewCreateInfo{});
    graphMotionImage_ = renderGraph_->createTransientImage(
        "Motion vectors and previous depth",
        vulkan::ImageCreateInfo{
            .extent = {targetExtent.width, targetExtent.height, 1},
            .format = motionFormat_,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        },
        vulkan::ImageViewCreateInfo{});
    graphNormalImage_ = renderGraph_->createTransientImage(
        "World normal and surface classification",
        vulkan::ImageCreateInfo{
            .extent = {targetExtent.width, targetExtent.height, 1},
            .format = normalFormat_,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        },
        vulkan::ImageViewCreateInfo{});
    const VkExtent2D bloomExtent{
        .width = post::halfExtent(targetExtent.width),
        .height = post::halfExtent(targetExtent.height),
    };
    const auto createBloomImage = [&](std::string_view name) {
        return renderGraph_->createTransientImage(
            std::string(name),
            vulkan::ImageCreateInfo{
                .extent = {bloomExtent.width, bloomExtent.height, 1},
                .format = hdrFormat_,
                .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            },
            vulkan::ImageViewCreateInfo{});
    };
    graphBloomLevels_.clear();
    if (config_.bloomEnabled) {
        graphBloomLevels_.push_back({
            .extent = bloomExtent,
            .filtered = createBloomImage("Bloom level 0 filtered"),
            .scratch = createBloomImage("Bloom level 0 scratch"),
        });
    }
    graphDepthImage_ = renderGraph_->createTransientImage(
        "Depth",
        vulkan::ImageCreateInfo{
            .extent = {targetExtent.width, targetExtent.height, 1},
            .format = depthFormat_,
            .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        },
        vulkan::ImageViewCreateInfo{.aspect = VK_IMAGE_ASPECT_DEPTH_BIT});
    graphHiZImages_.clear();
    hiZExtents_.clear();
    VkExtent2D hiZExtent = targetExtent;
    for (std::uint32_t level = 0; level < maxHiZLevels; ++level) {
        hiZExtents_.push_back(hiZExtent);
        graphHiZImages_.push_back(renderGraph_->createTransientImage(
            "Hi-Z level " + std::to_string(level),
            vulkan::ImageCreateInfo{
                .extent = {hiZExtent.width, hiZExtent.height, 1},
                .format = hiZFormat,
                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            },
            vulkan::ImageViewCreateInfo{}));
        if (hiZExtent.width == 1 && hiZExtent.height == 1) {
            break;
        }
        hiZExtent.width = std::max(1U, (hiZExtent.width + 1U) / 2U);
        hiZExtent.height = std::max(1U, (hiZExtent.height + 1U) / 2U);
    }
    graphVertexBuffer_ = renderGraph_->importBuffer(
        "Scene vertices", vertexBuffer_->handle(), 0, vertexBuffer_->size(),
        BufferState{
            .stages = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
            .access = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
        },
        QueueType::graphics, concurrentUpload);
    graphIndexBuffer_ =
        renderGraph_->importBuffer("Scene indices", indexBuffer_->handle(), 0, indexBuffer_->size(),
                                   BufferState{
                                       .stages = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
                                       .access = VK_ACCESS_2_INDEX_READ_BIT,
                                   },
                                   QueueType::graphics, concurrentUpload);
    graphInstanceBuffer_ = renderGraph_->importBuffer(
        "GPU instances", instanceBuffer_->handle(), 0, instanceBuffer_->size(),
        BufferState{
            .stages =
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        },
        QueueType::graphics, concurrentUpload);
    graphMaterialBuffer_ = renderGraph_->importBuffer(
        "GPU materials", materialBuffer_->handle(), 0, materialBuffer_->size(),
        BufferState{
            .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        },
        QueueType::graphics, concurrentUpload);
    graphTransformBuffer_ = renderGraph_->importBuffer(
        "GPU transforms", transformBuffer_->handle(), 0, transformBuffer_->size(),
        BufferState{
            .stages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        },
        QueueType::graphics, concurrentUpload);
    graphMeshletWorkBuffer_ = renderGraph_->importBuffer(
        "Meshlet LOD work", meshletWorkBuffer_->handle(), 0, meshletWorkBuffer_->size(),
        BufferState{
            .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        },
        QueueType::graphics, concurrentUpload);
    graphFrustumVisibleInstanceBuffer_ = renderGraph_->importBuffer(
        "Frustum-visible instances", frustumVisibleInstanceBuffer_->handle(), 0,
        frustumVisibleInstanceBuffer_->size(), {}, QueueType::graphics);
    graphFrustumIndirectDrawBuffer_ =
        renderGraph_->importBuffer("Frustum indirect draw", frustumIndirectDrawBuffer_->handle(), 0,
                                   frustumIndirectDrawBuffer_->size(), {}, QueueType::graphics);
    graphFrustumDrawCountBuffer_ =
        renderGraph_->importBuffer("Frustum draw count", frustumDrawCountBuffer_->handle(), 0,
                                   frustumDrawCountBuffer_->size(), {}, QueueType::graphics);
    graphVisibleInstanceBuffer_ =
        renderGraph_->importBuffer("Occlusion-visible instances", visibleInstanceBuffer_->handle(),
                                   0, visibleInstanceBuffer_->size(), {}, QueueType::graphics);
    graphIndirectDrawBuffer_ =
        renderGraph_->importBuffer("Occlusion indirect draw", indirectDrawBuffer_->handle(), 0,
                                   indirectDrawBuffer_->size(), {}, QueueType::graphics);
    graphDrawCountBuffer_ =
        renderGraph_->importBuffer("Visible draw count", drawCountBuffer_->handle(), 0,
                                   drawCountBuffer_->size(), {}, QueueType::graphics);
    for (std::size_t cascadeIndex = 0; cascadeIndex < directionalShadowCascadeCount;
         ++cascadeIndex) {
        graphShadowVisibleInstanceBuffers_[cascadeIndex] = renderGraph_->importBuffer(
            "Shadow-visible instances " + std::to_string(cascadeIndex),
            shadowVisibleInstanceBuffers_[cascadeIndex]->handle(), 0,
            shadowVisibleInstanceBuffers_[cascadeIndex]->size(), {}, QueueType::graphics);
        graphShadowIndirectDrawBuffers_[cascadeIndex] = renderGraph_->importBuffer(
            "Shadow indirect draw " + std::to_string(cascadeIndex),
            shadowIndirectDrawBuffers_[cascadeIndex]->handle(), 0,
            shadowIndirectDrawBuffers_[cascadeIndex]->size(), {}, QueueType::graphics);
        graphShadowDrawCountBuffers_[cascadeIndex] = renderGraph_->importBuffer(
            "Shadow draw count " + std::to_string(cascadeIndex),
            shadowDrawCountBuffers_[cascadeIndex]->handle(), 0,
            shadowDrawCountBuffers_[cascadeIndex]->size(), {}, QueueType::graphics);
    }
    graphUniformBuffer_ = renderGraph_->importBuffer(
        "Frame uniform", uniformBuffers_[0]->handle(), 0, uniformBuffers_[0]->size(),
        BufferState{
            .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .access = VK_ACCESS_2_UNIFORM_READ_BIT,
        },
        QueueType::graphics);
    graphCaptureBuffer_ = {};
    if (captureReadbackBuffer_) {
        graphCaptureBuffer_ =
            renderGraph_->importBuffer("Capture readback", captureReadbackBuffer_->handle(), 0,
                                       captureReadbackBuffer_->size(), {}, QueueType::graphics);
    }

    const QueueType cullingQueue =
        device_.queueFamilies().compute == device_.queueFamilies().graphics ? QueueType::compute
                                                                            : QueueType::graphics;
    const render_graph::PassHandle resetCullingPass = renderGraph_->addPass(
        "Reset GPU work counters", QueueType::graphics,
        [this](VkCommandBuffer commandBuffer, const render_graph::RenderGraph&, const void*) {
            vkCmdFillBuffer(commandBuffer, frustumDrawCountBuffer_->handle(), 0,
                            sizeof(std::uint32_t), 0U);
            vkCmdFillBuffer(commandBuffer, drawCountBuffer_->handle(), 0, sizeof(std::uint32_t),
                            0U);
            if (config_.shadowsEnabled) {
                for (const auto& shadowDrawCount : shadowDrawCountBuffers_) {
                    vkCmdFillBuffer(commandBuffer, shadowDrawCount->handle(), 0,
                                    sizeof(std::uint32_t), 0U);
                }
            }
        });
    renderGraph_->writeBuffer(resetCullingPass, graphFrustumDrawCountBuffer_,
                              BufferState{
                                  .stages = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                  .access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                              });
    renderGraph_->writeBuffer(resetCullingPass, graphDrawCountBuffer_,
                              BufferState{
                                  .stages = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                  .access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                              });
    if (config_.shadowsEnabled) {
        for (const render_graph::BufferHandle shadowCount : graphShadowDrawCountBuffers_) {
            renderGraph_->writeBuffer(resetCullingPass, shadowCount,
                                      BufferState{
                                          .stages = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                          .access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                      });
        }
    }

    const render_graph::PassHandle cullingPass = renderGraph_->addPass(
        "Compute frustum culling", cullingQueue,
        [this](VkCommandBuffer commandBuffer, const render_graph::RenderGraph&,
               const void* userData) {
            if (userData == nullptr) {
                throw std::runtime_error("Culling pass requires frame context");
            }
            recordCullingPass(commandBuffer, *static_cast<const GraphFrameContext*>(userData));
        });
    renderGraph_->readBuffer(cullingPass, graphUniformBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                 .access = VK_ACCESS_2_UNIFORM_READ_BIT,
                             });
    renderGraph_->readBuffer(cullingPass, graphInstanceBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                 .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             });
    renderGraph_->readBuffer(cullingPass, graphMeshletWorkBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                 .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             });
    renderGraph_->writeBuffer(cullingPass, graphFrustumVisibleInstanceBuffer_,
                              BufferState{
                                  .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                              });
    renderGraph_->writeBuffer(cullingPass, graphFrustumIndirectDrawBuffer_,
                              BufferState{
                                  .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                              });
    renderGraph_->writeBuffer(
        cullingPass, graphFrustumDrawCountBuffer_,
        BufferState{
            .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        });

    if (config_.shadowsEnabled) {
        for (std::size_t cascadeIndex = 0; cascadeIndex < directionalShadowCascadeCount;
             ++cascadeIndex) {
            const render_graph::PassHandle shadowCullingPass = renderGraph_->addPass(
                "Compute directional-shadow cascade " + std::to_string(cascadeIndex), cullingQueue,
                [this, cascadeIndex](VkCommandBuffer commandBuffer,
                                     const render_graph::RenderGraph&, const void* userData) {
                    if (userData == nullptr) {
                        throw std::runtime_error("Shadow-culling pass requires frame context");
                    }
                    recordShadowCullingPass(commandBuffer,
                                            *static_cast<const GraphFrameContext*>(userData),
                                            cascadeIndex);
                });
            renderGraph_->readBuffer(shadowCullingPass, graphUniformBuffer_,
                                     BufferState{
                                         .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         .access = VK_ACCESS_2_UNIFORM_READ_BIT,
                                     });
            renderGraph_->readBuffer(shadowCullingPass, graphInstanceBuffer_,
                                     BufferState{
                                         .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                                     });
            renderGraph_->readBuffer(shadowCullingPass, graphMeshletWorkBuffer_,
                                     BufferState{
                                         .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                                     });
            renderGraph_->writeBuffer(shadowCullingPass,
                                      graphShadowVisibleInstanceBuffers_[cascadeIndex],
                                      BufferState{
                                          .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                      });
            renderGraph_->writeBuffer(shadowCullingPass,
                                      graphShadowIndirectDrawBuffers_[cascadeIndex],
                                      BufferState{
                                          .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                      });
            renderGraph_->writeBuffer(shadowCullingPass, graphShadowDrawCountBuffers_[cascadeIndex],
                                      BufferState{
                                          .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                          .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                      });
        }
    }

    const render_graph::PassHandle occlusionPass = renderGraph_->addPass(
        "Compute Hi-Z occlusion culling", cullingQueue,
        [this](VkCommandBuffer commandBuffer, const render_graph::RenderGraph&,
               const void* userData) {
            if (userData == nullptr) {
                throw std::runtime_error("Occlusion-culling pass requires frame context");
            }
            recordOcclusionCullingPass(commandBuffer,
                                       *static_cast<const GraphFrameContext*>(userData));
        });
    renderGraph_->readBuffer(occlusionPass, graphUniformBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                 .access = VK_ACCESS_2_UNIFORM_READ_BIT,
                             });
    renderGraph_->readBuffer(occlusionPass, graphInstanceBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                 .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             });
    renderGraph_->readBuffer(occlusionPass, graphFrustumVisibleInstanceBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                 .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             });
    renderGraph_->readBuffer(occlusionPass, graphFrustumIndirectDrawBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                 .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             });
    renderGraph_->readBuffer(occlusionPass, graphFrustumDrawCountBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                 .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             });
    const ImageState occlusionSampledState{
        .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        .layout = VK_IMAGE_LAYOUT_GENERAL,
    };
    for (const render_graph::ImageHandle image : graphHiZImages_) {
        renderGraph_->readImage(occlusionPass, image, occlusionSampledState);
    }
    renderGraph_->writeBuffer(occlusionPass, graphVisibleInstanceBuffer_,
                              BufferState{
                                  .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                              });
    renderGraph_->writeBuffer(occlusionPass, graphIndirectDrawBuffer_,
                              BufferState{
                                  .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                  .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                              });
    renderGraph_->writeBuffer(
        occlusionPass, graphDrawCountBuffer_,
        BufferState{
            .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        });

    for (std::size_t cascadeIndex = 0; cascadeIndex < directionalShadowCascadeCount;
         ++cascadeIndex) {
        const render_graph::ImageHandle shadowImage = graphDirectionalShadowImages_[cascadeIndex];
        const render_graph::PassHandle shadowPass = renderGraph_->addPass(
            config_.shadowsEnabled
                ? "Directional shadow map cascade " + std::to_string(cascadeIndex)
                : "Clear disabled shadow fallback " + std::to_string(cascadeIndex),
            QueueType::graphics,
            [this, cascadeIndex, shadowImage](VkCommandBuffer commandBuffer,
                                              const render_graph::RenderGraph& graph,
                                              const void* userData) {
                if (userData == nullptr) {
                    throw std::runtime_error("Directional-shadow pass requires frame context");
                }
                recordShadowPass(commandBuffer, graph,
                                 *static_cast<const GraphFrameContext*>(userData), cascadeIndex,
                                 shadowImage);
            });
        if (config_.shadowsEnabled) {
            renderGraph_->readBuffer(shadowPass, graphVertexBuffer_,
                                     BufferState{
                                         .stages = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
                                         .access = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
                                     });
            renderGraph_->readBuffer(shadowPass, graphIndexBuffer_,
                                     BufferState{
                                         .stages = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
                                         .access = VK_ACCESS_2_INDEX_READ_BIT,
                                     });
            renderGraph_->readBuffer(shadowPass, graphInstanceBuffer_,
                                     BufferState{
                                         .stages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                         .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                                     });
            renderGraph_->readBuffer(shadowPass, graphTransformBuffer_,
                                     BufferState{
                                         .stages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                         .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                                     });
            renderGraph_->readBuffer(shadowPass, graphMaterialBuffer_,
                                     BufferState{
                                         .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                         .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                                     });
            renderGraph_->readBuffer(shadowPass, graphShadowVisibleInstanceBuffers_[cascadeIndex],
                                     BufferState{
                                         .stages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                         .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                                     });
            renderGraph_->readBuffer(shadowPass, graphShadowIndirectDrawBuffers_[cascadeIndex],
                                     BufferState{
                                         .stages = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                         .access = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                     });
            renderGraph_->readBuffer(shadowPass, graphShadowDrawCountBuffers_[cascadeIndex],
                                     BufferState{
                                         .stages = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                         .access = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                                     });
            renderGraph_->readBuffer(shadowPass, graphUniformBuffer_,
                                     BufferState{
                                         .stages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                         .access = VK_ACCESS_2_UNIFORM_READ_BIT,
                                     });
            for (const render_graph::ImageHandle texture : graphTextureImages_) {
                renderGraph_->readImage(shadowPass, texture, sampledImageState);
            }
        }
        renderGraph_->writeImage(shadowPass, shadowImage,
                                 ImageState{
                                     .stages = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                     .access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                     .layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                 });
    }

    const render_graph::PassHandle pbrPass = renderGraph_->addPass(
        "PBR raster", QueueType::graphics,
        [this](VkCommandBuffer commandBuffer, const render_graph::RenderGraph& graph,
               const void* userData) {
            if (userData == nullptr) {
                throw std::runtime_error("PBR raster pass requires frame context");
            }
            recordPbrPass(commandBuffer, graph, *static_cast<const GraphFrameContext*>(userData));
        });
    renderGraph_->readBuffer(pbrPass, graphVertexBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
                                 .access = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
                             });
    renderGraph_->readBuffer(pbrPass, graphIndexBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
                                 .access = VK_ACCESS_2_INDEX_READ_BIT,
                             });
    renderGraph_->readBuffer(pbrPass, graphInstanceBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                 .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             });
    renderGraph_->readBuffer(pbrPass, graphVisibleInstanceBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                 .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             });
    renderGraph_->readBuffer(pbrPass, graphMaterialBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                 .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             });
    renderGraph_->readBuffer(pbrPass, graphTransformBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                                 .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                             });
    renderGraph_->readBuffer(pbrPass, graphIndirectDrawBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                 .access = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                             });
    renderGraph_->readBuffer(pbrPass, graphDrawCountBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                 .access = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                             });
    renderGraph_->readBuffer(pbrPass, graphUniformBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                 .access = VK_ACCESS_2_UNIFORM_READ_BIT,
                             });
    for (const render_graph::ImageHandle texture : graphTextureImages_) {
        renderGraph_->readImage(pbrPass, texture, sampledImageState);
    }
    renderGraph_->readImage(pbrPass, graphEnvironmentImage_, sampledImageState);
    renderGraph_->readImage(pbrPass, graphDiffuseIrradianceImage_, sampledImageState);
    renderGraph_->readImage(pbrPass, graphPrefilteredSpecularImage_, sampledImageState);
    renderGraph_->readImage(pbrPass, graphBrdfLutImage_, sampledImageState);
    for (const render_graph::ImageHandle shadowImage : graphDirectionalShadowImages_) {
        renderGraph_->readImage(pbrPass, shadowImage,
                                ImageState{
                                    .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                    .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                                    .layout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                                });
    }
    renderGraph_->writeImage(pbrPass, graphHdrImage_,
                             ImageState{
                                 .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 .access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                 .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             });
    renderGraph_->writeImage(pbrPass, graphMotionImage_,
                             ImageState{
                                 .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 .access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                 .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             });
    renderGraph_->writeImage(pbrPass, graphNormalImage_,
                             ImageState{
                                 .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 .access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                 .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             });
    renderGraph_->writeImage(pbrPass, graphDepthImage_,
                             ImageState{
                                 .stages = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                           VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                                 .access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                                 .layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                             });

    const ImageState hiZWriteState{
        .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .layout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const ImageState computeSampledState{
        .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    for (std::size_t level = 0; level < graphHiZImages_.size(); ++level) {
        const render_graph::PassHandle hiZPass = renderGraph_->addPass(
            "Build Hi-Z level " + std::to_string(level), cullingQueue,
            [this, level](VkCommandBuffer commandBuffer, const render_graph::RenderGraph&,
                          const void*) { recordHiZPass(commandBuffer, level); });
        const render_graph::ImageHandle source =
            level == 0 ? graphDepthImage_ : graphHiZImages_[level - 1];
        renderGraph_->readImage(hiZPass, source, computeSampledState);
        renderGraph_->writeImage(hiZPass, graphHiZImages_[level], hiZWriteState);
    }

    if (config_.taaEnabled) {
        const render_graph::PassHandle taaPass = renderGraph_->addPass(
            "Temporal resolve", QueueType::graphics,
            [this](VkCommandBuffer commandBuffer, const render_graph::RenderGraph&,
                   const void* userData) {
                if (userData == nullptr) {
                    throw std::runtime_error("Temporal resolve requires frame context");
                }
                recordTaaPass(commandBuffer, *static_cast<const GraphFrameContext*>(userData));
            });
        renderGraph_->readImage(taaPass, graphHdrImage_, sampledImageState);
        renderGraph_->readImage(taaPass, graphMotionImage_, sampledImageState);
        renderGraph_->readImage(taaPass, graphDepthImage_, sampledImageState);
        renderGraph_->readImage(taaPass, graphNormalImage_, sampledImageState);
        renderGraph_->readImage(taaPass, graphHistoryColorReadImage_, historySampledState);
        renderGraph_->readImage(taaPass, graphHistoryDepthReadImage_, historySampledState);
        renderGraph_->readImage(taaPass, graphHistoryNormalReadImage_, historySampledState);
        renderGraph_->readBuffer(taaPass, graphUniformBuffer_,
                                 BufferState{
                                     .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                     .access = VK_ACCESS_2_UNIFORM_READ_BIT,
                                 });
        renderGraph_->writeImage(taaPass, graphHistoryColorWriteImage_,
                                 ImageState{
                                     .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     .access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                     .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 });
        renderGraph_->writeImage(taaPass, graphHistoryDepthWriteImage_,
                                 ImageState{
                                     .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     .access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                     .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 });
        renderGraph_->writeImage(taaPass, graphHistoryNormalWriteImage_,
                                 ImageState{
                                     .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                     .access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                     .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 });

        // Both ping-pong roles end every submitted frame in the same truthful
        // state. This avoids guessing which physical depth/normal image was
        // last sampled or written when it is rebound on the next frame.
        const render_graph::PassHandle preserveHistoryPass =
            renderGraph_->addPass("Preserve temporal histories", QueueType::graphics);
        renderGraph_->readImage(preserveHistoryPass, graphHistoryDepthWriteImage_,
                                historySampledState);
        renderGraph_->readImage(preserveHistoryPass, graphHistoryNormalWriteImage_,
                                historySampledState);
    }

    const render_graph::ImageHandle postProcessSource =
        config_.taaEnabled ? graphHistoryColorWriteImage_ : graphHdrImage_;
    if (config_.bloomEnabled) {
        if (graphBloomLevels_.empty()) {
            throw std::runtime_error("Bloom is enabled without a graph level");
        }
        const BloomLevelGraphResources& bloomLevel = graphBloomLevels_.front();
        const ImageState colorWriteState{
            .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };
        const render_graph::PassHandle bloomDownsamplePass = renderGraph_->addPass(
            "Bloom downsample", QueueType::graphics,
            [this](VkCommandBuffer commandBuffer, const render_graph::RenderGraph& graph,
                   const void* userData) {
                if (userData == nullptr) {
                    throw std::runtime_error("Bloom downsample requires frame context");
                }
                const auto& context = *static_cast<const GraphFrameContext*>(userData);
                recordBloomPass(commandBuffer, graph, graphBloomLevels_.front().filtered,
                                *bloomDownsamplePipeline_, context.frameIndex, context.frameIndex,
                                GpuTimingPass::bloomDownsample);
            });
        renderGraph_->readImage(bloomDownsamplePass, postProcessSource,
                                config_.taaEnabled ? historySampledState : sampledImageState);
        renderGraph_->writeImage(bloomDownsamplePass, bloomLevel.filtered, colorWriteState);

        const render_graph::PassHandle bloomHorizontalPass = renderGraph_->addPass(
            "Bloom horizontal", QueueType::graphics,
            [this](VkCommandBuffer commandBuffer, const render_graph::RenderGraph& graph,
                   const void* userData) {
                if (userData == nullptr) {
                    throw std::runtime_error("Bloom horizontal blur requires frame context");
                }
                const auto& context = *static_cast<const GraphFrameContext*>(userData);
                recordBloomPass(commandBuffer, graph, graphBloomLevels_.front().scratch,
                                *bloomBlurHorizontalPipeline_, framesInFlight, context.frameIndex,
                                GpuTimingPass::bloomHorizontal);
            });
        renderGraph_->readImage(bloomHorizontalPass, bloomLevel.filtered, sampledImageState);
        renderGraph_->writeImage(bloomHorizontalPass, bloomLevel.scratch, colorWriteState);

        const render_graph::PassHandle bloomVerticalPass = renderGraph_->addPass(
            "Bloom vertical", QueueType::graphics,
            [this](VkCommandBuffer commandBuffer, const render_graph::RenderGraph& graph,
                   const void* userData) {
                if (userData == nullptr) {
                    throw std::runtime_error("Bloom vertical blur requires frame context");
                }
                const auto& context = *static_cast<const GraphFrameContext*>(userData);
                recordBloomPass(commandBuffer, graph, graphBloomLevels_.front().filtered,
                                *bloomBlurVerticalPipeline_, framesInFlight + 1U,
                                context.frameIndex, GpuTimingPass::bloomVertical);
            });
        renderGraph_->readImage(bloomVerticalPass, bloomLevel.scratch, sampledImageState);
        renderGraph_->writeImage(bloomVerticalPass, bloomLevel.filtered, colorWriteState);
    }

    const render_graph::PassHandle toneMappingPass = renderGraph_->addPass(
        "Tone mapping", QueueType::graphics,
        [this](VkCommandBuffer commandBuffer, const render_graph::RenderGraph& graph,
               const void* userData) {
            if (userData == nullptr) {
                throw std::runtime_error("Tone-mapping pass requires frame context");
            }
            recordToneMappingPass(commandBuffer, graph,
                                  *static_cast<const GraphFrameContext*>(userData));
        });
    renderGraph_->readImage(toneMappingPass, postProcessSource,
                            config_.taaEnabled ? historySampledState : sampledImageState);
    if (config_.bloomEnabled) {
        renderGraph_->readImage(toneMappingPass, graphBloomLevels_.front().filtered,
                                sampledImageState);
    }
    renderGraph_->readImage(toneMappingPass, graphMotionImage_, sampledImageState);
    renderGraph_->readBuffer(toneMappingPass, graphUniformBuffer_,
                             BufferState{
                                 .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                 .access = VK_ACCESS_2_UNIFORM_READ_BIT,
                             });
    renderGraph_->writeImage(toneMappingPass, graphSwapchainImage_,
                             ImageState{
                                 .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 .access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                 .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                             });

    if (graphCaptureBuffer_) {
        const render_graph::PassHandle capturePass = renderGraph_->addPass(
            "Capture readback", QueueType::graphics,
            [this](VkCommandBuffer commandBuffer, const render_graph::RenderGraph& graph,
                   const void* userData) {
                if (userData == nullptr) {
                    throw std::runtime_error("Capture pass requires frame context");
                }
                if (static_cast<const GraphFrameContext*>(userData)->capture) {
                    recordCapturePass(commandBuffer, graph);
                }
            });
        renderGraph_->readImage(capturePass, graphSwapchainImage_,
                                ImageState{
                                    .stages = VK_PIPELINE_STAGE_2_COPY_BIT,
                                    .access = VK_ACCESS_2_TRANSFER_READ_BIT,
                                    .layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                });
        renderGraph_->writeBuffer(capturePass, graphCaptureBuffer_,
                                  BufferState{
                                      .stages = VK_PIPELINE_STAGE_2_COPY_BIT,
                                      .access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                  });
    }

    const render_graph::PassHandle presentPass =
        renderGraph_->addPass("Present transition", QueueType::graphics);
    renderGraph_->readImage(presentPass, graphSwapchainImage_,
                            ImageState{
                                .stages = VK_PIPELINE_STAGE_2_NONE,
                                .access = VK_ACCESS_2_NONE,
                                .layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                            });
    renderGraph_->compile();

    core::log(core::LogLevel::info, "Render graph compiled: Frustum + previous-frame Hi-Z culling",
              config_.shadowsEnabled
                  ? " -> directional-shadow culling/map -> multi-draw indirect PBR + "
                    "motion/normals -> Hi-Z build"
                  : " -> shadows disabled -> multi-draw indirect PBR + motion/normals -> "
                    "Hi-Z build",
              config_.taaEnabled ? " -> Temporal resolve" : "",
              config_.bloomEnabled ? " -> Bloom" : " -> Bloom skipped", " -> Tone mapping",
              graphCaptureBuffer_ ? " -> Capture" : "", " -> Present");

    const post::ResourceMemory memory = post::resourceMemory(
        targetExtent.width, targetExtent.height, config_.bloomEnabled, config_.taaEnabled);
    constexpr double bytesPerMiB = 1024.0 * 1024.0;
    core::log(core::LogLevel::info, "Post resources at ", targetExtent.width, "x",
              targetExtent.height, ": HDR ", static_cast<double>(memory.hdrBytes) / bytesPerMiB,
              " MiB, motion ", static_cast<double>(memory.motionBytes) / bytesPerMiB,
              " MiB, normals ", static_cast<double>(memory.normalBytes) / bytesPerMiB,
              " MiB, depth ", static_cast<double>(memory.depthBytes) / bytesPerMiB, " MiB; bloom ",
              memory.bloomLevels.size(), " current level(s), ",
              static_cast<double>(memory.bloomBytes()) / bytesPerMiB,
              " MiB transient total; temporal histories ",
              static_cast<double>(memory.temporalHistoryBytes) / bytesPerMiB,
              " MiB persistent; accounted total ",
              static_cast<double>(memory.totalBytes()) / bytesPerMiB, " MiB");
    const std::uint64_t shadowPixelBytes = depthFormat_ == VK_FORMAT_D16_UNORM ? 2U : 4U;
    const std::uint64_t shadowSize = config_.shadowsEnabled ? directionalShadowResolution : 1U;
    const std::uint64_t shadowImageBytes =
        shadowSize * shadowSize * shadowPixelBytes * directionalShadowCascadeCount;
    std::uint64_t shadowWorkBytes = 0;
    for (std::size_t cascadeIndex = 0; cascadeIndex < directionalShadowCascadeCount;
         ++cascadeIndex) {
        shadowWorkBytes += shadowVisibleInstanceBuffers_[cascadeIndex]->size() +
                           shadowIndirectDrawBuffers_[cascadeIndex]->size() +
                           shadowDrawCountBuffers_[cascadeIndex]->size();
    }
    core::log(core::LogLevel::info, "Directional shadow resources: ",
              config_.shadowsEnabled ? "three active cascades at " : "disabled fallbacks at ",
              shadowSize, "x", shadowSize, ", images ",
              static_cast<double>(shadowImageBytes) / bytesPerMiB, " MiB transient; GPU work ",
              static_cast<double>(shadowWorkBytes) / bytesPerMiB, " MiB persistent");
    for (std::size_t levelIndex = 0; levelIndex < memory.bloomLevels.size(); ++levelIndex) {
        const post::BloomLevelMemory& level = memory.bloomLevels[levelIndex];
        core::log(core::LogLevel::info, "Bloom level ", levelIndex, " at ", level.width, "x",
                  level.height, ": filtered ",
                  static_cast<double>(level.filteredBytes) / bytesPerMiB, " MiB, scratch ",
                  static_cast<double>(level.scratchBytes) / bytesPerMiB, " MiB");
    }
}

void Renderer::updateToneMappingDescriptor() {
    const VkDescriptorImageInfo currentColorInfo{
        .sampler = brdfLutSampler_->handle(),
        .imageView = renderGraph_->imageView(graphHdrImage_),
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkDescriptorImageInfo motionInfo{
        .sampler = hiZSampler_->handle(),
        .imageView = renderGraph_->imageView(graphMotionImage_),
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkDescriptorImageInfo currentDepthInfo{
        .sampler = hiZSampler_->handle(),
        .imageView = renderGraph_->imageView(graphDepthImage_),
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkDescriptorImageInfo currentNormalInfo{
        .sampler = brdfLutSampler_->handle(),
        .imageView = renderGraph_->imageView(graphNormalImage_),
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    for (std::size_t index = 0; index < framesInFlight; ++index) {
        const VkDescriptorImageInfo postProcessInfo{
            .sampler = brdfLutSampler_->handle(),
            .imageView = config_.taaEnabled ? historyColorViews_[index]->handle()
                                            : currentColorInfo.imageView,
            .imageLayout = config_.taaEnabled ? VK_IMAGE_LAYOUT_GENERAL
                                              : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkDescriptorImageInfo bloomInfo{
            .sampler = brdfLutSampler_->handle(),
            .imageView = config_.bloomEnabled
                             ? renderGraph_->imageView(graphBloomLevels_.front().filtered)
                             : postProcessInfo.imageView,
            .imageLayout = config_.bloomEnabled ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                : postProcessInfo.imageLayout,
        };
        const std::array toneWrites{
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = toneMappingDescriptorPool_->set(index),
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &postProcessInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = toneMappingDescriptorPool_->set(index),
                .dstBinding = 2,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &bloomInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = toneMappingDescriptorPool_->set(index),
                .dstBinding = 3,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &motionInfo,
            },
        };
        vkUpdateDescriptorSets(device_.handle(), static_cast<std::uint32_t>(toneWrites.size()),
                               toneWrites.data(), 0, nullptr);

        if (config_.taaEnabled) {
            const std::size_t readIndex = 1U - index;
            const VkDescriptorImageInfo historyColorInfo{
                .sampler = brdfLutSampler_->handle(),
                .imageView = historyColorViews_[readIndex]->handle(),
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };
            const VkDescriptorImageInfo historyDepthInfo{
                .sampler = hiZSampler_->handle(),
                .imageView = historyDepthViews_[readIndex]->handle(),
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };
            const VkDescriptorImageInfo historyNormalInfo{
                .sampler = brdfLutSampler_->handle(),
                .imageView = historyNormalViews_[readIndex]->handle(),
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };
            const std::array taaImageInfos{currentColorInfo, historyColorInfo, motionInfo,
                                           currentDepthInfo, historyDepthInfo, currentNormalInfo,
                                           historyNormalInfo};
            std::array<VkWriteDescriptorSet, 7> taaWrites{};
            for (std::size_t bindingIndex = 0; bindingIndex < taaWrites.size(); ++bindingIndex) {
                taaWrites[bindingIndex] = {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = taaDescriptorPool_->set(index),
                    .dstBinding = static_cast<std::uint32_t>(bindingIndex),
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &taaImageInfos[bindingIndex],
                };
            }
            vkUpdateDescriptorSets(device_.handle(), static_cast<std::uint32_t>(taaWrites.size()),
                                   taaWrites.data(), 0, nullptr);
        }

        if (config_.bloomEnabled) {
            const VkDescriptorBufferInfo bloomControlsInfo{
                .buffer = bloomControlsBuffer_->handle(),
                .offset = 0,
                .range = sizeof(BloomControls),
            };
            const VkWriteDescriptorSet bloomSourceWrite{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = bloomDescriptorPool_->set(index),
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &postProcessInfo,
            };
            const VkWriteDescriptorSet bloomControlsWrite{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = bloomDescriptorPool_->set(index),
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &bloomControlsInfo,
            };
            const std::array bloomWrites{bloomSourceWrite, bloomControlsWrite};
            vkUpdateDescriptorSets(device_.handle(), static_cast<std::uint32_t>(bloomWrites.size()),
                                   bloomWrites.data(), 0, nullptr);
        }
    }

    if (!config_.bloomEnabled) {
        return;
    }
    const std::array blurSources{
        VkDescriptorImageInfo{
            .sampler = brdfLutSampler_->handle(),
            .imageView = renderGraph_->imageView(graphBloomLevels_.front().filtered),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        },
        VkDescriptorImageInfo{
            .sampler = brdfLutSampler_->handle(),
            .imageView = renderGraph_->imageView(graphBloomLevels_.front().scratch),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        },
    };
    for (std::size_t index = 0; index < blurSources.size(); ++index) {
        const VkDescriptorBufferInfo bloomControlsInfo{
            .buffer = bloomControlsBuffer_->handle(),
            .offset = 0,
            .range = sizeof(BloomControls),
        };
        const VkWriteDescriptorSet sourceWrite{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = bloomDescriptorPool_->set(framesInFlight + index),
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &blurSources[index],
        };
        const VkWriteDescriptorSet controlsWrite{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = bloomDescriptorPool_->set(framesInFlight + index),
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &bloomControlsInfo,
        };
        const std::array writes{sourceWrite, controlsWrite};
        vkUpdateDescriptorSets(device_.handle(), static_cast<std::uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }
}

void Renderer::updateShadowDescriptor() {
    std::array<VkDescriptorImageInfo, directionalShadowCascadeCount> shadowInfos{};
    for (std::size_t cascadeIndex = 0; cascadeIndex < directionalShadowCascadeCount;
         ++cascadeIndex) {
        shadowInfos[cascadeIndex] = {
            .sampler = shadowSampler_->handle(),
            .imageView = renderGraph_->imageView(graphDirectionalShadowImages_[cascadeIndex]),
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
        };
    }
    for (std::size_t index = 0; index < framesInFlight; ++index) {
        const VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorPool_->set(index),
            .dstBinding = 10,
            .descriptorCount = static_cast<std::uint32_t>(shadowInfos.size()),
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = shadowInfos.data(),
        };
        vkUpdateDescriptorSets(device_.handle(), 1, &write, 0, nullptr);
    }
}

void Renderer::updateHiZDescriptors() {
    if (graphHiZImages_.empty() || graphHiZImages_.size() != hiZExtents_.size() ||
        graphHiZImages_.size() > maxHiZLevels) {
        throw std::runtime_error("Hi-Z graph resources are inconsistent");
    }

    for (std::size_t level = 0; level < graphHiZImages_.size(); ++level) {
        const render_graph::ImageHandle source =
            level == 0 ? graphDepthImage_ : graphHiZImages_[level - 1];
        const VkDescriptorImageInfo sourceInfo{
            .sampler = hiZSampler_->handle(),
            .imageView = renderGraph_->imageView(source),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkDescriptorImageInfo destinationInfo{
            .imageView = renderGraph_->imageView(graphHiZImages_[level]),
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const std::array writes{
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = hiZDescriptorPool_->set(level),
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &sourceInfo,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = hiZDescriptorPool_->set(level),
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &destinationInfo,
            },
        };
        vkUpdateDescriptorSets(device_.handle(), static_cast<std::uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }

    std::vector<VkDescriptorImageInfo> levelInfos;
    levelInfos.reserve(maxHiZLevels);
    for (const render_graph::ImageHandle image : graphHiZImages_) {
        levelInfos.push_back({
            .sampler = hiZSampler_->handle(),
            .imageView = renderGraph_->imageView(image),
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        });
    }
    levelInfos.resize(maxHiZLevels, levelInfos.back());
    for (std::size_t frame = 0; frame < framesInFlight; ++frame) {
        const VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = occlusionDescriptorPool_->set(frame),
            .dstBinding = 4,
            .descriptorCount = maxHiZLevels,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = levelInfos.data(),
        };
        vkUpdateDescriptorSets(device_.handle(), 1, &write, 0, nullptr);
    }
}

void Renderer::resetCamera() {
    constexpr float verticalFieldOfView = 42.0F;
    const float cameraDistance =
        sceneRadius_ / std::sin(glm::radians(verticalFieldOfView * 0.5F)) * 1.10F;
    cameraYaw_ = -0.5F * std::numbers::pi_v<float>;
    cameraPitch_ = 0.0F;
    cameraPosition_ = sceneCenter_ + glm::vec3(0.0F, 0.0F, cameraDistance);
    temporalScriptOrigin_ = cameraPosition_;
    temporalState_.reset();
    cameraMovedThisFrame_ = true;
    previousCameraUpdate_ = std::chrono::steady_clock::now();
}

void Renderer::updateCamera() {
    const glm::vec3 previousPosition = cameraPosition_;
    const glm::vec3 previousForward = cameraForward(cameraYaw_, cameraPitch_);
    const auto now = std::chrono::steady_clock::now();
    float deltaSeconds = 0.0F;
    if (previousCameraUpdate_.time_since_epoch().count() != 0) {
        deltaSeconds = std::chrono::duration<float>(now - previousCameraUpdate_).count();
    }
    previousCameraUpdate_ = now;
    deltaSeconds = std::clamp(deltaSeconds, 0.0F, 0.1F);

    const platform::RendererInput input = window_.consumeRendererInput();
    if (input.resetCamera) {
        resetCamera();
        return;
    }
    if (input.mouseLookActive) {
        constexpr float mouseSensitivity = 0.0025F;
        cameraYaw_ += input.mouseDeltaX * mouseSensitivity;
        cameraPitch_ -= input.mouseDeltaY * mouseSensitivity;
        cameraYaw_ = std::remainder(cameraYaw_, 2.0F * std::numbers::pi_v<float>);
        cameraPitch_ = std::clamp(cameraPitch_, glm::radians(-89.0F), glm::radians(89.0F));
    }
    constexpr float environmentRotationSpeed = std::numbers::pi_v<float> / 3.0F;
    const float environmentDirection = static_cast<float>(input.rotateEnvironmentRight) -
                                       static_cast<float>(input.rotateEnvironmentLeft);
    environmentRotationRadians_ = std::remainder(
        environmentRotationRadians_ + environmentDirection * environmentRotationSpeed *
                                          (input.speedBoost ? 3.0F : 1.0F) * deltaSeconds,
        2.0F * std::numbers::pi_v<float>);
    if (environmentDirection != 0.0F) {
        temporalState_.invalidateHistory();
    }

    const glm::vec3 forward = cameraForward(cameraYaw_, cameraPitch_);
    const glm::vec3 up(0.0F, 1.0F, 0.0F);
    const glm::vec3 right = glm::normalize(glm::cross(forward, up));
    glm::vec3 movement(0.0F);
    movement += input.moveForward ? forward : glm::vec3(0.0F);
    movement -= input.moveBackward ? forward : glm::vec3(0.0F);
    movement -= input.moveLeft ? right : glm::vec3(0.0F);
    movement += input.moveRight ? right : glm::vec3(0.0F);
    movement += input.moveUp ? up : glm::vec3(0.0F);
    movement -= input.moveDown ? up : glm::vec3(0.0F);
    const float movementLengthSquared = glm::dot(movement, movement);
    if (movementLengthSquared > 0.0F) {
        const float speed = sceneRadius_ * (input.speedBoost ? 6.0F : 1.5F);
        cameraPosition_ += movement / std::sqrt(movementLengthSquared) * speed * deltaSeconds;
    }

    if (config_.temporalCameraScript == TemporalCameraScript::translate) {
        constexpr float translationPerFrame = 0.01F;
        cameraPosition_ = temporalScriptOrigin_ + right * (static_cast<float>(renderedFrameCount_) *
                                                           translationPerFrame * sceneRadius_);
    } else if (config_.temporalCameraScript == TemporalCameraScript::teleport) {
        cameraPosition_ = temporalScriptOrigin_;
        if (renderedFrameCount_ >= 2U) {
            cameraPosition_ += right * sceneRadius_ * 1.5F;
        }
        if (renderedFrameCount_ == 2U) {
            temporalState_.reset();
            core::log(core::LogLevel::info,
                      "Temporal history reset for deterministic camera teleport");
        }
    } else if (config_.temporalCameraScript == TemporalCameraScript::rotate) {
        constexpr float rotationRadiansPerFrame = 0.025F;
        cameraPosition_ = temporalScriptOrigin_;
        cameraYaw_ = -0.5F * std::numbers::pi_v<float> +
                     static_cast<float>(renderedFrameCount_) * rotationRadiansPerFrame;
    } else if (config_.temporalCameraScript == TemporalCameraScript::approach) {
        constexpr float travelPerFrame = 0.08F;
        const glm::vec3 approachDirection = glm::normalize(sceneCenter_ - temporalScriptOrigin_);
        cameraPosition_ = temporalScriptOrigin_ + approachDirection *
                                                      static_cast<float>(renderedFrameCount_) *
                                                      travelPerFrame * sceneRadius_;
    }

    const glm::vec3 currentForward = cameraForward(cameraYaw_, cameraPitch_);
    const float translationDistance = glm::length(cameraPosition_ - previousPosition);
    const float forwardAgreement = glm::dot(previousForward, currentForward);
    cameraMovedThisFrame_ = translationDistance > std::max(sceneRadius_ * 0.000001F, 0.000001F) ||
                            forwardAgreement < 0.999999F;
    const bool largeTranslation = translationDistance > sceneRadius_ * 0.5F;
    const bool largeRotation = forwardAgreement < std::cos(glm::radians(30.0F));
    if (largeTranslation || largeRotation) {
        temporalState_.reset();
    }
}

void Renderer::updateFrameUniform(std::size_t frameIndex) {
    const VkExtent2D extent = swapchain_.extent();
    const float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    constexpr float verticalFieldOfView = 42.0F;
    const glm::vec3 forward = cameraForward(cameraYaw_, cameraPitch_);
    const glm::mat4 view =
        glm::lookAtRH(cameraPosition_, cameraPosition_ + forward, glm::vec3(0.0F, 1.0F, 0.0F));
    const float cameraDistance = glm::length(cameraPosition_ - sceneCenter_);
    const float nearPlane = std::max(cameraDistance - sceneRadius_ * 1.5F, sceneRadius_ * 0.001F);
    const float farPlane = std::max(cameraDistance + sceneRadius_ * 2.0F, sceneRadius_ * 4.0F);
    glm::mat4 projection =
        glm::perspectiveRH_ZO(glm::radians(verticalFieldOfView), aspect, nearPlane, farPlane);
    projection[1][1] *= -1.0F;
    const bool jitterEnabled =
        config_.taaEnabled && config_.postTestPattern == post::TestPattern::none;
    temporalState_.beginFrame(view, projection, extent.width, extent.height, renderedFrameCount_,
                              jitterEnabled);
    const temporal::CameraMatrices& currentCamera = temporalState_.current();
    const temporal::CameraMatrices& previousCamera = temporalState_.previousForRendering();
    const glm::vec3 previousCameraPosition = glm::vec3(glm::inverse(previousCamera.view)[3]);
    const glm::uvec2 visibilityMask = gpu_scene::packRenderLayerMask(mainView_.visibilityMask);
    constexpr std::array cascadeFractions{0.32F, 0.62F, 1.0F};
    const glm::vec3 directionalLightTravel = glm::normalize(glm::vec3(0.35F, -0.75F, -0.55F));
    const glm::vec3 lightReferenceUp = std::abs(directionalLightTravel.y) > 0.98F
                                           ? glm::vec3(1.0F, 0.0F, 0.0F)
                                           : glm::vec3(0.0F, 1.0F, 0.0F);
    const glm::vec3 lightRight =
        glm::normalize(glm::cross(lightReferenceUp, directionalLightTravel));
    const glm::vec3 lightUp = glm::cross(directionalLightTravel, lightRight);
    const float tangentHalfFov = std::tan(glm::radians(verticalFieldOfView * 0.5F));
    std::array<glm::vec4, directionalShadowCascadeCount> shadowCascades{};
    std::array<float, directionalShadowCascadeCount> shadowSplits{};
    std::array<float, directionalShadowCascadeCount> shadowDepthRadii{};
    float cascadeNear = nearPlane;
    for (std::size_t cascadeIndex = 0; cascadeIndex < directionalShadowCascadeCount;
         ++cascadeIndex) {
        const float cascadeFar =
            nearPlane + (farPlane - nearPlane) * cascadeFractions[cascadeIndex];
        const float centerDistance = 0.5F * (cascadeNear + cascadeFar);
        const float halfDepth = 0.5F * (cascadeFar - cascadeNear);
        const float halfHeight = cascadeFar * tangentHalfFov;
        const float halfWidth = halfHeight * aspect;
        const float radius = std::max(
            std::sqrt(halfDepth * halfDepth + halfHeight * halfHeight + halfWidth * halfWidth) *
                1.15F,
            sceneRadius_ * 0.02F);
        glm::vec3 center = cameraPosition_ + forward * centerDistance;
        const float worldUnitsPerTexel =
            (2.0F * radius) / static_cast<float>(directionalShadowResolution);
        const float lightX = glm::dot(center, lightRight);
        const float lightY = glm::dot(center, lightUp);
        center +=
            lightRight * (std::round(lightX / worldUnitsPerTexel) * worldUnitsPerTexel - lightX);
        center += lightUp * (std::round(lightY / worldUnitsPerTexel) * worldUnitsPerTexel - lightY);
        const float receiverLightDepth = glm::dot(center, directionalLightTravel);
        const float sceneLightDepth = glm::dot(sceneCenter_, directionalLightTravel);
        const float minimumLightDepth =
            std::min(receiverLightDepth - radius, sceneLightDepth - sceneRadius_);
        const float maximumLightDepth =
            std::max(receiverLightDepth + radius, sceneLightDepth + sceneRadius_);
        const float lightDepthCenter = 0.5F * (minimumLightDepth + maximumLightDepth);
        const float depthRadius =
            std::max(0.5F * (maximumLightDepth - minimumLightDepth) * 1.01F, sceneRadius_ * 0.02F);
        center += directionalLightTravel * (lightDepthCenter - receiverLightDepth);
        shadowCascades[cascadeIndex] = {center, config_.shadowsEnabled ? radius : 0.0F};
        shadowDepthRadii[cascadeIndex] = config_.shadowsEnabled ? depthRadius : 0.0F;
        shadowSplits[cascadeIndex] = cascadeFar;
        cascadeNear = cascadeFar;
    }
    const FrameUniform uniform{
        .viewProjection = currentCamera.jitteredViewProjection,
        .inverseViewProjection = glm::inverse(currentCamera.jitteredViewProjection),
        .unjitteredViewProjection = currentCamera.unjitteredViewProjection,
        .previousViewProjection = previousCamera.jitteredViewProjection,
        .previousUnjitteredViewProjection = previousCamera.unjitteredViewProjection,
        .cameraPositionExposure = {cameraPosition_, std::exp2(config_.exposureCompensationEv)},
        .directionalDirectionIntensity = {directionalLightTravel, 3.5F},
        .directionalColor = {1.0F, 0.92F, 0.82F,
                             config_.postTestPattern == post::TestPattern::none
                                 ? 0.0F
                                 : config_.postTestPosition.y},
        .pointPositionIntensity = {sceneCenter_ + glm::vec3(-0.60F, 0.45F, 0.85F) * sceneRadius_,
                                   3.0F * sceneRadius_ * sceneRadius_},
        .pointColorRange = {0.45F, 0.65F, 1.0F, sceneRadius_ * 4.0F},
        .environmentParameters =
            {environmentRotationRadians_,
             config_.postTestPattern == post::TestPattern::none ? 0.0F
                                                                : config_.postTestSourceValue,
             config_.postTestPattern == post::TestPattern::movingSquare
                 ? std::min(0.85F, 0.15F + static_cast<float>(renderedFrameCount_) * 0.05F)
                 : (config_.postTestPattern == post::TestPattern::none
                        ? 0.0F
                        : config_.postTestPosition.x),
             config_.postTestPattern == post::TestPattern::movingSquare
                 ? std::min(0.85F, 0.15F + static_cast<float>(renderedFrameCount_ > 0U
                                                                  ? renderedFrameCount_ - 1U
                                                                  : 0U) *
                                               0.05F)
                 : (config_.postTestPattern == post::TestPattern::none
                        ? 0.0F
                        : config_.postTestPosition.x)},
        .cullingParameters = {config_.occlusionCullingEnabled && !cameraMovedThisFrame_ &&
                                      hiZReady_ && temporalState_.previousCameraValid()
                                  ? 1U
                                  : 0U,
                              static_cast<std::uint32_t>(hiZExtents_.size()), extent.width,
                              extent.height},
        .visibilityMask = {visibilityMask, static_cast<std::uint32_t>(config_.postTestPattern),
                           std::bit_cast<std::uint32_t>(config_.bloomIntensity)},
        .temporalJitter = {currentCamera.jitterPixels, previousCamera.jitterPixels},
        .temporalFlags = {config_.taaEnabled && temporalState_.historyValid() ? 1U : 0U,
                          static_cast<std::uint32_t>(config_.temporalDebugView),
                          config_.taaEnabled ? 1U : 0U, config_.bloomEnabled ? 1U : 0U},
        .shadowSceneCenterRadius = shadowCascades[0],
        .shadowCascadeCenterRadius1 = shadowCascades[1],
        .shadowCascadeCenterRadius2 = shadowCascades[2],
        .shadowCascadeSplits = {shadowSplits[0], shadowSplits[1], shadowSplits[2],
                                config_.shadowsEnabled
                                    ? static_cast<float>(directionalShadowCascadeCount)
                                    : 0.0F},
        .shadowCascadeDepthRadii = {shadowDepthRadii[0], shadowDepthRadii[1], shadowDepthRadii[2],
                                    0.0F},
        .previousCameraPosition = {previousCameraPosition, 1.0F},
    };
    uniformBuffers_[frameIndex]->write(std::as_bytes(std::span(&uniform, 1)));
}

void Renderer::recreateSwapchain() {
    const auto [width, height] = window_.waitForDrawableSize();
    (void)window_.consumeFramebufferResize();
    if (width == 0 || height == 0) {
        return;
    }
    vulkan::check(vkDeviceWaitIdle(device_.handle()), "vkDeviceWaitIdle(swapchain recreation)");
    for (VkSemaphore semaphore : renderFinishedSemaphores_) {
        vkDestroySemaphore(device_.handle(), semaphore, nullptr);
    }
    renderFinishedSemaphores_.clear();
    const VkFormat previousFormat = swapchain_.format();
    swapchain_.recreate(width, height);
    renderGraph_.reset();
    createCaptureBuffer();
    createTemporalResources();
    createRenderGraph();
    updateShadowDescriptor();
    updateToneMappingDescriptor();
    updateHiZDescriptors();
    hiZReady_ = false;
    if (swapchain_.format() != previousFormat) {
        createGraphicsPipelines();
    }
    const VkSemaphoreCreateInfo semaphoreInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    renderFinishedSemaphores_.reserve(swapchain_.imageCount());
    for (std::uint32_t index = 0; index < swapchain_.imageCount(); ++index) {
        VkSemaphore semaphore = VK_NULL_HANDLE;
        vulkan::check(vkCreateSemaphore(device_.handle(), &semaphoreInfo, nullptr, &semaphore),
                      "vkCreateSemaphore(render finished after resize)");
        renderFinishedSemaphores_.push_back(semaphore);
    }
    imageInitialized_.assign(swapchain_.imageCount(), false);
}

void Renderer::recordFrame(VkCommandBuffer commandBuffer, std::uint32_t imageIndex,
                           std::size_t frameIndex, bool capture) {
    const VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vulkan::check(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");
    if (timestampQueryPool_ != VK_NULL_HANDLE) {
        constexpr std::uint32_t queriesPerFrame =
            static_cast<std::uint32_t>(gpuTimingPassCount * 2U);
        vkCmdResetQueryPool(commandBuffer, timestampQueryPool_,
                            static_cast<std::uint32_t>(frameIndex) * queriesPerFrame,
                            queriesPerFrame);
    }

    const render_graph::ImageState swapchainState{
        // The acquire semaphore waits at COLOR_ATTACHMENT_OUTPUT. Matching the
        // imported stage orders the presentation engine's read before the
        // graph's layout-transition write under synchronization validation,
        // including the first acquisition whose discardable layout is UNDEFINED.
        .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .access = VK_ACCESS_2_NONE,
        .layout = imageInitialized_[imageIndex] ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                                : VK_IMAGE_LAYOUT_UNDEFINED,
    };
    renderGraph_->bindImportedImage(graphSwapchainImage_, swapchain_.image(imageIndex),
                                    swapchainState, render_graph::QueueType::graphics);
    renderGraph_->bindImportedBuffer(graphUniformBuffer_, uniformBuffers_[frameIndex]->handle(), 0,
                                     uniformBuffers_[frameIndex]->size(),
                                     render_graph::BufferState{
                                         .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                                   VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                         .access = VK_ACCESS_2_UNIFORM_READ_BIT,
                                     },
                                     render_graph::QueueType::graphics);
    const render_graph::BufferState frustumVisibleState =
        hiZReady_ ? render_graph::BufferState{
                         .stages = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                     }
                   : render_graph::BufferState{};
    const render_graph::BufferState frustumIndirectState = frustumVisibleState;
    const render_graph::BufferState frustumCountState = frustumVisibleState;
    const render_graph::BufferState visibleState =
        hiZReady_ ? render_graph::BufferState{
                         .stages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                         .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                     }
                   : render_graph::BufferState{};
    const render_graph::BufferState indirectState =
        hiZReady_ ? render_graph::BufferState{
                         .stages = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                         .access = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
                     }
                   : render_graph::BufferState{};
    renderGraph_->bindImportedBuffer(graphFrustumVisibleInstanceBuffer_,
                                     frustumVisibleInstanceBuffer_->handle(), 0,
                                     frustumVisibleInstanceBuffer_->size(), frustumVisibleState,
                                     render_graph::QueueType::graphics);
    renderGraph_->bindImportedBuffer(graphFrustumIndirectDrawBuffer_,
                                     frustumIndirectDrawBuffer_->handle(), 0,
                                     frustumIndirectDrawBuffer_->size(), frustumIndirectState,
                                     render_graph::QueueType::graphics);
    renderGraph_->bindImportedBuffer(
        graphFrustumDrawCountBuffer_, frustumDrawCountBuffer_->handle(), 0,
        frustumDrawCountBuffer_->size(), frustumCountState, render_graph::QueueType::graphics);
    renderGraph_->bindImportedBuffer(graphVisibleInstanceBuffer_, visibleInstanceBuffer_->handle(),
                                     0, visibleInstanceBuffer_->size(), visibleState,
                                     render_graph::QueueType::graphics);
    renderGraph_->bindImportedBuffer(graphIndirectDrawBuffer_, indirectDrawBuffer_->handle(), 0,
                                     indirectDrawBuffer_->size(), indirectState,
                                     render_graph::QueueType::graphics);
    renderGraph_->bindImportedBuffer(graphDrawCountBuffer_, drawCountBuffer_->handle(), 0,
                                     drawCountBuffer_->size(), indirectState,
                                     render_graph::QueueType::graphics);
    const render_graph::BufferState shadowVisibleState =
        hiZReady_ && config_.shadowsEnabled
            ? render_graph::BufferState{
                  .stages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                  .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
              }
            : render_graph::BufferState{};
    const render_graph::BufferState shadowIndirectState =
        hiZReady_ && config_.shadowsEnabled ? indirectState : render_graph::BufferState{};
    for (std::size_t cascadeIndex = 0; cascadeIndex < directionalShadowCascadeCount;
         ++cascadeIndex) {
        renderGraph_->bindImportedBuffer(graphShadowVisibleInstanceBuffers_[cascadeIndex],
                                         shadowVisibleInstanceBuffers_[cascadeIndex]->handle(), 0,
                                         shadowVisibleInstanceBuffers_[cascadeIndex]->size(),
                                         shadowVisibleState, render_graph::QueueType::graphics);
        renderGraph_->bindImportedBuffer(graphShadowIndirectDrawBuffers_[cascadeIndex],
                                         shadowIndirectDrawBuffers_[cascadeIndex]->handle(), 0,
                                         shadowIndirectDrawBuffers_[cascadeIndex]->size(),
                                         shadowIndirectState, render_graph::QueueType::graphics);
        renderGraph_->bindImportedBuffer(graphShadowDrawCountBuffers_[cascadeIndex],
                                         shadowDrawCountBuffers_[cascadeIndex]->handle(), 0,
                                         shadowDrawCountBuffers_[cascadeIndex]->size(),
                                         shadowIndirectState, render_graph::QueueType::graphics);
    }
    const std::size_t historyWriteIndex = frameIndex;
    const std::size_t historyReadIndex = 1U - historyWriteIndex;
    if (config_.taaEnabled) {
        const render_graph::ImageState historyInitialState =
            temporalImageLayoutsInitialized_
                ? render_graph::ImageState{
                      .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                      .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                      .layout = VK_IMAGE_LAYOUT_GENERAL,
                  }
                : render_graph::ImageState{};
        renderGraph_->bindImportedImage(graphHistoryColorReadImage_,
                                        historyColorImages_[historyReadIndex]->handle(),
                                        historyInitialState, render_graph::QueueType::graphics);
        renderGraph_->bindImportedImage(graphHistoryColorWriteImage_,
                                        historyColorImages_[historyWriteIndex]->handle(),
                                        historyInitialState, render_graph::QueueType::graphics);
        renderGraph_->bindImportedImage(graphHistoryDepthReadImage_,
                                        historyDepthImages_[historyReadIndex]->handle(),
                                        historyInitialState, render_graph::QueueType::graphics);
        renderGraph_->bindImportedImage(graphHistoryDepthWriteImage_,
                                        historyDepthImages_[historyWriteIndex]->handle(),
                                        historyInitialState, render_graph::QueueType::graphics);
        renderGraph_->bindImportedImage(graphHistoryNormalReadImage_,
                                        historyNormalImages_[historyReadIndex]->handle(),
                                        historyInitialState, render_graph::QueueType::graphics);
        renderGraph_->bindImportedImage(graphHistoryNormalWriteImage_,
                                        historyNormalImages_[historyWriteIndex]->handle(),
                                        historyInitialState, render_graph::QueueType::graphics);
    }
    if (captureReadbackBuffer_) {
        renderGraph_->bindImportedBuffer(graphCaptureBuffer_, captureReadbackBuffer_->handle(), 0,
                                         captureReadbackBuffer_->size(), {},
                                         render_graph::QueueType::graphics);
    }
    const GraphFrameContext context{
        .imageIndex = imageIndex,
        .frameIndex = frameIndex,
        .historyReadIndex = historyReadIndex,
        .historyWriteIndex = historyWriteIndex,
        .capture = capture,
    };
    renderGraph_->execute(commandBuffer, render_graph::QueueType::graphics, &context);
    hiZReady_ = true;

    vulkan::check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
}

void Renderer::recordCullingPass(VkCommandBuffer commandBuffer, const GraphFrameContext& context) {
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      frustumCullPipeline_->handle());
    const VkDescriptorSet descriptorSet = cullingDescriptorPool_->set(context.frameIndex);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            frustumCullPipeline_->layout(), 0, 1, &descriptorSet, 0, nullptr);
    constexpr std::uint32_t workgroupSize = 64;
    const std::uint32_t workgroupCount = (sceneWorkItemCount_ + workgroupSize - 1U) / workgroupSize;
    vkCmdDispatch(commandBuffer, workgroupCount, 1, 1);
}

void Renderer::recordShadowCullingPass(VkCommandBuffer commandBuffer,
                                       const GraphFrameContext& context, std::size_t cascadeIndex) {
    if (cascadeIndex >= directionalShadowCascadeCount) {
        throw std::runtime_error("Directional-shadow cascade index is out of range");
    }
    if (cascadeIndex == 0) {
        writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::shadowCulling, false);
    }
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, shadowCullPipeline_->handle());
    const std::size_t shadowSetIndex = cascadeIndex * framesInFlight + context.frameIndex;
    const VkDescriptorSet descriptorSet =
        cullingDescriptorPool_->set(framesInFlight + shadowSetIndex);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            shadowCullPipeline_->layout(), 0, 1, &descriptorSet, 0, nullptr);
    const std::uint32_t pushedCascadeIndex = static_cast<std::uint32_t>(cascadeIndex);
    vkCmdPushConstants(commandBuffer, shadowCullPipeline_->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(pushedCascadeIndex), &pushedCascadeIndex);
    constexpr std::uint32_t workgroupSize = 64;
    const std::uint32_t workgroupCount = (sceneWorkItemCount_ + workgroupSize - 1U) / workgroupSize;
    vkCmdDispatch(commandBuffer, workgroupCount, 1, 1);
    if (cascadeIndex + 1U == directionalShadowCascadeCount) {
        writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::shadowCulling, true);
    }
}

void Renderer::recordShadowPass(VkCommandBuffer commandBuffer,
                                const render_graph::RenderGraph& graph,
                                const GraphFrameContext& context, std::size_t cascadeIndex,
                                render_graph::ImageHandle shadowImage) {
    if (cascadeIndex >= directionalShadowCascadeCount) {
        throw std::runtime_error("Directional-shadow cascade index is out of range");
    }
    const VkClearValue clearDepth{.depthStencil = {.depth = 1.0F, .stencil = 0}};
    const VkRenderingAttachmentInfo depthAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = graph.imageView(shadowImage),
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = clearDepth,
    };
    const std::uint32_t size = config_.shadowsEnabled ? directionalShadowResolution : 1U;
    const VkExtent2D extent{size, size};
    const VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {.offset = {0, 0}, .extent = extent},
        .layerCount = 1,
        .pDepthAttachment = &depthAttachment,
    };
    if (config_.shadowsEnabled && cascadeIndex == 0) {
        writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::shadowMap, false);
    }
    vkCmdBeginRendering(commandBuffer, &renderingInfo);
    if (!config_.shadowsEnabled) {
        vkCmdEndRendering(commandBuffer);
        return;
    }
    const VkViewport viewport{.x = 0.0F,
                              .y = 0.0F,
                              .width = static_cast<float>(extent.width),
                              .height = static_cast<float>(extent.height),
                              .minDepth = 0.0F,
                              .maxDepth = 1.0F};
    const VkRect2D scissor{.offset = {0, 0}, .extent = extent};
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline_->handle());
    const std::size_t shadowSetIndex = cascadeIndex * framesInFlight + context.frameIndex;
    const std::array descriptorSets{shadowDescriptorPool_->set(shadowSetIndex),
                                    bindlessDescriptorPool_->set(0)};
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline_->layout(), 0,
        static_cast<std::uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, nullptr);
    const std::uint32_t pushedCascadeIndex = static_cast<std::uint32_t>(cascadeIndex);
    vkCmdPushConstants(commandBuffer, shadowPipeline_->layout(), VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(pushedCascadeIndex), &pushedCascadeIndex);
    const VkBuffer vertexBuffer = vertexBuffer_->handle();
    constexpr VkDeviceSize vertexOffset = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &vertexOffset);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer_->handle(), 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexedIndirectCount(commandBuffer, shadowIndirectDrawBuffers_[cascadeIndex]->handle(),
                                  0, shadowDrawCountBuffers_[cascadeIndex]->handle(), 0,
                                  sceneWorkItemCount_, sizeof(VkDrawIndexedIndirectCommand));
    vkCmdEndRendering(commandBuffer);
    if (cascadeIndex + 1U == directionalShadowCascadeCount) {
        writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::shadowMap, true);
    }
}

void Renderer::recordOcclusionCullingPass(VkCommandBuffer commandBuffer,
                                          const GraphFrameContext& context) {
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      occlusionCullPipeline_->handle());
    const VkDescriptorSet descriptorSet = occlusionDescriptorPool_->set(context.frameIndex);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            occlusionCullPipeline_->layout(), 0, 1, &descriptorSet, 0, nullptr);
    constexpr std::uint32_t workgroupSize = 64;
    const std::uint32_t workgroupCount = (sceneWorkItemCount_ + workgroupSize - 1U) / workgroupSize;
    vkCmdDispatch(commandBuffer, workgroupCount, 1, 1);
}

void Renderer::recordHiZPass(VkCommandBuffer commandBuffer, std::size_t level) {
    if (level >= hiZExtents_.size()) {
        throw std::runtime_error("Hi-Z pass level is out of range");
    }
    const vulkan::ComputePipeline& pipeline = level == 0 ? *hiZCopyPipeline_ : *hiZReducePipeline_;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.handle());
    const VkDescriptorSet descriptorSet = hiZDescriptorPool_->set(level);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.layout(), 0, 1,
                            &descriptorSet, 0, nullptr);
    constexpr std::uint32_t workgroupSize = 8;
    const VkExtent2D extent = hiZExtents_[level];
    vkCmdDispatch(commandBuffer, (extent.width + workgroupSize - 1U) / workgroupSize,
                  (extent.height + workgroupSize - 1U) / workgroupSize, 1);
}

void Renderer::recordPbrPass(VkCommandBuffer commandBuffer, const render_graph::RenderGraph& graph,
                             const GraphFrameContext& context) {
    const std::size_t frameIndex = context.frameIndex;

    const VkClearValue clearColor{{{0.002F, 0.004F, 0.008F, 1.0F}}};
    const VkClearValue clearMotion{{{0.0F, 0.0F, 1.0F, 0.0F}}};
    const VkClearValue clearNormal{{{0.5F, 0.5F, 1.0F, 0.0F}}};
    const std::array colorAttachments{
        VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = graph.imageView(graphHdrImage_),
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = clearColor,
        },
        VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = graph.imageView(graphMotionImage_),
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = clearMotion,
        },
        VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = graph.imageView(graphNormalImage_),
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = clearNormal,
        },
    };
    const VkClearValue clearDepth{.depthStencil = {.depth = 1.0F, .stencil = 0}};
    const VkRenderingAttachmentInfo depthAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = graph.imageView(graphDepthImage_),
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = clearDepth,
    };
    const VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {.offset = {0, 0}, .extent = swapchain_.extent()},
        .layerCount = 1,
        .colorAttachmentCount = static_cast<std::uint32_t>(colorAttachments.size()),
        .pColorAttachments = colorAttachments.data(),
        .pDepthAttachment = &depthAttachment,
    };
    vkCmdBeginRendering(commandBuffer, &renderingInfo);
    const VkViewport viewport{
        .x = 0.0F,
        .y = 0.0F,
        .width = static_cast<float>(swapchain_.extent().width),
        .height = static_cast<float>(swapchain_.extent().height),
        .minDepth = 0.0F,
        .maxDepth = 1.0F,
    };
    const VkRect2D scissor{.offset = {0, 0}, .extent = swapchain_.extent()};
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    const std::array descriptorSets{descriptorPool_->set(frameIndex),
                                    bindlessDescriptorPool_->set(0)};
    if (config_.postTestPattern != post::TestPattern::none) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          postTestPipeline_->handle());
        const VkDescriptorSet descriptorSet = descriptorPool_->set(frameIndex);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                postTestPipeline_->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        vkCmdEndRendering(commandBuffer);
        return;
    }
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline_->handle());
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline_->layout(),
                            0, static_cast<std::uint32_t>(descriptorSets.size()),
                            descriptorSets.data(), 0, nullptr);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_->handle());
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_->layout(), 0,
        static_cast<std::uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, nullptr);
    const VkBuffer vertexBuffer = vertexBuffer_->handle();
    constexpr VkDeviceSize vertexOffset = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &vertexOffset);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer_->handle(), 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexedIndirectCount(commandBuffer, indirectDrawBuffer_->handle(), 0,
                                  drawCountBuffer_->handle(), 0, sceneWorkItemCount_,
                                  sizeof(VkDrawIndexedIndirectCommand));
    vkCmdEndRendering(commandBuffer);
}

void Renderer::recordTaaPass(VkCommandBuffer commandBuffer, const GraphFrameContext& context) {
    if (!config_.taaEnabled || context.historyWriteIndex >= historyColorViews_.size() ||
        context.historyWriteIndex >= historyNormalViews_.size()) {
        throw std::runtime_error("Temporal resolve has invalid history state");
    }
    const std::array attachments{
        VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = historyColorViews_[context.historyWriteIndex]->handle(),
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        },
        VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = historyDepthViews_[context.historyWriteIndex]->handle(),
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        },
        VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = historyNormalViews_[context.historyWriteIndex]->handle(),
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        },
    };
    const VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {.offset = {0, 0}, .extent = swapchain_.extent()},
        .layerCount = 1,
        .colorAttachmentCount = static_cast<std::uint32_t>(attachments.size()),
        .pColorAttachments = attachments.data(),
    };
    vkCmdBeginRendering(commandBuffer, &renderingInfo);
    const VkViewport viewport{
        .x = 0.0F,
        .y = 0.0F,
        .width = static_cast<float>(swapchain_.extent().width),
        .height = static_cast<float>(swapchain_.extent().height),
        .minDepth = 0.0F,
        .maxDepth = 1.0F,
    };
    const VkRect2D scissor{.offset = {0, 0}, .extent = swapchain_.extent()};
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, taaPipeline_->handle());
    const VkDescriptorSet descriptorSet = taaDescriptorPool_->set(context.frameIndex);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, taaPipeline_->layout(),
                            0, 1, &descriptorSet, 0, nullptr);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    vkCmdEndRendering(commandBuffer);
}

void Renderer::recordBloomPass(VkCommandBuffer commandBuffer,
                               const render_graph::RenderGraph& graph,
                               render_graph::ImageHandle target,
                               const vulkan::GraphicsPipeline& pipeline,
                               std::size_t descriptorSetIndex, std::size_t frameIndex,
                               GpuTimingPass timingPass) {
    writeGpuTimestamp(commandBuffer, frameIndex, timingPass, false);
    const VkExtent2D extent{
        .width = post::halfExtent(swapchain_.extent().width),
        .height = post::halfExtent(swapchain_.extent().height),
    };
    const VkRenderingAttachmentInfo colorAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = graph.imageView(target),
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    const VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {.offset = {0, 0}, .extent = extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
    };
    vkCmdBeginRendering(commandBuffer, &renderingInfo);
    const VkViewport viewport{
        .x = 0.0F,
        .y = 0.0F,
        .width = static_cast<float>(extent.width),
        .height = static_cast<float>(extent.height),
        .minDepth = 0.0F,
        .maxDepth = 1.0F,
    };
    const VkRect2D scissor{.offset = {0, 0}, .extent = extent};
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.handle());
    const VkDescriptorSet descriptorSet = bloomDescriptorPool_->set(descriptorSetIndex);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout(), 0, 1,
                            &descriptorSet, 0, nullptr);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    vkCmdEndRendering(commandBuffer);
    writeGpuTimestamp(commandBuffer, frameIndex, timingPass, true);
}

void Renderer::recordToneMappingPass(VkCommandBuffer commandBuffer,
                                     const render_graph::RenderGraph&,
                                     const GraphFrameContext& context) {
    writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::toneMapping, false);
    const VkRenderingAttachmentInfo colorAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = swapchain_.imageView(context.imageIndex),
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    };
    const VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {.offset = {0, 0}, .extent = swapchain_.extent()},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
    };
    vkCmdBeginRendering(commandBuffer, &renderingInfo);
    const VkViewport viewport{
        .x = 0.0F,
        .y = 0.0F,
        .width = static_cast<float>(swapchain_.extent().width),
        .height = static_cast<float>(swapchain_.extent().height),
        .minDepth = 0.0F,
        .maxDepth = 1.0F,
    };
    const VkRect2D scissor{.offset = {0, 0}, .extent = swapchain_.extent()};
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      toneMappingPipeline_->handle());
    const VkDescriptorSet descriptorSet = toneMappingDescriptorPool_->set(context.frameIndex);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            toneMappingPipeline_->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    vkCmdEndRendering(commandBuffer);
    writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::toneMapping, true);
}

void Renderer::recordCapturePass(VkCommandBuffer commandBuffer,
                                 const render_graph::RenderGraph& graph) {
    const VkBufferImageCopy copy{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource =
            {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {swapchain_.extent().width, swapchain_.extent().height, 1},
    };
    vkCmdCopyImageToBuffer(commandBuffer, graph.image(graphSwapchainImage_),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, graph.buffer(graphCaptureBuffer_),
                           1, &copy);
}

void Renderer::writeCapture() const {
    if (!captureReadbackBuffer_ || captureReadbackBuffer_->mappedData() == nullptr) {
        throw std::runtime_error("Capture readback buffer is not host visible");
    }
    captureReadbackBuffer_->invalidate();
    const VkExtent2D extent = swapchain_.extent();
    const std::size_t byteCount = static_cast<std::size_t>(extent.width) * extent.height * 4U;
    const auto* source = static_cast<const std::uint8_t*>(captureReadbackBuffer_->mappedData());
    std::vector<std::uint8_t> rgba(source, source + byteCount);
    if (swapchain_.format() == VK_FORMAT_B8G8R8A8_SRGB ||
        swapchain_.format() == VK_FORMAT_B8G8R8A8_UNORM) {
        for (std::size_t offset = 0; offset < rgba.size(); offset += 4U) {
            std::swap(rgba[offset], rgba[offset + 2U]);
        }
    } else if (swapchain_.format() != VK_FORMAT_R8G8B8A8_SRGB &&
               swapchain_.format() != VK_FORMAT_R8G8B8A8_UNORM) {
        throw std::runtime_error("Capture does not support the selected swapchain format");
    }
    assets::writeRgba8Png(config_.capturePath, extent.width, extent.height, rgba);
    core::log(core::LogLevel::info, "Captured frame ", config_.captureFrame, " to ",
              config_.capturePath.string());
}

} // namespace renderer
