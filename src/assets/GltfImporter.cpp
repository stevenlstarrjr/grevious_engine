#include "renderer/assets/GltfImporter.hpp"

#include "renderer/assets/TangentGenerator.hpp"

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace renderer::assets {
namespace {

namespace fs = std::filesystem;
namespace rs = renderer::scene;

constexpr auto supportedExtensions =
    fastgltf::Extensions::KHR_mesh_quantization | fastgltf::Extensions::KHR_lights_punctual |
    fastgltf::Extensions::KHR_materials_emissive_strength;
constexpr std::array<std::string_view, 3> supportedExtensionNames = {
    "KHR_mesh_quantization",
    "KHR_lights_punctual",
    "KHR_materials_emissive_strength",
};

[[noreturn]] void fail(const fs::path& path, std::string message) {
    throw GltfImportError("glTF import failed for '" + path.string() + "': " + std::move(message));
}

std::string copyString(std::string_view value) { return {value.data(), value.size()}; }

std::string lowerExtension(const fs::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension;
}

template <typename Value>
void requireAtMost(const fs::path& path, Value value, Value maximum, std::string_view label) {
    if (value > maximum) {
        fail(path, std::string(label) + " exceeds the configured limit (" + std::to_string(value) +
                       " > " + std::to_string(maximum) + ')');
    }
}

template <typename Value>
void addBounded(const fs::path& path, Value& total, Value value, Value maximum,
                std::string_view label) {
    if (value > maximum || total > maximum - value) {
        fail(path, std::string(label) + " exceeds the configured aggregate limit");
    }
    total += value;
}

bool supportedExtension(std::string_view extension) {
    return std::find(supportedExtensionNames.begin(), supportedExtensionNames.end(), extension) !=
           supportedExtensionNames.end();
}

fs::path resolveExternalPath(const fs::path& sourcePath, const fastgltf::URI& uri,
                             const GltfImportOptions& options) {
    if (!uri.valid() || !uri.isLocalPath() || uri.isDataUri()) {
        fail(sourcePath, "external resources must use valid local relative paths");
    }

    const fs::path referenced = uri.fspath();
    if (referenced.is_absolute() && !options.allowExternalFilesOutsideSourceDirectory) {
        fail(sourcePath, "absolute external resource path is disallowed: " + referenced.string());
    }

    std::error_code error;
    const fs::path sourceDirectory = fs::weakly_canonical(sourcePath.parent_path(), error);
    if (error) {
        fail(sourcePath, "could not resolve the source directory: " + error.message());
    }
    const fs::path candidate = fs::weakly_canonical(
        referenced.is_absolute() ? referenced : sourceDirectory / referenced, error);
    if (error) {
        fail(sourcePath, "could not resolve external resource '" + referenced.string() +
                             "': " + error.message());
    }

    if (!options.allowExternalFilesOutsideSourceDirectory) {
        const fs::path relative = candidate.lexically_relative(sourceDirectory);
        if (relative.empty() || relative.is_absolute() ||
            (!relative.empty() && *relative.begin() == "..")) {
            fail(sourcePath,
                 "external resource escapes the source directory: " + referenced.string());
        }
    }
    return candidate;
}

std::uintmax_t externalFileBytes(const fs::path& sourcePath, const fastgltf::sources::URI& source,
                                 const GltfImportOptions& options) {
    const fs::path resolved = resolveExternalPath(sourcePath, source.uri, options);
    std::error_code error;
    if (!fs::is_regular_file(resolved, error) || error) {
        fail(sourcePath, "external resource is not a regular file: " + resolved.string());
    }
    const std::uintmax_t size = fs::file_size(resolved, error);
    if (error) {
        fail(sourcePath,
             "could not inspect external resource '" + resolved.string() + "': " + error.message());
    }
    if (source.fileByteOffset > size) {
        fail(sourcePath,
             "external resource byte offset exceeds its file size: " + resolved.string());
    }
    requireAtMost(sourcePath, size - source.fileByteOffset, options.limits.maxExternalFileBytes,
                  "external resource size");
    return size - source.fileByteOffset;
}

fastgltf::Asset parseAsset(const fs::path& sourcePath, fastgltf::Options parseOptions) {
    auto mapped = fastgltf::MappedGltfFile::FromPath(sourcePath);
    if (mapped.error() != fastgltf::Error::None) {
        fail(sourcePath, "could not map source file (" +
                             copyString(fastgltf::getErrorName(mapped.error())) + ')');
    }

    fastgltf::Parser parser(supportedExtensions);
    auto loaded = parser.loadGltf(mapped.get(), sourcePath.parent_path(), parseOptions);
    if (loaded.error() != fastgltf::Error::None) {
        fail(sourcePath, copyString(fastgltf::getErrorMessage(loaded.error())) + " [" +
                             copyString(fastgltf::getErrorName(loaded.error())) + ']');
    }
    fastgltf::Asset asset = std::move(loaded).operator*();
    const fastgltf::Error validationError = fastgltf::validate(asset);
    if (validationError != fastgltf::Error::None) {
        fail(sourcePath,
             "asset validation failed: " + copyString(fastgltf::getErrorMessage(validationError)) +
                 " [" + copyString(fastgltf::getErrorName(validationError)) + ']');
    }
    return asset;
}

void verifyExtensionPolicy(const fs::path& sourcePath, const fastgltf::Asset& asset) {
    for (const auto& extension : asset.extensionsUsed) {
        if (!supportedExtension(extension)) {
            fail(sourcePath, "unsupported extension '" + copyString(extension) +
                                 "'; the importer never silently discards extension data");
        }
    }
}

void verifyFeaturePolicy(const fs::path& sourcePath, const fastgltf::Asset& asset) {
    if (!asset.animations.empty()) {
        fail(sourcePath, "animations are not supported by the source Scene IR yet");
    }
    if (!asset.skins.empty()) {
        fail(sourcePath, "skins are not supported by the source Scene IR yet");
    }
    if (!asset.materialVariants.empty()) {
        fail(sourcePath, "material variants are not supported yet");
    }

    for (std::size_t meshIndex = 0; meshIndex < asset.meshes.size(); ++meshIndex) {
        const auto& mesh = asset.meshes[meshIndex];
        if (!mesh.weights.empty()) {
            fail(sourcePath,
                 "mesh[" + std::to_string(meshIndex) + "] contains unsupported morph weights");
        }
        for (std::size_t primitiveIndex = 0; primitiveIndex < mesh.primitives.size();
             ++primitiveIndex) {
            const auto& primitive = mesh.primitives[primitiveIndex];
            const std::string prefix = "mesh[" + std::to_string(meshIndex) + "].primitive[" +
                                       std::to_string(primitiveIndex) + ']';
            if (!primitive.targets.empty()) {
                fail(sourcePath, prefix + " contains unsupported morph targets");
            }
            if (primitive.dracoCompression) {
                fail(sourcePath, prefix + " uses unsupported Draco compression");
            }
            if (!primitive.mappings.empty()) {
                fail(sourcePath, prefix + " contains unsupported material-variant mappings");
            }
            for (const auto& attribute : primitive.attributes) {
                constexpr std::array<std::string_view, 6> supportedAttributes = {
                    "POSITION", "NORMAL", "TANGENT", "TEXCOORD_0", "TEXCOORD_1", "COLOR_0",
                };
                if (std::find(supportedAttributes.begin(), supportedAttributes.end(),
                              attribute.name) == supportedAttributes.end()) {
                    fail(sourcePath, prefix + " contains unsupported vertex attribute '" +
                                         copyString(attribute.name) + '\'');
                }
            }
        }
    }

    for (std::size_t nodeIndex = 0; nodeIndex < asset.nodes.size(); ++nodeIndex) {
        const auto& node = asset.nodes[nodeIndex];
        if (node.skinIndex.has_value() || !node.weights.empty()) {
            fail(sourcePath, "node[" + std::to_string(nodeIndex) +
                                 "] contains unsupported skin or morph state");
        }
        if (!node.instancingAttributes.empty()) {
            fail(sourcePath,
                 "node[" + std::to_string(nodeIndex) + "] uses unsupported GPU instancing data");
        }
    }
}

void verifyLimits(const fs::path& sourcePath, const fastgltf::Asset& asset,
                  const GltfImportOptions& options, bool inspectExternalFiles) {
    const GltfImportLimits& limits = options.limits;
    requireAtMost(sourcePath, asset.accessors.size(), limits.maxAccessors, "accessor count");
    requireAtMost(sourcePath, asset.images.size(), limits.maxImages, "image count");
    requireAtMost(sourcePath, asset.materials.size(), limits.maxMaterials, "material count");
    requireAtMost(sourcePath, asset.meshes.size(), limits.maxMeshes, "mesh count");
    requireAtMost(sourcePath, asset.nodes.size(), limits.maxNodes, "node count");

    std::size_t primitiveCount = 0;
    std::size_t vertexCount = 0;
    std::size_t indexCount = 0;
    for (const auto& mesh : asset.meshes) {
        addBounded(sourcePath, primitiveCount, mesh.primitives.size(), limits.maxPrimitives,
                   "primitive count");
        for (const auto& primitive : mesh.primitives) {
            const auto position = primitive.findAttribute("POSITION");
            if (position != primitive.attributes.end()) {
                addBounded(sourcePath, vertexCount, asset.accessors[position->accessorIndex].count,
                           limits.maxVertices, "vertex count");
            }
            if (primitive.indicesAccessor.has_value()) {
                addBounded(sourcePath, indexCount,
                           asset.accessors[*primitive.indicesAccessor].count, limits.maxIndices,
                           "index count");
            }
        }
    }

    std::uintmax_t totalBufferBytes = 0;
    for (const auto& buffer : asset.buffers) {
        addBounded(sourcePath, totalBufferBytes, static_cast<std::uintmax_t>(buffer.byteLength),
                   limits.maxTotalBufferBytes, "buffer bytes");
        if (inspectExternalFiles) {
            if (const auto* uri = std::get_if<fastgltf::sources::URI>(&buffer.data)) {
                const std::uintmax_t available = externalFileBytes(sourcePath, *uri, options);
                if (buffer.byteLength > available) {
                    fail(sourcePath, "external buffer is shorter than its declared byteLength");
                }
            }
        }
    }

    std::uintmax_t totalImageBytes = 0;
    for (const auto& image : asset.images) {
        const std::uintmax_t bytes = std::visit(
            fastgltf::visitor{
                [&](const fastgltf::sources::URI& uri) -> std::uintmax_t {
                    return inspectExternalFiles ? externalFileBytes(sourcePath, uri, options) : 0;
                },
                [](const fastgltf::sources::Array& array) -> std::uintmax_t {
                    return array.bytes.size();
                },
                [](const fastgltf::sources::Vector& vector) -> std::uintmax_t {
                    return vector.bytes.size();
                },
                [](const fastgltf::sources::ByteView& view) -> std::uintmax_t {
                    return view.bytes.size();
                },
                [&](const fastgltf::sources::BufferView& view) -> std::uintmax_t {
                    return asset.bufferViews[view.bufferViewIndex].byteLength;
                },
                [](const auto&) -> std::uintmax_t { return 0; },
            },
            image.data);
        addBounded(sourcePath, totalImageBytes, bytes, limits.maxTotalImageBytes, "image bytes");
    }
}

std::span<const std::byte> bufferBytes(const fs::path& sourcePath, const fastgltf::Buffer& buffer) {
    return std::visit(
        fastgltf::visitor{
            [](const fastgltf::sources::Array& array) -> std::span<const std::byte> {
                return {array.bytes.data(), array.bytes.size()};
            },
            [](const fastgltf::sources::Vector& vector) -> std::span<const std::byte> {
                return vector.bytes;
            },
            [](const fastgltf::sources::ByteView& view) -> std::span<const std::byte> {
                return {view.bytes.data(), view.bytes.size()};
            },
            [&](const auto&) -> std::span<const std::byte> {
                fail(sourcePath, "parser did not materialize a referenced buffer");
            },
        },
        buffer.data);
}

std::string mimeTypeName(fastgltf::MimeType mimeType) {
    switch (mimeType) {
    case fastgltf::MimeType::None:
        return {};
    case fastgltf::MimeType::JPEG:
        return "image/jpeg";
    case fastgltf::MimeType::PNG:
        return "image/png";
    case fastgltf::MimeType::KTX2:
        return "image/ktx2";
    case fastgltf::MimeType::DDS:
        return "image/vnd-ms.dds";
    case fastgltf::MimeType::GltfBuffer:
        return "model/gltf-buffer";
    case fastgltf::MimeType::OctetStream:
        return "application/octet-stream";
    case fastgltf::MimeType::WEBP:
        return "image/webp";
    }
    return {};
}

rs::Image importImage(const fs::path& sourcePath, const fastgltf::Asset& asset,
                      const fastgltf::Image& image) {
    rs::Image imported;
    imported.name = copyString(image.name);
    std::visit(fastgltf::visitor{
                   [&](const fastgltf::sources::URI& uri) {
                       imported.sourceUri = copyString(uri.uri.string());
                       imported.mimeType = mimeTypeName(uri.mimeType);
                   },
                   [&](const fastgltf::sources::Array& array) {
                       imported.encodedData.assign(array.bytes.data(),
                                                   array.bytes.data() + array.bytes.size());
                       imported.mimeType = mimeTypeName(array.mimeType);
                   },
                   [&](const fastgltf::sources::Vector& vector) {
                       imported.encodedData = vector.bytes;
                       imported.mimeType = mimeTypeName(vector.mimeType);
                   },
                   [&](const fastgltf::sources::ByteView& view) {
                       imported.encodedData.assign(view.bytes.begin(), view.bytes.end());
                       imported.mimeType = mimeTypeName(view.mimeType);
                   },
                   [&](const fastgltf::sources::BufferView& source) {
                       const auto& view = asset.bufferViews[source.bufferViewIndex];
                       const std::span<const std::byte> bytes =
                           bufferBytes(sourcePath, asset.buffers[view.bufferIndex]);
                       if (view.byteOffset > bytes.size() ||
                           view.byteLength > bytes.size() - view.byteOffset) {
                           fail(sourcePath, "image bufferView lies outside its loaded buffer");
                       }
                       imported.encodedData.assign(
                           bytes.begin() + static_cast<std::ptrdiff_t>(view.byteOffset),
                           bytes.begin() +
                               static_cast<std::ptrdiff_t>(view.byteOffset + view.byteLength));
                       imported.mimeType = mimeTypeName(source.mimeType);
                   },
                   [&](const auto&) {
                       fail(sourcePath, "image data source is unsupported or unavailable");
                   },
               },
               image.data);
    return imported;
}

rs::WrapMode convertWrap(fastgltf::Wrap wrap) {
    switch (wrap) {
    case fastgltf::Wrap::ClampToEdge:
        return rs::WrapMode::clampToEdge;
    case fastgltf::Wrap::MirroredRepeat:
        return rs::WrapMode::mirroredRepeat;
    case fastgltf::Wrap::Repeat:
        return rs::WrapMode::repeat;
    }
    return rs::WrapMode::repeat;
}

void applyMinFilter(fastgltf::Filter source, rs::Sampler& sampler) {
    switch (source) {
    case fastgltf::Filter::Nearest:
        sampler.minFilter = rs::Filter::nearest;
        sampler.mipFilter = rs::MipFilter::none;
        break;
    case fastgltf::Filter::Linear:
        sampler.minFilter = rs::Filter::linear;
        sampler.mipFilter = rs::MipFilter::none;
        break;
    case fastgltf::Filter::NearestMipMapNearest:
        sampler.minFilter = rs::Filter::nearest;
        sampler.mipFilter = rs::MipFilter::nearest;
        break;
    case fastgltf::Filter::LinearMipMapNearest:
        sampler.minFilter = rs::Filter::linear;
        sampler.mipFilter = rs::MipFilter::nearest;
        break;
    case fastgltf::Filter::NearestMipMapLinear:
        sampler.minFilter = rs::Filter::nearest;
        sampler.mipFilter = rs::MipFilter::linear;
        break;
    case fastgltf::Filter::LinearMipMapLinear:
        sampler.minFilter = rs::Filter::linear;
        sampler.mipFilter = rs::MipFilter::linear;
        break;
    }
}

rs::Sampler importSampler(const fastgltf::Sampler& sampler) {
    rs::Sampler imported{.name = copyString(sampler.name)};
    if (sampler.minFilter.has_value()) {
        applyMinFilter(*sampler.minFilter, imported);
    }
    if (sampler.magFilter.has_value()) {
        imported.magFilter = *sampler.magFilter == fastgltf::Filter::Nearest ? rs::Filter::nearest
                                                                             : rs::Filter::linear;
    }
    imported.wrapU = convertWrap(sampler.wrapS);
    imported.wrapV = convertWrap(sampler.wrapT);
    return imported;
}

template <typename OptionalTextureInfo>
rs::TextureBinding importBinding(const fs::path& sourcePath, const OptionalTextureInfo& binding,
                                 const std::vector<rs::TextureHandle>& textures) {
    if (!binding.has_value()) {
        return {};
    }
    if (binding->textureIndex >= textures.size()) {
        fail(sourcePath, "material texture index is outside the texture array");
    }
    if (binding->texCoordIndex >= 2) {
        fail(sourcePath, "only TEXCOORD_0 and TEXCOORD_1 are supported");
    }
    return {.texture = textures[binding->textureIndex],
            .texCoord = static_cast<std::uint32_t>(binding->texCoordIndex)};
}

rs::MaterialQueue convertMaterialQueue(fastgltf::AlphaMode mode) {
    switch (mode) {
    case fastgltf::AlphaMode::Opaque:
        return rs::MaterialQueue::opaque;
    case fastgltf::AlphaMode::Mask:
        return rs::MaterialQueue::alphaMasked;
    case fastgltf::AlphaMode::Blend:
        return rs::MaterialQueue::transparent;
    }
    return rs::MaterialQueue::opaque;
}

rs::Material importMaterial(const fs::path& sourcePath, const fastgltf::Material& material,
                            const std::vector<rs::TextureHandle>& textures) {
    rs::Material imported;
    imported.name = copyString(material.name);
    imported.baseColorFactor = {
        material.pbrData.baseColorFactor[0], material.pbrData.baseColorFactor[1],
        material.pbrData.baseColorFactor[2], material.pbrData.baseColorFactor[3]};
    imported.metallicFactor = material.pbrData.metallicFactor;
    imported.roughnessFactor = material.pbrData.roughnessFactor;
    imported.emissiveFactor = {
        material.emissiveFactor[0] * material.emissiveStrength,
        material.emissiveFactor[1] * material.emissiveStrength,
        material.emissiveFactor[2] * material.emissiveStrength,
    };
    imported.baseColorTexture =
        importBinding(sourcePath, material.pbrData.baseColorTexture, textures);
    imported.metallicRoughnessTexture =
        importBinding(sourcePath, material.pbrData.metallicRoughnessTexture, textures);
    imported.normalTexture = importBinding(sourcePath, material.normalTexture, textures);
    imported.occlusionTexture = importBinding(sourcePath, material.occlusionTexture, textures);
    imported.emissiveTexture = importBinding(sourcePath, material.emissiveTexture, textures);
    if (material.normalTexture.has_value()) {
        imported.normalScale = material.normalTexture->scale;
    }
    if (material.occlusionTexture.has_value()) {
        imported.occlusionStrength = material.occlusionTexture->strength;
    }
    imported.queue = convertMaterialQueue(material.alphaMode);
    imported.alphaCutoff = material.alphaCutoff;
    imported.doubleSided = material.doubleSided;
    return imported;
}

const fastgltf::Accessor& accessor(const fs::path& sourcePath, const fastgltf::Asset& asset,
                                   std::size_t index, fastgltf::AccessorType requiredType,
                                   std::string_view semantic) {
    if (index >= asset.accessors.size()) {
        fail(sourcePath, std::string(semantic) + " accessor index is out of range");
    }
    const auto& result = asset.accessors[index];
    if (result.type != requiredType) {
        fail(sourcePath, std::string(semantic) + " accessor has an incompatible element type");
    }
    return result;
}

std::vector<glm::vec2> readVec2(const fs::path& sourcePath, const fastgltf::Asset& asset,
                                std::size_t index, std::string_view semantic) {
    const auto& source = accessor(sourcePath, asset, index, fastgltf::AccessorType::Vec2, semantic);
    std::vector<glm::vec2> result;
    result.reserve(source.count);
    fastgltf::iterateAccessor<fastgltf::math::fvec2>(
        asset, source, [&](const auto& value) { result.emplace_back(value[0], value[1]); });
    return result;
}

std::vector<glm::vec3> readVec3(const fs::path& sourcePath, const fastgltf::Asset& asset,
                                std::size_t index, std::string_view semantic) {
    const auto& source = accessor(sourcePath, asset, index, fastgltf::AccessorType::Vec3, semantic);
    std::vector<glm::vec3> result;
    result.reserve(source.count);
    fastgltf::iterateAccessor<fastgltf::math::fvec3>(asset, source, [&](const auto& value) {
        result.emplace_back(value[0], value[1], value[2]);
    });
    return result;
}

std::vector<glm::vec4> readVec4(const fs::path& sourcePath, const fastgltf::Asset& asset,
                                std::size_t index, std::string_view semantic) {
    const auto& source = accessor(sourcePath, asset, index, fastgltf::AccessorType::Vec4, semantic);
    std::vector<glm::vec4> result;
    result.reserve(source.count);
    fastgltf::iterateAccessor<fastgltf::math::fvec4>(asset, source, [&](const auto& value) {
        result.emplace_back(value[0], value[1], value[2], value[3]);
    });
    return result;
}

std::vector<glm::vec4> readColors(const fs::path& sourcePath, const fastgltf::Asset& asset,
                                  std::size_t index) {
    if (index >= asset.accessors.size()) {
        fail(sourcePath, "COLOR_0 accessor index is out of range");
    }
    const auto& source = asset.accessors[index];
    std::vector<glm::vec4> result;
    result.reserve(source.count);
    if (source.type == fastgltf::AccessorType::Vec3) {
        fastgltf::iterateAccessor<fastgltf::math::fvec3>(asset, source, [&](const auto& value) {
            result.emplace_back(value[0], value[1], value[2], 1.0F);
        });
    } else if (source.type == fastgltf::AccessorType::Vec4) {
        fastgltf::iterateAccessor<fastgltf::math::fvec4>(asset, source, [&](const auto& value) {
            result.emplace_back(value[0], value[1], value[2], value[3]);
        });
    } else {
        fail(sourcePath, "COLOR_0 accessor must contain VEC3 or VEC4 elements");
    }
    return result;
}

std::vector<std::uint32_t> readIndices(const fs::path& sourcePath, const fastgltf::Asset& asset,
                                       std::size_t index) {
    const auto& source =
        accessor(sourcePath, asset, index, fastgltf::AccessorType::Scalar, "indices");
    if (source.componentType != fastgltf::ComponentType::UnsignedByte &&
        source.componentType != fastgltf::ComponentType::UnsignedShort &&
        source.componentType != fastgltf::ComponentType::UnsignedInt) {
        fail(sourcePath, "index accessor must use an unsigned integer component type");
    }
    std::vector<std::uint32_t> result;
    result.reserve(source.count);
    fastgltf::iterateAccessor<std::uint32_t>(asset, source,
                                             [&](std::uint32_t value) { result.push_back(value); });
    return result;
}

rs::PrimitiveTopology convertTopology(const fs::path& sourcePath, fastgltf::PrimitiveType type) {
    switch (type) {
    case fastgltf::PrimitiveType::Points:
        return rs::PrimitiveTopology::points;
    case fastgltf::PrimitiveType::Lines:
        return rs::PrimitiveTopology::lines;
    case fastgltf::PrimitiveType::LineStrip:
        return rs::PrimitiveTopology::lineStrip;
    case fastgltf::PrimitiveType::Triangles:
        return rs::PrimitiveTopology::triangles;
    case fastgltf::PrimitiveType::TriangleStrip:
        return rs::PrimitiveTopology::triangleStrip;
    case fastgltf::PrimitiveType::TriangleFan:
        return rs::PrimitiveTopology::triangleFan;
    case fastgltf::PrimitiveType::LineLoop:
        fail(sourcePath, "LINE_LOOP topology has no Vulkan primitive-topology equivalent");
    }
    fail(sourcePath, "unknown primitive topology");
}

rs::MeshPrimitive importPrimitive(const fs::path& sourcePath, const fastgltf::Asset& asset,
                                  const fastgltf::Primitive& primitive,
                                  const std::vector<rs::MaterialHandle>& materials,
                                  std::string name) {
    const auto position = primitive.findAttribute("POSITION");
    if (position == primitive.attributes.end()) {
        fail(sourcePath, "mesh primitive is missing the required POSITION attribute");
    }

    rs::MeshPrimitive imported;
    imported.name = std::move(name);
    imported.topology = convertTopology(sourcePath, primitive.type);
    imported.positions = readVec3(sourcePath, asset, position->accessorIndex, "POSITION");

    if (const auto normal = primitive.findAttribute("NORMAL");
        normal != primitive.attributes.end()) {
        imported.normals = readVec3(sourcePath, asset, normal->accessorIndex, "NORMAL");
    }
    if (const auto tangent = primitive.findAttribute("TANGENT");
        tangent != primitive.attributes.end()) {
        imported.tangents = readVec4(sourcePath, asset, tangent->accessorIndex, "TANGENT");
    }
    if (const auto uv = primitive.findAttribute("TEXCOORD_0"); uv != primitive.attributes.end()) {
        imported.texCoords[0] = readVec2(sourcePath, asset, uv->accessorIndex, "TEXCOORD_0");
    }
    if (const auto uv = primitive.findAttribute("TEXCOORD_1"); uv != primitive.attributes.end()) {
        imported.texCoords[1] = readVec2(sourcePath, asset, uv->accessorIndex, "TEXCOORD_1");
    }
    if (const auto color = primitive.findAttribute("COLOR_0");
        color != primitive.attributes.end()) {
        imported.colors = readColors(sourcePath, asset, color->accessorIndex);
    }
    if (primitive.indicesAccessor.has_value()) {
        imported.indices = readIndices(sourcePath, asset, *primitive.indicesAccessor);
    }
    if (primitive.materialIndex.has_value()) {
        if (*primitive.materialIndex >= materials.size()) {
            fail(sourcePath, "mesh primitive material index is outside the material array");
        }
        imported.material = materials[*primitive.materialIndex];
        const fastgltf::Material& material = asset.materials[*primitive.materialIndex];
        if (imported.tangents.empty() && material.normalTexture.has_value()) {
            generateMikkTangents(imported,
                                 static_cast<std::uint32_t>(material.normalTexture->texCoordIndex));
        }
    }
    return imported;
}

rs::Camera importCamera(const fastgltf::Camera& camera) {
    rs::Camera imported;
    imported.name = copyString(camera.name);
    std::visit(fastgltf::visitor{
                   [&](const fastgltf::Camera::Perspective& perspective) {
                       imported.type = rs::CameraType::perspective;
                       imported.verticalFieldOfView = perspective.yfov;
                       if (perspective.aspectRatio.has_value()) {
                           imported.aspectRatio = *perspective.aspectRatio;
                       }
                       imported.zNear = perspective.znear;
                       if (perspective.zfar.has_value()) {
                           imported.zFar = *perspective.zfar;
                       }
                   },
                   [&](const fastgltf::Camera::Orthographic& orthographic) {
                       imported.type = rs::CameraType::orthographic;
                       imported.xMagnitude = orthographic.xmag;
                       imported.yMagnitude = orthographic.ymag;
                       imported.zNear = orthographic.znear;
                       imported.zFar = orthographic.zfar;
                   },
               },
               camera.camera);
    return imported;
}

rs::Light importLight(const fastgltf::Light& light) {
    rs::Light imported;
    imported.name = copyString(light.name);
    switch (light.type) {
    case fastgltf::LightType::Directional:
        imported.type = rs::LightType::directional;
        break;
    case fastgltf::LightType::Point:
        imported.type = rs::LightType::point;
        break;
    case fastgltf::LightType::Spot:
        imported.type = rs::LightType::spot;
        break;
    }
    imported.color = {light.color[0], light.color[1], light.color[2]};
    imported.intensity = light.intensity;
    if (light.range.has_value()) {
        imported.range = *light.range;
    }
    if (light.innerConeAngle.has_value()) {
        imported.innerConeAngle = *light.innerConeAngle;
    }
    if (light.outerConeAngle.has_value()) {
        imported.outerConeAngle = *light.outerConeAngle;
    }
    return imported;
}

glm::mat4 importTransform(const fastgltf::Node& node) {
    const fastgltf::math::fmat4x4 source = fastgltf::getLocalTransformMatrix(node);
    glm::mat4 result(1.0F);
    for (glm::mat4::length_type column = 0; column < 4; ++column) {
        for (glm::vec4::length_type row = 0; row < 4; ++row) {
            result[column][row] = source[column][row];
        }
    }
    return result;
}

struct SelectedNodes {
    std::vector<bool> included;
    std::vector<std::size_t> roots;
    std::string sceneName;
};

SelectedNodes selectNodes(const fs::path& sourcePath, const fastgltf::Asset& asset,
                          const GltfImportOptions& options) {
    SelectedNodes selected;
    selected.included.assign(asset.nodes.size(), false);
    if (asset.scenes.empty()) {
        if (options.sceneIndex.has_value()) {
            fail(sourcePath, "a scene index was requested, but the asset has no scenes array");
        }
        std::vector<std::size_t> parentCounts(asset.nodes.size(), 0);
        for (const auto& node : asset.nodes) {
            for (const std::size_t child : node.children) {
                ++parentCounts[child];
            }
        }
        selected.included.assign(asset.nodes.size(), true);
        for (std::size_t index = 0; index < parentCounts.size(); ++index) {
            if (parentCounts[index] == 0) {
                selected.roots.push_back(index);
            }
        }
        return selected;
    }

    const std::size_t sceneIndex = options.sceneIndex.value_or(
        asset.defaultScene.has_value() ? *asset.defaultScene : std::size_t{0});
    if (sceneIndex >= asset.scenes.size()) {
        fail(sourcePath, "selected scene index is outside the scenes array");
    }
    const auto& sourceScene = asset.scenes[sceneIndex];
    selected.sceneName = copyString(sourceScene.name);
    selected.roots.assign(sourceScene.nodeIndices.begin(), sourceScene.nodeIndices.end());

    std::vector<std::size_t> stack = selected.roots;
    while (!stack.empty()) {
        const std::size_t index = stack.back();
        stack.pop_back();
        if (index >= asset.nodes.size()) {
            fail(sourcePath, "selected scene references a node outside the nodes array");
        }
        if (selected.included[index]) {
            continue;
        }
        selected.included[index] = true;
        const auto& children = asset.nodes[index].children;
        stack.insert(stack.end(), children.begin(), children.end());
    }
    return selected;
}

} // namespace

scene::Scene importGltf(const fs::path& sourcePath, const GltfImportOptions& options) {
    if (sourcePath.empty()) {
        fail(sourcePath, "source path is empty");
    }
    const std::string extension = lowerExtension(sourcePath);
    if (extension != ".gltf" && extension != ".glb") {
        fail(sourcePath, "source extension must be .gltf or .glb");
    }

    std::error_code error;
    if (!fs::is_regular_file(sourcePath, error) || error) {
        fail(sourcePath, "source is not a readable regular file");
    }
    const std::uintmax_t sourceSize = fs::file_size(sourcePath, error);
    if (error) {
        fail(sourcePath, "could not inspect source file: " + error.message());
    }
    requireAtMost(sourcePath, sourceSize, options.limits.maxSourceFileBytes, "source file size");

    // The preflight parse leaves external resources as URIs so their paths and
    // sizes can be checked before fastgltf allocates and loads them.
    fastgltf::Asset preflight = parseAsset(sourcePath, fastgltf::Options::None);
    verifyExtensionPolicy(sourcePath, preflight);
    verifyFeaturePolicy(sourcePath, preflight);
    verifyLimits(sourcePath, preflight, options, true);

    fastgltf::Asset asset = parseAsset(sourcePath, fastgltf::Options::LoadExternalBuffers |
                                                       fastgltf::Options::LoadExternalImages);
    verifyExtensionPolicy(sourcePath, asset);
    verifyFeaturePolicy(sourcePath, asset);
    verifyLimits(sourcePath, asset, options, false);

    const SelectedNodes selected = selectNodes(sourcePath, asset, options);
    const std::string sceneName =
        selected.sceneName.empty() ? sourcePath.stem().string() : selected.sceneName;
    rs::Scene result(sceneName, sourcePath.string());

    std::vector<rs::ImageHandle> images;
    images.reserve(asset.images.size());
    for (const auto& image : asset.images) {
        images.push_back(result.addImage(importImage(sourcePath, asset, image)));
    }

    std::vector<rs::SamplerHandle> samplers;
    samplers.reserve(asset.samplers.size());
    for (const auto& sampler : asset.samplers) {
        samplers.push_back(result.addSampler(importSampler(sampler)));
    }

    std::vector<rs::TextureHandle> textures;
    textures.reserve(asset.textures.size());
    for (const auto& texture : asset.textures) {
        if (!texture.imageIndex.has_value()) {
            fail(sourcePath, "texture has no core image source");
        }
        if (*texture.imageIndex >= images.size()) {
            fail(sourcePath, "texture image index is outside the images array");
        }
        rs::Texture imported;
        imported.name = copyString(texture.name);
        imported.image = images[*texture.imageIndex];
        if (texture.samplerIndex.has_value()) {
            if (*texture.samplerIndex >= samplers.size()) {
                fail(sourcePath, "texture sampler index is outside the samplers array");
            }
            imported.sampler = samplers[*texture.samplerIndex];
        }
        textures.push_back(result.addTexture(std::move(imported)));
    }

    std::vector<rs::MaterialHandle> materials;
    materials.reserve(asset.materials.size());
    for (const auto& material : asset.materials) {
        materials.push_back(result.addMaterial(importMaterial(sourcePath, material, textures)));
    }

    std::vector<rs::MeshHandle> meshes;
    meshes.reserve(asset.meshes.size());
    for (std::size_t meshIndex = 0; meshIndex < asset.meshes.size(); ++meshIndex) {
        const auto& mesh = asset.meshes[meshIndex];
        rs::Mesh imported;
        imported.name = copyString(mesh.name);
        imported.primitives.reserve(mesh.primitives.size());
        for (std::size_t primitiveIndex = 0; primitiveIndex < mesh.primitives.size();
             ++primitiveIndex) {
            std::string primitiveName =
                imported.name.empty() ? "mesh[" + std::to_string(meshIndex) + ']' : imported.name;
            primitiveName += ".primitive[" + std::to_string(primitiveIndex) + ']';
            imported.primitives.push_back(result.addPrimitive(
                importPrimitive(sourcePath, asset, mesh.primitives[primitiveIndex], materials,
                                std::move(primitiveName))));
        }
        meshes.push_back(result.addMesh(std::move(imported)));
    }

    std::vector<rs::CameraHandle> cameras;
    cameras.reserve(asset.cameras.size());
    for (const auto& camera : asset.cameras) {
        cameras.push_back(result.addCamera(importCamera(camera)));
    }

    std::vector<rs::LightHandle> lights;
    lights.reserve(asset.lights.size());
    for (const auto& light : asset.lights) {
        lights.push_back(result.addLight(importLight(light)));
    }

    std::vector<rs::NodeHandle> nodes(asset.nodes.size());
    for (std::size_t index = 0; index < asset.nodes.size(); ++index) {
        if (!selected.included[index]) {
            continue;
        }
        const auto& node = asset.nodes[index];
        rs::Node imported;
        imported.name = copyString(node.name);
        imported.localTransform = importTransform(node);
        if (node.meshIndex.has_value()) {
            imported.mesh = meshes[*node.meshIndex];
        }
        if (node.cameraIndex.has_value()) {
            imported.camera = cameras[*node.cameraIndex];
        }
        if (node.lightIndex.has_value()) {
            imported.light = lights[*node.lightIndex];
        }
        nodes[index] = result.addNode(std::move(imported));
    }

    for (std::size_t index = 0; index < asset.nodes.size(); ++index) {
        if (!selected.included[index]) {
            continue;
        }
        for (const std::size_t child : asset.nodes[index].children) {
            if (!selected.included[child]) {
                fail(sourcePath, "selected hierarchy contains an excluded child node");
            }
            result.addChild(nodes[index], nodes[child]);
        }
    }
    for (const std::size_t root : selected.roots) {
        result.addRoot(nodes[root]);
    }

    const rs::ValidationReport report = result.validate();
    if (!report.ok()) {
        fail(sourcePath, report.describe());
    }
    return result;
}

} // namespace renderer::assets
