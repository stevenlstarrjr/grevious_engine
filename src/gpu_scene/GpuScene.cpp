#include "renderer/gpu_scene/GpuScene.hpp"

#include "renderer/scene/Scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat3x3.hpp>
#include <glm/matrix.hpp>

namespace renderer::gpu_scene {

bool EmissiveTriangleDistribution::valid() const {
    float previousCdf = 0.0F;
    float probabilitySum = 0.0F;
    for (const EmissiveTriangleRecord& triangle : triangles) {
        if (!std::isfinite(triangle.sampling.x) || !std::isfinite(triangle.sampling.y) ||
            !std::isfinite(triangle.sampling.z) || !std::isfinite(triangle.sampling.w) ||
            triangle.sampling.x < previousCdf || triangle.sampling.x > 1.0F ||
            triangle.sampling.y < 0.0F || triangle.sampling.z < 0.0F ||
            triangle.sampling.w < 0.0F) {
            return false;
        }
        previousCdf = triangle.sampling.x;
        probabilitySum += triangle.sampling.y;
    }
    return triangles.empty() || (previousCdf == 1.0F && std::abs(probabilitySum - 1.0F) <= 1.0e-4F);
}

SceneRecords makePbrSphereGrid() {
    constexpr std::uint32_t columns = 7;
    constexpr std::uint32_t rows = 5;
    SceneRecords scene;
    scene.instances.reserve(columns * rows);
    scene.materials.reserve(columns * rows);
    scene.transforms.reserve(columns * rows);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t column = 0; column < columns; ++column) {
            const std::uint32_t materialIndex = static_cast<std::uint32_t>(scene.materials.size());
            const glm::vec3 center{(static_cast<float>(column) - 3.0F) * 1.05F,
                                   (static_cast<float>(row) - 2.0F) * 1.05F, 0.0F};
            constexpr float radius = 0.42F;
            const std::uint32_t transformIndex =
                static_cast<std::uint32_t>(scene.transforms.size());
            scene.instances.push_back({
                .centerRadius = {center, radius},
                .meshMaterial = {0U, materialIndex, transformIndex, 0U},
            });
            scene.materials.push_back({
                .baseColorFactor = {0.95F, 0.46F, 0.10F, 1.0F},
                .emissiveAlphaCutoff = {0.0F, 0.0F, 0.0F, 0.5F},
                .metallicRoughnessOcclusionNormal = {static_cast<float>(column) / 6.0F,
                                                     std::max(static_cast<float>(row) / 4.0F,
                                                              0.10F),
                                                     1.0F, 1.0F},
                .textureIndices = {0U, 0U, 0U, 0U},
                .textureState = {0U, baseColorTextureBit, 0U, 0U},
            });
            const glm::mat4 world = glm::translate(glm::mat4(1.0F), center) *
                                    glm::scale(glm::mat4(1.0F), glm::vec3(radius));
            scene.transforms.push_back({
                .world = world,
                .normalToWorld = glm::transpose(glm::inverse(world)),
                .previousWorld = world,
            });
        }
    }
    return scene;
}

MaterialRecord packMaterial(const scene::Material& material,
                            const MaterialTextureDescriptorTable& descriptors) {
    MaterialRecord packed{
        .baseColorFactor = material.baseColorFactor,
        .emissiveAlphaCutoff = {material.emissiveFactor, material.alphaCutoff},
        .metallicRoughnessOcclusionNormal = {material.metallicFactor, material.roughnessFactor,
                                             material.occlusionStrength, material.normalScale},
    };

    const auto bindTexture = [&](const scene::TextureBinding& binding,
                                 std::span<const std::uint32_t> descriptorIndices,
                                 MaterialTextureBit presenceBit, std::uint32_t& destination) {
        if (!binding.texture.valid()) {
            return;
        }
        const std::size_t textureIndex = binding.texture.index();
        if (textureIndex >= descriptorIndices.size() ||
            descriptorIndices[textureIndex] == invalidTextureDescriptor) {
            throw std::runtime_error("A material texture has no resident bindless descriptor");
        }
        destination = descriptorIndices[textureIndex];
        packed.textureState.y |= presenceBit;
        if (binding.texCoord == 1U) {
            packed.textureState.w |= presenceBit;
        }
    };
    bindTexture(material.baseColorTexture, descriptors.baseColor, baseColorTextureBit,
                packed.textureIndices.x);
    bindTexture(material.metallicRoughnessTexture, descriptors.metallicRoughness,
                metallicRoughnessTextureBit, packed.textureIndices.y);
    bindTexture(material.normalTexture, descriptors.normal, normalTextureBit,
                packed.textureIndices.z);
    bindTexture(material.occlusionTexture, descriptors.occlusion, occlusionTextureBit,
                packed.textureIndices.w);
    bindTexture(material.emissiveTexture, descriptors.emissive, emissiveTextureBit,
                packed.textureState.x);

    switch (material.queue) {
    case scene::MaterialQueue::opaque:
        break;
    case scene::MaterialQueue::alphaMasked:
        packed.textureState.z |= alphaMaskMaterial;
        break;
    case scene::MaterialQueue::transparent:
        packed.textureState.z |= alphaBlendMaterial;
        break;
    case scene::MaterialQueue::additive:
        packed.textureState.z |= alphaBlendMaterial | additiveMaterial;
        break;
    }
    if (material.doubleSided) {
        packed.textureState.z |= doubleSidedMaterial;
    }
    if (material.rainReceiver) {
        packed.textureState.z |= rainReceiverMaterial;
    }
    if (material.wetCoat) {
        packed.textureState.z |= wetCoatMaterial;
    }
    return packed;
}

MaterialRecord packMaterial(const scene::Material& material,
                            std::span<const std::uint32_t> textureDescriptorIndices) {
    return packMaterial(material, MaterialTextureDescriptorTable{
                                      .baseColor = textureDescriptorIndices,
                                      .metallicRoughness = textureDescriptorIndices,
                                      .normal = textureDescriptorIndices,
                                      .occlusion = textureDescriptorIndices,
                                      .emissive = textureDescriptorIndices,
                                  });
}

std::vector<MaterialRecord> packMaterials(const scene::Scene& scene,
                                          const MaterialTextureDescriptorTable& descriptors) {
    std::vector<MaterialRecord> packed;
    packed.reserve(scene.materials().size());
    for (const scene::Material& material : scene.materials()) {
        packed.push_back(packMaterial(material, descriptors));
    }
    return packed;
}

std::vector<MaterialRecord> packMaterials(const scene::Scene& scene,
                                          std::span<const std::uint32_t> textureDescriptorIndices) {
    return packMaterials(scene, MaterialTextureDescriptorTable{
                                    .baseColor = textureDescriptorIndices,
                                    .metallicRoughness = textureDescriptorIndices,
                                    .normal = textureDescriptorIndices,
                                    .occlusion = textureDescriptorIndices,
                                    .emissive = textureDescriptorIndices,
                                });
}

PackedScene packScene(const scene::Scene& scene,
                      const MaterialTextureDescriptorTable& descriptors) {
    const scene::ValidationReport validation = scene.validate();
    if (!validation.ok()) {
        throw std::runtime_error(validation.describe());
    }

    PackedScene packed;
    packed.materials = packMaterials(scene, descriptors);
    std::uint32_t defaultMaterialIndex = invalidTextureDescriptor;
    auto resolveMaterial = [&](scene::MaterialHandle material) {
        if (material.valid()) {
            return material.index();
        }
        if (defaultMaterialIndex == invalidTextureDescriptor) {
            if (packed.materials.size() >= std::numeric_limits<std::uint32_t>::max()) {
                throw std::length_error("GPU material index space is exhausted");
            }
            defaultMaterialIndex = static_cast<std::uint32_t>(packed.materials.size());
            packed.materials.emplace_back();
        }
        return defaultMaterialIndex;
    };

    std::vector<std::uint32_t> primitiveIndices(scene.primitives().size(),
                                                invalidTextureDescriptor);
    packed.primitives.reserve(scene.primitives().size());
    for (std::size_t primitiveIndex = 0; primitiveIndex < scene.primitives().size();
         ++primitiveIndex) {
        const scene::MeshPrimitive& primitive = scene.primitives()[primitiveIndex];
        if (primitive.topology != scene::PrimitiveTopology::triangles) {
            throw std::runtime_error("GPU scene packing currently supports triangle lists only");
        }
        if (primitive.normals.size() != primitive.positions.size()) {
            throw std::runtime_error("GPU scene packing requires one normal per vertex");
        }
        if (primitive.material.valid() &&
            scene.material(primitive.material).normalTexture.texture.valid() &&
            primitive.tangents.size() != primitive.positions.size()) {
            throw std::runtime_error(
                "A normal-mapped primitive requires imported or generated tangents");
        }
        if (primitive.positions.size() > std::numeric_limits<std::uint32_t>::max() ||
            packed.vertices.size() >
                static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::length_error("GPU vertex offset exceeds indexed-draw limits");
        }
        if (packed.indices.size() >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::length_error("GPU first index exceeds indexed-draw limits");
        }
        const std::int32_t vertexOffset = static_cast<std::int32_t>(packed.vertices.size());
        const std::uint32_t firstIndex = static_cast<std::uint32_t>(packed.indices.size());
        for (std::size_t vertexIndex = 0; vertexIndex < primitive.positions.size(); ++vertexIndex) {
            if (glm::dot(primitive.normals[vertexIndex], primitive.normals[vertexIndex]) <=
                1.0e-20F) {
                throw std::runtime_error("GPU scene packing rejects a zero-length normal");
            }
            const glm::vec3 normal = glm::normalize(primitive.normals[vertexIndex]);
            glm::vec4 tangent{1.0F, 0.0F, 0.0F, 1.0F};
            if (!primitive.tangents.empty()) {
                tangent = primitive.tangents[vertexIndex];
            } else {
                const glm::vec3 axis = std::abs(normal.z) < 0.999F ? glm::vec3(0.0F, 0.0F, 1.0F)
                                                                   : glm::vec3(0.0F, 1.0F, 0.0F);
                tangent = {glm::normalize(glm::cross(axis, normal)), 1.0F};
            }
            packed.vertices.push_back({
                .position = primitive.positions[vertexIndex],
                .normal = normal,
                .uv0 = primitive.texCoords[0].empty() ? glm::vec2(0.0F)
                                                      : primitive.texCoords[0][vertexIndex],
                .tangent = tangent,
                .uv1 = primitive.texCoords[1].empty() ? glm::vec2(0.0F)
                                                      : primitive.texCoords[1][vertexIndex],
                .color = primitive.colors.empty() ? glm::vec4(1.0F)
                                                  : primitive.colors[vertexIndex],
            });
        }
        if (primitive.indices.empty()) {
            for (std::size_t index = 0; index < primitive.positions.size(); ++index) {
                packed.indices.push_back(static_cast<std::uint32_t>(index));
            }
        } else {
            packed.indices.insert(packed.indices.end(), primitive.indices.begin(),
                                  primitive.indices.end());
        }
        if (packed.indices.size() - firstIndex >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::length_error("GPU primitive index count exceeds draw limits");
        }
        const scene::AxisAlignedBounds bounds = scene::computeBounds(primitive);
        const glm::vec3 center = (bounds.minimum + bounds.maximum) * 0.5F;
        const float radius = glm::length(bounds.maximum - bounds.minimum) * 0.5F;
        if (packed.primitives.size() >= std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("GPU primitive index space is exhausted");
        }
        const std::uint32_t packedPrimitiveIndex =
            static_cast<std::uint32_t>(packed.primitives.size());
        primitiveIndices[primitiveIndex] = packedPrimitiveIndex;
        packed.primitives.push_back({
            .indexCount = static_cast<std::uint32_t>(packed.indices.size() - firstIndex),
            .firstIndex = firstIndex,
            .vertexOffset = vertexOffset,
            .materialIndex = resolveMaterial(primitive.material),
            .centerRadius = {center, radius},
        });
    }

    const scene::SceneEvaluation evaluation = scene::evaluate(scene);
    packed.transforms.reserve(evaluation.meshInstances.size());
    for (const scene::MeshInstance& meshInstance : evaluation.meshInstances) {
        const glm::mat4 world = meshInstance.worldTransform;
        const float determinant = glm::determinant(glm::mat3(world));
        const std::array basis{glm::vec3(world[0]), glm::vec3(world[1]), glm::vec3(world[2])};
        const std::array lengths{glm::length(basis[0]), glm::length(basis[1]),
                                 glm::length(basis[2])};
        const float basisVolume = lengths[0] * lengths[1] * lengths[2];
        constexpr float relativeDeterminantTolerance = 1.0e-6F;
        if (!std::isfinite(determinant) || !std::isfinite(basisVolume) ||
            basisVolume <= std::numeric_limits<float>::min() ||
            std::abs(determinant) <= relativeDeterminantTolerance * basisVolume) {
            throw std::runtime_error("GPU scene packing rejects a singular mesh transform");
        }
        if (packed.transforms.size() >= std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("GPU transform index space is exhausted");
        }
        const std::uint32_t transformIndex = static_cast<std::uint32_t>(packed.transforms.size());
        packed.transforms.push_back({
            .world = world,
            .normalToWorld = glm::transpose(glm::inverse(world)),
            .previousWorld = world,
        });
        const scene::Mesh& mesh = scene.mesh(meshInstance.mesh);
        for (const scene::PrimitiveHandle primitiveHandle : mesh.primitives) {
            const std::uint32_t residentPrimitiveIndex = primitiveIndices[primitiveHandle.index()];
            const PrimitiveRecord& residentPrimitive = packed.primitives[residentPrimitiveIndex];
            const glm::vec3 localCenter = glm::vec3(residentPrimitive.centerRadius);
            const glm::vec3 worldCenter = glm::vec3(world * glm::vec4(localCenter, 1.0F));
            constexpr float orthogonalityTolerance = 1.0e-5F;
            const bool orthogonal = std::abs(glm::dot(basis[0], basis[1])) <=
                                        orthogonalityTolerance * lengths[0] * lengths[1] &&
                                    std::abs(glm::dot(basis[0], basis[2])) <=
                                        orthogonalityTolerance * lengths[0] * lengths[2] &&
                                    std::abs(glm::dot(basis[1], basis[2])) <=
                                        orthogonalityTolerance * lengths[1] * lengths[2];
            // A rotation-scale matrix has an exact maximum stretch equal to
            // its longest basis vector. Arbitrary/sheared matrices retain the
            // conservative Frobenius-norm bound used by GPU culling.
            const float maximumScale =
                orthogonal ? std::max({lengths[0], lengths[1], lengths[2]})
                           : std::sqrt(lengths[0] * lengths[0] + lengths[1] * lengths[1] +
                                       lengths[2] * lengths[2]);
            packed.instances.push_back({
                .centerRadius = {worldCenter, residentPrimitive.centerRadius.w * maximumScale},
                .meshMaterial = {residentPrimitiveIndex, residentPrimitive.materialIndex,
                                 transformIndex,
                                 determinant < 0.0F ? mirroredTransformInstance : 0U},
                .visibilityLayer = {packRenderLayerMask(meshInstance.visibilityMask), 0U, 0U},
            });
        }
    }
    packed.primitiveLods.reserve(packed.primitives.size());
    packed.lodSources.reserve(packed.primitives.size());
    for (std::size_t primitiveIndex = 0; primitiveIndex < packed.primitives.size();
         ++primitiveIndex) {
        const std::uint32_t packedIndex = static_cast<std::uint32_t>(primitiveIndex);
        packed.primitiveLods.push_back({.firstLod = packedIndex, .lodCount = 1U});
        packed.lodSources.push_back(
            {.primitiveIndex = packedIndex, .minimumProjectedDiameter = 0.0F});
    }
    MeshletHierarchy hierarchy =
        buildMeshletHierarchy(packed.vertices, packed.indices, packed.primitives,
                              packed.primitiveLods, packed.lodSources);
    packed.primitiveLods = std::move(hierarchy.primitiveLods);
    packed.lods = std::move(hierarchy.lods);
    packed.meshlets = std::move(hierarchy.meshlets);
    return packed;
}

PackedScene packScene(const scene::Scene& scene,
                      std::span<const std::uint32_t> textureDescriptorIndices) {
    return packScene(scene, MaterialTextureDescriptorTable{
                                .baseColor = textureDescriptorIndices,
                                .metallicRoughness = textureDescriptorIndices,
                                .normal = textureDescriptorIndices,
                                .occlusion = textureDescriptorIndices,
                                .emissive = textureDescriptorIndices,
                            });
}

EmissiveTriangleDistribution buildEmissiveTriangleDistribution(
    std::span<const VertexRecord> vertices, std::span<const std::uint32_t> indices,
    std::span<const PrimitiveRecord> primitives, std::span<const PrimitiveLodRange> primitiveLods,
    std::span<const LodSourceRecord> lodSources, std::span<InstanceRecord> instances,
    std::span<const MaterialRecord> materials, std::span<const TransformRecord> transforms,
    RenderLayerMask viewMask) {
    if (vertices.empty() || indices.empty() || primitives.empty() || primitiveLods.empty() ||
        lodSources.empty() || instances.empty() || materials.empty() || transforms.empty()) {
        throw std::invalid_argument(
            "Emissive-triangle construction requires complete resident scene data");
    }

    EmissiveTriangleDistribution distribution;
    double totalWeight = 0.0;
    for (std::size_t instanceIndex = 0; instanceIndex < instances.size(); ++instanceIndex) {
        InstanceRecord& instance = instances[instanceIndex];
        instance.visibilityLayer.z = invalidEmissiveTriangleIndex;
        instance.visibilityLayer.w = 0U;
        if (!instanceVisible(instance, viewMask)) {
            continue;
        }
        const std::uint32_t primitiveGroup = instance.meshMaterial.x;
        const std::uint32_t materialIndex = instance.meshMaterial.y;
        const std::uint32_t transformIndex = instance.meshMaterial.z;
        if (primitiveGroup >= primitiveLods.size() || materialIndex >= materials.size() ||
            transformIndex >= transforms.size()) {
            throw std::runtime_error(
                "Emissive-triangle instance references invalid scene residency");
        }

        const glm::vec3 emission =
            glm::max(glm::vec3(materials[materialIndex].emissiveAlphaCutoff), glm::vec3(0.0F));
        if (!std::isfinite(emission.x) || !std::isfinite(emission.y) ||
            !std::isfinite(emission.z)) {
            throw std::runtime_error("Emissive-triangle material has nonfinite radiance");
        }
        const float luminance = glm::dot(emission, glm::vec3(0.2126F, 0.7152F, 0.0722F));
        if (luminance <= 0.0F) {
            continue;
        }

        const PrimitiveLodRange lodRange = primitiveLods[primitiveGroup];
        const std::uint64_t lodEnd =
            static_cast<std::uint64_t>(lodRange.firstLod) + lodRange.lodCount;
        if (lodRange.lodCount == 0U || lodEnd > lodSources.size()) {
            throw std::runtime_error("Emissive-triangle instance has an invalid LOD range");
        }
        const std::uint32_t primitiveIndex = lodSources[lodRange.firstLod].primitiveIndex;
        if (primitiveIndex >= primitives.size()) {
            throw std::runtime_error("Emissive-triangle LOD references an invalid primitive");
        }
        const PrimitiveRecord& primitive = primitives[primitiveIndex];
        const std::uint64_t indexEnd =
            static_cast<std::uint64_t>(primitive.firstIndex) + primitive.indexCount;
        if (primitive.indexCount == 0U || primitive.indexCount % 3U != 0U ||
            indexEnd > indices.size() || primitive.vertexOffset < 0) {
            throw std::runtime_error(
                "Emissive-triangle source is not a valid indexed triangle list");
        }

        const std::uint32_t triangleCount = primitive.indexCount / 3U;
        if (distribution.triangles.size() > std::numeric_limits<std::uint32_t>::max() ||
            triangleCount >
                std::numeric_limits<std::uint32_t>::max() - distribution.triangles.size()) {
            throw std::length_error("Emissive-triangle light index space is exhausted");
        }
        instance.visibilityLayer.z = static_cast<std::uint32_t>(distribution.triangles.size());
        instance.visibilityLayer.w = triangleCount;
        const glm::mat4& world = transforms[transformIndex].world;
        for (std::uint32_t triangleIndex = 0U; triangleIndex < triangleCount; ++triangleIndex) {
            std::array<glm::vec3, 3> positions{};
            for (std::uint32_t corner = 0U; corner < 3U; ++corner) {
                const std::uint32_t localVertex =
                    indices[static_cast<std::size_t>(primitive.firstIndex) +
                            static_cast<std::size_t>(triangleIndex) * 3U + corner];
                const std::uint64_t vertexIndex =
                    static_cast<std::uint64_t>(primitive.vertexOffset) + localVertex;
                if (vertexIndex >= vertices.size()) {
                    throw std::runtime_error(
                        "Emissive triangle references a vertex outside residency");
                }
                positions[corner] = glm::vec3(
                    world *
                    glm::vec4(vertices[static_cast<std::size_t>(vertexIndex)].position, 1.0F));
            }
            const float area = 0.5F * glm::length(glm::cross(positions[1] - positions[0],
                                                             positions[2] - positions[0]));
            if (!std::isfinite(area)) {
                throw std::runtime_error("Emissive triangle has nonfinite world area");
            }
            const float weight = area * luminance;
            totalWeight += static_cast<double>(weight);
            distribution.triangles.push_back({
                .identity = {static_cast<std::uint32_t>(instanceIndex), triangleIndex,
                             materialIndex, 0U},
                .sampling = {0.0F, 0.0F, area, weight},
            });
        }
    }

    if (!(totalWeight > 0.0) || !std::isfinite(totalWeight)) {
        distribution.triangles.clear();
        for (InstanceRecord& instance : instances) {
            instance.visibilityLayer.z = invalidEmissiveTriangleIndex;
            instance.visibilityLayer.w = 0U;
        }
        return distribution;
    }

    double cdf = 0.0;
    for (EmissiveTriangleRecord& triangle : distribution.triangles) {
        const float probability =
            static_cast<float>(static_cast<double>(triangle.sampling.w) / totalWeight);
        cdf += static_cast<double>(probability);
        triangle.sampling.x = static_cast<float>(std::min(cdf, 1.0));
        triangle.sampling.y = probability;
    }
    distribution.triangles.back().sampling.x = 1.0F;
    if (!distribution.valid()) {
        throw std::runtime_error("Emissive-triangle sampling distribution is invalid");
    }
    return distribution;
}

MeshletHierarchy buildMeshletHierarchy(std::span<const VertexRecord> vertices,
                                       std::span<const std::uint32_t> indices,
                                       std::span<const PrimitiveRecord> primitives,
                                       std::span<const PrimitiveLodRange> primitiveLods,
                                       std::span<const LodSourceRecord> lodSources,
                                       std::uint32_t maximumTrianglesPerMeshlet) {
    if (maximumTrianglesPerMeshlet == 0 || maximumTrianglesPerMeshlet > maxMeshletTriangles) {
        throw std::invalid_argument("Meshlet triangle capacity is outside the supported range");
    }
    if (primitiveLods.empty() || lodSources.empty()) {
        throw std::invalid_argument("Meshlet hierarchy requires primitive LOD sources");
    }

    MeshletHierarchy hierarchy;
    hierarchy.primitiveLods.reserve(primitiveLods.size());
    hierarchy.lods.reserve(lodSources.size());
    const std::uint32_t maximumIndicesPerMeshlet = maximumTrianglesPerMeshlet * 3U;

    for (const PrimitiveLodRange& sourceRange : primitiveLods) {
        const std::uint64_t sourceEnd = static_cast<std::uint64_t>(sourceRange.firstLod) +
                                        static_cast<std::uint64_t>(sourceRange.lodCount);
        if (sourceRange.lodCount == 0 || sourceEnd > lodSources.size()) {
            throw std::runtime_error("Primitive LOD range is empty or out of bounds");
        }
        if (hierarchy.lods.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("GPU LOD index space is exhausted");
        }
        PrimitiveLodRange packedRange{
            .firstLod = static_cast<std::uint32_t>(hierarchy.lods.size()),
            .lodCount = sourceRange.lodCount,
        };

        for (std::uint32_t localLod = 0; localLod < sourceRange.lodCount; ++localLod) {
            const std::size_t sourceIndex =
                static_cast<std::size_t>(sourceRange.firstLod) + static_cast<std::size_t>(localLod);
            const LodSourceRecord& source = lodSources[sourceIndex];
            if (source.primitiveIndex >= primitives.size() ||
                !std::isfinite(source.minimumProjectedDiameter) ||
                source.minimumProjectedDiameter < 0.0F) {
                throw std::runtime_error("LOD source contains an invalid primitive or threshold");
            }
            if (localLod + 1U < sourceRange.lodCount) {
                const float nextThreshold = lodSources[sourceIndex + 1U].minimumProjectedDiameter;
                if (!std::isfinite(nextThreshold) ||
                    source.minimumProjectedDiameter <= nextThreshold) {
                    throw std::runtime_error(
                        "Primitive LOD thresholds must decrease from finest to coarsest");
                }
            } else if (source.minimumProjectedDiameter != 0.0F) {
                throw std::runtime_error(
                    "The coarsest primitive LOD must cover zero projected diameter");
            }

            const PrimitiveRecord& primitive = primitives[source.primitiveIndex];
            const std::uint64_t primitiveEnd = static_cast<std::uint64_t>(primitive.firstIndex) +
                                               static_cast<std::uint64_t>(primitive.indexCount);
            if (primitive.indexCount == 0 || primitive.indexCount % 3U != 0U ||
                primitiveEnd > indices.size()) {
                throw std::runtime_error(
                    "Meshlet source must be a nonempty in-bounds triangle list");
            }
            if (hierarchy.meshlets.size() > std::numeric_limits<std::uint32_t>::max()) {
                throw std::length_error("GPU meshlet index space is exhausted");
            }
            const std::uint32_t firstMeshlet =
                static_cast<std::uint32_t>(hierarchy.meshlets.size());
            for (std::uint32_t first = primitive.firstIndex;
                 first < primitive.firstIndex + primitive.indexCount;) {
                std::vector<std::uint32_t> uniqueVertices;
                uniqueVertices.reserve(maxMeshletVertices);
                std::uint32_t indexCount = 0;
                while (indexCount < maximumIndicesPerMeshlet &&
                       first + indexCount < primitive.firstIndex + primitive.indexCount) {
                    const std::uint32_t triangleFirst = first + indexCount;
                    std::uint32_t newVertexCount = 0;
                    for (std::uint32_t corner = 0; corner < 3U; ++corner) {
                        const std::uint32_t vertex = indices[triangleFirst + corner];
                        if (std::find(uniqueVertices.begin(), uniqueVertices.end(), vertex) ==
                            uniqueVertices.end()) {
                            ++newVertexCount;
                        }
                    }
                    if (!uniqueVertices.empty() &&
                        uniqueVertices.size() + newVertexCount > maxMeshletVertices) {
                        break;
                    }
                    for (std::uint32_t corner = 0; corner < 3U; ++corner) {
                        const std::uint32_t vertex = indices[triangleFirst + corner];
                        if (std::find(uniqueVertices.begin(), uniqueVertices.end(), vertex) ==
                            uniqueVertices.end()) {
                            uniqueVertices.push_back(vertex);
                        }
                    }
                    indexCount += 3U;
                }
                if (indexCount == 0) {
                    throw std::runtime_error("One triangle exceeds the meshlet vertex capacity");
                }
                glm::vec3 minimum(std::numeric_limits<float>::infinity());
                glm::vec3 maximum(-std::numeric_limits<float>::infinity());
                std::vector<glm::vec3> triangleNormals;
                triangleNormals.reserve(indexCount / 3U);
                for (std::uint32_t offset = 0; offset < indexCount; ++offset) {
                    const std::int64_t vertexIndex =
                        static_cast<std::int64_t>(primitive.vertexOffset) +
                        static_cast<std::int64_t>(indices[first + offset]);
                    if (vertexIndex < 0 ||
                        static_cast<std::uint64_t>(vertexIndex) >= vertices.size()) {
                        throw std::runtime_error(
                            "Meshlet source references a vertex outside the unified buffer");
                    }
                    const glm::vec3 position =
                        vertices[static_cast<std::size_t>(vertexIndex)].position;
                    minimum = glm::min(minimum, position);
                    maximum = glm::max(maximum, position);
                }
                for (std::uint32_t offset = 0; offset < indexCount; offset += 3U) {
                    std::array<glm::vec3, 3> triangle{};
                    for (std::uint32_t corner = 0; corner < 3U; ++corner) {
                        const std::int64_t vertexIndex =
                            static_cast<std::int64_t>(primitive.vertexOffset) +
                            static_cast<std::int64_t>(indices[first + offset + corner]);
                        triangle[corner] = vertices[static_cast<std::size_t>(vertexIndex)].position;
                    }
                    const glm::vec3 normal =
                        glm::cross(triangle[1] - triangle[0], triangle[2] - triangle[0]);
                    if (glm::dot(normal, normal) > 1.0e-20F) {
                        triangleNormals.push_back(glm::normalize(normal));
                    }
                }
                const glm::vec3 center = (minimum + maximum) * 0.5F;
                glm::vec3 coneAxis(0.0F, 0.0F, 1.0F);
                float coneCutoff = -1.0F;
                if (!triangleNormals.empty()) {
                    glm::vec3 normalSum(0.0F);
                    for (const glm::vec3& normal : triangleNormals) {
                        normalSum += normal;
                    }
                    if (glm::dot(normalSum, normalSum) > 1.0e-20F) {
                        coneAxis = glm::normalize(normalSum);
                        coneCutoff = 1.0F;
                        for (const glm::vec3& normal : triangleNormals) {
                            coneCutoff = std::min(coneCutoff, glm::dot(coneAxis, normal));
                        }
                        if (coneCutoff < 0.0F) {
                            coneCutoff = -1.0F;
                        }
                    }
                }
                hierarchy.meshlets.push_back({
                    .indexCount = indexCount,
                    .firstIndex = first,
                    .vertexOffset = primitive.vertexOffset,
                    .sourcePrimitive = source.primitiveIndex,
                    .centerRadius = {center, glm::length(maximum - minimum) * 0.5F},
                    .normalCone = {coneAxis, coneCutoff},
                });
                first += indexCount;
            }
            const std::size_t meshletCount = hierarchy.meshlets.size() - firstMeshlet;
            if (meshletCount > std::numeric_limits<std::uint32_t>::max()) {
                throw std::length_error("One LOD exceeds the GPU meshlet count limit");
            }
            const float maximumProjectedDiameter =
                localLod == 0U ? std::numeric_limits<float>::max()
                               : lodSources[sourceIndex - 1U].minimumProjectedDiameter;
            hierarchy.lods.push_back({
                .firstMeshlet = firstMeshlet,
                .meshletCount = static_cast<std::uint32_t>(meshletCount),
                .minimumProjectedDiameter = source.minimumProjectedDiameter,
                .maximumProjectedDiameter = maximumProjectedDiameter,
            });
        }
        hierarchy.primitiveLods.push_back(packedRange);
    }
    return hierarchy;
}

MeshletHierarchy buildSingleLodMeshletHierarchy(std::span<const VertexRecord> vertices,
                                                std::span<const std::uint32_t> indices,
                                                std::span<const PrimitiveRecord> primitives,
                                                std::uint32_t maximumTrianglesPerMeshlet) {
    if (primitives.empty()) {
        throw std::invalid_argument("Single-LOD meshlet construction requires primitives");
    }
    if (primitives.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("Primitive count exceeds the GPU LOD index space");
    }
    std::vector<PrimitiveLodRange> ranges;
    std::vector<LodSourceRecord> sources;
    ranges.reserve(primitives.size());
    sources.reserve(primitives.size());
    for (std::size_t primitiveIndex = 0; primitiveIndex < primitives.size(); ++primitiveIndex) {
        const std::uint32_t packedIndex = static_cast<std::uint32_t>(primitiveIndex);
        ranges.push_back({.firstLod = packedIndex, .lodCount = 1U});
        sources.push_back({.primitiveIndex = packedIndex, .minimumProjectedDiameter = 0.0F});
    }
    return buildMeshletHierarchy(vertices, indices, primitives, ranges, sources,
                                 maximumTrianglesPerMeshlet);
}

std::vector<MeshletWorkRecord>
makeMeshletWorkRecords(std::span<const PrimitiveLodRange> primitiveLods,
                       std::span<const LodRecord> lods, std::span<const MeshletRecord> meshlets,
                       std::span<const InstanceRecord> instances) {
    if (instances.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("Instance count exceeds the GPU work-record index space");
    }
    std::vector<MeshletWorkRecord> work;
    for (std::size_t instanceIndex = 0; instanceIndex < instances.size(); ++instanceIndex) {
        const std::uint32_t primitiveGroup = instances[instanceIndex].meshMaterial.x;
        if (primitiveGroup >= primitiveLods.size()) {
            throw std::runtime_error("GPU instance references an invalid primitive LOD group");
        }
        const PrimitiveLodRange range = primitiveLods[primitiveGroup];
        const std::uint64_t lodEnd = static_cast<std::uint64_t>(range.firstLod) + range.lodCount;
        if (range.lodCount == 0 || lodEnd > lods.size()) {
            throw std::runtime_error("GPU primitive has an invalid packed LOD range");
        }
        for (std::uint32_t localLod = 0; localLod < range.lodCount; ++localLod) {
            const LodRecord& lod = lods[static_cast<std::size_t>(range.firstLod) + localLod];
            const std::uint64_t meshletEnd =
                static_cast<std::uint64_t>(lod.firstMeshlet) + lod.meshletCount;
            if (lod.meshletCount == 0 || meshletEnd > meshlets.size() ||
                !std::isfinite(lod.minimumProjectedDiameter) ||
                !std::isfinite(lod.maximumProjectedDiameter) ||
                lod.minimumProjectedDiameter < 0.0F ||
                lod.maximumProjectedDiameter <= lod.minimumProjectedDiameter) {
                throw std::runtime_error("GPU LOD contains an invalid meshlet range or band");
            }
            for (std::uint32_t meshletOffset = 0; meshletOffset < lod.meshletCount;
                 ++meshletOffset) {
                if (work.size() >= std::numeric_limits<std::uint32_t>::max()) {
                    throw std::length_error("Meshlet work count exceeds indirect-count limits");
                }
                const MeshletRecord& meshlet =
                    meshlets[static_cast<std::size_t>(lod.firstMeshlet) + meshletOffset];
                work.push_back({
                    .indexCount = meshlet.indexCount,
                    .firstIndex = meshlet.firstIndex,
                    .vertexOffset = meshlet.vertexOffset,
                    .instanceIndex = static_cast<std::uint32_t>(instanceIndex),
                    .projectedDiameterRange = {lod.minimumProjectedDiameter,
                                               lod.maximumProjectedDiameter},
                    .lodIndex = localLod,
                });
            }
        }
    }
    return work;
}

std::uint32_t selectLod(std::span<const LodRecord> lods, float projectedDiameter) {
    if (!std::isfinite(projectedDiameter) || projectedDiameter < 0.0F) {
        return invalidLodIndex;
    }
    for (std::size_t index = 0; index < lods.size(); ++index) {
        const LodRecord& lod = lods[index];
        const bool withinMaximum =
            projectedDiameter < lod.maximumProjectedDiameter ||
            (lod.maximumProjectedDiameter == std::numeric_limits<float>::max() &&
             projectedDiameter <= lod.maximumProjectedDiameter);
        if (projectedDiameter >= lod.minimumProjectedDiameter && withinMaximum) {
            if (index > std::numeric_limits<std::uint32_t>::max()) {
                return invalidLodIndex;
            }
            return static_cast<std::uint32_t>(index);
        }
    }
    return invalidLodIndex;
}

std::size_t countLayerVisibleInstances(std::span<const InstanceRecord> instances,
                                       RenderLayerMask viewMask) {
    return static_cast<std::size_t>(std::count_if(instances.begin(), instances.end(),
                                                  [viewMask](const InstanceRecord& instance) {
                                                      return instanceVisible(instance, viewMask);
                                                  }));
}

} // namespace renderer::gpu_scene
