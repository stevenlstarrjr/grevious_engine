#include "renderer/ray_tracing/AccelerationStructurePlan.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace renderer::ray_tracing {
namespace {

bool materialIsOpaque(const gpu_scene::MaterialRecord& material) {
    constexpr std::uint32_t nonOpaqueFlags =
        gpu_scene::alphaMaskMaterial | gpu_scene::alphaBlendMaterial | gpu_scene::additiveMaterial;
    return (material.textureState.z & nonOpaqueFlags) == 0U;
}

} // namespace

std::array<float, 12> makeInstanceTransform3x4(const glm::mat4& world) {
    std::array<float, 12> transform{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 4; ++column) {
            const float value = world[column][row];
            if (!std::isfinite(value)) {
                throw std::invalid_argument("RT instance transform contains a non-finite value");
            }
            transform[static_cast<std::size_t>(row * 4 + column)] = value;
        }
    }
    return transform;
}

SceneBuildPlan makeSceneBuildPlan(std::size_t vertexCount, std::span<const std::uint32_t> indices,
                                  std::span<const gpu_scene::PrimitiveRecord> primitives,
                                  std::span<const gpu_scene::PrimitiveLodRange> primitiveLods,
                                  std::span<const gpu_scene::LodSourceRecord> lodSources,
                                  std::span<const gpu_scene::InstanceRecord> instances,
                                  std::span<const gpu_scene::MaterialRecord> materials,
                                  std::size_t transformCount) {
    if (vertexCount == 0U || vertexCount > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("RT build planning requires a nonempty 32-bit vertex range");
    }
    if (indices.empty() || primitives.empty() || primitiveLods.empty() || lodSources.empty() ||
        instances.empty() || materials.empty() || transformCount == 0U) {
        throw std::invalid_argument("RT build planning requires complete resident scene data");
    }
    if (primitiveLods.size() > maximumInstanceCustomIndex + std::size_t{1}) {
        throw std::length_error("RT primitive-group count exceeds its 24-bit instance mapping");
    }
    if (instances.size() > maximumInstanceCustomIndex + std::size_t{1}) {
        throw std::length_error("RT instance count exceeds Vulkan's 24-bit custom index");
    }

    SceneBuildPlan plan;
    plan.bottomLevels.reserve(primitiveLods.size());
    for (std::size_t groupIndex = 0; groupIndex < primitiveLods.size(); ++groupIndex) {
        const gpu_scene::PrimitiveLodRange range = primitiveLods[groupIndex];
        const std::uint64_t lodEnd = static_cast<std::uint64_t>(range.firstLod) + range.lodCount;
        if (range.lodCount == 0U || lodEnd > lodSources.size()) {
            throw std::runtime_error("RT primitive group has an invalid LOD range");
        }
        const gpu_scene::LodSourceRecord& finest = lodSources[range.firstLod];
        if (finest.primitiveIndex >= primitives.size()) {
            throw std::runtime_error("RT LOD references an invalid primitive");
        }
        const gpu_scene::PrimitiveRecord& primitive = primitives[finest.primitiveIndex];
        const std::uint64_t indexEnd =
            static_cast<std::uint64_t>(primitive.firstIndex) + primitive.indexCount;
        if (primitive.indexCount == 0U || primitive.indexCount % 3U != 0U ||
            indexEnd > indices.size() || primitive.vertexOffset < 0) {
            throw std::runtime_error("RT BLAS geometry is not a valid indexed triangle list");
        }

        const std::uint32_t firstVertex = static_cast<std::uint32_t>(primitive.vertexOffset);
        std::uint32_t maximumVertex = 0U;
        for (std::uint32_t offset = 0; offset < primitive.indexCount; ++offset) {
            const std::uint32_t localVertex =
                indices[static_cast<std::size_t>(primitive.firstIndex) + offset];
            const std::uint64_t referenced = static_cast<std::uint64_t>(firstVertex) + localVertex;
            if (referenced >= vertexCount) {
                throw std::runtime_error("RT BLAS index references a vertex outside residency");
            }
            maximumVertex = std::max(maximumVertex, localVertex);
        }

        const bool primitiveOpaque = primitive.materialIndex < materials.size()
                                         ? materialIsOpaque(materials[primitive.materialIndex])
                                         : true;
        plan.bottomLevels.push_back({
            .primitiveGroup = static_cast<std::uint32_t>(groupIndex),
            .primitiveIndex = finest.primitiveIndex,
            .firstIndex = primitive.firstIndex,
            .indexCount = primitive.indexCount,
            .firstVertex = firstVertex,
            .maximumVertex = maximumVertex,
            .opaque = primitiveOpaque,
        });
        plan.uniqueTriangleCount += primitive.indexCount / 3U;
    }

    plan.topLevelInstances.reserve(instances.size());
    for (std::size_t instanceIndex = 0; instanceIndex < instances.size(); ++instanceIndex) {
        const gpu_scene::InstanceRecord& instance = instances[instanceIndex];
        const std::uint32_t primitiveGroup = instance.meshMaterial.x;
        const std::uint32_t materialIndex = instance.meshMaterial.y;
        const std::uint32_t transformIndex = instance.meshMaterial.z;
        if (primitiveGroup >= plan.bottomLevels.size() || materialIndex >= materials.size() ||
            transformIndex >= transformCount) {
            throw std::runtime_error(
                "RT instance references invalid geometry, material, or transform");
        }
        const gpu_scene::MaterialRecord& material = materials[materialIndex];
        if (!materialIsOpaque(material)) {
            // BLAS geometry flags cannot vary per TLAS instance. One cutout use
            // makes every shared use conservative/non-opaque for future alpha tests.
            plan.bottomLevels[primitiveGroup].opaque = false;
        }
        const std::uint32_t flags = instance.meshMaterial.w;
        plan.topLevelInstances.push_back({
            .sourceInstanceIndex = static_cast<std::uint32_t>(instanceIndex),
            .blasIndex = primitiveGroup,
            .transformIndex = transformIndex,
            .materialIndex = materialIndex,
            .doubleSided = (material.textureState.z & gpu_scene::doubleSidedMaterial) != 0U,
            .mirrored = (flags & gpu_scene::mirroredTransformInstance) != 0U,
        });
        plan.instancedTriangleCount += plan.bottomLevels[primitiveGroup].indexCount / 3U;
    }
    return plan;
}

} // namespace renderer::ray_tracing
