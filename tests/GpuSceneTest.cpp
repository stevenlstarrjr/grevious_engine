#include "renderer/gpu_scene/GpuScene.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void testMeshletLodWorkGeneration() {
    using namespace renderer::gpu_scene;
    const std::array vertices{
        VertexRecord{.position = {-1.0F, -1.0F, 0.0F}},
        VertexRecord{.position = {1.0F, -1.0F, 0.0F}},
        VertexRecord{.position = {0.0F, 1.0F, 0.0F}},
    };
    std::vector<std::uint32_t> indices;
    indices.reserve(65U * 3U);
    for (std::uint32_t triangle = 0; triangle < 65U; ++triangle) {
        indices.insert(indices.end(), {0U, 1U, 2U});
    }
    const std::array primitives{
        PrimitiveRecord{.indexCount = static_cast<std::uint32_t>(indices.size())},
        PrimitiveRecord{.indexCount = static_cast<std::uint32_t>(indices.size())},
    };
    const std::array primitiveLods{PrimitiveLodRange{.firstLod = 0U, .lodCount = 2U}};
    const std::array lodSources{
        LodSourceRecord{.primitiveIndex = 0U, .minimumProjectedDiameter = 96.0F},
        LodSourceRecord{.primitiveIndex = 1U, .minimumProjectedDiameter = 0.0F},
    };
    const MeshletHierarchy hierarchy =
        buildMeshletHierarchy(vertices, indices, primitives, primitiveLods, lodSources);
    require(hierarchy.primitiveLods.size() == 1U && hierarchy.lods.size() == 2U &&
                hierarchy.meshlets.size() == 4U,
            "Meshlet hierarchy did not partition both 65-triangle LODs");
    require(hierarchy.meshlets[0].indexCount == maxMeshletTriangles * 3U &&
                hierarchy.meshlets[1].indexCount == 3U &&
                hierarchy.meshlets[0].centerRadius.w > 0.0F,
            "Meshlet triangle bounds or local bounds are invalid");
    require(selectLod(hierarchy.lods, 96.0F) == 0U && selectLod(hierarchy.lods, 95.0F) == 1U &&
                selectLod(hierarchy.lods, -1.0F) == invalidLodIndex,
            "Projected-size LOD selection does not honor exact boundaries");

    const std::array instances{InstanceRecord{}};
    const std::vector<MeshletWorkRecord> work = makeMeshletWorkRecords(
        hierarchy.primitiveLods, hierarchy.lods, hierarchy.meshlets, instances);
    require(work.size() == 4U && work[0].instanceIndex == 0U && work[0].lodIndex == 0U &&
                work[2].lodIndex == 1U && work[0].projectedDiameterRange.x == 96.0F &&
                work[2].projectedDiameterRange.y == 96.0F,
            "Flattened meshlet work lost its instance or LOD selection band");
}

void testMeshletVertexCapacity() {
    using namespace renderer::gpu_scene;
    std::vector<VertexRecord> vertices(66U);
    std::vector<std::uint32_t> indices;
    indices.reserve(vertices.size());
    for (std::uint32_t index = 0; index < vertices.size(); ++index) {
        vertices[index].position = {static_cast<float>(index), 0.0F, 0.0F};
        indices.push_back(index);
    }
    const std::array primitives{
        PrimitiveRecord{.indexCount = static_cast<std::uint32_t>(indices.size())}};
    const MeshletHierarchy hierarchy =
        buildSingleLodMeshletHierarchy(vertices, indices, primitives);
    require(hierarchy.meshlets.size() == 2U && hierarchy.meshlets[0].indexCount == 21U * 3U &&
                hierarchy.meshlets[1].indexCount == 3U,
            "Meshlet construction exceeded its 64-vertex capacity");
}

} // namespace

int main() {
    try {
        const auto scene = renderer::gpu_scene::makePbrSphereGrid();
        require(scene.instances.size() == 35 && scene.materials.size() == 35 &&
                    scene.transforms.size() == 35,
                "GPU PBR grid does not contain 35 instances, materials, and transforms");
        require(scene.materials.front().metallicRoughnessOcclusionNormal.x == 0.0F &&
                    scene.materials.back().metallicRoughnessOcclusionNormal.x == 1.0F,
                "GPU PBR grid does not span the metallic range");
        require(std::abs(scene.materials.front().metallicRoughnessOcclusionNormal.y - 0.10F) <
                        1.0e-6F &&
                    scene.materials.back().metallicRoughnessOcclusionNormal.y == 1.0F,
                "GPU PBR grid does not span its stable roughness range");
        for (std::size_t index = 0; index < scene.instances.size(); ++index) {
            require(
                scene.instances[index].centerRadius.w > 0.0F &&
                    scene.instances[index].meshMaterial.y == static_cast<std::uint32_t>(index) &&
                    scene.instances[index].meshMaterial.z == static_cast<std::uint32_t>(index) &&
                    scene.materials[index].metallicRoughnessOcclusionNormal.z >= 0.0F &&
                    scene.materials[index].metallicRoughnessOcclusionNormal.z <= 1.0F &&
                    scene.materials[index].textureIndices.x == 0U &&
                    renderer::gpu_scene::unpackRenderLayerMask(
                        glm::uvec2(scene.instances[index].visibilityLayer)) ==
                        renderer::defaultInstanceRenderLayerMask &&
                    (scene.materials[index].textureState.y &
                     renderer::gpu_scene::baseColorTextureBit) != 0U,
                "GPU instance/material relationship is invalid");
        }
        testMeshletLodWorkGeneration();
        testMeshletVertexCapacity();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
