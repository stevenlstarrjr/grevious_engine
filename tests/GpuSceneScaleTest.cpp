#include "renderer/gpu_scene/GpuScene.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace gpu = renderer::gpu_scene;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void appendRepeatedTriangles(std::vector<std::uint32_t>& indices, std::uint32_t count) {
    for (std::uint32_t triangle = 0; triangle < count; ++triangle) {
        indices.insert(indices.end(), {0U, 1U, 2U});
    }
}

} // namespace

int main() {
    try {
        const std::array vertices{
            gpu::VertexRecord{.position = {-1.0F, -1.0F, 0.0F}},
            gpu::VertexRecord{.position = {1.0F, -1.0F, 0.0F}},
            gpu::VertexRecord{.position = {0.0F, 1.0F, 0.0F}},
        };
        std::vector<std::uint32_t> indices;
        appendRepeatedTriangles(indices, 257U);
        const std::uint32_t mediumFirst = static_cast<std::uint32_t>(indices.size());
        appendRepeatedTriangles(indices, 65U);
        const std::uint32_t coarseFirst = static_cast<std::uint32_t>(indices.size());
        appendRepeatedTriangles(indices, 17U);
        const std::array primitives{
            gpu::PrimitiveRecord{.indexCount = 257U * 3U, .firstIndex = 0U},
            gpu::PrimitiveRecord{.indexCount = 65U * 3U, .firstIndex = mediumFirst},
            gpu::PrimitiveRecord{.indexCount = 17U * 3U, .firstIndex = coarseFirst},
        };
        const std::array primitiveLods{gpu::PrimitiveLodRange{.firstLod = 0U, .lodCount = 3U}};
        const std::array lodSources{
            gpu::LodSourceRecord{.primitiveIndex = 0U, .minimumProjectedDiameter = 256.0F},
            gpu::LodSourceRecord{.primitiveIndex = 1U, .minimumProjectedDiameter = 64.0F},
            gpu::LodSourceRecord{.primitiveIndex = 2U, .minimumProjectedDiameter = 0.0F},
        };
        const gpu::MeshletHierarchy hierarchy =
            gpu::buildMeshletHierarchy(vertices, indices, primitives, primitiveLods, lodSources);
        require(hierarchy.meshlets.size() == 8U && hierarchy.lods[0].meshletCount == 5U &&
                    hierarchy.lods[1].meshletCount == 2U && hierarchy.lods[2].meshletCount == 1U,
                "Three-level hierarchy produced an unexpected meshlet count");

        constexpr std::size_t instanceCount = 32U * 1024U;
        std::vector<gpu::InstanceRecord> instances(instanceCount);
        for (std::size_t index = 0; index < instances.size(); ++index) {
            instances[index].centerRadius = {static_cast<float>(index % 256U),
                                             static_cast<float>(index / 256U), 0.0F, 1.0F};
            instances[index].meshMaterial.x = 0U;
        }
        const std::vector<gpu::MeshletWorkRecord> work = gpu::makeMeshletWorkRecords(
            hierarchy.primitiveLods, hierarchy.lods, hierarchy.meshlets, instances);
        require(work.size() == instanceCount * hierarchy.meshlets.size(),
                "Scale scene lost or duplicated flattened work candidates");
        require(work.front().instanceIndex == 0U && work.front().lodIndex == 0U &&
                    work[7].lodIndex == 2U && work[8].instanceIndex == 1U &&
                    work.back().instanceIndex == instanceCount - 1U,
                "Scale work ordering is not deterministic by instance, LOD, and meshlet");
        require(gpu::selectLod(hierarchy.lods, 512.0F) == 0U &&
                    gpu::selectLod(hierarchy.lods, 255.0F) == 1U &&
                    gpu::selectLod(hierarchy.lods, 63.0F) == 2U,
                "Scale hierarchy projected-size selection is invalid");
        require(gpu::countLayerVisibleInstances(instances, renderer::defaultViewVisibilityMask) ==
                    instanceCount,
                "Scale scene default visibility eligibility is unstable");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
