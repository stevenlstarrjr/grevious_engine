#pragma once

#include "renderer/visibility/Visibility.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace renderer::scene {
class Scene;
struct Material;
} // namespace renderer::scene

namespace renderer::gpu_scene {

// Plain, fixed-layout records are the CPU/GPU residency boundary. Importers and
// future scripting adapters produce higher-level scene commands; only the
// renderer packs those commands into these records.
struct alignas(16) InstanceRecord {
    glm::vec4 centerRadius{0.0F};
    glm::uvec4 meshMaterial{0U};
    // Low/high 32-bit words of RenderLayerMask. z/w identify the first
    // emissive-triangle light and triangle count when this instance emits.
    // Splitting the mask avoids making optional shaderInt64 a requirement.
    glm::uvec4 visibilityLayer{static_cast<std::uint32_t>(defaultInstanceRenderLayerMask), 0U, 0U,
                               0U};
};

enum InstanceFlag : std::uint32_t {
    mirroredTransformInstance = 1U << 0U,
};

struct VertexRecord {
    glm::vec3 position{0.0F};
    glm::vec3 normal{0.0F, 0.0F, 1.0F};
    glm::vec2 uv0{0.0F};
    glm::vec4 tangent{1.0F, 0.0F, 0.0F, 1.0F};
    glm::vec2 uv1{0.0F};
};

struct alignas(16) TransformRecord {
    glm::mat4 world{1.0F};
    glm::mat4 normalToWorld{1.0F};
    glm::mat4 previousWorld{1.0F};
};

struct alignas(16) PrimitiveRecord {
    std::uint32_t indexCount = 0;
    std::uint32_t firstIndex = 0;
    std::int32_t vertexOffset = 0;
    std::uint32_t materialIndex = 0;
    glm::vec4 centerRadius{0.0F};
};

inline constexpr std::uint32_t maxMeshletVertices = 64;
inline constexpr std::uint32_t maxMeshletTriangles = 64;
inline constexpr std::uint32_t invalidLodIndex = ~std::uint32_t{0};

// A meshlet is a bounded, contiguous triangle range in the unified index
// buffer. This baseline does not duplicate/remap vertices; the record is
// directly consumable by indexed indirect drawing on every required GPU.
struct alignas(16) MeshletRecord {
    std::uint32_t indexCount = 0;
    std::uint32_t firstIndex = 0;
    std::int32_t vertexOffset = 0;
    std::uint32_t sourcePrimitive = 0;
    glm::vec4 centerRadius{0.0F};
    // Object-space cone axis and minimum triangle-normal dot. A negative
    // cutoff disables future cone culling for degenerate/opposed clusters.
    glm::vec4 normalCone{0.0F, 0.0F, 1.0F, -1.0F};
};

// Primitive groups point at consecutive LOD sources. Sources are ordered from
// finest to coarsest and carry the minimum projected diameter, in pixels, at
// which they are selected. The coarsest source must have a zero threshold.
struct PrimitiveLodRange {
    std::uint32_t firstLod = 0;
    std::uint32_t lodCount = 0;
};

struct LodSourceRecord {
    std::uint32_t primitiveIndex = 0;
    float minimumProjectedDiameter = 0.0F;
};

struct alignas(16) LodRecord {
    std::uint32_t firstMeshlet = 0;
    std::uint32_t meshletCount = 0;
    float minimumProjectedDiameter = 0.0F;
    float maximumProjectedDiameter = 0.0F;
};

// Candidate GPU work is flattened per instance, LOD, and meshlet. Compute
// culling selects one LOD band, compacts surviving candidates, and writes a
// matching VkDrawIndexedIndirectCommand plus visible-instance index.
struct alignas(16) MeshletWorkRecord {
    std::uint32_t indexCount = 0;
    std::uint32_t firstIndex = 0;
    std::int32_t vertexOffset = 0;
    std::uint32_t instanceIndex = 0;
    glm::vec2 projectedDiameterRange{0.0F};
    std::uint32_t lodIndex = 0;
    std::uint32_t reserved = 0;
};

struct alignas(16) MaterialRecord {
    glm::vec4 baseColorFactor{1.0F};
    // RGB emissive factor, alpha cutoff.
    glm::vec4 emissiveAlphaCutoff{0.0F, 0.0F, 0.0F, 0.5F};
    // Metallic, roughness, occlusion strength, normal scale.
    glm::vec4 metallicRoughnessOcclusionNormal{1.0F};
    // Base-color, metallic/roughness, normal, occlusion texture indices.
    glm::uvec4 textureIndices{0U};
    // Emissive texture index, texture-presence mask, material flags, UV1 mask.
    glm::uvec4 textureState{0U};
};

// One record per triangle in an emissive instance's finest resident RT LOD.
// identity is instance index, local triangle index, material index, reserved.
// sampling is normalized CDF, selection probability, world area, power proxy.
struct alignas(16) EmissiveTriangleRecord {
    glm::uvec4 identity{0U};
    glm::vec4 sampling{0.0F};
};

struct EmissiveTriangleDistribution {
    std::vector<EmissiveTriangleRecord> triangles;

    [[nodiscard]] bool valid() const;
};

enum MaterialTextureBit : std::uint32_t {
    baseColorTextureBit = 1U << 0U,
    metallicRoughnessTextureBit = 1U << 1U,
    normalTextureBit = 1U << 2U,
    occlusionTextureBit = 1U << 3U,
    emissiveTextureBit = 1U << 4U,
};

enum MaterialFlag : std::uint32_t {
    alphaMaskMaterial = 1U << 0U,
    alphaBlendMaterial = 1U << 1U,
    doubleSidedMaterial = 1U << 2U,
    additiveMaterial = 1U << 3U,
};

inline constexpr std::uint32_t invalidTextureDescriptor = ~std::uint32_t{0};
inline constexpr std::uint32_t invalidEmissiveTriangleIndex = ~std::uint32_t{0};

struct DrawIndexedIndirectRecord {
    std::uint32_t indexCount = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t firstIndex = 0;
    std::int32_t vertexOffset = 0;
    std::uint32_t firstInstance = 0;
};

struct SceneRecords {
    std::vector<InstanceRecord> instances;
    std::vector<MaterialRecord> materials;
    std::vector<TransformRecord> transforms;
};

struct PackedScene {
    std::vector<VertexRecord> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<PrimitiveRecord> primitives;
    std::vector<TransformRecord> transforms;
    std::vector<InstanceRecord> instances;
    std::vector<MaterialRecord> materials;
    std::vector<PrimitiveLodRange> primitiveLods;
    // Retained independently from meshlets because BLAS construction consumes
    // whole primitive LODs rather than raster meshlet work records.
    std::vector<LodSourceRecord> lodSources;
    std::vector<LodRecord> lods;
    std::vector<MeshletRecord> meshlets;
};

struct MeshletHierarchy {
    std::vector<PrimitiveLodRange> primitiveLods;
    std::vector<LodRecord> lods;
    std::vector<MeshletRecord> meshlets;
};

[[nodiscard]] constexpr glm::uvec2 packRenderLayerMask(RenderLayerMask mask) {
    return {static_cast<std::uint32_t>(mask), static_cast<std::uint32_t>(mask >> 32U)};
}

[[nodiscard]] constexpr RenderLayerMask unpackRenderLayerMask(const glm::uvec2& mask) {
    return static_cast<RenderLayerMask>(mask.x) | (static_cast<RenderLayerMask>(mask.y) << 32U);
}

[[nodiscard]] constexpr bool instanceVisible(const InstanceRecord& instance,
                                             RenderLayerMask viewMask) {
    const glm::uvec2 packedView = packRenderLayerMask(viewMask);
    return ((instance.visibilityLayer.x & packedView.x) |
            (instance.visibilityLayer.y & packedView.y)) != 0U;
}

[[nodiscard]] std::size_t countLayerVisibleInstances(std::span<const InstanceRecord> instances,
                                                     RenderLayerMask viewMask);

static_assert(offsetof(InstanceRecord, centerRadius) == 0);
static_assert(offsetof(InstanceRecord, meshMaterial) == 16);
static_assert(offsetof(InstanceRecord, visibilityLayer) == 32);
static_assert(sizeof(InstanceRecord) == 12 * sizeof(float));
static_assert(alignof(InstanceRecord) == 16);
static_assert(offsetof(VertexRecord, position) == 0);
static_assert(offsetof(VertexRecord, normal) == 3 * sizeof(float));
static_assert(offsetof(VertexRecord, uv0) == 6 * sizeof(float));
static_assert(offsetof(VertexRecord, tangent) == 8 * sizeof(float));
static_assert(offsetof(VertexRecord, uv1) == 12 * sizeof(float));
static_assert(sizeof(VertexRecord) == 14 * sizeof(float));
static_assert(sizeof(TransformRecord) == 48 * sizeof(float));
static_assert(alignof(TransformRecord) == 16);
static_assert(sizeof(PrimitiveRecord) == 8 * sizeof(std::uint32_t));
static_assert(alignof(PrimitiveRecord) == 16);
static_assert(sizeof(MeshletRecord) == 12 * sizeof(std::uint32_t));
static_assert(alignof(MeshletRecord) == 16);
static_assert(sizeof(PrimitiveLodRange) == 2 * sizeof(std::uint32_t));
static_assert(sizeof(LodSourceRecord) == 2 * sizeof(std::uint32_t));
static_assert(sizeof(LodRecord) == 4 * sizeof(std::uint32_t));
static_assert(alignof(LodRecord) == 16);
static_assert(sizeof(MeshletWorkRecord) == 8 * sizeof(std::uint32_t));
static_assert(alignof(MeshletWorkRecord) == 16);
static_assert(sizeof(MaterialRecord) == 20 * sizeof(float));
static_assert(alignof(MaterialRecord) == 16);
static_assert(sizeof(EmissiveTriangleRecord) == 8 * sizeof(std::uint32_t));
static_assert(alignof(EmissiveTriangleRecord) == 16);
static_assert(sizeof(DrawIndexedIndirectRecord) == 5 * sizeof(std::uint32_t));

[[nodiscard]] SceneRecords makePbrSphereGrid();
[[nodiscard]] MaterialRecord
packMaterial(const scene::Material& material,
             std::span<const std::uint32_t> textureDescriptorIndices = {});
[[nodiscard]] std::vector<MaterialRecord>
packMaterials(const scene::Scene& scene,
              std::span<const std::uint32_t> textureDescriptorIndices = {});
[[nodiscard]] PackedScene packScene(const scene::Scene& scene,
                                    std::span<const std::uint32_t> textureDescriptorIndices = {});
[[nodiscard]] EmissiveTriangleDistribution buildEmissiveTriangleDistribution(
    std::span<const VertexRecord> vertices, std::span<const std::uint32_t> indices,
    std::span<const PrimitiveRecord> primitives,
    std::span<const PrimitiveLodRange> primitiveLods,
    std::span<const LodSourceRecord> lodSources, std::span<InstanceRecord> instances,
    std::span<const MaterialRecord> materials, std::span<const TransformRecord> transforms,
    RenderLayerMask viewMask = defaultViewVisibilityMask);
[[nodiscard]] MeshletHierarchy buildMeshletHierarchy(
    std::span<const VertexRecord> vertices, std::span<const std::uint32_t> indices,
    std::span<const PrimitiveRecord> primitives, std::span<const PrimitiveLodRange> primitiveLods,
    std::span<const LodSourceRecord> lodSources,
    std::uint32_t maximumTrianglesPerMeshlet = maxMeshletTriangles);
[[nodiscard]] MeshletHierarchy
buildSingleLodMeshletHierarchy(std::span<const VertexRecord> vertices,
                               std::span<const std::uint32_t> indices,
                               std::span<const PrimitiveRecord> primitives,
                               std::uint32_t maximumTrianglesPerMeshlet = maxMeshletTriangles);
[[nodiscard]] std::vector<MeshletWorkRecord>
makeMeshletWorkRecords(std::span<const PrimitiveLodRange> primitiveLods,
                       std::span<const LodRecord> lods, std::span<const MeshletRecord> meshlets,
                       std::span<const InstanceRecord> instances);
[[nodiscard]] std::uint32_t selectLod(std::span<const LodRecord> lods, float projectedDiameter);

} // namespace renderer::gpu_scene
