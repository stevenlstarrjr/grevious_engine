#include "renderer/gpu_scene/GpuScene.hpp"
#include "renderer/scene/Scene.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

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

void testEmissiveTriangleDistribution() {
    using namespace renderer::gpu_scene;
    const std::array vertices{
        VertexRecord{.position = {0.0F, 0.0F, 0.0F}},
        VertexRecord{.position = {1.0F, 0.0F, 0.0F}},
        VertexRecord{.position = {0.0F, 1.0F, 0.0F}},
    };
    const std::array indices{0U, 1U, 2U};
    const std::array primitives{PrimitiveRecord{.indexCount = 3U, .materialIndex = 0U}};
    const std::array ranges{PrimitiveLodRange{.firstLod = 0U, .lodCount = 1U}};
    const std::array sources{
        LodSourceRecord{.primitiveIndex = 0U, .minimumProjectedDiameter = 0.0F}};
    const glm::mat4 scaled = glm::scale(glm::mat4(1.0F), glm::vec3(2.0F));
    const std::array transforms{
        TransformRecord{},
        TransformRecord{.world = scaled,
                        .normalToWorld = glm::transpose(glm::inverse(scaled)),
                        .previousWorld = scaled},
    };
    const std::array materials{
        MaterialRecord{.emissiveAlphaCutoff = {1.0F, 1.0F, 1.0F, 0.5F}},
        MaterialRecord{},
    };
    std::array instances{
        InstanceRecord{.meshMaterial = {0U, 0U, 0U, 0U}},
        InstanceRecord{.meshMaterial = {0U, 0U, 1U, 0U}},
        InstanceRecord{.meshMaterial = {0U, 1U, 0U, 0U}},
        InstanceRecord{.meshMaterial = {0U, 0U, 0U, 0U},
                       .visibilityLayer = {1U << 20U, 0U, 0U, 0U}},
    };
    const EmissiveTriangleDistribution distribution = buildEmissiveTriangleDistribution(
        vertices, indices, primitives, ranges, sources, instances, materials, transforms);
    require(distribution.valid() && distribution.triangles.size() == 2U,
            "Emissive distribution did not retain both emitting instances");
    require(std::abs(distribution.triangles[0].sampling.y - 0.2F) < 1.0e-6F &&
                std::abs(distribution.triangles[1].sampling.y - 0.8F) < 1.0e-6F &&
                distribution.triangles[0].sampling.x == distribution.triangles[0].sampling.y &&
                distribution.triangles[1].sampling.x == 1.0F,
            "Emissive distribution is not world-area/luminance weighted");
    require(instances[0].visibilityLayer.z == 0U && instances[0].visibilityLayer.w == 1U &&
                instances[1].visibilityLayer.z == 1U && instances[1].visibilityLayer.w == 1U &&
                instances[2].visibilityLayer.z == invalidEmissiveTriangleIndex &&
                instances[2].visibilityLayer.w == 0U &&
                instances[3].visibilityLayer.z == invalidEmissiveTriangleIndex &&
                instances[3].visibilityLayer.w == 0U,
            "Emissive instance lookup metadata is incorrect");
}

renderer::scene::Scene makeScaledTriangleScene(const glm::vec3& scale) {
    namespace rs = renderer::scene;
    rs::Scene source("scaled triangle");
    rs::MeshPrimitive primitive;
    primitive.positions = {{-1.0F, -1.0F, 0.0F}, {1.0F, -1.0F, 0.0F}, {0.0F, 1.0F, 0.0F}};
    primitive.normals = {{0.0F, 0.0F, 1.0F}, {0.0F, 0.0F, 1.0F}, {0.0F, 0.0F, 1.0F}};
    primitive.indices = {0U, 1U, 2U};
    const rs::PrimitiveHandle primitiveHandle = source.addPrimitive(std::move(primitive));
    const rs::MeshHandle meshHandle = source.addMesh(rs::Mesh{.primitives = {primitiveHandle}});
    const rs::NodeHandle node = source.addNode(
        rs::Node{.localTransform = glm::scale(glm::mat4(1.0F), scale), .mesh = meshHandle});
    source.addRoot(node);
    return source;
}

void testScaleRelativeTransformValidation() {
    using namespace renderer::gpu_scene;
    const PackedScene tiny = packScene(makeScaledTriangleScene(glm::vec3(1.0e-4F)));
    require(tiny.instances.size() == 1U && tiny.transforms.size() == 1U &&
                tiny.instances.front().centerRadius.w > 0.0F,
            "GPU scene packing rejected a small but invertible mesh transform");

    bool collapsedRejected = false;
    try {
        (void)packScene(makeScaledTriangleScene(glm::vec3(1.0F, 0.0F, 1.0F)));
    } catch (const std::runtime_error& error) {
        collapsedRejected =
            std::string_view(error.what()).find("singular") != std::string_view::npos;
    }
    require(collapsedRejected, "GPU scene packing accepted a collapsed mesh transform");
}

renderer::scene::Scene makeVertexColorTriangleScene(bool includeColors) {
    namespace rs = renderer::scene;
    rs::Scene source("vertex color triangle");
    rs::MeshPrimitive primitive;
    primitive.positions = {{-1.0F, -1.0F, 0.0F}, {1.0F, -1.0F, 0.0F},
                           {0.0F, 1.0F, 0.0F}};
    primitive.normals = {{0.0F, 0.0F, 1.0F}, {0.0F, 0.0F, 1.0F},
                         {0.0F, 0.0F, 1.0F}};
    if (includeColors) {
        primitive.colors = {{1.0F, 0.0F, 0.0F, 0.25F},
                            {0.0F, 1.0F, 0.0F, 0.50F},
                            {0.0F, 0.0F, 1.0F, 0.75F}};
    }
    primitive.indices = {0U, 1U, 2U};
    const rs::PrimitiveHandle primitiveHandle = source.addPrimitive(std::move(primitive));
    const rs::MeshHandle meshHandle = source.addMesh(rs::Mesh{.primitives = {primitiveHandle}});
    const rs::NodeHandle node = source.addNode(rs::Node{.mesh = meshHandle});
    source.addRoot(node);
    return source;
}

void testVertexColorPackingAndDefault() {
    using namespace renderer::gpu_scene;
    const PackedScene colored = packScene(makeVertexColorTriangleScene(true));
    require(colored.vertices.size() == 3U &&
                colored.vertices[0].color == glm::vec4(1.0F, 0.0F, 0.0F, 0.25F) &&
                colored.vertices[1].color == glm::vec4(0.0F, 1.0F, 0.0F, 0.50F) &&
                colored.vertices[2].color == glm::vec4(0.0F, 0.0F, 1.0F, 0.75F),
            "GPU scene packing dropped or changed glTF COLOR_0 RGBA");

    const PackedScene uncolored = packScene(makeVertexColorTriangleScene(false));
    require(uncolored.vertices.size() == 3U &&
                uncolored.vertices[0].color == glm::vec4(1.0F) &&
                uncolored.vertices[1].color == glm::vec4(1.0F) &&
                uncolored.vertices[2].color == glm::vec4(1.0F),
            "A missing COLOR_0 attribute did not pack as white");
}

void testOneTextureCanUseDistinctSemanticDescriptors() {
    namespace rs = renderer::scene;
    using namespace renderer::gpu_scene;
    const rs::TextureHandle shared = rs::TextureHandle::fromIndex(0U);
    rs::Material material;
    material.baseColorTexture.texture = shared;
    material.metallicRoughnessTexture.texture = shared;
    material.normalTexture.texture = shared;
    material.occlusionTexture.texture = shared;
    material.emissiveTexture.texture = shared;
    constexpr std::array<std::uint32_t, 1> baseColor{11U};
    constexpr std::array<std::uint32_t, 1> metallicRoughness{22U};
    constexpr std::array<std::uint32_t, 1> normal{33U};
    constexpr std::array<std::uint32_t, 1> occlusion{44U};
    constexpr std::array<std::uint32_t, 1> emissive{55U};
    const MaterialRecord packed = packMaterial(
        material, MaterialTextureDescriptorTable{
                      .baseColor = baseColor,
                      .metallicRoughness = metallicRoughness,
                      .normal = normal,
                      .occlusion = occlusion,
                      .emissive = emissive,
                  });
    require(packed.textureIndices == glm::uvec4(11U, 22U, 33U, 44U) &&
                packed.textureState.x == 55U,
            "One source texture reused across sRGB, data, and normal roles lost its "
            "usage-specific descriptors");
}

void testWeatherMaterialRolesPackIndependently() {
    namespace rs = renderer::scene;
    using namespace renderer::gpu_scene;
    rs::Material material;
    material.rainReceiver = true;
    material.wetCoat = true;
    const MaterialRecord packed = packMaterial(material, std::span<const std::uint32_t>{});
    require((packed.textureState.z & rainReceiverMaterial) != 0U &&
                (packed.textureState.z & wetCoatMaterial) != 0U,
            "Weather material roles were not preserved in GPU material flags");
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
        testEmissiveTriangleDistribution();
        testScaleRelativeTransformValidation();
        testVertexColorPackingAndDefault();
        testOneTextureCanUseDistinctSemanticDescriptors();
        testWeatherMaterialRolesPackIndependently();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
