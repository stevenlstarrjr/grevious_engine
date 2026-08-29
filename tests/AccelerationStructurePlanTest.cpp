#include "renderer/ray_tracing/AccelerationStructurePlan.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using renderer::gpu_scene::InstanceRecord;
using renderer::gpu_scene::LodSourceRecord;
using renderer::gpu_scene::MaterialRecord;
using renderer::gpu_scene::PrimitiveLodRange;
using renderer::gpu_scene::PrimitiveRecord;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void testFinestLodAndSharedInstances() {
    const std::vector<std::uint32_t> indices{
        0U, 1U, 2U, 2U, 3U, 0U, 4U, 5U, 6U,
    };
    const std::array<PrimitiveRecord, 2> primitives{
        PrimitiveRecord{.indexCount = 6U, .firstIndex = 0U, .vertexOffset = 0},
        PrimitiveRecord{.indexCount = 3U, .firstIndex = 6U, .vertexOffset = 0},
    };
    const std::array<PrimitiveLodRange, 1> primitiveLods{
        PrimitiveLodRange{.firstLod = 0U, .lodCount = 2U},
    };
    const std::array<LodSourceRecord, 2> lodSources{
        LodSourceRecord{.primitiveIndex = 0U, .minimumProjectedDiameter = 100.0F},
        LodSourceRecord{.primitiveIndex = 1U, .minimumProjectedDiameter = 0.0F},
    };
    MaterialRecord texturedOpaqueMaterial{};
    texturedOpaqueMaterial.textureIndices = {1U, 2U, 3U, 4U};
    texturedOpaqueMaterial.textureState = {
        5U,
        renderer::gpu_scene::baseColorTextureBit |
            renderer::gpu_scene::metallicRoughnessTextureBit |
            renderer::gpu_scene::normalTextureBit | renderer::gpu_scene::occlusionTextureBit |
            renderer::gpu_scene::emissiveTextureBit,
        0U,
        renderer::gpu_scene::normalTextureBit,
    };
    const std::array<MaterialRecord, 1> materials{texturedOpaqueMaterial};
    const std::array<InstanceRecord, 2> instances{
        InstanceRecord{.meshMaterial = {0U, 0U, 0U, 0U}},
        InstanceRecord{.meshMaterial = {0U, 0U, 1U, 0U}},
    };

    const renderer::ray_tracing::SceneBuildPlan plan = renderer::ray_tracing::makeSceneBuildPlan(
        7U, indices, primitives, primitiveLods, lodSources, instances, materials, 2U);

    require(plan.bottomLevels.size() == 1U, "shared geometry must produce one BLAS");
    require(plan.topLevelInstances.size() == 2U, "both objects must produce TLAS instances");
    require(plan.bottomLevels[0].primitiveIndex == 0U, "planner must select the finest LOD");
    require(plan.bottomLevels[0].maximumVertex == 3U, "BLAS maximum vertex is incorrect");
    require(plan.uniqueTriangleCount == 2U, "unique triangle count is incorrect");
    require(plan.instancedTriangleCount == 4U, "instanced triangle count is incorrect");
    require(renderer::ray_tracing::supportsOpaquePathTracing(plan),
            "opaque textured scene should be accepted by the path tracer");
}

void testVulkanInstanceTransformLayout() {
    glm::mat4 world(1.0F);
    world[0][0] = 2.0F;
    world[1][1] = 3.0F;
    world[2][2] = 4.0F;
    world[3] = {5.0F, 6.0F, 7.0F, 1.0F};
    const std::array<float, 12> actual = renderer::ray_tracing::makeInstanceTransform3x4(world);
    const std::array<float, 12> expected{
        2.0F, 0.0F, 0.0F, 5.0F, 0.0F, 3.0F, 0.0F, 6.0F, 0.0F, 0.0F, 4.0F, 7.0F,
    };
    require(actual == expected, "Vulkan instance transform is not row-major 3x4");

    world[0][0] = std::numeric_limits<float>::infinity();
    bool rejected = false;
    try {
        static_cast<void>(renderer::ray_tracing::makeInstanceTransform3x4(world));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "non-finite RT transform must be rejected");
}

void testChangingTransformResolution() {
    std::array<renderer::gpu_scene::TransformRecord, 2> transforms{};
    transforms[0].world = glm::mat4(1.0F);
    transforms[1].world = glm::translate(glm::mat4(1.0F), glm::vec3(2.0F, 3.0F, 4.0F));
    renderer::ray_tracing::TlasInstancePlan instance{.transformIndex = 1U};

    const std::array<float, 12> initial =
        renderer::ray_tracing::resolveInstanceTransform3x4(instance, transforms);
    require(initial[3] == 2.0F && initial[7] == 3.0F && initial[11] == 4.0F,
            "TLAS instance did not resolve its selected transform");

    transforms[1].world = glm::translate(glm::mat4(1.0F), glm::vec3(-5.0F, 6.0F, 7.0F));
    const std::array<float, 12> changed =
        renderer::ray_tracing::resolveInstanceTransform3x4(instance, transforms);
    require(changed[3] == -5.0F && changed[7] == 6.0F && changed[11] == 7.0F,
            "TLAS instance did not propagate a changed world transform");
    require(changed != initial, "changed TLAS transform produced stale packed data");

    instance.transformIndex = 2U;
    bool rejected = false;
    try {
        static_cast<void>(renderer::ray_tracing::resolveInstanceTransform3x4(instance, transforms));
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    require(rejected, "out-of-range TLAS transform mapping must be rejected");
}

void testVertexOffsetUsesLocalMaximumIndex() {
    const std::array<std::uint32_t, 3> indices{0U, 1U, 2U};
    const std::array<PrimitiveRecord, 1> primitives{
        PrimitiveRecord{.indexCount = 3U, .firstIndex = 0U, .vertexOffset = 4},
    };
    const std::array<PrimitiveLodRange, 1> primitiveLods{
        PrimitiveLodRange{.firstLod = 0U, .lodCount = 1U},
    };
    const std::array<LodSourceRecord, 1> lodSources{
        LodSourceRecord{.primitiveIndex = 0U, .minimumProjectedDiameter = 0.0F},
    };
    const std::array<InstanceRecord, 1> instances{InstanceRecord{}};
    const std::array<MaterialRecord, 1> materials{MaterialRecord{}};

    const renderer::ray_tracing::SceneBuildPlan plan = renderer::ray_tracing::makeSceneBuildPlan(
        7U, indices, primitives, primitiveLods, lodSources, instances, materials, 1U);
    require(plan.bottomLevels[0].firstVertex == 4U, "vertex base offset is incorrect");
    require(plan.bottomLevels[0].maximumVertex == 2U,
            "maximum vertex must remain relative to the BLAS vertex base");
}

void testMaterialAndTransformFlags() {
    const std::array<std::uint32_t, 3> indices{0U, 1U, 2U};
    const std::array<PrimitiveRecord, 1> primitives{
        PrimitiveRecord{.indexCount = 3U, .firstIndex = 0U, .vertexOffset = 0},
    };
    const std::array<PrimitiveLodRange, 1> primitiveLods{
        PrimitiveLodRange{.firstLod = 0U, .lodCount = 1U},
    };
    const std::array<LodSourceRecord, 1> lodSources{
        LodSourceRecord{.primitiveIndex = 0U, .minimumProjectedDiameter = 0.0F},
    };
    MaterialRecord material{};
    material.textureState.z =
        renderer::gpu_scene::alphaMaskMaterial | renderer::gpu_scene::doubleSidedMaterial;
    const std::array<MaterialRecord, 1> materials{material};
    const std::array<InstanceRecord, 1> instances{
        InstanceRecord{
            .meshMaterial = {0U, 0U, 0U, renderer::gpu_scene::mirroredTransformInstance},
            .visibilityLayer = {0U, 1U << 15U, 0U, 0U},
        },
    };

    const renderer::ray_tracing::SceneBuildPlan plan = renderer::ray_tracing::makeSceneBuildPlan(
        3U, indices, primitives, primitiveLods, lodSources, instances, materials, 1U);

    require(!plan.bottomLevels[0].opaque, "alpha-tested geometry must not be marked opaque");
    require(plan.topLevelInstances[0].rayShadowMaterialMode ==
                renderer::ray_tracing::RayShadowMaterialMode::alphaMasked,
            "alpha-tested material was not classified for ray candidate evaluation");
    require(renderer::ray_tracing::supportsRayQueryShadows(plan),
            "alpha-tested material should be supported by ray-query shadows");
    require(!renderer::ray_tracing::supportsOpaquePathTracing(plan),
            "alpha-tested material must be rejected by the initial path tracer");
    require(renderer::ray_tracing::supportsPathTracing(plan),
            "alpha-tested material should be supported by path any-hit evaluation");
    require(plan.topLevelInstances[0].doubleSided, "double-sided state was not propagated");
    require(plan.topLevelInstances[0].mirrored, "mirrored transform state was not propagated");
    require(plan.topLevelInstances[0].renderLayerMask == (1ULL << 47U),
            "full-width renderer layer mask was not propagated");
    require(renderer::ray_tracing::resolveInstanceVisibilityMask(plan.topLevelInstances[0],
                                                                 1ULL << 47U) ==
                renderer::ray_tracing::defaultRayVisibilityMask,
            "visible high renderer layer did not enable the Vulkan ray mask");
    require(renderer::ray_tracing::resolveInstanceVisibilityMask(plan.topLevelInstances[0],
                                                                 1ULL << 15U) == 0U,
            "hidden renderer layer remained visible to ray traversal");
}

void testUnsupportedMaterialRequiresRasterFallback() {
    const std::array<std::uint32_t, 3> indices{0U, 1U, 2U};
    const std::array<PrimitiveRecord, 1> primitives{
        PrimitiveRecord{.indexCount = 3U, .firstIndex = 0U, .vertexOffset = 0},
    };
    const std::array<PrimitiveLodRange, 1> primitiveLods{
        PrimitiveLodRange{.firstLod = 0U, .lodCount = 1U},
    };
    const std::array<LodSourceRecord, 1> lodSources{
        LodSourceRecord{.primitiveIndex = 0U, .minimumProjectedDiameter = 0.0F},
    };
    const std::array<InstanceRecord, 1> instances{InstanceRecord{}};
    constexpr std::array unsupportedFlags{renderer::gpu_scene::alphaBlendMaterial,
                                          renderer::gpu_scene::additiveMaterial};
    for (const std::uint32_t flag : unsupportedFlags) {
        MaterialRecord material{};
        material.textureState.z = flag;
        const std::array<MaterialRecord, 1> materials{material};
        const renderer::ray_tracing::SceneBuildPlan plan =
            renderer::ray_tracing::makeSceneBuildPlan(3U, indices, primitives, primitiveLods,
                                                      lodSources, instances, materials, 1U);
        require(plan.topLevelInstances[0].rayShadowMaterialMode ==
                    renderer::ray_tracing::RayShadowMaterialMode::unsupported,
                "blended/additive material was not classified as unsupported");
        require(!renderer::ray_tracing::supportsRayQueryShadows(plan),
                "blended/additive material must retain the raster shadow fallback");
        require(!renderer::ray_tracing::supportsOpaquePathTracing(plan),
                "blended/additive material must be rejected by the initial path tracer");
        require(!renderer::ray_tracing::supportsPathTracing(plan),
                "blended/additive material must remain unsupported by path tracing");
    }
}

void testInvalidGeometryIsRejected() {
    const std::array<std::uint32_t, 3> indices{0U, 1U, 9U};
    const std::array<PrimitiveRecord, 1> primitives{
        PrimitiveRecord{.indexCount = 3U, .firstIndex = 0U, .vertexOffset = 0},
    };
    const std::array<PrimitiveLodRange, 1> primitiveLods{
        PrimitiveLodRange{.firstLod = 0U, .lodCount = 1U},
    };
    const std::array<LodSourceRecord, 1> lodSources{
        LodSourceRecord{.primitiveIndex = 0U, .minimumProjectedDiameter = 0.0F},
    };
    const std::array<InstanceRecord, 1> instances{InstanceRecord{}};
    const std::array<MaterialRecord, 1> materials{MaterialRecord{}};

    bool rejected = false;
    try {
        static_cast<void>(renderer::ray_tracing::makeSceneBuildPlan(
            3U, indices, primitives, primitiveLods, lodSources, instances, materials, 1U));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "out-of-range vertex index must be rejected");
}

void testInvalidLodRangeIsRejected() {
    const std::array<std::uint32_t, 3> indices{0U, 1U, 2U};
    const std::array<PrimitiveRecord, 1> primitives{
        PrimitiveRecord{.indexCount = 3U, .firstIndex = 0U, .vertexOffset = 0},
    };
    const std::array<PrimitiveLodRange, 1> primitiveLods{
        PrimitiveLodRange{.firstLod = 1U, .lodCount = 1U},
    };
    const std::array<LodSourceRecord, 1> lodSources{
        LodSourceRecord{.primitiveIndex = 0U, .minimumProjectedDiameter = 0.0F},
    };
    const std::array<InstanceRecord, 1> instances{InstanceRecord{}};
    const std::array<MaterialRecord, 1> materials{MaterialRecord{}};

    bool rejected = false;
    try {
        static_cast<void>(renderer::ray_tracing::makeSceneBuildPlan(
            3U, indices, primitives, primitiveLods, lodSources, instances, materials, 1U));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "out-of-range LOD range must be rejected");
}

} // namespace

int main() {
    try {
        testFinestLodAndSharedInstances();
        testVulkanInstanceTransformLayout();
        testChangingTransformResolution();
        testVertexOffsetUsesLocalMaximumIndex();
        testMaterialAndTransformFlags();
        testUnsupportedMaterialRequiresRasterFallback();
        testInvalidGeometryIsRejected();
        testInvalidLodRangeIsRejected();
        std::cout << "acceleration structure plan tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "acceleration structure plan test failure: " << error.what() << '\n';
        return 1;
    }
}
