#pragma once

#include "renderer/gpu_scene/GpuScene.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace renderer::ray_tracing {

inline constexpr std::uint32_t maximumInstanceCustomIndex = (1U << 24U) - 1U;
inline constexpr std::uint8_t defaultRayVisibilityMask = 0xffU;

struct BlasGeometryPlan {
    std::uint32_t primitiveGroup = 0;
    std::uint32_t primitiveIndex = 0;
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t firstVertex = 0;
    std::uint32_t maximumVertex = 0;
    bool opaque = true;
};

struct TlasInstancePlan {
    std::uint32_t sourceInstanceIndex = 0;
    std::uint32_t blasIndex = 0;
    std::uint32_t transformIndex = 0;
    std::uint32_t materialIndex = 0;
    std::uint8_t visibilityMask = defaultRayVisibilityMask;
    bool doubleSided = false;
    bool mirrored = false;
};

struct SceneBuildPlan {
    std::vector<BlasGeometryPlan> bottomLevels;
    std::vector<TlasInstancePlan> topLevelInstances;
    std::uint64_t uniqueTriangleCount = 0;
    std::uint64_t instancedTriangleCount = 0;
};

// Converts GLM's column-major affine matrix into Vulkan's row-major 3x4
// acceleration-structure instance transform. Kept CPU-testable because a
// transposition error moves or shears every ray-traced object.
[[nodiscard]] std::array<float, 12> makeInstanceTransform3x4(const glm::mat4& world);

// The current static foundation selects the finest resident LOD for each
// primitive group. This is intentionally a planner boundary: later GPU-driven
// RT LOD selection can replace the policy without changing Vulkan AS ownership.
[[nodiscard]] SceneBuildPlan
makeSceneBuildPlan(std::size_t vertexCount, std::span<const std::uint32_t> indices,
                   std::span<const gpu_scene::PrimitiveRecord> primitives,
                   std::span<const gpu_scene::PrimitiveLodRange> primitiveLods,
                   std::span<const gpu_scene::LodSourceRecord> lodSources,
                   std::span<const gpu_scene::InstanceRecord> instances,
                   std::span<const gpu_scene::MaterialRecord> materials,
                   std::size_t transformCount);

} // namespace renderer::ray_tracing
