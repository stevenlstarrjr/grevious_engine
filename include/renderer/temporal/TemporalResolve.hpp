#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <cstdint>

namespace renderer::temporal {

inline constexpr float temporalNormalDotThreshold = 0.85F;
inline constexpr float pathReconstructionEquivalentSampleCount = 4.0F;

struct HistoryRejectionInput {
    bool historyValid = false;
    bool motionValid = false;
    glm::vec2 previousUv{0.0F};
    float expectedPreviousDepth = 1.0F;
    float historyDepth = 1.0F;
    float depthThreshold = 0.00075F;
    glm::vec3 currentNormal{0.0F, 0.0F, 1.0F};
    glm::vec3 historyNormal{0.0F, 0.0F, 1.0F};
    bool currentSurface = false;
    bool historySurface = false;
};

// Path reconstruction validates sky history from reprojection and the
// surface/sky class alone. Surface history additionally requires matching
// device depth and first-hit geometric normals.
struct PathHistoryRejectionInput {
    bool historyValid = false;
    bool motionValid = false;
    glm::vec2 previousUv{0.0F};
    float expectedPreviousDepth = 1.0F;
    float historyDepth = 1.0F;
    float depthThreshold = 0.00075F;
    glm::vec3 currentGeometricNormal{0.0F, 0.0F, 1.0F};
    glm::vec3 historyGeometricNormal{0.0F, 0.0F, 1.0F};
    bool currentSurface = false;
    bool historySurface = false;
};

// CPU references for the corresponding TAA shader operations. They keep
// reprojection convention and rejection thresholds directly testable without
// requiring a Vulkan device.
[[nodiscard]] glm::vec2 reprojectHistoryUv(const glm::vec2& currentRasterUv,
                                           const glm::vec2& unjitteredMotionUv,
                                           const glm::vec2& currentJitterPixels,
                                           const glm::vec2& previousJitterPixels,
                                           std::uint32_t width, std::uint32_t height);

[[nodiscard]] bool acceptsHistory(const HistoryRejectionInput& input);

[[nodiscard]] bool acceptsPathHistory(const PathHistoryRejectionInput& input);

[[nodiscard]] float temporalHistoryWeight(const glm::vec2& motionUv, std::uint32_t width,
                                          std::uint32_t height);

[[nodiscard]] float pathReconstructionHistoryWeight(const glm::vec2& motionUv,
                                                     std::uint32_t width,
                                                     std::uint32_t height,
                                                     std::uint32_t sampleIndex);

[[nodiscard]] glm::vec2 pathTemporalMoments(float currentLuminance,
                                            const glm::vec2& historyMoments,
                                            float historyWeight);

[[nodiscard]] float pathTemporalVariance(const glm::vec2& moments);

[[nodiscard]] float pathSpatialFilterStrength(std::uint32_t sampleIndex);

[[nodiscard]] glm::vec3 clipHistoryColor(const glm::vec3& history,
                                         const glm::vec3& neighborhoodMinimum,
                                         const glm::vec3& neighborhoodMaximum);

} // namespace renderer::temporal
