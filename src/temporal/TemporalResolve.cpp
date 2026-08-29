#include "renderer/temporal/TemporalResolve.hpp"

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace renderer::temporal {
namespace {

bool finite(const glm::vec2& value) { return std::isfinite(value.x) && std::isfinite(value.y); }

bool finite(const glm::vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool validExtent(std::uint32_t width, std::uint32_t height) { return width != 0U && height != 0U; }

} // namespace

glm::vec2 reprojectHistoryUv(const glm::vec2& currentRasterUv, const glm::vec2& unjitteredMotionUv,
                             const glm::vec2& currentJitterPixels,
                             const glm::vec2& previousJitterPixels, std::uint32_t width,
                             std::uint32_t height) {
    if (!validExtent(width, height)) {
        throw std::invalid_argument("Temporal reprojection requires a nonzero render extent");
    }
    const glm::vec2 extent(static_cast<float>(width), static_cast<float>(height));
    return currentRasterUv - unjitteredMotionUv +
           (previousJitterPixels - currentJitterPixels) / extent;
}

bool acceptsHistory(const HistoryRejectionInput& input) {
    if (!input.historyValid || !input.motionValid || !finite(input.previousUv) ||
        input.previousUv.x < 0.0F || input.previousUv.y < 0.0F || input.previousUv.x > 1.0F ||
        input.previousUv.y > 1.0F || !std::isfinite(input.expectedPreviousDepth) ||
        !std::isfinite(input.historyDepth) || !std::isfinite(input.depthThreshold) ||
        input.depthThreshold < 0.0F ||
        std::abs(input.historyDepth - input.expectedPreviousDepth) > input.depthThreshold ||
        input.currentSurface != input.historySurface || !finite(input.currentNormal) ||
        !finite(input.historyNormal)) {
        return false;
    }
    const float currentLength = glm::length(input.currentNormal);
    const float historyLength = glm::length(input.historyNormal);
    if (currentLength <= 1.0e-4F || historyLength <= 1.0e-4F) {
        return false;
    }
    return glm::dot(input.currentNormal / currentLength, input.historyNormal / historyLength) >=
           temporalNormalDotThreshold;
}

bool acceptsPathHistory(const PathHistoryRejectionInput& input) {
    if (!input.historyValid || !input.motionValid || !finite(input.previousUv) ||
        input.previousUv.x < 0.0F || input.previousUv.y < 0.0F || input.previousUv.x > 1.0F ||
        input.previousUv.y > 1.0F || input.currentSurface != input.historySurface) {
        return false;
    }
    if (!input.currentSurface) {
        return true;
    }
    if (!std::isfinite(input.expectedPreviousDepth) || !std::isfinite(input.historyDepth) ||
        !std::isfinite(input.depthThreshold) || input.depthThreshold < 0.0F ||
        std::abs(input.historyDepth - input.expectedPreviousDepth) > input.depthThreshold ||
        !finite(input.currentGeometricNormal) || !finite(input.historyGeometricNormal)) {
        return false;
    }
    const float currentLength = glm::length(input.currentGeometricNormal);
    const float historyLength = glm::length(input.historyGeometricNormal);
    if (currentLength <= 1.0e-4F || historyLength <= 1.0e-4F) {
        return false;
    }
    return glm::dot(input.currentGeometricNormal / currentLength,
                    input.historyGeometricNormal / historyLength) >=
           temporalNormalDotThreshold;
}

float temporalHistoryWeight(const glm::vec2& motionUv, std::uint32_t width, std::uint32_t height) {
    if (!validExtent(width, height)) {
        throw std::invalid_argument("Temporal history weighting requires a nonzero render extent");
    }
    const glm::vec2 motionPixels =
        motionUv * glm::vec2(static_cast<float>(width), static_cast<float>(height));
    return std::clamp(0.92F - glm::length(motionPixels) * 0.08F, 0.10F, 0.92F);
}

float pathReconstructionHistoryWeight(const glm::vec2& motionUv, std::uint32_t width,
                                      std::uint32_t height, std::uint32_t sampleIndex) {
    const float convergenceWeight =
        pathReconstructionEquivalentSampleCount /
        (static_cast<float>(sampleIndex) + pathReconstructionEquivalentSampleCount + 1.0F);
    return std::min(temporalHistoryWeight(motionUv, width, height), convergenceWeight);
}

glm::vec2 pathTemporalMoments(float currentLuminance, const glm::vec2& historyMoments,
                              float historyWeight) {
    if (!std::isfinite(currentLuminance) || currentLuminance < 0.0F ||
        !finite(historyMoments) || glm::any(glm::lessThan(historyMoments, glm::vec2(0.0F))) ||
        !std::isfinite(historyWeight) || historyWeight < 0.0F || historyWeight > 1.0F) {
        throw std::invalid_argument(
            "Path temporal moments require finite nonnegative values and a unit weight");
    }
    const float boundedLuminance = std::min(currentLuminance, 255.0F);
    const glm::vec2 currentMoments(boundedLuminance,
                                   boundedLuminance * boundedLuminance);
    const glm::vec2 boundedHistory = glm::clamp(historyMoments, glm::vec2(0.0F),
                                                glm::vec2(255.0F, 65504.0F));
    return glm::mix(currentMoments, boundedHistory, historyWeight);
}

float pathTemporalVariance(const glm::vec2& moments) {
    if (!finite(moments) || glm::any(glm::lessThan(moments, glm::vec2(0.0F)))) {
        throw std::invalid_argument("Path temporal variance requires finite moments");
    }
    return std::clamp(moments.y - moments.x * moments.x, 0.0F, 65504.0F);
}

float pathSpatialFilterStrength(std::uint32_t sampleIndex) {
    return std::clamp(4.0F / (static_cast<float>(sampleIndex) + 4.0F), 0.0F, 1.0F);
}

glm::vec3 clipHistoryColor(const glm::vec3& history, const glm::vec3& neighborhoodMinimum,
                           const glm::vec3& neighborhoodMaximum) {
    if (!finite(history) || !finite(neighborhoodMinimum) || !finite(neighborhoodMaximum) ||
        glm::any(glm::greaterThan(neighborhoodMinimum, neighborhoodMaximum))) {
        throw std::invalid_argument("Temporal color clipping requires finite ordered bounds");
    }
    return glm::clamp(history, neighborhoodMinimum, neighborhoodMaximum);
}

} // namespace renderer::temporal
