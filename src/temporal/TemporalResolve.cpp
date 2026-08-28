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

float temporalHistoryWeight(const glm::vec2& motionUv, std::uint32_t width, std::uint32_t height) {
    if (!validExtent(width, height)) {
        throw std::invalid_argument("Temporal history weighting requires a nonzero render extent");
    }
    const glm::vec2 motionPixels =
        motionUv * glm::vec2(static_cast<float>(width), static_cast<float>(height));
    return std::clamp(0.92F - glm::length(motionPixels) * 0.08F, 0.10F, 0.92F);
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
