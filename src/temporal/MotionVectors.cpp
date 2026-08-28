#include "renderer/temporal/MotionVectors.hpp"

#include <glm/common.hpp>
#include <glm/vec4.hpp>

#include <cmath>

namespace renderer::temporal {
namespace {

bool finite(const glm::vec4& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) &&
           std::isfinite(value.w);
}

bool finite(const glm::vec2& value) {
    return std::isfinite(value.x) && std::isfinite(value.y);
}

} // namespace

RigidMotionSample projectRigidMotion(
    const glm::vec3& localPosition, const glm::mat4& currentWorld,
    const glm::mat4& previousWorld, const glm::mat4& currentUnjitteredViewProjection,
    const glm::mat4& previousUnjitteredViewProjection) {
    const glm::vec4 currentClip = currentUnjitteredViewProjection *
                                  currentWorld * glm::vec4(localPosition, 1.0F);
    const glm::vec4 previousClip = previousUnjitteredViewProjection *
                                   previousWorld * glm::vec4(localPosition, 1.0F);
    constexpr float minimumW = 1.0e-6F;
    if (!finite(currentClip) || !finite(previousClip) || currentClip.w <= minimumW ||
        previousClip.w <= minimumW) {
        return {};
    }
    const glm::vec2 currentUv = glm::vec2(currentClip) / currentClip.w * 0.5F + 0.5F;
    const glm::vec2 previousUv = glm::vec2(previousClip) / previousClip.w * 0.5F + 0.5F;
    const float previousDepth = previousClip.z / previousClip.w;
    if (!finite(currentUv) || !finite(previousUv) || !std::isfinite(previousDepth)) {
        return {};
    }
    return {
        .motionUv = currentUv - previousUv,
        .previousDeviceDepth = previousDepth,
        .valid = true,
    };
}

} // namespace renderer::temporal
