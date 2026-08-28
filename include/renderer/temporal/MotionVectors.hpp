#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace renderer::temporal {

// CPU reference for the rigid motion-vector shader convention. Motion is
// current unjittered UV minus previous unjittered UV.
struct RigidMotionSample {
    glm::vec2 motionUv{0.0F};
    float previousDeviceDepth = 0.0F;
    bool valid = false;
};

[[nodiscard]] RigidMotionSample projectRigidMotion(
    const glm::vec3& localPosition, const glm::mat4& currentWorld,
    const glm::mat4& previousWorld, const glm::mat4& currentUnjitteredViewProjection,
    const glm::mat4& previousUnjitteredViewProjection);

} // namespace renderer::temporal
