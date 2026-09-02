#pragma once

#include <glm/vec3.hpp>

namespace renderer::weather {

// Camera-local precipitation controls. Rain is rendered as a bounded visual
// field after temporal reconstruction; splash and wet-material shading consume
// the same settings without making visual drops physics bodies.
struct RainSettings {
    bool enabled = false;
    // Normalized precipitation rate. Zero is dry; one is heavy rain.
    float intensity = 0.65F;
    float fallSpeedMetersPerSecond = 24.0F;
    float streakLengthMeters = 0.18F;
    float maximumDistanceMeters = 32.0F;
    // World-space air velocity. The application shares this horizontal vector
    // with cloud advection; the rain shader adds smooth small-scale gusts.
    glm::vec3 windVelocityMetersPerSecond{0.0F};
};

} // namespace renderer::weather
