#pragma once

#include <glm/vec3.hpp>

#include <optional>

namespace renderer::materials {

struct MetallicRoughnessSample {
    glm::vec3 baseColor{1.0F};
    float metallic = 0.0F;
    float perceptualRoughness = 1.0F;
};

// glTF stores perceptual roughness. Its isotropic GGX alpha is roughness^2;
// this is intentionally distinct from PBRT's legacy scene-parameter remap.
[[nodiscard]] float perceptualRoughnessToAlpha(float roughness);

[[nodiscard]] float trowbridgeReitzD(float normalDotHalf, float alpha);
[[nodiscard]] float smithLambda(float normalDotDirection, float alpha);
[[nodiscard]] float smithMaskingShadowing(float normalDotView, float normalDotLight,
                                          float alpha);
[[nodiscard]] glm::vec3 fresnelSchlick(float directionDotHalf, const glm::vec3& f0);
[[nodiscard]] float fresnelDielectric(float cosThetaIncident, float relativeIor);

// Returns the BRDF value (not multiplied by N.L or incident radiance).
[[nodiscard]] glm::vec3 evaluateMetallicRoughnessBrdf(
    const MetallicRoughnessSample& material, const glm::vec3& normal,
    const glm::vec3& viewDirection, const glm::vec3& lightDirection);

[[nodiscard]] float inverseSquareAttenuation(float distance,
                                             std::optional<float> range = std::nullopt);

// Incident radiance is linear HDR radiance. The result is intentionally not
// clamped or tone mapped.
[[nodiscard]] glm::vec3 evaluateDirectLighting(
    const MetallicRoughnessSample& material, const glm::vec3& normal,
    const glm::vec3& viewDirection, const glm::vec3& lightDirection,
    const glm::vec3& incidentRadiance);

} // namespace renderer::materials
