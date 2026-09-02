#pragma once

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <optional>

namespace renderer::materials {

struct MetallicRoughnessSample {
    glm::vec3 baseColor{1.0F};
    float metallic = 0.0F;
    float perceptualRoughness = 1.0F;
};

struct MetallicRoughnessBsdfSample {
    glm::vec3 direction{0.0F};
    glm::vec3 throughput{0.0F};
    float pdf = 0.0F;
};

// glTF effective base color is the component-wise product of the material
// factor, sampled base-color texture, and linear COLOR_0 value. Missing
// textures and vertex colors are represented by white.
[[nodiscard]] glm::vec4 effectiveGltfBaseColor(const glm::vec4& factor,
                                               const glm::vec4& textureSample,
                                               const glm::vec4& vertexColor);

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

// Returns the probability of selecting the visible-GGX lobe. The remaining
// probability selects a cosine-weighted diffuse direction.
[[nodiscard]] float metallicRoughnessSpecularSampleProbability(
    const MetallicRoughnessSample& material, const glm::vec3& normal,
    const glm::vec3& viewDirection);

// Solid-angle PDF of the complete diffuse/visible-GGX sampling mixture.
[[nodiscard]] float metallicRoughnessSamplePdf(
    const MetallicRoughnessSample& material, const glm::vec3& normal,
    const glm::vec3& viewDirection, const glm::vec3& lightDirection);

// randomSample contains one lobe-selection variate followed by a 2D lobe
// sample. A zero PDF represents a rejected visible-normal reflection.
[[nodiscard]] MetallicRoughnessBsdfSample sampleMetallicRoughnessBsdf(
    const MetallicRoughnessSample& material, const glm::vec3& normal,
    const glm::vec3& viewDirection, const glm::vec3& randomSample);

// PBRT-style throughput-based Russian roulette. Throughputs at or above one
// continue deterministically; lower values are bounded away from zero and one.
[[nodiscard]] float pathRussianRouletteSurvivalProbability(const glm::vec3& throughput);

// Converts a triangle selection probability and uniform area density to the
// solid-angle density used by direct-light MIS at a shading point.
[[nodiscard]] float areaLightSolidAnglePdf(float selectionProbability, float area,
                                           float distanceSquared, float absoluteCosine);

[[nodiscard]] float inverseSquareAttenuation(float distance,
                                             std::optional<float> range = std::nullopt);

// Incident radiance is linear HDR radiance. The result is intentionally not
// clamped or tone mapped.
[[nodiscard]] glm::vec3 evaluateDirectLighting(
    const MetallicRoughnessSample& material, const glm::vec3& normal,
    const glm::vec3& viewDirection, const glm::vec3& lightDirection,
    const glm::vec3& incidentRadiance);

} // namespace renderer::materials
