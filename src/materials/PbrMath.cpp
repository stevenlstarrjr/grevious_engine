#include "renderer/materials/PbrMath.hpp"

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace renderer::materials {
namespace {

// Portions derived from pbrt-v4, Copyright (c) 1998-2020 Matt Pharr,
// Wenzel Jakob, and Greg Humphreys; Apache-2.0.
// Mathematical source of truth: pbrt-v4 revision 5f7a606806a4ac7b939131ded9d7a30ebd02416e,
// src/pbrt/util/scattering.h (FrDielectric and TrowbridgeReitzDistribution) and
// src/pbrt/bxdfs.h (rough microfacet reflection). See PBRT_PROVENANCE.md for the
// exact-port versus realtime-adaptation boundary.
constexpr float minimumAlpha = 1.0e-4F;

float saturate(float value) { return std::clamp(value, 0.0F, 1.0F); }

float square(float value) { return value * value; }

float pow5(float value) {
    const float squared = value * value;
    return squared * squared * value;
}

glm::vec3 finiteOrZero(glm::vec3 value) {
    if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z)) {
        return glm::vec3(0.0F);
    }
    return value;
}

} // namespace

float perceptualRoughnessToAlpha(float roughness) {
    const float clamped = saturate(roughness);
    return std::max(clamped * clamped, minimumAlpha);
}

float trowbridgeReitzD(float normalDotHalf, float alpha) {
    const float noH = saturate(normalDotHalf);
    if (noH <= 0.0F) {
        return 0.0F;
    }
    const float safeAlpha = std::max(alpha, minimumAlpha);
    const float alphaSquared = square(safeAlpha);
    const float cosineSquared = square(noH);
    const float tangentSquared = (1.0F - cosineSquared) / cosineSquared;
    const float cosineFourth = square(cosineSquared);
    if (!std::isfinite(tangentSquared) || cosineFourth < 1.0e-16F) {
        return 0.0F;
    }
    const float exponent = tangentSquared / alphaSquared;
    return 1.0F /
           (std::numbers::pi_v<float> * alphaSquared * cosineFourth * square(1.0F + exponent));
}

float smithLambda(float normalDotDirection, float alpha) {
    const float noW = saturate(normalDotDirection);
    if (noW <= 0.0F) {
        return std::numeric_limits<float>::infinity();
    }
    const float safeAlpha = std::max(alpha, minimumAlpha);
    const float tanThetaSquared = std::max(0.0F, 1.0F - noW * noW) / (noW * noW);
    return (std::sqrt(1.0F + square(safeAlpha) * tanThetaSquared) - 1.0F) * 0.5F;
}

float smithMaskingShadowing(float normalDotView, float normalDotLight, float alpha) {
    if (normalDotView <= 0.0F || normalDotLight <= 0.0F) {
        return 0.0F;
    }
    return 1.0F / (1.0F + smithLambda(normalDotView, alpha) + smithLambda(normalDotLight, alpha));
}

glm::vec3 fresnelSchlick(float directionDotHalf, const glm::vec3& f0) {
    const float weight = pow5(1.0F - saturate(directionDotHalf));
    const glm::vec3 clampedF0 = glm::clamp(f0, glm::vec3(0.0F), glm::vec3(1.0F));
    return clampedF0 + (glm::vec3(1.0F) - clampedF0) * weight;
}

float fresnelDielectric(float cosThetaIncident, float relativeIor) {
    float cosine = std::clamp(cosThetaIncident, -1.0F, 1.0F);
    float eta = std::max(relativeIor, minimumAlpha);
    if (cosine < 0.0F) {
        eta = 1.0F / eta;
        cosine = -cosine;
    }
    const float sinThetaIncidentSquared = std::max(0.0F, 1.0F - cosine * cosine);
    const float sinThetaTransmittedSquared = sinThetaIncidentSquared / (eta * eta);
    if (sinThetaTransmittedSquared >= 1.0F) {
        return 1.0F;
    }
    const float cosThetaTransmitted = std::sqrt(1.0F - sinThetaTransmittedSquared);
    const float parallel =
        (eta * cosine - cosThetaTransmitted) / (eta * cosine + cosThetaTransmitted);
    const float perpendicular =
        (cosine - eta * cosThetaTransmitted) / (cosine + eta * cosThetaTransmitted);
    return (parallel * parallel + perpendicular * perpendicular) * 0.5F;
}

glm::vec3 evaluateMetallicRoughnessBrdf(const MetallicRoughnessSample& material,
                                        const glm::vec3& normal, const glm::vec3& viewDirection,
                                        const glm::vec3& lightDirection) {
    const glm::vec3 n = glm::normalize(normal);
    const glm::vec3 v = glm::normalize(viewDirection);
    const glm::vec3 l = glm::normalize(lightDirection);
    const float noV = saturate(glm::dot(n, v));
    const float noL = saturate(glm::dot(n, l));
    if (noV <= 0.0F || noL <= 0.0F) {
        return glm::vec3(0.0F);
    }

    const glm::vec3 halfUnnormalized = v + l;
    const float halfLengthSquared = glm::dot(halfUnnormalized, halfUnnormalized);
    if (halfLengthSquared <= std::numeric_limits<float>::epsilon()) {
        return glm::vec3(0.0F);
    }
    const glm::vec3 h = halfUnnormalized / std::sqrt(halfLengthSquared);
    const float noH = saturate(glm::dot(n, h));
    const float voH = saturate(glm::dot(v, h));

    const glm::vec3 baseColor = glm::clamp(material.baseColor, glm::vec3(0.0F), glm::vec3(1.0F));
    const float metallic = saturate(material.metallic);
    const float alpha = perceptualRoughnessToAlpha(material.perceptualRoughness);
    const glm::vec3 f0 = glm::mix(glm::vec3(0.04F), baseColor, metallic);
    const glm::vec3 fresnel = fresnelSchlick(voH, f0);
    const float distribution = trowbridgeReitzD(noH, alpha);
    const float geometry = smithMaskingShadowing(noV, noL, alpha);
    const glm::vec3 specular =
        fresnel * (distribution * geometry / std::max(4.0F * noV * noL, 1.0e-7F));

    const glm::vec3 diffuseWeight = (glm::vec3(1.0F) - fresnel) * (1.0F - metallic);
    const glm::vec3 diffuse = diffuseWeight * baseColor / std::numbers::pi_v<float>;
    return finiteOrZero(diffuse + specular);
}

float inverseSquareAttenuation(float distance, std::optional<float> range) {
    const float safeDistanceSquared = std::max(distance * distance, 1.0e-6F);
    float attenuation = 1.0F / safeDistanceSquared;
    if (range.has_value()) {
        if (*range <= 0.0F || distance >= *range) {
            return 0.0F;
        }
        const float ratio = distance / *range;
        const float window = saturate(1.0F - square(square(ratio)));
        attenuation *= window * window;
    }
    return attenuation;
}

glm::vec3 evaluateDirectLighting(const MetallicRoughnessSample& material, const glm::vec3& normal,
                                 const glm::vec3& viewDirection, const glm::vec3& lightDirection,
                                 const glm::vec3& incidentRadiance) {
    const glm::vec3 n = glm::normalize(normal);
    const glm::vec3 l = glm::normalize(lightDirection);
    const float noL = saturate(glm::dot(n, l));
    const glm::vec3 radiance = glm::max(incidentRadiance, glm::vec3(0.0F));
    return finiteOrZero(evaluateMetallicRoughnessBrdf(material, n, viewDirection, l) * radiance *
                        noL);
}

} // namespace renderer::materials
