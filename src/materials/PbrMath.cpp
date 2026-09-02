#include "renderer/materials/PbrMath.hpp"

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>

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

float luminance(const glm::vec3& value) {
    return glm::dot(value, glm::vec3(0.2126F, 0.7152F, 0.0722F));
}

glm::vec3 safeNormalize(const glm::vec3& value, const glm::vec3& fallback) {
    const float lengthSquared = glm::dot(value, value);
    if (!std::isfinite(lengthSquared) || lengthSquared <= std::numeric_limits<float>::epsilon()) {
        return fallback;
    }
    return value / std::sqrt(lengthSquared);
}

void basisFromNormal(const glm::vec3& normal, glm::vec3& tangent, glm::vec3& bitangent) {
    const glm::vec3 reference =
        std::abs(normal.z) < 0.999F ? glm::vec3(0.0F, 0.0F, 1.0F) : glm::vec3(1.0F, 0.0F, 0.0F);
    tangent = glm::normalize(glm::cross(reference, normal));
    bitangent = glm::cross(normal, tangent);
}

glm::vec3 sampleVisibleGgx(const glm::vec3& localView, float alpha, const glm::vec2& sample) {
    glm::vec3 stretched = safeNormalize(
        glm::vec3(alpha * localView.x, alpha * localView.y, localView.z),
        glm::vec3(0.0F, 0.0F, 1.0F));
    if (stretched.z < 0.0F) {
        stretched = -stretched;
    }
    const glm::vec3 tangent =
        stretched.z < 0.99999F
            ? glm::normalize(glm::cross(glm::vec3(0.0F, 0.0F, 1.0F), stretched))
            : glm::vec3(1.0F, 0.0F, 0.0F);
    const glm::vec3 bitangent = glm::cross(stretched, tangent);
    const float radius = std::sqrt(sample.x);
    const float angle = 2.0F * std::numbers::pi_v<float> * sample.y;
    glm::vec2 disk(radius * std::cos(angle), radius * std::sin(angle));
    const float horizon = std::sqrt(std::max(0.0F, 1.0F - disk.x * disk.x));
    const float blend = (1.0F + stretched.z) * 0.5F;
    disk.y = std::lerp(horizon, disk.y, blend);
    const float projectedZ = std::sqrt(std::max(0.0F, 1.0F - glm::dot(disk, disk)));
    const glm::vec3 hemisphere =
        disk.x * tangent + disk.y * bitangent + projectedZ * stretched;
    return safeNormalize(
        glm::vec3(alpha * hemisphere.x, alpha * hemisphere.y,
                  std::max(1.0e-6F, hemisphere.z)),
        glm::vec3(0.0F, 0.0F, 1.0F));
}

float visibleGgxReflectionPdf(float perceptualRoughness, const glm::vec3& normal,
                              const glm::vec3& viewDirection,
                              const glm::vec3& lightDirection) {
    const float noV = saturate(glm::dot(normal, viewDirection));
    const float noL = saturate(glm::dot(normal, lightDirection));
    const glm::vec3 halfUnnormalized = viewDirection + lightDirection;
    const float halfLengthSquared = glm::dot(halfUnnormalized, halfUnnormalized);
    if (noV <= 0.0F || noL <= 0.0F ||
        halfLengthSquared <= std::numeric_limits<float>::epsilon()) {
        return 0.0F;
    }
    const glm::vec3 halfVector = halfUnnormalized / std::sqrt(halfLengthSquared);
    const float noH = saturate(glm::dot(normal, halfVector));
    const float voH = saturate(glm::dot(viewDirection, halfVector));
    if (noH <= 0.0F || voH <= 0.0F) {
        return 0.0F;
    }
    const float alpha = perceptualRoughnessToAlpha(perceptualRoughness);
    const float masking = 1.0F / (1.0F + smithLambda(noV, alpha));
    const float pdf = trowbridgeReitzD(noH, alpha) * masking / (4.0F * noV);
    return std::isfinite(pdf) ? std::max(pdf, 0.0F) : 0.0F;
}

} // namespace

glm::vec4 effectiveGltfBaseColor(const glm::vec4& factor, const glm::vec4& textureSample,
                                 const glm::vec4& vertexColor) {
    return glm::clamp(factor * textureSample * vertexColor, glm::vec4(0.0F), glm::vec4(1.0F));
}

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

float metallicRoughnessSpecularSampleProbability(const MetallicRoughnessSample& material,
                                                  const glm::vec3& normal,
                                                  const glm::vec3& viewDirection) {
    const glm::vec3 n = safeNormalize(normal, glm::vec3(0.0F, 0.0F, 1.0F));
    const glm::vec3 v = safeNormalize(viewDirection, n);
    const float noV = saturate(glm::dot(n, v));
    const glm::vec3 baseColor =
        glm::clamp(material.baseColor, glm::vec3(0.0F), glm::vec3(1.0F));
    const float metallic = saturate(material.metallic);
    const glm::vec3 f0 = glm::mix(glm::vec3(0.04F), baseColor, metallic);
    const float specularWeight = luminance(fresnelSchlick(noV, f0));
    const float diffuseWeight = luminance(baseColor) * (1.0F - metallic) *
                                std::max(0.0F, 1.0F - specularWeight);
    const float totalWeight = specularWeight + diffuseWeight;
    if (totalWeight <= std::numeric_limits<float>::epsilon()) {
        return 0.5F;
    }
    return std::clamp(specularWeight / totalWeight, 0.05F, 0.95F);
}

float metallicRoughnessSamplePdf(const MetallicRoughnessSample& material,
                                 const glm::vec3& normal, const glm::vec3& viewDirection,
                                 const glm::vec3& lightDirection) {
    const glm::vec3 n = safeNormalize(normal, glm::vec3(0.0F, 0.0F, 1.0F));
    const glm::vec3 v = safeNormalize(viewDirection, n);
    const glm::vec3 l = safeNormalize(lightDirection, n);
    const float noL = saturate(glm::dot(n, l));
    const float noV = saturate(glm::dot(n, v));
    if (noL <= 0.0F || noV <= 0.0F) {
        return 0.0F;
    }
    const float specularProbability =
        metallicRoughnessSpecularSampleProbability(material, n, v);
    const float diffusePdf = noL / std::numbers::pi_v<float>;
    const float specularPdf =
        visibleGgxReflectionPdf(material.perceptualRoughness, n, v, l);
    const float pdf = std::lerp(diffusePdf, specularPdf, specularProbability);
    return std::isfinite(pdf) ? std::max(pdf, 0.0F) : 0.0F;
}

MetallicRoughnessBsdfSample sampleMetallicRoughnessBsdf(
    const MetallicRoughnessSample& material, const glm::vec3& normal,
    const glm::vec3& viewDirection, const glm::vec3& randomSample) {
    const glm::vec3 n = safeNormalize(normal, glm::vec3(0.0F, 0.0F, 1.0F));
    const glm::vec3 v = safeNormalize(viewDirection, n);
    if (glm::dot(n, v) <= 0.0F) {
        return {};
    }
    const float maximumSample = std::nextafter(1.0F, 0.0F);
    const glm::vec3 sample =
        glm::clamp(randomSample, glm::vec3(0.0F), glm::vec3(maximumSample));
    const float specularProbability =
        metallicRoughnessSpecularSampleProbability(material, n, v);

    glm::vec3 direction{};
    if (sample.x < specularProbability) {
        glm::vec3 tangent{};
        glm::vec3 bitangent{};
        basisFromNormal(n, tangent, bitangent);
        const glm::vec3 localView(glm::dot(v, tangent), glm::dot(v, bitangent), glm::dot(v, n));
        const glm::vec3 localHalf = sampleVisibleGgx(
            localView, perceptualRoughnessToAlpha(material.perceptualRoughness),
            glm::vec2(sample.y, sample.z));
        const glm::vec3 halfVector = safeNormalize(
            tangent * localHalf.x + bitangent * localHalf.y + n * localHalf.z, n);
        direction = glm::reflect(-v, halfVector);
    } else {
        const float radius = std::sqrt(sample.y);
        const float angle = 2.0F * std::numbers::pi_v<float> * sample.z;
        const glm::vec3 localDirection(radius * std::cos(angle), radius * std::sin(angle),
                                       std::sqrt(std::max(0.0F, 1.0F - sample.y)));
        glm::vec3 tangent{};
        glm::vec3 bitangent{};
        basisFromNormal(n, tangent, bitangent);
        direction = tangent * localDirection.x + bitangent * localDirection.y +
                    n * localDirection.z;
    }
    direction = safeNormalize(direction, -n);
    const float noL = saturate(glm::dot(n, direction));
    if (noL <= 0.0F) {
        return {};
    }

    const float pdf = metallicRoughnessSamplePdf(material, n, v, direction);
    if (!std::isfinite(pdf) || pdf <= 0.0F) {
        return {};
    }
    const glm::vec3 throughput =
        evaluateMetallicRoughnessBrdf(material, n, v, direction) * (noL / pdf);
    if (!std::isfinite(throughput.x) || !std::isfinite(throughput.y) ||
        !std::isfinite(throughput.z)) {
        return {};
    }
    return {.direction = direction,
            .throughput = glm::max(throughput, glm::vec3(0.0F)),
            .pdf = pdf};
}

float pathRussianRouletteSurvivalProbability(const glm::vec3& throughput) {
    if (!std::isfinite(throughput.x) || !std::isfinite(throughput.y) ||
        !std::isfinite(throughput.z)) {
        return 0.0F;
    }
    const float maximumComponent =
        std::max({throughput.x, throughput.y, throughput.z});
    if (maximumComponent <= 0.0F) {
        return 0.0F;
    }
    if (maximumComponent >= 1.0F) {
        return 1.0F;
    }
    return std::clamp(maximumComponent, 0.05F, 0.95F);
}

float areaLightSolidAnglePdf(float selectionProbability, float area,
                             float distanceSquared, float absoluteCosine) {
    if (!std::isfinite(selectionProbability) || !std::isfinite(area) ||
        !std::isfinite(distanceSquared) || !std::isfinite(absoluteCosine) ||
        selectionProbability <= 0.0F || area <= 0.0F || distanceSquared <= 0.0F ||
        absoluteCosine <= 0.0F) {
        return 0.0F;
    }
    const float pdf = selectionProbability * distanceSquared / (area * absoluteCosine);
    return std::isfinite(pdf) ? pdf : 0.0F;
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
