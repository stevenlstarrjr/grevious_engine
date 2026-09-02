#include "renderer/lighting/DirectionalLightControls.hpp"
#include "renderer/materials/PbrMath.hpp"

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

namespace rm = renderer::materials;
namespace rl = renderer::lighting;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

bool close(float lhs, float rhs, float tolerance = 1.0e-5F) {
    return std::abs(lhs - rhs) <= tolerance;
}

bool closeRelative(float lhs, float rhs, float tolerance = 1.0e-5F) {
    return std::abs(lhs - rhs) <= tolerance * std::max(1.0F, std::abs(rhs));
}

bool close(const glm::vec3& lhs, const glm::vec3& rhs, float tolerance = 1.0e-5F) {
    return close(lhs.x, rhs.x, tolerance) && close(lhs.y, rhs.y, tolerance) &&
           close(lhs.z, rhs.z, tolerance);
}

bool close(const glm::vec4& lhs, const glm::vec4& rhs, float tolerance = 1.0e-5F) {
    return close(lhs.x, rhs.x, tolerance) && close(lhs.y, rhs.y, tolerance) &&
           close(lhs.z, rhs.z, tolerance) && close(lhs.w, rhs.w, tolerance);
}

bool closeRelative(const glm::vec3& lhs, const glm::vec3& rhs, float tolerance = 1.0e-5F) {
    return closeRelative(lhs.x, rhs.x, tolerance) && closeRelative(lhs.y, rhs.y, tolerance) &&
           closeRelative(lhs.z, rhs.z, tolerance);
}

bool finiteNonNegative(const glm::vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) &&
           value.x >= 0.0F && value.y >= 0.0F && value.z >= 0.0F;
}

float pbrtIsotropicTrowbridgeReitzD(float normalDotHalf, float alpha) {
    if (normalDotHalf <= 0.0F) {
        return 0.0F;
    }
    const float cosineSquared = normalDotHalf * normalDotHalf;
    const float tangentSquared = (1.0F - cosineSquared) / cosineSquared;
    const float cosineFourth = cosineSquared * cosineSquared;
    if (!std::isfinite(tangentSquared) || cosineFourth < 1.0e-16F) {
        return 0.0F;
    }
    const float exponent = tangentSquared / (alpha * alpha);
    return 1.0F / (std::numbers::pi_v<float> * alpha * alpha * cosineFourth * (1.0F + exponent) *
                   (1.0F + exponent));
}

float pbrtIsotropicSmithLambda(float normalDotDirection, float alpha) {
    const float cosineSquared = normalDotDirection * normalDotDirection;
    const float tangentSquared = (1.0F - cosineSquared) / cosineSquared;
    return (std::sqrt(1.0F + alpha * alpha * tangentSquared) - 1.0F) * 0.5F;
}

std::uint32_t sampleHash(std::uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    value ^= value >> 16U;
    return value;
}

float unitSample(std::uint32_t value) {
    return static_cast<float>(sampleHash(value) >> 8U) * (1.0F / 16777216.0F);
}

glm::vec3 bsdfRandomSample(std::uint32_t index) {
    return {unitSample(index ^ 0x68bc21ebU), unitSample(index ^ 0x02e5be93U),
            unitSample(index ^ 0x967a889bU)};
}

void testTrowbridgeReitzAndSmithReferenceValues() {
    constexpr float alpha = 0.25F;
    const float expectedNormalD = 1.0F / (std::numbers::pi_v<float> * alpha * alpha);
    require(close(rm::trowbridgeReitzD(1.0F, alpha), expectedNormalD, 1.0e-4F),
            "GGX NDF disagrees with the PBRT Trowbridge-Reitz normal-incidence value");
    require(close(rm::smithLambda(1.0F, alpha), 0.0F),
            "Smith lambda must be zero at normal incidence");
    require(close(rm::smithMaskingShadowing(1.0F, 1.0F, alpha), 1.0F),
            "Smith masking-shadowing must be one at normal incidence");

    const float geometry = rm::smithMaskingShadowing(0.35F, 0.72F, alpha);
    require(geometry > 0.0F && geometry < 1.0F &&
                close(geometry, rm::smithMaskingShadowing(0.72F, 0.35F, alpha)),
            "Smith masking-shadowing is not bounded and symmetric");
    require(close(rm::perceptualRoughnessToAlpha(0.5F), 0.25F),
            "glTF perceptual roughness did not map to squared GGX alpha");

    constexpr std::array alphas{0.01F, 0.1F, 0.25F, 0.6F, 1.0F};
    constexpr std::array cosines{0.03F, 0.15F, 0.4F, 0.75F, 1.0F};
    for (const float testAlpha : alphas) {
        for (const float cosine : cosines) {
            const float rendererD = rm::trowbridgeReitzD(cosine, testAlpha);
            const float pbrtD = pbrtIsotropicTrowbridgeReitzD(cosine, testAlpha);
            require(closeRelative(rendererD, pbrtD, 2.0e-5F),
                    "GGX NDF drifted from PBRT's original-form isotropic equation");
            require(close(rm::smithLambda(cosine, testAlpha),
                          pbrtIsotropicSmithLambda(cosine, testAlpha), 2.0e-5F),
                    "Smith lambda drifted from PBRT's original-form isotropic equation");
        }
    }
    require(rm::trowbridgeReitzD(0.0F, alpha) == 0.0F,
            "GGX NDF does not preserve PBRT's exact grazing boundary");
    require(rm::trowbridgeReitzD(1.0e-5F, alpha) == 0.0F,
            "GGX NDF does not preserve PBRT's near-grazing precision guard");
}

void testFresnelEndpointsAndExactReference() {
    const glm::vec3 f0(0.04F, 0.25F, 0.8F);
    require(close(rm::fresnelSchlick(1.0F, f0), f0),
            "Fresnel-Schlick does not equal F0 at normal incidence");
    require(close(rm::fresnelSchlick(0.0F, f0), glm::vec3(1.0F)),
            "Fresnel-Schlick does not approach one at grazing incidence");

    constexpr float ior = 1.5F;
    constexpr float expectedF0 = ((ior - 1.0F) * (ior - 1.0F)) / ((ior + 1.0F) * (ior + 1.0F));
    require(close(rm::fresnelDielectric(1.0F, ior), expectedF0),
            "Exact dielectric Fresnel disagrees with the analytic normal-incidence value");
    require(close(rm::fresnelDielectric(0.0F, ior), 1.0F),
            "Exact dielectric Fresnel does not reach one at grazing incidence");
}

void testReciprocityAndFiniteEdgeCases() {
    const rm::MetallicRoughnessSample material{
        .baseColor = {0.72F, 0.18F, 0.05F},
        .metallic = 0.65F,
        .perceptualRoughness = 0.31F,
    };
    const glm::vec3 normal(0.0F, 0.0F, 1.0F);
    const glm::vec3 view = glm::normalize(glm::vec3(0.3F, -0.1F, 0.95F));
    const glm::vec3 light = glm::normalize(glm::vec3(-0.2F, 0.4F, 0.89F));
    const glm::vec3 forward = rm::evaluateMetallicRoughnessBrdf(material, normal, view, light);
    const glm::vec3 reverse = rm::evaluateMetallicRoughnessBrdf(material, normal, light, view);
    require(close(forward, reverse, 2.0e-5F), "Metallic/roughness BRDF violates reciprocity");

    constexpr std::array roughnesses{0.0F, 1.0e-6F, 0.1F, 0.5F, 1.0F};
    for (const float roughness : roughnesses) {
        rm::MetallicRoughnessSample edge = material;
        edge.perceptualRoughness = roughness;
        const glm::vec3 result = rm::evaluateMetallicRoughnessBrdf(
            edge, normal, glm::normalize(glm::vec3(1.0F, 0.0F, 1.0e-5F)),
            glm::normalize(glm::vec3(-0.99F, 0.0F, 0.1F)));
        require(finiteNonNegative(result), "BRDF produced a non-finite or negative edge result");
    }
}

glm::vec3
integrateDirectionalHemisphericalReflectance(const rm::MetallicRoughnessSample& material) {
    constexpr int thetaSteps = 160;
    constexpr int phiSteps = 320;
    constexpr float deltaTheta = (0.5F * std::numbers::pi_v<float>) / thetaSteps;
    constexpr float deltaPhi = (2.0F * std::numbers::pi_v<float>) / phiSteps;
    const glm::vec3 normal(0.0F, 0.0F, 1.0F);
    const glm::vec3 view = normal;
    glm::vec3 integral(0.0F);
    for (int thetaIndex = 0; thetaIndex < thetaSteps; ++thetaIndex) {
        const float theta = (static_cast<float>(thetaIndex) + 0.5F) * deltaTheta;
        const float sinTheta = std::sin(theta);
        const float cosTheta = std::cos(theta);
        for (int phiIndex = 0; phiIndex < phiSteps; ++phiIndex) {
            const float phi = (static_cast<float>(phiIndex) + 0.5F) * deltaPhi;
            const glm::vec3 light(sinTheta * std::cos(phi), sinTheta * std::sin(phi), cosTheta);
            integral += rm::evaluateMetallicRoughnessBrdf(material, normal, view, light) *
                        (cosTheta * sinTheta * deltaTheta * deltaPhi);
        }
    }
    return integral;
}

void testWhiteFurnaceEnergyBound() {
    constexpr std::array materials{
        rm::MetallicRoughnessSample{
            .baseColor = {1.0F, 1.0F, 1.0F}, .metallic = 0.0F, .perceptualRoughness = 0.15F},
        rm::MetallicRoughnessSample{
            .baseColor = {1.0F, 1.0F, 1.0F}, .metallic = 0.0F, .perceptualRoughness = 1.0F},
        rm::MetallicRoughnessSample{
            .baseColor = {0.9F, 0.6F, 0.2F}, .metallic = 1.0F, .perceptualRoughness = 0.2F},
        rm::MetallicRoughnessSample{
            .baseColor = {0.9F, 0.6F, 0.2F}, .metallic = 1.0F, .perceptualRoughness = 0.8F},
    };
    for (const auto& material : materials) {
        const glm::vec3 reflectance = integrateDirectionalHemisphericalReflectance(material);
        require(finiteNonNegative(reflectance) && reflectance.x <= 1.005F &&
                    reflectance.y <= 1.005F && reflectance.z <= 1.005F,
                "White-furnace integration found an energy-gaining BRDF");
    }
}

void testSampledBsdfPdfAccounting() {
    constexpr std::array materials{
        rm::MetallicRoughnessSample{
            .baseColor = {0.9F, 0.8F, 0.7F}, .metallic = 0.0F, .perceptualRoughness = 0.65F},
        rm::MetallicRoughnessSample{
            .baseColor = {0.9F, 0.6F, 0.2F}, .metallic = 1.0F, .perceptualRoughness = 0.35F},
        rm::MetallicRoughnessSample{
            .baseColor = {0.7F, 0.2F, 0.1F}, .metallic = 0.55F, .perceptualRoughness = 0.8F},
    };
    const glm::vec3 normal(0.0F, 0.0F, 1.0F);
    constexpr std::uint32_t sampleCount = 16'384U;
    for (std::size_t materialIndex = 0; materialIndex < materials.size(); ++materialIndex) {
        const glm::vec3 view = materialIndex == 2
                                   ? glm::normalize(glm::vec3(0.75F, -0.2F, 0.63F))
                                   : normal;
        const float specularProbability =
            rm::metallicRoughnessSpecularSampleProbability(materials[materialIndex], normal, view);
        require(std::isfinite(specularProbability) && specularProbability >= 0.05F &&
                    specularProbability <= 0.95F,
                "BSDF lobe-selection probability is invalid");

        std::uint32_t validSamples = 0U;
        for (std::uint32_t index = 0; index < sampleCount; ++index) {
            const auto sample = rm::sampleMetallicRoughnessBsdf(
                materials[materialIndex], normal, view,
                bsdfRandomSample(index + static_cast<std::uint32_t>(materialIndex) * sampleCount));
            if (sample.pdf == 0.0F) {
                continue;
            }
            ++validSamples;
            require(std::isfinite(sample.pdf) && sample.pdf > 0.0F &&
                        finiteNonNegative(sample.throughput) &&
                        close(glm::dot(sample.direction, sample.direction), 1.0F, 2.0e-5F) &&
                        glm::dot(normal, sample.direction) > 0.0F,
                    "Sampled BSDF produced an invalid direction, PDF, or throughput");
            const float evaluatedPdf = rm::metallicRoughnessSamplePdf(
                materials[materialIndex], normal, view, sample.direction);
            require(closeRelative(sample.pdf, evaluatedPdf, 2.0e-5F),
                    "Stored BSDF PDF differs from mixture PDF evaluation");
            const glm::vec3 expected = rm::evaluateMetallicRoughnessBrdf(
                                           materials[materialIndex], normal, view, sample.direction) *
                                       glm::dot(normal, sample.direction);
            require(closeRelative(sample.throughput * sample.pdf, expected, 2.0e-4F),
                    "BSDF throughput does not equal f * absDot / PDF");
        }
        require(validSamples > sampleCount / 3U,
                "Visible-GGX rejection discarded an unreasonable number of samples");
    }

    const rm::MetallicRoughnessSample whiteDiffuse{
        .baseColor = {1.0F, 1.0F, 1.0F}, .metallic = 0.0F, .perceptualRoughness = 0.5F};
    const rm::MetallicRoughnessSample polishedMetal{
        .baseColor = {0.9F, 0.7F, 0.3F}, .metallic = 1.0F, .perceptualRoughness = 0.2F};
    const float diffuseProbability =
        rm::metallicRoughnessSpecularSampleProbability(whiteDiffuse, normal, normal);
    const float metalProbability =
        rm::metallicRoughnessSpecularSampleProbability(polishedMetal, normal, normal);
    const float grazingProbability = rm::metallicRoughnessSpecularSampleProbability(
        whiteDiffuse, normal, glm::normalize(glm::vec3(1.0F, 0.0F, 0.08F)));
    require(metalProbability > diffuseProbability && grazingProbability > diffuseProbability,
            "Lobe selection does not favor metallic and grazing specular transport");
}

void testSampledBsdfMatchesReferenceIntegration() {
    constexpr std::array materials{
        rm::MetallicRoughnessSample{
            .baseColor = {0.8F, 0.6F, 0.3F}, .metallic = 0.0F, .perceptualRoughness = 0.65F},
        rm::MetallicRoughnessSample{
            .baseColor = {0.75F, 0.5F, 0.2F}, .metallic = 1.0F, .perceptualRoughness = 0.55F},
    };
    constexpr std::uint32_t sampleCount = 131'072U;
    const glm::vec3 normal(0.0F, 0.0F, 1.0F);
    for (std::size_t materialIndex = 0; materialIndex < materials.size(); ++materialIndex) {
        glm::vec3 estimate(0.0F);
        for (std::uint32_t index = 0; index < sampleCount; ++index) {
            const auto sample = rm::sampleMetallicRoughnessBsdf(
                materials[materialIndex], normal, normal,
                bsdfRandomSample(index + static_cast<std::uint32_t>(materialIndex) * sampleCount));
            estimate += sample.throughput;
        }
        estimate /= static_cast<float>(sampleCount);
        const glm::vec3 reference = integrateDirectionalHemisphericalReflectance(
            materials[materialIndex]);
        require(close(estimate, reference, 0.012F),
                "Sampled BSDF estimator disagrees with reference hemispherical integration");
    }
}

void testPathRussianRouletteSurvivalProbability() {
    require(close(rm::pathRussianRouletteSurvivalProbability(glm::vec3(0.0F)), 0.0F) &&
                close(rm::pathRussianRouletteSurvivalProbability(
                          glm::vec3(-0.5F, -0.1F, 0.0F)),
                      0.0F),
            "Russian roulette did not reject nonpositive throughput");
    require(close(rm::pathRussianRouletteSurvivalProbability(
                      glm::vec3(0.01F, 0.02F, 0.03F)),
                  0.05F) &&
                close(rm::pathRussianRouletteSurvivalProbability(
                          glm::vec3(0.1F, 0.4F, 0.2F)),
                      0.4F) &&
                close(rm::pathRussianRouletteSurvivalProbability(
                          glm::vec3(0.99F, 0.2F, 0.1F)),
                      0.95F) &&
                close(rm::pathRussianRouletteSurvivalProbability(
                          glm::vec3(1.0F, 0.2F, 0.1F)),
                      1.0F),
            "Russian roulette survival probability violated its bounds");
    require(close(rm::pathRussianRouletteSurvivalProbability(
                      glm::vec3(std::numeric_limits<float>::infinity())),
                  0.0F) &&
                close(rm::pathRussianRouletteSurvivalProbability(
                          glm::vec3(std::numeric_limits<float>::quiet_NaN())),
                      0.0F),
            "Russian roulette accepted nonfinite throughput");

    const glm::vec3 throughput(0.2F, 0.4F, 0.1F);
    const float survival = rm::pathRussianRouletteSurvivalProbability(throughput);
    require(close(throughput / survival * survival, throughput),
            "Russian roulette survival compensation is biased");
}

void testAreaLightSolidAnglePdf() {
    require(close(rm::areaLightSolidAnglePdf(0.25F, 2.0F, 16.0F, 0.5F), 4.0F),
            "Area-light density did not convert to solid angle");
    require(close(rm::areaLightSolidAnglePdf(0.0F, 2.0F, 16.0F, 0.5F), 0.0F) &&
                close(rm::areaLightSolidAnglePdf(0.25F, 0.0F, 16.0F, 0.5F), 0.0F) &&
                close(rm::areaLightSolidAnglePdf(0.25F, 2.0F, 16.0F, 0.0F), 0.0F) &&
                close(rm::areaLightSolidAnglePdf(
                          std::numeric_limits<float>::quiet_NaN(), 2.0F, 16.0F, 0.5F),
                      0.0F),
            "Area-light density accepted a singular or nonfinite input");
}

void testLightUnitsAndHdrPreservation() {
    require(close(rm::inverseSquareAttenuation(2.0F), 0.25F) &&
                close(rm::inverseSquareAttenuation(4.0F), 0.0625F),
            "Point-light attenuation is not inverse square");
    require(close(rm::inverseSquareAttenuation(10.0F, 10.0F), 0.0F),
            "Finite-range attenuation does not reach zero at the range");

    const rm::MetallicRoughnessSample whiteDiffuse{
        .baseColor = {1.0F, 1.0F, 1.0F},
        .metallic = 0.0F,
        .perceptualRoughness = 0.8F,
    };
    const glm::vec3 direct =
        rm::evaluateDirectLighting(whiteDiffuse, {0.0F, 0.0F, 1.0F}, {0.0F, 0.0F, 1.0F},
                                   {0.0F, 0.0F, 1.0F}, {100.0F, 100.0F, 100.0F});
    require(direct.x > 1.0F && direct.y > 1.0F && direct.z > 1.0F,
            "Direct-light evaluation incorrectly clamped HDR radiance");
}

void testEffectiveGltfBaseColor() {
    const glm::vec4 factor(0.8F, 0.6F, 0.4F, 0.5F);
    const glm::vec4 textureSample(0.5F, 0.25F, 1.0F, 0.8F);
    const glm::vec4 vertexColor(0.25F, 0.5F, 0.75F, 0.4F);
    require(close(rm::effectiveGltfBaseColor(factor, textureSample, vertexColor),
                  glm::vec4(0.1F, 0.075F, 0.3F, 0.16F)),
            "glTF base color did not multiply factor, texture, and COLOR_0 RGBA");
    require(close(rm::effectiveGltfBaseColor(factor, textureSample, glm::vec4(1.0F)),
                  factor * textureSample),
            "Missing COLOR_0 did not default to white");
}

void testDirectionalLightAndShadowControlsAreIndependent() {
    constexpr std::uint32_t cascadeCount = 3U;
    const rl::DirectionalLightControls litAndShadowed =
        rl::makeDirectionalLightControls(3.5F, true, cascadeCount);
    require(close(litAndShadowed.intensity, 3.5F) &&
                litAndShadowed.activeShadowCascadeCount == cascadeCount,
            "Enabled directional light did not retain its shadow controls");

    const rl::DirectionalLightControls litWithoutShadows =
        rl::makeDirectionalLightControls(3.5F, false, cascadeCount);
    require(close(litWithoutShadows.intensity, 3.5F) &&
                litWithoutShadows.activeShadowCascadeCount == 0U,
            "Disabling directional shadows incorrectly disabled the light");

    const rl::DirectionalLightControls disabledLight =
        rl::makeDirectionalLightControls(0.0F, true, cascadeCount);
    require(close(disabledLight.intensity, 0.0F) &&
                disabledLight.activeShadowCascadeCount == 0U,
            "A disabled directional light retained shadow work");
}

} // namespace

int main() {
    try {
        testTrowbridgeReitzAndSmithReferenceValues();
        testFresnelEndpointsAndExactReference();
        testReciprocityAndFiniteEdgeCases();
        testWhiteFurnaceEnergyBound();
        testSampledBsdfPdfAccounting();
        testSampledBsdfMatchesReferenceIntegration();
        testPathRussianRouletteSurvivalProbability();
        testAreaLightSolidAnglePdf();
        testLightUnitsAndHdrPreservation();
        testEffectiveGltfBaseColor();
        testDirectionalLightAndShadowControlsAreIndependent();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
