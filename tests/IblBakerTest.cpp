#include "renderer/lighting/Atmosphere.hpp"
#include "renderer/lighting/IblBaker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

float luminanceVariance(const renderer::lighting::FloatImage& image) {
    double sum = 0.0;
    double squaredSum = 0.0;
    const std::size_t count = image.texels.size() / image.channels;
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t offset = index * image.channels;
        const double luminance = 0.2126 * image.texels[offset] +
                                 0.7152 * image.texels[offset + 1] +
                                 0.0722 * image.texels[offset + 2];
        sum += luminance;
        squaredSum += luminance * luminance;
    }
    const double mean = sum / static_cast<double>(count);
    return static_cast<float>(squaredSum / static_cast<double>(count) - mean * mean);
}

float luminanceAt(const renderer::lighting::FloatImage& image, std::uint32_t x,
                  std::uint32_t y) {
    const std::size_t offset =
        (static_cast<std::size_t>(y) * image.width + x) * image.channels;
    return 0.2126F * image.texels[offset] + 0.7152F * image.texels[offset + 1U] +
           0.0722F * image.texels[offset + 2U];
}

void verifyAliasDistribution(
    const renderer::lighting::EnvironmentSamplingDistribution& distribution) {
    require(distribution.valid(), "Environment sampling distribution is invalid");
    const std::size_t count = distribution.entries.size();
    std::vector<double> reconstructed(count, 0.0);
    double massSum = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const auto& entry = distribution.entries[index];
        require(std::isfinite(entry.threshold) && entry.threshold >= 0.0F &&
                    entry.threshold <= 1.0F && entry.aliasIndex < count &&
                    std::isfinite(entry.probabilityMass) && entry.probabilityMass >= 0.0F,
                "Environment alias entry is out of range");
        const double threshold = static_cast<double>(entry.threshold);
        reconstructed[index] += threshold / static_cast<double>(count);
        reconstructed[entry.aliasIndex] +=
            (1.0 - threshold) / static_cast<double>(count);
        massSum += entry.probabilityMass;
    }
    require(std::abs(massSum - 1.0) < 1.0e-5,
            "Environment probability masses are not normalized");
    for (std::size_t index = 0; index < count; ++index) {
        require(std::abs(reconstructed[index] -
                         static_cast<double>(distribution.entries[index].probabilityMass)) <
                    1.0e-6,
                "Environment alias table does not reproduce its probability masses");
    }
}

} // namespace

int main() {
    try {
        namespace rl = renderer::lighting;
        const rl::TimeOfDaySunSettings equinox{
            .latitudeDegrees = 0.0F,
            .dayOfYear = 81U,
        };
        const glm::vec3 noonSun = rl::timeOfDaySunDirection(12.0F, equinox);
        const glm::vec3 midnightSun = rl::timeOfDaySunDirection(0.0F, equinox);
        const glm::vec3 sunriseSun = rl::timeOfDaySunDirection(6.0F, equinox);
        const glm::vec3 sunsetSun = rl::timeOfDaySunDirection(18.0F, equinox);
        require(noonSun.z > 0.999F && midnightSun.z < -0.999F,
                "Equatorial equinox Sun did not cross the zenith and nadir");
        require(sunriseSun.x > 0.999F && std::abs(sunriseSun.z) < 1.0e-4F &&
                    sunsetSun.x < -0.999F && std::abs(sunsetSun.z) < 1.0e-4F,
                "Equatorial equinox sunrise or sunset direction is incorrect");

        const rl::TimeOfDaySunSettings summer{};
        constexpr std::array<float, 6> testHours{0.0F, 6.0F, 9.25F,
                                                 12.0F, 18.0F, 23.75F};
        for (const float hour : testHours) {
            const glm::vec3 direction = rl::timeOfDaySunDirection(hour, summer);
            const float recovered = rl::timeOfDayHourFromSunDirection(direction, summer);
            const float difference =
                std::abs(std::remainder(recovered - hour, 24.0F));
            require(std::abs(glm::length(direction) - 1.0F) < 1.0e-5F &&
                        difference < 1.0e-3F,
                    "Time-of-day Sun direction failed its hour round trip");
        }
        rl::AtmosphereDemoView advancingView{};
        rl::applyTimeOfDaySun(advancingView, 10.0F, summer);
        rl::advanceTimeOfDaySun(advancingView, 2.0F, summer);
        require(glm::dot(rl::atmosphereDemoSunDirection(advancingView),
                         rl::timeOfDaySunDirection(12.0F, summer)) > 0.99999F,
                "Incremental time-of-day Sun motion diverged from the astronomical path");

        const rl::AtmosphereDemoView surfaceView = rl::atmosphereDemoPreset(1U);
        rl::AtmosphereDemoView smoothSunView = surfaceView;
        smoothSunView.sunAzimuthAngleRadians +=
            std::numbers::pi_v<float> / 180.0F;
        require(!rl::atmosphereHistoryResetRequired(surfaceView, smoothSunView),
                "Smooth Sun motion incorrectly requires a history reset");
        require(rl::atmosphereHistoryResetRequired(
                    surfaceView, rl::atmosphereDemoPreset(8U)),
                "Surface-to-orbit transition did not require a history reset");
        rl::AtmosphereDemoView exposureJump = surfaceView;
        exposureJump.exposure *= 4.0F;
        require(rl::atmosphereHistoryResetRequired(surfaceView, exposureJump),
                "Large atmosphere exposure change did not require a history reset");

        rl::AtmosphereSettings atmosphereSettings;
        atmosphereSettings.environmentWidth = 64;
        atmosphereSettings.environmentHeight = 32;
        atmosphereSettings.clouds.enabled = true;
        const rl::FloatImage atmosphere =
            rl::makePrecomputedAtmosphereEnvironment(atmosphereSettings);
        require(atmosphere.valid(), "Precomputed atmosphere environment is invalid");
        for (const float value : atmosphere.texels) {
            require(std::isfinite(value) && value >= 0.0F,
                    "Precomputed atmosphere contains invalid radiance");
        }
        const float zenithLuminance = luminanceAt(atmosphere, 32U, 0U);
        const float horizonLuminance = luminanceAt(atmosphere, 32U, 15U);
        require(zenithLuminance > 0.0F && horizonLuminance > 0.0F,
                "Precomputed atmosphere lost sky radiance");
        require(luminanceVariance(atmosphere) > 1.0e-5F,
                "Precomputed atmosphere lost directional scattering variation");

        rl::AtmosphereSettings invalidCloudAtmosphere = atmosphereSettings;
        invalidCloudAtmosphere.clouds.thicknessMeters =
            invalidCloudAtmosphere.topRadiusMeters -
            invalidCloudAtmosphere.bottomRadiusMeters;
        bool rejectedInvalidClouds = false;
        try {
            (void)rl::makePrecomputedAtmosphereEnvironment(invalidCloudAtmosphere);
        } catch (const std::runtime_error&) {
            rejectedInvalidClouds = true;
        }
        require(rejectedInvalidClouds,
                "Atmosphere accepted a cloud shell outside its physical extent");

        rl::FloatImage environment = rl::makeProceduralStudioEnvironment(64, 32);
        require(environment.valid() &&
                    *std::max_element(environment.texels.begin(), environment.texels.end()) >
                        1.0F,
                "Procedural environment is not valid HDR radiance");

        const rl::IblAssets ibl = rl::bakeImageBasedLighting(
            std::move(environment),
            {.irradianceWidth = 8,
             .irradianceHeight = 4,
             .specularWidth = 16,
             .specularHeight = 8,
             .specularMipLevels = 5,
             .brdfLutSize = 16,
             .sampleCount = 32});
        const rl::EnvironmentSamplingDistribution distribution =
            rl::buildEnvironmentSamplingDistribution(ibl.environment);
        verifyAliasDistribution(distribution);
        require(distribution.width == ibl.environment.width &&
                    distribution.height == ibl.environment.height,
                "Environment distribution dimensions do not match the source image");

        rl::FloatImage blackEnvironment{
            .width = 12,
            .height = 6,
            .channels = 4,
            .texels = std::vector<float>(12U * 6U * 4U, 0.0F),
        };
        const rl::EnvironmentSamplingDistribution blackDistribution =
            rl::buildEnvironmentSamplingDistribution(blackEnvironment);
        verifyAliasDistribution(blackDistribution);
        const double deltaPhi = 2.0 * std::numbers::pi_v<double> / 12.0;
        for (std::uint32_t y = 0; y < blackDistribution.height; ++y) {
            const double theta0 = std::numbers::pi_v<double> * static_cast<double>(y) / 6.0;
            const double theta1 =
                std::numbers::pi_v<double> * static_cast<double>(y + 1U) / 6.0;
            const double expectedMass =
                deltaPhi * (std::cos(theta0) - std::cos(theta1)) /
                (4.0 * std::numbers::pi_v<double>);
            for (std::uint32_t x = 0; x < blackDistribution.width; ++x) {
                const std::size_t index =
                    static_cast<std::size_t>(y) * blackDistribution.width + x;
                require(std::abs(blackDistribution.entries[index].probabilityMass -
                                 expectedMass) < 1.0e-7,
                        "Black environment did not fall back to uniform-sphere sampling");
            }
        }
        require(ibl.environment.valid() && ibl.diffuseIrradiance.valid(),
                "Environment or diffuse irradiance bake is invalid");
        require(ibl.prefilteredSpecular.levels.size() == 5 &&
                    ibl.prefilteredSpecular.levels.front().width == 16 &&
                    ibl.prefilteredSpecular.levels.back().width == 1,
                "Specular prefilter mip chain is incomplete");
        require(ibl.brdfLut.valid() && ibl.brdfLut.channels == 2,
                "BRDF integration LUT is invalid");
        require(luminanceVariance(ibl.prefilteredSpecular.levels.back()) <
                    luminanceVariance(ibl.prefilteredSpecular.levels.front()),
                "Rough specular prefilter did not reduce environment variance");

        for (const float value : ibl.diffuseIrradiance.texels) {
            require(std::isfinite(value) && value >= 0.0F,
                    "Diffuse irradiance contains invalid radiance");
        }
        for (const float value : ibl.brdfLut.texels) {
            require(std::isfinite(value) && value >= 0.0F && value <= 1.0F,
                    "BRDF LUT contains an invalid split-sum coefficient");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
