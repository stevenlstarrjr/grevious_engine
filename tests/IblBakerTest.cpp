#include "renderer/lighting/IblBaker.hpp"

#include <algorithm>
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
