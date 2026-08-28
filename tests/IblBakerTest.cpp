#include "renderer/lighting/IblBaker.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>

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
