#pragma once

#include "renderer/assets/ImageDecoder.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace renderer::lighting {

struct FloatImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 4;
    std::vector<float> texels;

    [[nodiscard]] bool valid() const;
};

struct FloatMipChain {
    std::vector<FloatImage> levels;
};

struct IblBakeSettings {
    std::uint32_t irradianceWidth = 32;
    std::uint32_t irradianceHeight = 16;
    std::uint32_t specularWidth = 128;
    std::uint32_t specularHeight = 64;
    std::uint32_t specularMipLevels = 8;
    std::uint32_t brdfLutSize = 64;
    std::uint32_t sampleCount = 64;
};

struct IblAssets {
    FloatImage environment;
    FloatImage diffuseIrradiance;
    FloatMipChain prefilteredSpecular;
    FloatImage brdfLut;
};

struct alignas(16) EnvironmentAliasEntry {
    float threshold = 1.0F;
    std::uint32_t aliasIndex = 0;
    float probabilityMass = 0.0F;
    float reserved = 0.0F;
};

struct EnvironmentSamplingDistribution {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<EnvironmentAliasEntry> entries;

    [[nodiscard]] bool valid() const;
};

static_assert(sizeof(EnvironmentAliasEntry) == 4 * sizeof(std::uint32_t));

[[nodiscard]] FloatImage toLinearFloatImage(const assets::DecodedImage& image);
[[nodiscard]] FloatImage makeProceduralStudioEnvironment(std::uint32_t width = 256,
                                                         std::uint32_t height = 128);
[[nodiscard]] EnvironmentSamplingDistribution
buildEnvironmentSamplingDistribution(const FloatImage& environment);
[[nodiscard]] IblAssets bakeImageBasedLighting(FloatImage environment,
                                               const IblBakeSettings& settings = {});

} // namespace renderer::lighting
