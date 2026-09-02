#include "renderer/procedural/NoiseVolume.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace renderer::procedural {
namespace {

struct Vec3 {
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
};

std::uint32_t hash(std::uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    value ^= value >> 16U;
    return value;
}

std::uint32_t hashCell(std::uint32_t x, std::uint32_t y, std::uint32_t z,
                       std::uint32_t seed) {
    return hash(x * 0x9e3779b9U ^ y * 0x85ebca6bU ^ z * 0xc2b2ae35U ^ seed);
}

float unitFloat(std::uint32_t value) {
    return static_cast<float>(value >> 8U) * (1.0F / 16777216.0F);
}

float fade(float value) {
    return value * value * value * (value * (value * 6.0F - 15.0F) + 10.0F);
}

float lerp(float a, float b, float weight) {
    return a + (b - a) * weight;
}

std::uint32_t wrapCell(int value, std::uint32_t period) {
    const int signedPeriod = static_cast<int>(period);
    int wrapped = value % signedPeriod;
    if (wrapped < 0) {
        wrapped += signedPeriod;
    }
    return static_cast<std::uint32_t>(wrapped);
}

float latticeGradientDot(int x, int y, int z, float dx, float dy, float dz,
                         std::uint32_t period, std::uint32_t seed) {
    constexpr std::array gradients{
        Vec3{1.0F, 1.0F, 0.0F}, Vec3{-1.0F, 1.0F, 0.0F},
        Vec3{1.0F, -1.0F, 0.0F}, Vec3{-1.0F, -1.0F, 0.0F},
        Vec3{1.0F, 0.0F, 1.0F}, Vec3{-1.0F, 0.0F, 1.0F},
        Vec3{1.0F, 0.0F, -1.0F}, Vec3{-1.0F, 0.0F, -1.0F},
        Vec3{0.0F, 1.0F, 1.0F}, Vec3{0.0F, -1.0F, 1.0F},
        Vec3{0.0F, 1.0F, -1.0F}, Vec3{0.0F, -1.0F, -1.0F},
    };
    const std::uint32_t value = hashCell(
        wrapCell(x, period), wrapCell(y, period), wrapCell(z, period), seed);
    const Vec3 gradient = gradients[value % gradients.size()];
    return (gradient.x * dx + gradient.y * dy + gradient.z * dz) * 0.70710678F;
}

float periodicGradientNoise(Vec3 point, std::uint32_t period,
                            std::uint32_t seed) {
    const float scaledX = point.x * static_cast<float>(period);
    const float scaledY = point.y * static_cast<float>(period);
    const float scaledZ = point.z * static_cast<float>(period);
    const int x = static_cast<int>(std::floor(scaledX));
    const int y = static_cast<int>(std::floor(scaledY));
    const int z = static_cast<int>(std::floor(scaledZ));
    const float fx = fade(scaledX - static_cast<float>(x));
    const float fy = fade(scaledY - static_cast<float>(y));
    const float fz = fade(scaledZ - static_cast<float>(z));

    const float localX = scaledX - static_cast<float>(x);
    const float localY = scaledY - static_cast<float>(y);
    const float localZ = scaledZ - static_cast<float>(z);
    const float x00 = lerp(
        latticeGradientDot(x, y, z, localX, localY, localZ, period, seed),
        latticeGradientDot(x + 1, y, z, localX - 1.0F, localY, localZ,
                           period, seed),
        fx);
    const float x10 = lerp(
        latticeGradientDot(x, y + 1, z, localX, localY - 1.0F, localZ,
                           period, seed),
        latticeGradientDot(x + 1, y + 1, z, localX - 1.0F,
                           localY - 1.0F, localZ, period, seed),
        fx);
    const float x01 = lerp(
        latticeGradientDot(x, y, z + 1, localX, localY, localZ - 1.0F,
                           period, seed),
        latticeGradientDot(x + 1, y, z + 1, localX - 1.0F, localY,
                           localZ - 1.0F, period, seed),
        fx);
    const float x11 = lerp(
        latticeGradientDot(x, y + 1, z + 1, localX, localY - 1.0F,
                           localZ - 1.0F, period, seed),
        latticeGradientDot(x + 1, y + 1, z + 1, localX - 1.0F,
                           localY - 1.0F, localZ - 1.0F, period, seed),
        fx);
    const float noise = lerp(lerp(x00, x10, fy), lerp(x01, x11, fy), fz);
    return std::clamp(0.5F + noise * 0.82F, 0.0F, 1.0F);
}

float periodicFbm(Vec3 point, std::uint32_t firstPeriod,
                  std::uint32_t seed) {
    float value = 0.0F;
    float normalization = 0.0F;
    float amplitude = 0.58F;
    std::uint32_t period = firstPeriod;
    for (std::uint32_t octave = 0U; octave < 4U; ++octave) {
        value += periodicGradientNoise(point, period,
                                       seed + octave * 0x68bc21ebU) * amplitude;
        normalization += amplitude;
        amplitude *= 0.5F;
        period *= 2U;
    }
    return value / normalization;
}

class WorleyGrid {
  public:
    WorleyGrid(std::uint32_t period, std::uint32_t seed)
        : period_(period), points_(static_cast<std::size_t>(period) * period * period) {
        for (std::uint32_t z = 0U; z < period_; ++z) {
            for (std::uint32_t y = 0U; y < period_; ++y) {
                for (std::uint32_t x = 0U; x < period_; ++x) {
                    const std::uint32_t cellHash = hashCell(x, y, z, seed);
                    points_[index(x, y, z)] = {
                        unitFloat(hash(cellHash ^ 0xa511e9b3U)),
                        unitFloat(hash(cellHash ^ 0x63d83595U)),
                        unitFloat(hash(cellHash ^ 0x9e3779b9U)),
                    };
                }
            }
        }
    }

    [[nodiscard]] float cellular(Vec3 point) const {
        const float scaledX = point.x * static_cast<float>(period_);
        const float scaledY = point.y * static_cast<float>(period_);
        const float scaledZ = point.z * static_cast<float>(period_);
        const int baseX = static_cast<int>(std::floor(scaledX));
        const int baseY = static_cast<int>(std::floor(scaledY));
        const int baseZ = static_cast<int>(std::floor(scaledZ));
        float minimumSquaredDistance = std::numeric_limits<float>::max();
        for (int z = -1; z <= 1; ++z) {
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    const int cellX = baseX + x;
                    const int cellY = baseY + y;
                    const int cellZ = baseZ + z;
                    const Vec3 feature = points_[index(
                        wrapCell(cellX, period_), wrapCell(cellY, period_),
                        wrapCell(cellZ, period_))];
                    const float dx = static_cast<float>(cellX) + feature.x - scaledX;
                    const float dy = static_cast<float>(cellY) + feature.y - scaledY;
                    const float dz = static_cast<float>(cellZ) + feature.z - scaledZ;
                    minimumSquaredDistance = std::min(
                        minimumSquaredDistance, dx * dx + dy * dy + dz * dz);
                }
            }
        }
        return std::clamp(1.0F - std::sqrt(minimumSquaredDistance) * 0.92F,
                          0.0F, 1.0F);
    }

  private:
    [[nodiscard]] std::size_t index(std::uint32_t x, std::uint32_t y,
                                    std::uint32_t z) const {
        return (static_cast<std::size_t>(z) * period_ + y) * period_ + x;
    }

    std::uint32_t period_ = 0U;
    std::vector<Vec3> points_;
};

std::uint8_t quantize(float value) {
    return static_cast<std::uint8_t>(
        std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F));
}

NoiseVolumeLevel makeLevel(std::uint32_t extent, std::uint32_t seed,
                           bool detail) {
    const std::array periods = detail ? std::array{3U, 6U, 12U}
                                      : std::array{4U, 8U, 16U};
    const WorleyGrid low(periods[0], seed ^ 0x30f2a91bU);
    const WorleyGrid medium(periods[1], seed ^ 0x91e10da5U);
    const WorleyGrid high(periods[2], seed ^ 0x6c8e9cf5U);
    NoiseVolumeLevel level{
        .extent = extent,
        .texels = std::vector<std::uint8_t>(
            static_cast<std::size_t>(extent) * extent * extent * 4U),
    };
    const float inverseExtent = 1.0F / static_cast<float>(extent);
    for (std::uint32_t z = 0U; z < extent; ++z) {
        for (std::uint32_t y = 0U; y < extent; ++y) {
            for (std::uint32_t x = 0U; x < extent; ++x) {
                const Vec3 point{(static_cast<float>(x) + 0.5F) * inverseExtent,
                                 (static_cast<float>(y) + 0.5F) * inverseExtent,
                                 (static_cast<float>(z) + 0.5F) * inverseExtent};
                const float fbm = periodicFbm(point, detail ? 3U : 4U,
                                              seed ^ 0x243f6a88U);
                const float cellularLow = low.cellular(point);
                const float cellularMedium = medium.cellular(point);
                const float cellularHigh = high.cellular(point);
                const std::size_t texel =
                    ((static_cast<std::size_t>(z) * extent + y) * extent + x) * 4U;
                if (detail) {
                    level.texels[texel + 0U] = quantize(cellularLow);
                    level.texels[texel + 1U] = quantize(cellularMedium);
                    level.texels[texel + 2U] = quantize(cellularHigh);
                    level.texels[texel + 3U] = quantize(fbm);
                } else {
                    level.texels[texel + 0U] = quantize(fbm);
                    level.texels[texel + 1U] = quantize(cellularLow);
                    level.texels[texel + 2U] = quantize(cellularMedium);
                    level.texels[texel + 3U] = quantize(cellularHigh);
                }
            }
        }
    }
    return level;
}

NoiseVolumeLevel downsample(const NoiseVolumeLevel& source) {
    const std::uint32_t extent = std::max(1U, source.extent / 2U);
    NoiseVolumeLevel result{
        .extent = extent,
        .texels = std::vector<std::uint8_t>(
            static_cast<std::size_t>(extent) * extent * extent * 4U),
    };
    for (std::uint32_t z = 0U; z < extent; ++z) {
        for (std::uint32_t y = 0U; y < extent; ++y) {
            for (std::uint32_t x = 0U; x < extent; ++x) {
                std::array<std::uint32_t, 4> sum{};
                for (std::uint32_t dz = 0U; dz < 2U; ++dz) {
                    for (std::uint32_t dy = 0U; dy < 2U; ++dy) {
                        for (std::uint32_t dx = 0U; dx < 2U; ++dx) {
                            const std::uint32_t sourceX = std::min(x * 2U + dx,
                                                                  source.extent - 1U);
                            const std::uint32_t sourceY = std::min(y * 2U + dy,
                                                                  source.extent - 1U);
                            const std::uint32_t sourceZ = std::min(z * 2U + dz,
                                                                  source.extent - 1U);
                            const std::size_t sourceTexel =
                                ((static_cast<std::size_t>(sourceZ) * source.extent + sourceY) *
                                     source.extent +
                                 sourceX) *
                                4U;
                            for (std::size_t channel = 0U; channel < sum.size(); ++channel) {
                                sum[channel] += source.texels[sourceTexel + channel];
                            }
                        }
                    }
                }
                const std::size_t destinationTexel =
                    ((static_cast<std::size_t>(z) * extent + y) * extent + x) * 4U;
                for (std::size_t channel = 0U; channel < sum.size(); ++channel) {
                    result.texels[destinationTexel + channel] =
                        static_cast<std::uint8_t>((sum[channel] + 4U) / 8U);
                }
            }
        }
    }
    return result;
}

NoiseVolume makeVolume(std::uint32_t extent, std::uint32_t seed, bool detail) {
    if (extent < 4U || (extent & (extent - 1U)) != 0U) {
        throw std::runtime_error("Noise volume extent must be a power of two of at least four");
    }
    NoiseVolume volume;
    volume.levels.push_back(makeLevel(extent, seed, detail));
    while (volume.levels.back().extent > 1U) {
        volume.levels.push_back(downsample(volume.levels.back()));
    }
    return volume;
}

} // namespace

bool NoiseVolumeLevel::valid() const {
    const std::size_t expected =
        static_cast<std::size_t>(extent) * extent * extent * 4U;
    return extent > 0U && texels.size() == expected;
}

bool NoiseVolume::valid() const {
    if (levels.empty() || !levels.front().valid()) {
        return false;
    }
    std::uint32_t expectedExtent = levels.front().extent;
    for (const NoiseVolumeLevel& level : levels) {
        if (!level.valid() || level.extent != expectedExtent) {
            return false;
        }
        expectedExtent = std::max(1U, expectedExtent / 2U);
    }
    return levels.back().extent == 1U;
}

CloudNoiseVolumes makeCloudNoiseVolumes(std::uint32_t baseExtent,
                                        std::uint32_t detailExtent,
                                        std::uint32_t seed) {
    return {
        .base = makeVolume(baseExtent, seed, false),
        .detail = makeVolume(detailExtent, seed ^ 0xb7e15162U, true),
    };
}

} // namespace renderer::procedural
