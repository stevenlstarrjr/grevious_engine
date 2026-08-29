#include "renderer/lighting/IblBaker.hpp"

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace renderer::lighting {
namespace {

constexpr float pi = std::numbers::pi_v<float>;
constexpr float twoPi = 2.0F * pi;

float srgbToLinear(float value) {
    return value <= 0.04045F ? value / 12.92F : std::pow((value + 0.055F) / 1.055F, 2.4F);
}

glm::vec2 hammersley(std::uint32_t index, std::uint32_t count) {
    std::uint32_t bits = index;
    bits = (bits << 16U) | (bits >> 16U);
    bits = ((bits & 0x00ff00ffU) << 8U) | ((bits & 0xff00ff00U) >> 8U);
    bits = ((bits & 0x0f0f0f0fU) << 4U) | ((bits & 0xf0f0f0f0U) >> 4U);
    bits = ((bits & 0x33333333U) << 2U) | ((bits & 0xccccccccU) >> 2U);
    bits = ((bits & 0x55555555U) << 1U) | ((bits & 0xaaaaaaaaU) >> 1U);
    constexpr float inverseTwoTo32 = 2.3283064365386963e-10F;
    return {static_cast<float>(index) / static_cast<float>(count),
            static_cast<float>(bits) * inverseTwoTo32};
}

glm::vec2 directionToUv(glm::vec3 direction) {
    direction = glm::normalize(direction);
    float u = std::atan2(direction.z, direction.x) / twoPi + 0.5F;
    u -= std::floor(u);
    const float v = std::acos(glm::clamp(direction.y, -1.0F, 1.0F)) / pi;
    return {u, v};
}

glm::vec3 uvToDirection(float u, float v) {
    const float phi = (u - 0.5F) * twoPi;
    const float theta = v * pi;
    const float sinTheta = std::sin(theta);
    return {sinTheta * std::cos(phi), std::cos(theta), sinTheta * std::sin(phi)};
}

glm::vec3 texelRgb(const FloatImage& image, std::uint32_t x, std::uint32_t y) {
    const std::size_t offset = (static_cast<std::size_t>(y) * image.width + x) * image.channels;
    return {image.texels[offset], image.texels[offset + 1], image.texels[offset + 2]};
}

void setTexel(FloatImage& image, std::uint32_t x, std::uint32_t y, glm::vec3 value) {
    const std::size_t offset = (static_cast<std::size_t>(y) * image.width + x) * image.channels;
    image.texels[offset] = value.x;
    image.texels[offset + 1] = value.y;
    image.texels[offset + 2] = value.z;
    if (image.channels == 4) {
        image.texels[offset + 3] = 1.0F;
    }
}

void setTexel(FloatImage& image, std::uint32_t x, std::uint32_t y, glm::vec2 value) {
    const std::size_t offset = (static_cast<std::size_t>(y) * image.width + x) * image.channels;
    image.texels[offset] = value.x;
    image.texels[offset + 1] = value.y;
}

glm::vec3 sampleLatLong(const FloatImage& image, glm::vec3 direction) {
    const glm::vec2 uv = directionToUv(direction);
    const float x = uv.x * static_cast<float>(image.width) - 0.5F;
    const float y = uv.y * static_cast<float>(image.height) - 0.5F;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const float tx = x - std::floor(x);
    const float ty = y - std::floor(y);
    const auto wrapX = [&image](int coordinate) {
        const int width = static_cast<int>(image.width);
        const int wrapped = coordinate % width;
        return static_cast<std::uint32_t>(wrapped < 0 ? wrapped + width : wrapped);
    };
    const auto clampY = [&image](int coordinate) {
        return static_cast<std::uint32_t>(
            std::clamp(coordinate, 0, static_cast<int>(image.height) - 1));
    };
    const glm::vec3 a = glm::mix(texelRgb(image, wrapX(x0), clampY(y0)),
                                 texelRgb(image, wrapX(x0 + 1), clampY(y0)), tx);
    const glm::vec3 b = glm::mix(texelRgb(image, wrapX(x0), clampY(y0 + 1)),
                                 texelRgb(image, wrapX(x0 + 1), clampY(y0 + 1)), tx);
    return glm::mix(a, b, ty);
}

void basisFromNormal(glm::vec3 normal, glm::vec3& tangent, glm::vec3& bitangent) {
    const glm::vec3 up =
        std::abs(normal.y) < 0.999F ? glm::vec3(0.0F, 1.0F, 0.0F) : glm::vec3(1.0F, 0.0F, 0.0F);
    tangent = glm::normalize(glm::cross(up, normal));
    bitangent = glm::cross(normal, tangent);
}

glm::vec3 localToWorld(glm::vec3 local, glm::vec3 normal) {
    glm::vec3 tangent{};
    glm::vec3 bitangent{};
    basisFromNormal(normal, tangent, bitangent);
    return glm::normalize(local.x * tangent + local.y * bitangent + local.z * normal);
}

glm::vec3 sampleCosineHemisphere(glm::vec2 sample) {
    const float radius = std::sqrt(sample.x);
    const float phi = twoPi * sample.y;
    return {radius * std::cos(phi), radius * std::sin(phi),
            std::sqrt(std::max(0.0F, 1.0F - sample.x))};
}

glm::vec3 sampleVisibleGgx(glm::vec3 view, float alpha, glm::vec2 sample) {
    // Derived from pbrt-v4, Copyright (c) 1998-2020 Matt Pharr, Wenzel Jakob,
    // and Greg Humphreys; Apache-2.0. This is a CPU port of
    // TrowbridgeReitzDistribution::Sample_wm. PBRT's
    // anisotropic alpha_x/alpha_y collapses to glTF's isotropic alpha here.
    glm::vec3 stretched = glm::normalize(glm::vec3(alpha * view.x, alpha * view.y, view.z));
    if (stretched.z < 0.0F) {
        stretched = -stretched;
    }
    const glm::vec3 tangent =
        stretched.z < 0.99999F ? glm::normalize(glm::cross(glm::vec3(0.0F, 0.0F, 1.0F), stretched))
                               : glm::vec3(1.0F, 0.0F, 0.0F);
    const glm::vec3 bitangent = glm::cross(stretched, tangent);
    const float radius = std::sqrt(sample.x);
    const float phi = twoPi * sample.y;
    glm::vec2 disk{radius * std::cos(phi), radius * std::sin(phi)};
    const float horizon = std::sqrt(std::max(0.0F, 1.0F - disk.x * disk.x));
    const float blend = (1.0F + stretched.z) * 0.5F;
    disk.y = glm::mix(horizon, disk.y, blend);
    const float projectedZ = std::sqrt(std::max(0.0F, 1.0F - glm::dot(disk, disk)));
    const glm::vec3 hemisphere = disk.x * tangent + disk.y * bitangent + projectedZ * stretched;
    return glm::normalize(
        glm::vec3(alpha * hemisphere.x, alpha * hemisphere.y, std::max(1.0e-6F, hemisphere.z)));
}

float smithLambda(glm::vec3 direction, float alpha) {
    const float cosine = std::abs(direction.z);
    if (cosine <= 1.0e-7F) {
        return std::numeric_limits<float>::infinity();
    }
    const float tangentSquared = std::max(0.0F, 1.0F - cosine * cosine) / (cosine * cosine);
    return (std::sqrt(1.0F + alpha * alpha * tangentSquared) - 1.0F) * 0.5F;
}

float smithG1(glm::vec3 direction, float alpha) {
    return 1.0F / (1.0F + smithLambda(direction, alpha));
}

float smithG(glm::vec3 view, glm::vec3 light, float alpha) {
    return 1.0F / (1.0F + smithLambda(view, alpha) + smithLambda(light, alpha));
}

FloatImage bakeDiffuseIrradiance(const FloatImage& environment, std::uint32_t width,
                                 std::uint32_t height, std::uint32_t sampleCount) {
    FloatImage result{.width = width,
                      .height = height,
                      .channels = 4,
                      .texels = std::vector<float>(static_cast<std::size_t>(width) * height * 4)};
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const glm::vec3 normal =
                uvToDirection((static_cast<float>(x) + 0.5F) / static_cast<float>(width),
                              (static_cast<float>(y) + 0.5F) / static_cast<float>(height));
            glm::vec3 sum{0.0F};
            for (std::uint32_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
                const glm::vec3 local =
                    sampleCosineHemisphere(hammersley(sampleIndex, sampleCount));
                sum += sampleLatLong(environment, localToWorld(local, normal));
            }
            setTexel(result, x, y, sum * (pi / static_cast<float>(sampleCount)));
        }
    }
    return result;
}

FloatMipChain bakePrefilteredSpecular(const FloatImage& environment,
                                      const IblBakeSettings& settings) {
    FloatMipChain result;
    result.levels.reserve(settings.specularMipLevels);
    for (std::uint32_t level = 0; level < settings.specularMipLevels; ++level) {
        const std::uint32_t width = std::max(1U, settings.specularWidth >> level);
        const std::uint32_t height = std::max(1U, settings.specularHeight >> level);
        FloatImage image{.width = width,
                         .height = height,
                         .channels = 4,
                         .texels =
                             std::vector<float>(static_cast<std::size_t>(width) * height * 4)};
        const float roughness =
            settings.specularMipLevels > 1
                ? static_cast<float>(level) / static_cast<float>(settings.specularMipLevels - 1)
                : 0.0F;
        const float alpha = std::max(roughness * roughness, 1.0e-4F);
        for (std::uint32_t y = 0; y < height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) {
                const glm::vec3 reflection =
                    uvToDirection((static_cast<float>(x) + 0.5F) / static_cast<float>(width),
                                  (static_cast<float>(y) + 0.5F) / static_cast<float>(height));
                glm::vec3 sum{0.0F};
                float totalWeight = 0.0F;
                for (std::uint32_t sampleIndex = 0; sampleIndex < settings.sampleCount;
                     ++sampleIndex) {
                    const glm::vec3 halfLocal =
                        sampleVisibleGgx(glm::vec3(0.0F, 0.0F, 1.0F), alpha,
                                         hammersley(sampleIndex, settings.sampleCount));
                    const glm::vec3 lightLocal =
                        2.0F * glm::dot(glm::vec3(0.0F, 0.0F, 1.0F), halfLocal) * halfLocal -
                        glm::vec3(0.0F, 0.0F, 1.0F);
                    if (lightLocal.z <= 0.0F) {
                        continue;
                    }
                    const float weight = lightLocal.z;
                    sum +=
                        sampleLatLong(environment, localToWorld(lightLocal, reflection)) * weight;
                    totalWeight += weight;
                }
                setTexel(image, x, y,
                         totalWeight > 0.0F ? sum / totalWeight
                                            : sampleLatLong(environment, reflection));
            }
        }
        result.levels.push_back(std::move(image));
    }
    return result;
}

FloatImage bakeBrdfLut(std::uint32_t size, std::uint32_t sampleCount) {
    FloatImage result{.width = size,
                      .height = size,
                      .channels = 2,
                      .texels = std::vector<float>(static_cast<std::size_t>(size) * size * 2)};
    for (std::uint32_t y = 0; y < size; ++y) {
        const float roughness = (static_cast<float>(y) + 0.5F) / static_cast<float>(size);
        const float alpha = std::max(roughness * roughness, 1.0e-4F);
        for (std::uint32_t x = 0; x < size; ++x) {
            const float noV = (static_cast<float>(x) + 0.5F) / static_cast<float>(size);
            const glm::vec3 view{std::sqrt(std::max(0.0F, 1.0F - noV * noV)), 0.0F, noV};
            const float viewMasking = smithG1(view, alpha);
            glm::vec2 integrated{0.0F};
            for (std::uint32_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
                const glm::vec3 halfVector =
                    sampleVisibleGgx(view, alpha, hammersley(sampleIndex, sampleCount));
                const glm::vec3 light = 2.0F * glm::dot(view, halfVector) * halfVector - view;
                if (light.z <= 0.0F) {
                    continue;
                }
                const float visibility = smithG(view, light, alpha) / viewMasking;
                const float fresnelWeight =
                    std::pow(1.0F - glm::clamp(glm::dot(view, halfVector), 0.0F, 1.0F), 5.0F);
                integrated.x += (1.0F - fresnelWeight) * visibility;
                integrated.y += fresnelWeight * visibility;
            }
            setTexel(result, x, y, integrated / static_cast<float>(sampleCount));
        }
    }
    return result;
}

void validateSettings(const IblBakeSettings& settings) {
    if (settings.irradianceWidth == 0 || settings.irradianceHeight == 0 ||
        settings.specularWidth == 0 || settings.specularHeight == 0 ||
        settings.specularMipLevels == 0 || settings.brdfLutSize == 0 || settings.sampleCount == 0 ||
        settings.specularMipLevels > 32) {
        throw std::runtime_error("IBL bake settings must be nonzero and bounded");
    }
    const std::uint32_t maxDimension = std::max(settings.specularWidth, settings.specularHeight);
    const std::uint32_t maximumMips = 32U - std::countl_zero(maxDimension);
    if (settings.specularMipLevels > maximumMips) {
        throw std::runtime_error("IBL specular mip count exceeds its image dimensions");
    }
}

} // namespace

bool FloatImage::valid() const {
    return width != 0 && height != 0 && (channels == 2 || channels == 4) &&
           texels.size() == static_cast<std::size_t>(width) * height * channels;
}

bool EnvironmentSamplingDistribution::valid() const {
    return width != 0 && height != 0 &&
           entries.size() == static_cast<std::size_t>(width) * height;
}

EnvironmentSamplingDistribution
buildEnvironmentSamplingDistribution(const FloatImage& environment) {
    if (!environment.valid() || environment.channels < 3U) {
        throw std::runtime_error(
            "Environment importance sampling requires a valid RGB environment");
    }

    const std::size_t entryCount =
        static_cast<std::size_t>(environment.width) * environment.height;
    std::vector<double> weights(entryCount);
    double totalWeight = 0.0;
    const double deltaPhi = 2.0 * std::numbers::pi_v<double> /
                            static_cast<double>(environment.width);
    for (std::uint32_t y = 0; y < environment.height; ++y) {
        const double theta0 = std::numbers::pi_v<double> * static_cast<double>(y) /
                              static_cast<double>(environment.height);
        const double theta1 = std::numbers::pi_v<double> * static_cast<double>(y + 1U) /
                              static_cast<double>(environment.height);
        const double texelSolidAngle = deltaPhi * (std::cos(theta0) - std::cos(theta1));
        for (std::uint32_t x = 0; x < environment.width; ++x) {
            const glm::vec3 rgb = texelRgb(environment, x, y);
            if (!std::isfinite(rgb.x) || !std::isfinite(rgb.y) || !std::isfinite(rgb.z)) {
                throw std::runtime_error(
                    "Environment importance sampling requires finite RGB radiance");
            }
            const double luminance = std::max(
                0.0, 0.2126 * static_cast<double>(rgb.x) +
                         0.7152 * static_cast<double>(rgb.y) +
                         0.0722 * static_cast<double>(rgb.z));
            const std::size_t index = static_cast<std::size_t>(y) * environment.width + x;
            weights[index] = luminance * texelSolidAngle;
            totalWeight += weights[index];
        }
    }

    if (!(totalWeight > std::numeric_limits<double>::epsilon())) {
        totalWeight = 0.0;
        for (std::uint32_t y = 0; y < environment.height; ++y) {
            const double theta0 = std::numbers::pi_v<double> * static_cast<double>(y) /
                                  static_cast<double>(environment.height);
            const double theta1 = std::numbers::pi_v<double> * static_cast<double>(y + 1U) /
                                  static_cast<double>(environment.height);
            const double texelSolidAngle = deltaPhi * (std::cos(theta0) - std::cos(theta1));
            for (std::uint32_t x = 0; x < environment.width; ++x) {
                const std::size_t index =
                    static_cast<std::size_t>(y) * environment.width + x;
                weights[index] = texelSolidAngle;
                totalWeight += texelSolidAngle;
            }
        }
    }

    EnvironmentSamplingDistribution result{
        .width = environment.width,
        .height = environment.height,
        .entries = std::vector<EnvironmentAliasEntry>(entryCount),
    };
    std::vector<double> scaled(entryCount);
    std::vector<std::uint32_t> small;
    std::vector<std::uint32_t> large;
    small.reserve(entryCount);
    large.reserve(entryCount);
    for (std::size_t index = 0; index < entryCount; ++index) {
        const double mass = weights[index] / totalWeight;
        result.entries[index].probabilityMass = static_cast<float>(mass);
        result.entries[index].aliasIndex = static_cast<std::uint32_t>(index);
        scaled[index] = mass * static_cast<double>(entryCount);
        (scaled[index] < 1.0 ? small : large).push_back(static_cast<std::uint32_t>(index));
    }

    while (!small.empty() && !large.empty()) {
        const std::uint32_t smallIndex = small.back();
        small.pop_back();
        const std::uint32_t largeIndex = large.back();
        large.pop_back();
        result.entries[smallIndex].threshold =
            static_cast<float>(std::clamp(scaled[smallIndex], 0.0, 1.0));
        result.entries[smallIndex].aliasIndex = largeIndex;
        scaled[largeIndex] = scaled[largeIndex] + scaled[smallIndex] - 1.0;
        (scaled[largeIndex] < 1.0 ? small : large).push_back(largeIndex);
    }
    for (const std::uint32_t index : small) {
        result.entries[index].threshold = 1.0F;
        result.entries[index].aliasIndex = index;
    }
    for (const std::uint32_t index : large) {
        result.entries[index].threshold = 1.0F;
        result.entries[index].aliasIndex = index;
    }
    return result;
}

FloatImage toLinearFloatImage(const assets::DecodedImage& image) {
    if (!image.valid()) {
        throw std::runtime_error("Cannot convert an invalid decoded image");
    }
    FloatImage result{.width = image.width,
                      .height = image.height,
                      .channels = 4,
                      .texels = std::vector<float>(image.texelCount() * 4)};
    if (image.componentType == assets::ImageComponentType::float32) {
        result.texels = image.float32;
        return result;
    }
    for (std::size_t index = 0; index < image.unorm8.size(); ++index) {
        const float normalized = static_cast<float>(image.unorm8[index]) / 255.0F;
        const bool colorChannel = index % 4 != 3;
        result.texels[index] = colorChannel && image.colorSpace == assets::ImageColorSpace::srgb
                                   ? srgbToLinear(normalized)
                                   : normalized;
    }
    return result;
}

FloatImage makeProceduralStudioEnvironment(std::uint32_t width, std::uint32_t height) {
    if (width < 2 || height < 2) {
        throw std::runtime_error("Procedural environment dimensions must be at least 2x2");
    }
    FloatImage result{.width = width,
                      .height = height,
                      .channels = 4,
                      .texels = std::vector<float>(static_cast<std::size_t>(width) * height * 4)};
    const glm::vec3 keyDirection = glm::normalize(glm::vec3(-0.45F, 0.58F, 0.68F));
    const glm::vec3 rimDirection = glm::normalize(glm::vec3(0.72F, 0.18F, -0.67F));
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const glm::vec3 direction =
                uvToDirection((static_cast<float>(x) + 0.5F) / static_cast<float>(width),
                              (static_cast<float>(y) + 0.5F) / static_cast<float>(height));
            const float skyBlend = glm::clamp(direction.y * 0.5F + 0.5F, 0.0F, 1.0F);
            glm::vec3 radiance = glm::mix(glm::vec3(0.012F, 0.015F, 0.022F),
                                          glm::vec3(0.20F, 0.31F, 0.52F), skyBlend);
            radiance += glm::vec3(26.0F, 17.0F, 8.0F) *
                        std::exp((glm::dot(direction, keyDirection) - 1.0F) / 0.008F);
            radiance += glm::vec3(3.5F, 6.0F, 11.0F) *
                        std::exp((glm::dot(direction, rimDirection) - 1.0F) / 0.025F);
            setTexel(result, x, y, radiance);
        }
    }
    return result;
}

IblAssets bakeImageBasedLighting(FloatImage environment, const IblBakeSettings& settings) {
    if (!environment.valid() || environment.channels != 4) {
        throw std::runtime_error("IBL environment must be a valid RGBA float image");
    }
    validateSettings(settings);
    IblAssets result;
    result.environment = std::move(environment);
    result.diffuseIrradiance =
        bakeDiffuseIrradiance(result.environment, settings.irradianceWidth,
                              settings.irradianceHeight, settings.sampleCount);
    result.prefilteredSpecular = bakePrefilteredSpecular(result.environment, settings);
    result.brdfLut = bakeBrdfLut(settings.brdfLutSize, settings.sampleCount);
    return result;
}

} // namespace renderer::lighting
