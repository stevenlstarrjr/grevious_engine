#include "renderer/post/PostProcessing.hpp"

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace renderer::post {
namespace {

std::size_t pixelIndex(const HdrImage& image, std::uint32_t x, std::uint32_t y) {
    if (x >= image.width || y >= image.height) {
        throw std::out_of_range("Post-processing image coordinate is out of range");
    }
    return static_cast<std::size_t>(y) * image.width + x;
}

glm::vec3 blurSample(const HdrImage& source, glm::vec2 uv, glm::vec2 texelStep) {
    glm::vec3 color = sampleLinearClamp(source, uv) * blurCenterWeight;
    color += sampleLinearClamp(source, uv + texelStep * blurNearOffset) * blurNearWeight;
    color += sampleLinearClamp(source, uv - texelStep * blurNearOffset) * blurNearWeight;
    color += sampleLinearClamp(source, uv + texelStep * blurFarOffset) * blurFarWeight;
    color += sampleLinearClamp(source, uv - texelStep * blurFarOffset) * blurFarWeight;
    return color;
}

} // namespace

HdrImage::HdrImage(std::uint32_t imageWidth, std::uint32_t imageHeight, glm::vec3 clearValue)
    : width(imageWidth), height(imageHeight),
      pixels(static_cast<std::size_t>(imageWidth) * imageHeight, clearValue) {
    if (imageWidth == 0U || imageHeight == 0U) {
        throw std::invalid_argument("Post-processing images require non-zero dimensions");
    }
}

bool HdrImage::valid() const {
    return width > 0U && height > 0U &&
           pixels.size() == static_cast<std::size_t>(width) * height;
}

const glm::vec3& HdrImage::at(std::uint32_t x, std::uint32_t y) const {
    return pixels.at(pixelIndex(*this, x, y));
}

glm::vec3& HdrImage::at(std::uint32_t x, std::uint32_t y) {
    return pixels.at(pixelIndex(*this, x, y));
}

std::uint64_t ResourceMemory::bloomBytes() const {
    return std::accumulate(bloomLevels.begin(), bloomLevels.end(), std::uint64_t{0},
                           [](std::uint64_t total, const BloomLevelMemory& level) {
                               return total + level.totalBytes();
                           });
}

std::uint64_t ResourceMemory::totalBytes() const {
    return hdrBytes + motionBytes + normalBytes + depthBytes + bloomBytes() +
           temporalHistoryBytes;
}

std::uint32_t halfExtent(std::uint32_t extent) {
    return std::max(1U, extent / 2U + extent % 2U);
}

float luminance(const glm::vec3& color) {
    return glm::dot(color, glm::vec3{0.2126F, 0.7152F, 0.0722F});
}

glm::vec3 bloomExtract(const glm::vec3& color) {
    const float brightness = std::max({color.r, color.g, color.b});
    float soft = glm::clamp((brightness - bloomThreshold + bloomKnee) / (2.0F * bloomKnee),
                            0.0F, 1.0F);
    soft = soft * soft * bloomKnee;
    const float contribution =
        std::max(brightness - bloomThreshold, soft) / std::max(brightness, 1.0e-5F);
    return color * contribution;
}

HdrImage makeTestPattern(std::uint32_t width, std::uint32_t height, TestPattern pattern,
                         float sourceValue, float centerX) {
    if (!std::isfinite(sourceValue) || sourceValue < 0.0F || sourceValue > 65504.0F) {
        throw std::invalid_argument("Post test source value must be finite and in [0, 65504]");
    }
    HdrImage image(width, height);
    const std::uint32_t centerColumn = std::min(
        width - 1U, static_cast<std::uint32_t>(glm::clamp(centerX, 0.0F, 1.0F) *
                                              static_cast<float>(width - 1U)));
    const std::uint32_t centerRow = height / 2U;
    const glm::vec3 source{sourceValue};
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            bool bright = false;
            switch (pattern) {
            case TestPattern::none:
            case TestPattern::black:
                break;
            case TestPattern::constant:
                bright = true;
                break;
            case TestPattern::impulse:
                bright = x == centerColumn && y == centerRow;
                break;
            case TestPattern::square:
            case TestPattern::movingSquare:
                bright = std::abs(static_cast<int>(x) - static_cast<int>(centerColumn)) <= 2 &&
                         std::abs(static_cast<int>(y) - static_cast<int>(centerRow)) <= 2;
                break;
            case TestPattern::horizontalLine:
                bright = y == centerRow;
                break;
            case TestPattern::verticalLine:
                bright = x == centerColumn;
                break;
            case TestPattern::multipleSources:
                bright = (x == width / 4U && y == height / 4U) ||
                         (x == centerColumn && y == centerRow) ||
                         (x == (width * 3U) / 4U && y == (height * 3U) / 4U);
                break;
            case TestPattern::checkerboard:
                bright = ((x + y) & 1U) == 0U;
                break;
            }
            if (bright) {
                image.at(x, y) = source;
            }
        }
    }
    return image;
}

glm::vec3 sampleLinearClamp(const HdrImage& image, glm::vec2 uv) {
    if (!image.valid()) {
        throw std::invalid_argument("Cannot sample an invalid post-processing image");
    }
    const float sourceX = uv.x * static_cast<float>(image.width) - 0.5F;
    const float sourceY = uv.y * static_cast<float>(image.height) - 0.5F;
    const int x0 = static_cast<int>(std::floor(sourceX));
    const int y0 = static_cast<int>(std::floor(sourceY));
    const float tx = sourceX - static_cast<float>(x0);
    const float ty = sourceY - static_cast<float>(y0);
    const auto clampedX = [&](int x) {
        return static_cast<std::uint32_t>(std::clamp(x, 0, static_cast<int>(image.width) - 1));
    };
    const auto clampedY = [&](int y) {
        return static_cast<std::uint32_t>(std::clamp(y, 0, static_cast<int>(image.height) - 1));
    };
    const glm::vec3 top = glm::mix(image.at(clampedX(x0), clampedY(y0)),
                                   image.at(clampedX(x0 + 1), clampedY(y0)), tx);
    const glm::vec3 bottom = glm::mix(image.at(clampedX(x0), clampedY(y0 + 1)),
                                      image.at(clampedX(x0 + 1), clampedY(y0 + 1)), tx);
    return glm::mix(top, bottom, ty);
}

HdrImage downsampleBloom(const HdrImage& source) {
    if (!source.valid()) {
        throw std::invalid_argument("Cannot downsample an invalid post-processing image");
    }
    HdrImage output(halfExtent(source.width), halfExtent(source.height));
    const glm::vec2 sourceTexel{1.0F / static_cast<float>(source.width),
                                1.0F / static_cast<float>(source.height)};
    for (std::uint32_t y = 0; y < output.height; ++y) {
        for (std::uint32_t x = 0; x < output.width; ++x) {
            const glm::vec2 uv{(static_cast<float>(x) + 0.5F) / static_cast<float>(output.width),
                               (static_cast<float>(y) + 0.5F) /
                                   static_cast<float>(output.height)};
            glm::vec3 color = sampleLinearClamp(source, uv) * 0.5F;
            color += sampleLinearClamp(source, uv + sourceTexel * glm::vec2{-1.0F, -1.0F}) *
                     0.125F;
            color += sampleLinearClamp(source, uv + sourceTexel * glm::vec2{1.0F, -1.0F}) *
                     0.125F;
            color += sampleLinearClamp(source, uv + sourceTexel * glm::vec2{-1.0F, 1.0F}) *
                     0.125F;
            color += sampleLinearClamp(source, uv + sourceTexel * glm::vec2{1.0F, 1.0F}) *
                     0.125F;
            output.at(x, y) = bloomExtract(color);
        }
    }
    return output;
}

HdrImage blurHorizontal(const HdrImage& source) {
    if (!source.valid()) {
        throw std::invalid_argument("Cannot blur an invalid post-processing image");
    }
    HdrImage output(source.width, source.height);
    const glm::vec2 texelStep{1.0F / static_cast<float>(source.width), 0.0F};
    for (std::uint32_t y = 0; y < source.height; ++y) {
        for (std::uint32_t x = 0; x < source.width; ++x) {
            const glm::vec2 uv{(static_cast<float>(x) + 0.5F) / static_cast<float>(source.width),
                               (static_cast<float>(y) + 0.5F) /
                                   static_cast<float>(source.height)};
            output.at(x, y) = blurSample(source, uv, texelStep);
        }
    }
    return output;
}

HdrImage blurVertical(const HdrImage& source) {
    if (!source.valid()) {
        throw std::invalid_argument("Cannot blur an invalid post-processing image");
    }
    HdrImage output(source.width, source.height);
    const glm::vec2 texelStep{0.0F, 1.0F / static_cast<float>(source.height)};
    for (std::uint32_t y = 0; y < source.height; ++y) {
        for (std::uint32_t x = 0; x < source.width; ++x) {
            const glm::vec2 uv{(static_cast<float>(x) + 0.5F) / static_cast<float>(source.width),
                               (static_cast<float>(y) + 0.5F) /
                                   static_cast<float>(source.height)};
            output.at(x, y) = blurSample(source, uv, texelStep);
        }
    }
    return output;
}

float exposureMultiplier(float exposureEv) {
    return std::exp2(exposureEv);
}

glm::vec3 applyExposure(const glm::vec3& color, float exposureEv) {
    return color * exposureMultiplier(exposureEv);
}

glm::vec3 acesFitted(const glm::vec3& color) {
    constexpr float a = 2.51F;
    constexpr float b = 0.03F;
    constexpr float c = 2.43F;
    constexpr float d = 0.59F;
    constexpr float e = 0.14F;
    return glm::clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0F,
                      1.0F);
}

glm::vec3 compositeToneMap(const glm::vec3& hdr, const glm::vec3& bloom, float exposureEv) {
    return acesFitted(applyExposure(hdr + bloom * bloomIntensity, exposureEv));
}

ImageMetrics measure(const HdrImage& image) {
    if (!image.valid()) {
        throw std::invalid_argument("Cannot measure an invalid post-processing image");
    }
    ImageMetrics metrics;
    glm::dvec2 weightedPosition{0.0};
    for (std::uint32_t y = 0; y < image.height; ++y) {
        for (std::uint32_t x = 0; x < image.width; ++x) {
            const glm::vec3 color = image.at(x, y);
            metrics.finite = metrics.finite && std::isfinite(color.r) && std::isfinite(color.g) &&
                             std::isfinite(color.b);
            const float value = luminance(color);
            metrics.maximumLuminance = std::max(metrics.maximumLuminance, value);
            metrics.totalLuminance += static_cast<double>(value);
            weightedPosition += glm::dvec2{static_cast<double>(x), static_cast<double>(y)} *
                                static_cast<double>(value);
        }
    }
    metrics.meanLuminance = metrics.totalLuminance /
                            static_cast<double>(static_cast<std::uint64_t>(image.width) *
                                                image.height);
    if (metrics.totalLuminance > std::numeric_limits<double>::epsilon()) {
        metrics.luminanceCentroid = weightedPosition / metrics.totalLuminance;
    }
    return metrics;
}

ResourceMemory resourceMemory(std::uint32_t width, std::uint32_t height, bool bloomEnabled,
                              bool taaEnabled) {
    const std::uint64_t pixelCount = static_cast<std::uint64_t>(width) * height;
    const std::uint64_t halfPixelCount = static_cast<std::uint64_t>(halfExtent(width)) *
                                         halfExtent(height);
    ResourceMemory memory{
        .hdrBytes = pixelCount * 8U,
        .motionBytes = pixelCount * 8U,
        .normalBytes = pixelCount * 8U,
        .depthBytes = pixelCount * 4U,
        .bloomLevels = {},
    };
    if (bloomEnabled) {
        memory.bloomLevels.push_back({
            .width = halfExtent(width),
            .height = halfExtent(height),
            .filteredBytes = halfPixelCount * 8U,
            .scratchBytes = halfPixelCount * 8U,
        });
    }
    if (taaEnabled) {
        // Double-buffered RGBA16F color, R32F depth, and RGBA16F normal histories.
        memory.temporalHistoryBytes = pixelCount * (2U * 8U + 2U * 4U + 2U * 8U);
    }
    return memory;
}

} // namespace renderer::post
