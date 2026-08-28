#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace renderer::post {

inline constexpr float bloomThreshold = 1.2F;
inline constexpr float bloomKnee = 0.6F;
inline constexpr float bloomIntensity = 0.08F;
inline constexpr float blurCenterWeight = 0.227027F;
inline constexpr float blurNearWeight = 0.316216F;
inline constexpr float blurFarWeight = 0.070270F;
inline constexpr float blurNearOffset = 1.384615F;
inline constexpr float blurFarOffset = 3.230769F;

enum class TestPattern : std::uint32_t {
    none = 0,
    black = 1,
    constant = 2,
    impulse = 3,
    square = 4,
    horizontalLine = 5,
    verticalLine = 6,
    multipleSources = 7,
    checkerboard = 8,
    movingSquare = 9,
};

struct HdrImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<glm::vec3> pixels;

    HdrImage() = default;
    HdrImage(std::uint32_t imageWidth, std::uint32_t imageHeight,
             glm::vec3 clearValue = glm::vec3{0.0F});

    [[nodiscard]] bool valid() const;
    [[nodiscard]] const glm::vec3& at(std::uint32_t x, std::uint32_t y) const;
    [[nodiscard]] glm::vec3& at(std::uint32_t x, std::uint32_t y);
};

struct ImageMetrics {
    bool finite = true;
    float maximumLuminance = 0.0F;
    double meanLuminance = 0.0;
    double totalLuminance = 0.0;
    glm::dvec2 luminanceCentroid{0.0};
};

struct BloomLevelMemory {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t filteredBytes = 0;
    std::uint64_t scratchBytes = 0;

    [[nodiscard]] std::uint64_t totalBytes() const { return filteredBytes + scratchBytes; }
};

struct ResourceMemory {
    std::uint64_t hdrBytes = 0;
    std::uint64_t motionBytes = 0;
    std::uint64_t normalBytes = 0;
    std::uint64_t depthBytes = 0;
    std::vector<BloomLevelMemory> bloomLevels;
    std::uint64_t temporalHistoryBytes = 0;

    [[nodiscard]] std::uint64_t bloomBytes() const;
    [[nodiscard]] std::uint64_t totalBytes() const;
};

[[nodiscard]] std::uint32_t halfExtent(std::uint32_t extent);
[[nodiscard]] float luminance(const glm::vec3& color);
[[nodiscard]] glm::vec3 bloomExtract(const glm::vec3& color);
[[nodiscard]] HdrImage makeTestPattern(std::uint32_t width, std::uint32_t height,
                                       TestPattern pattern, float sourceValue,
                                       float centerX = 0.5F);
[[nodiscard]] glm::vec3 sampleLinearClamp(const HdrImage& image, glm::vec2 uv);
[[nodiscard]] HdrImage downsampleBloom(const HdrImage& source);
[[nodiscard]] HdrImage blurHorizontal(const HdrImage& source);
[[nodiscard]] HdrImage blurVertical(const HdrImage& source);
[[nodiscard]] float exposureMultiplier(float exposureEv);
[[nodiscard]] glm::vec3 applyExposure(const glm::vec3& color, float exposureEv);
[[nodiscard]] glm::vec3 acesFitted(const glm::vec3& color);
[[nodiscard]] glm::vec3 compositeToneMap(const glm::vec3& hdr, const glm::vec3& bloom,
                                         float exposureEv);
[[nodiscard]] ImageMetrics measure(const HdrImage& image);
[[nodiscard]] ResourceMemory resourceMemory(std::uint32_t width, std::uint32_t height,
                                             bool bloomEnabled, bool taaEnabled);

} // namespace renderer::post
