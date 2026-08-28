#include "renderer/post/PostProcessing.hpp"

#include <glm/common.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {

using renderer::post::HdrImage;
using renderer::post::TestPattern;

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "post-processing test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

bool near(float actual, float expected, float tolerance = 1.0e-5F) {
    return std::abs(actual - expected) <= tolerance;
}

double rowEnergy(const HdrImage& image, std::uint32_t row) {
    double result = 0.0;
    for (std::uint32_t x = 0; x < image.width; ++x) {
        result += renderer::post::luminance(image.at(x, row));
    }
    return result;
}

double columnEnergy(const HdrImage& image, std::uint32_t column) {
    double result = 0.0;
    for (std::uint32_t y = 0; y < image.height; ++y) {
        result += renderer::post::luminance(image.at(column, y));
    }
    return result;
}

void testThresholdAndBlack() {
    using namespace renderer::post;
    require(bloomExtract(glm::vec3{0.0F}) == glm::vec3{0.0F}, "black extraction is non-zero");
    require(bloomExtract(glm::vec3{bloomThreshold - bloomKnee - 1.0e-4F}) == glm::vec3{0.0F},
            "value below soft-knee support blooms");
    const float atThreshold = bloomExtract(glm::vec3{bloomThreshold}).r;
    require(near(atThreshold, bloomKnee * 0.25F, 2.0e-5F),
            "soft-knee response at threshold changed");
    require(bloomExtract(glm::vec3{bloomThreshold - 1.0e-3F}).r < atThreshold,
            "soft-knee response below threshold is not increasing");
    require(bloomExtract(glm::vec3{bloomThreshold + 1.0e-3F}).r > atThreshold,
            "threshold response is not increasing");

    HdrImage black = makeTestPattern(17, 11, TestPattern::black, 65504.0F);
    const HdrImage downsampled = downsampleBloom(black);
    const HdrImage horizontal = blurHorizontal(downsampled);
    const HdrImage vertical = blurVertical(horizontal);
    require(measure(downsampled).totalLuminance == 0.0, "black downsample leaked energy");
    require(measure(horizontal).totalLuminance == 0.0, "black horizontal blur leaked energy");
    require(measure(vertical).totalLuminance == 0.0, "black vertical blur leaked energy");
    require(compositeToneMap(glm::vec3{0.0F}, glm::vec3{0.0F}, 0.0F) == glm::vec3{0.0F},
            "black tone-map result is non-zero");
}

void testDarkLeakageAndEnergy() {
    using namespace renderer::post;
    const HdrImage dark = makeTestPattern(19, 13, TestPattern::constant, 0.5F);
    const ImageMetrics darkMetrics = measure(blurVertical(blurHorizontal(downsampleBloom(dark))));
    require(darkMetrics.totalLuminance == 0.0 && darkMetrics.maximumLuminance == 0.0F,
            "sub-knee dark field leaks bloom");

    double previousEnergy = -1.0;
    for (const float value : std::array{1.0F, 2.0F, 4.0F, 8.0F, 16.0F, 65504.0F}) {
        const HdrImage source = makeTestPattern(33, 25, TestPattern::square, value);
        const ImageMetrics metrics = measure(blurVertical(blurHorizontal(downsampleBloom(source))));
        require(metrics.finite, "finite HDR source generated non-finite bloom");
        require(metrics.totalLuminance >= previousEnergy, "bloom energy is non-monotonic");
        previousEnergy = metrics.totalLuminance;
    }
}

void testIndependentBlurAxesAndSymmetry() {
    using namespace renderer::post;
    HdrImage impulse(31, 31);
    impulse.at(15, 15) = glm::vec3{16.0F};
    const HdrImage horizontal = blurHorizontal(impulse);
    for (std::uint32_t y = 0; y < horizontal.height; ++y) {
        if (y != 15U) {
            require(rowEnergy(horizontal, y) == 0.0, "horizontal blur spread vertically");
        }
    }
    require(rowEnergy(horizontal, 15) > 15.9 && rowEnergy(horizontal, 15) < 16.1,
            "horizontal kernel is not normalized");

    const HdrImage verticalOnly = blurVertical(impulse);
    for (std::uint32_t x = 0; x < verticalOnly.width; ++x) {
        if (x != 15U) {
            require(columnEnergy(verticalOnly, x) == 0.0, "vertical blur spread horizontally");
        }
    }
    require(columnEnergy(verticalOnly, 15) > 15.9 && columnEnergy(verticalOnly, 15) < 16.1,
            "vertical kernel is not normalized");

    const HdrImage combined = blurVertical(horizontal);
    const ImageMetrics metrics = measure(combined);
    require(metrics.finite, "combined impulse kernel is non-finite");
    require(std::abs(metrics.luminanceCentroid.x - 15.0) < 1.0e-5 &&
                std::abs(metrics.luminanceCentroid.y - 15.0) < 1.0e-5,
            "combined impulse kernel is not centered");
    for (std::uint32_t offset = 0; offset <= 10; ++offset) {
        require(near(combined.at(15U - offset, 15).r, combined.at(15U + offset, 15).r),
                "combined kernel has horizontal bias");
        require(near(combined.at(15, 15U - offset).r, combined.at(15, 15U + offset).r),
                "combined kernel has vertical bias");
        require(near(combined.at(15U + offset, 15).r, combined.at(15, 15U + offset).r),
                "combined kernel axes are asymmetric");
    }
    require(combined.at(15, 15).r > combined.at(20, 15).r,
            "combined impulse kernel does not decay");

    HdrImage brighter(31, 31);
    brighter.at(15, 15) = glm::vec3{64.0F};
    const HdrImage brighterCombined = blurVertical(blurHorizontal(brighter));
    for (std::uint32_t offset = 0; offset <= 10; ++offset) {
        require(near(combined.at(15U + offset, 15).r / 16.0F,
                     brighterCombined.at(15U + offset, 15).r / 64.0F, 1.0e-6F),
                "blur radius or normalized shape changed with source brightness");
    }
}

void testDownsampleAndEdges() {
    using namespace renderer::post;
    require(halfExtent(1) == 1 && halfExtent(2) == 1 && halfExtent(3) == 2 &&
                halfExtent(1279) == 640,
            "ceil-half extent convention changed");
    for (const auto [width, height] : std::array<std::pair<std::uint32_t, std::uint32_t>, 4>{
             {{16, 12}, {17, 13}, {1, 1}, {1279, 719}}}) {
        const HdrImage uniform = makeTestPattern(width, height, TestPattern::constant, 4.0F);
        const HdrImage result = downsampleBloom(uniform);
        require(result.width == halfExtent(width) && result.height == halfExtent(height),
                "downsample dimensions are wrong");
        const glm::vec3 expected = bloomExtract(glm::vec3{4.0F});
        for (const glm::vec3 color : result.pixels) {
            require(glm::all(glm::lessThanEqual(glm::abs(color - expected), glm::vec3{1.0e-5F})),
                    "uniform downsample does not preserve filtered value");
        }
    }

    for (const TestPattern pattern : {TestPattern::impulse, TestPattern::square,
                                      TestPattern::checkerboard, TestPattern::horizontalLine,
                                      TestPattern::verticalLine, TestPattern::multipleSources}) {
        const ImageMetrics metrics = measure(downsampleBloom(makeTestPattern(17, 13, pattern, 32.0F)));
        require(metrics.finite && metrics.totalLuminance >= 0.0,
                "controlled downsample pattern is invalid");
    }

    HdrImage twoByTwo(17, 13);
    twoByTwo.at(8, 6) = glm::vec3{32.0F};
    twoByTwo.at(9, 6) = glm::vec3{32.0F};
    twoByTwo.at(8, 7) = glm::vec3{32.0F};
    twoByTwo.at(9, 7) = glm::vec3{32.0F};
    require(measure(downsampleBloom(twoByTwo)).totalLuminance > 0.0,
            "2x2 source was lost during downsampling");

    HdrImage clampProbe(3, 3);
    clampProbe.at(0, 0) = glm::vec3{7.0F};
    clampProbe.at(2, 2) = glm::vec3{11.0F};
    require(near(sampleLinearClamp(clampProbe, {-1.0F, -1.0F}).r, 7.0F) &&
                near(sampleLinearClamp(clampProbe, {2.0F, 2.0F}).r, 11.0F),
            "post sampler does not clamp at image edges");

    for (const glm::uvec2 position : {glm::uvec2{0, 0}, glm::uvec2{30, 0}, glm::uvec2{0, 30},
                                      glm::uvec2{30, 30}, glm::uvec2{0, 15},
                                      glm::uvec2{30, 15}, glm::uvec2{15, 0},
                                      glm::uvec2{15, 30}, glm::uvec2{15, 15}}) {
        HdrImage edge(31, 31);
        edge.at(position.x, position.y) = glm::vec3{16.0F};
        const ImageMetrics metrics = measure(blurVertical(blurHorizontal(edge)));
        require(metrics.finite && metrics.totalLuminance > 0.0,
                "edge impulse was lost or became non-finite");
    }
}

void testExposureAndToneMapping() {
    using namespace renderer::post;
    require(near(exposureMultiplier(1.0F), 2.0F), "EV +1 is not 2x");
    require(near(exposureMultiplier(-1.0F), 0.5F), "EV -1 is not 0.5x");
    require(near(applyExposure(glm::vec3{2.0F}, -1.0F).r, 1.0F),
            "linear exposure application is wrong");

    glm::vec3 previous{-1.0F};
    for (const float value :
         std::array{0.0F, 1.0e-6F, 0.01F, 0.18F, 1.0F, 4.0F, 64.0F, 65504.0F}) {
        const glm::vec3 mapped = acesFitted(glm::vec3{value});
        require(std::isfinite(mapped.r) && mapped.r >= 0.0F && mapped.r <= 1.0F,
                "ACES fit produced invalid output");
        require(mapped.r >= previous.r, "ACES fit is non-monotonic");
        previous = mapped;
    }
    require(acesFitted(glm::vec3{0.0F}) == glm::vec3{0.0F}, "ACES black is not black");
    const glm::vec3 color = acesFitted(glm::vec3{0.25F, 1.0F, 4.0F});
    require(color.r < color.g && color.g < color.b, "ACES fit unexpectedly inverted color order");
    for (const float ev : {-4.0F, -2.0F, 0.0F, 2.0F, 4.0F}) {
        const glm::vec3 result = compositeToneMap(glm::vec3{4.0F}, glm::vec3{2.0F}, ev);
        require(std::isfinite(result.r), "exposure/tone-map chain is non-finite");
    }
}

void testResourceAccounting() {
    using namespace renderer::post;
    const ResourceMemory even = resourceMemory(1280, 720, true, true);
    require(even.bloomLevels.size() == 1 &&
                even.bloomLevels.front().filteredBytes == 640ULL * 360ULL * 8ULL &&
                even.bloomLevels.front().scratchBytes == 640ULL * 360ULL * 8ULL,
            "even bloom memory accounting is wrong");
    require(even.temporalHistoryBytes == 1280ULL * 720ULL * 40ULL,
            "temporal memory accounting is wrong");
    const ResourceMemory odd = resourceMemory(1279, 719, true, false);
    require(odd.bloomLevels.size() == 1 && odd.bloomLevels.front().width == 640 &&
                odd.bloomLevels.front().height == 360,
            "odd bloom memory accounting is wrong");
    require(resourceMemory(1280, 720, false, true).bloomLevels.empty(),
            "disabled bloom still owns transient memory");
    require(even.totalBytes() > even.temporalHistoryBytes, "total memory accounting is incomplete");
}

} // namespace

int main() {
    testThresholdAndBlack();
    testDarkLeakageAndEnergy();
    testIndependentBlurAxesAndSymmetry();
    testDownsampleAndEdges();
    testExposureAndToneMapping();
    testResourceAccounting();
    std::cout << "post-processing reference tests passed\n";
    return EXIT_SUCCESS;
}
