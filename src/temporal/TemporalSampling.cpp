#include "renderer/temporal/TemporalSampling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace renderer::temporal {

float radicalInverse(std::uint32_t base, std::uint64_t index) {
    if (base < 2U) {
        throw std::invalid_argument("Radical inverse requires a base of at least two");
    }

    // Derived from pbrt-v4 RadicalInverse(), Copyright (c) 1998-2020 Matt
    // Pharr, Wenzel Jakob, and Greg Humphreys; Apache-2.0. The renderer accepts
    // the prime base directly and uses only bases two and three for TAA. See
    // PBRT_PROVENANCE.md for the exact source revision.
    const std::uint64_t limit = std::numeric_limits<std::uint64_t>::max() / base - base;
    const double inverseBase = 1.0 / static_cast<double>(base);
    double inverseBaseM = 1.0;
    std::uint64_t reversedDigits = 0;
    while (index != 0 && reversedDigits < limit) {
        const std::uint64_t next = index / base;
        const std::uint64_t digit = index - next * base;
        reversedDigits = reversedDigits * base + digit;
        inverseBaseM *= inverseBase;
        index = next;
    }
    const double value = static_cast<double>(reversedDigits) * inverseBaseM;
    return static_cast<float>(std::min(value, std::nextafter(1.0, 0.0)));
}

glm::vec2 taaJitterPixels(std::uint64_t frameIndex) {
    glm::vec2 center(0.0F);
    for (std::uint32_t index = 1; index <= taaJitterPeriod; ++index) {
        center += glm::vec2(radicalInverse(2U, index), radicalInverse(3U, index));
    }
    center /= static_cast<float>(taaJitterPeriod);

    const std::uint64_t sequenceIndex = frameIndex % taaJitterPeriod + 1U;
    return glm::vec2(radicalInverse(2U, sequenceIndex), radicalInverse(3U, sequenceIndex)) - center;
}

} // namespace renderer::temporal
