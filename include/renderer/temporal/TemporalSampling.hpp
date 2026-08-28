#pragma once

#include <glm/vec2.hpp>

#include <cstdint>

namespace renderer::temporal {

inline constexpr std::uint32_t taaJitterPeriod = 8;

// PBRT-style radical inverse for a caller-supplied prime base. The renderer
// needs only bases two and three for its compact TAA camera sequence.
[[nodiscard]] float radicalInverse(std::uint32_t base, std::uint64_t index);

// Returns a zero-mean subpixel offset in approximately [-0.5, 0.5)^2. The
// finite cycle is centered explicitly so a static camera does not drift.
[[nodiscard]] glm::vec2 taaJitterPixels(std::uint64_t frameIndex);

} // namespace renderer::temporal
