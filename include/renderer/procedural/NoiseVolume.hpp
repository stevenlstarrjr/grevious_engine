#pragma once

#include <cstdint>
#include <vector>

namespace renderer::procedural {

// One RGBA8 level of a tileable three-dimensional procedural texture. Keeping
// the generated data renderer-agnostic lets the same assets feed clouds,
// terrain, fog, materials, and simulation systems.
struct NoiseVolumeLevel {
    std::uint32_t extent = 0U;
    std::vector<std::uint8_t> texels;

    [[nodiscard]] bool valid() const;
};

struct NoiseVolume {
    std::vector<NoiseVolumeLevel> levels;

    [[nodiscard]] bool valid() const;
};

struct CloudNoiseVolumes {
    NoiseVolume base;
    NoiseVolume detail;
};

// Generates deterministic, seamlessly tileable Perlin-Worley volumes.
// Base R is a broad gradient FBM field and GBA are successively finer cellular
// fields. Detail RGB contains cellular erosion octaves and A contains FBM.
[[nodiscard]] CloudNoiseVolumes makeCloudNoiseVolumes(
    std::uint32_t baseExtent = 128U,
    std::uint32_t detailExtent = 32U,
    std::uint32_t seed = 0x47454e45U);

} // namespace renderer::procedural
