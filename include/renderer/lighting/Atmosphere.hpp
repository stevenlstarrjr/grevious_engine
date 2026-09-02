#pragma once

#include "renderer/lighting/IblBaker.hpp"

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <cstdint>
#include <filesystem>
#include <vector>

namespace renderer::lighting {

// Real-time weather-scale clouds use an analytic planetary shell and
// procedural density. This is intentionally renderer-neutral: imported
// NanoVDB or simulated sparse fields can later provide density through the
// same volume boundary without changing atmosphere or presentation ownership.
struct CloudSettings {
    bool enabled = false;
    float coverage = 0.42F;
    float density = 1.0F;
    float baseAltitudeMeters = 1'500.0F;
    // Keep weather fronts at a useful regional scale while allowing the local
    // density field to grow individual cumulus bodies several kilometers tall.
    float thicknessMeters = 5'500.0F;
    float horizontalScaleMeters = 12'000.0F;
    glm::vec3 windDirection{1.0F, 0.0F, 0.25F};
    float windSpeedMetersPerSecond = 18.0F;
};

// Earth-like defaults from Eric Bruneton's 2017 implementation of
// "Precomputed Atmospheric Scattering". Distances and density scale heights
// are expressed in meters; radianceScale maps the physical spectral samples to
// Genesis's scene-linear exposure convention.
struct AtmosphereSettings {
    std::uint32_t environmentWidth = 384;
    std::uint32_t environmentHeight = 192;
    glm::vec3 sunDirection = glm::normalize(glm::vec3(-0.35F, 0.75F, 0.55F));
    float observerAltitudeMeters = 2.0F;
    float bottomRadiusMeters = 6'360'000.0F;
    float topRadiusMeters = 6'420'000.0F;
    float rayleighScaleHeightMeters = 8'000.0F;
    float mieScaleHeightMeters = 1'200.0F;
    glm::vec3 rayleighScatteringPerMeter{5.802e-6F, 13.558e-6F, 33.100e-6F};
    glm::vec3 mieScatteringPerMeter{3.996e-6F};
    glm::vec3 mieExtinctionPerMeter{4.440e-6F};
    glm::vec3 ozoneExtinctionPerMeter{0.650e-6F, 1.881e-6F, 0.085e-6F};
    glm::vec3 solarIrradiance{1.474F, 1.8504F, 1.91198F};
    glm::vec3 groundAlbedo{0.1F};
    float miePhaseFunctionG = 0.8F;
    float radianceScale = 10.0F;
    bool ozoneEnabled = true;
    CloudSettings clouds{};
};

inline constexpr std::uint32_t brunetonTransmittanceWidth = 256U;
inline constexpr std::uint32_t brunetonTransmittanceHeight = 64U;
inline constexpr std::uint32_t brunetonScatteringWidth = 256U;
inline constexpr std::uint32_t brunetonScatteringHeight = 128U;
inline constexpr std::uint32_t brunetonScatteringDepth = 32U;
inline constexpr std::uint32_t brunetonIrradianceWidth = 64U;
inline constexpr std::uint32_t brunetonIrradianceHeight = 16U;

// The six runtime values used by the original C++ and WebGL demos. Keeping
// this as renderer-neutral state lets SDL own input while Qt can drive the
// same atmosphere API later.
struct AtmosphereDemoView {
    float viewDistanceMeters = 9'000.0F;
    float viewZenithAngleRadians = 1.47F;
    float viewAzimuthAngleRadians = -0.1F;
    float sunZenithAngleRadians = 1.3F;
    float sunAzimuthAngleRadians = 2.9F;
    float exposure = 10.0F;
};

// Lightweight astronomical Sun model for interactive day/night playback.
// Day 172 is approximately the June solstice; latitude is geodetic degrees.
struct TimeOfDaySunSettings {
    float latitudeDegrees = 35.0F;
    std::uint32_t dayOfYear = 172U;
};

[[nodiscard]] AtmosphereDemoView atmosphereDemoPreset(std::uint32_t preset);
[[nodiscard]] glm::vec3 atmosphereDemoCameraFromEarthCenterKm(
    const AtmosphereDemoView& view, float bottomRadiusMeters = 6'360'000.0F);
[[nodiscard]] glm::vec3 atmosphereDemoSunDirection(const AtmosphereDemoView& view);
[[nodiscard]] glm::vec3 timeOfDaySunDirection(
    float localSolarHour, const TimeOfDaySunSettings& settings = {});
[[nodiscard]] float timeOfDayHourFromSunDirection(
    const glm::vec3& direction, const TimeOfDaySunSettings& settings = {});
void applyTimeOfDaySun(AtmosphereDemoView& view, float localSolarHour,
                       const TimeOfDaySunSettings& settings = {});
void advanceTimeOfDaySun(AtmosphereDemoView& view, float elapsedSolarHours,
                         const TimeOfDaySunSettings& settings = {});
[[nodiscard]] bool atmosphereHistoryResetRequired(
    const AtmosphereDemoView& previous, const AtmosphereDemoView& current);

struct AtmospherePrecomputedLuts {
    std::vector<float> transmittance;
    std::vector<float> scattering;
    std::vector<float> irradiance;

    [[nodiscard]] bool valid() const;
};

// Loads the official little-endian RGBA32F lookup tables published with the
// reference WebGL demo. An empty directory resolves to Genesis's vendored copy.
[[nodiscard]] AtmospherePrecomputedLuts loadAtmospherePrecomputedLuts(
    const std::filesystem::path& directory = {});

// Precomputes an HDR latitude-longitude environment. The solar disk is
// intentionally excluded: Genesis renders it as a camera-visible source while
// its matching directional light supplies direct illumination and shadows.
// This avoids counting the Sun twice in IBL and path-traced transport.
[[nodiscard]] FloatImage
makePrecomputedAtmosphereEnvironment(const AtmosphereSettings& settings = {});

} // namespace renderer::lighting
