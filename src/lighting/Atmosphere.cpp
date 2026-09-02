/*
 * This implementation is based on the physical model and Earth parameters in
 * Eric Bruneton's "Precomputed Atmospheric Scattering: a New Implementation".
 * Copyright (c) 2017 Eric Bruneton. All rights reserved.
 * Distributed under the BSD 3-Clause license; see
 * thirdparty/BrunetonAtmosphere/LICENSE.txt.
 *
 * Genesis uses a Vulkan-independent CPU precomputation here. A 2D optical-depth
 * LUT is evaluated once, then reused while integrating the camera-level sky
 * into the renderer's native latitude-longitude HDR environment representation.
 */

#include "renderer/lighting/Atmosphere.hpp"

#include <glm/common.hpp>
#include <glm/exponential.hpp>
#include <glm/geometric.hpp>
#include <glm/vector_relational.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace renderer::lighting {
namespace {

constexpr float pi = std::numbers::pi_v<float>;
constexpr float twoPi = 2.0F * pi;
constexpr std::uint32_t transmittanceWidth = 256U;
constexpr std::uint32_t transmittanceHeight = 64U;
constexpr std::uint32_t transmittanceSteps = 48U;
constexpr std::uint32_t skySteps = 64U;

struct DensitySample {
    float rayleigh = 0.0F;
    float mie = 0.0F;
    float ozone = 0.0F;
};

struct TransmittanceLut {
    std::vector<glm::vec3> texels;
};

glm::vec3 uvToDirection(float u, float v) {
    const float phi = (u - 0.5F) * twoPi;
    const float theta = v * pi;
    const float sinTheta = std::sin(theta);
    return {sinTheta * std::cos(phi), std::cos(theta), sinTheta * std::sin(phi)};
}

void setTexel(FloatImage& image, std::uint32_t x, std::uint32_t y,
              const glm::vec3& value) {
    const std::size_t offset =
        (static_cast<std::size_t>(y) * image.width + x) * image.channels;
    image.texels[offset] = value.x;
    image.texels[offset + 1U] = value.y;
    image.texels[offset + 2U] = value.z;
    image.texels[offset + 3U] = 1.0F;
}

void validate(const AtmosphereSettings& settings) {
    const auto finitePositive = [](float value) {
        return std::isfinite(value) && value > 0.0F;
    };
    if (settings.environmentWidth < 2U || settings.environmentHeight < 2U ||
        !finitePositive(settings.bottomRadiusMeters) ||
        !finitePositive(settings.topRadiusMeters) ||
        settings.topRadiusMeters <= settings.bottomRadiusMeters ||
        !finitePositive(settings.rayleighScaleHeightMeters) ||
        !finitePositive(settings.mieScaleHeightMeters) ||
        !std::isfinite(settings.observerAltitudeMeters) ||
        settings.observerAltitudeMeters < 0.0F ||
        settings.observerAltitudeMeters >=
            settings.topRadiusMeters - settings.bottomRadiusMeters ||
        !std::isfinite(settings.miePhaseFunctionG) ||
        std::abs(settings.miePhaseFunctionG) >= 1.0F ||
        !finitePositive(settings.radianceScale) ||
        !std::isfinite(glm::length(settings.sunDirection)) ||
        glm::length(settings.sunDirection) < 1.0e-5F) {
        throw std::runtime_error("Atmosphere settings are invalid or out of range");
    }
    const auto finiteNonNegative = [](const glm::vec3& value) {
        return std::isfinite(value.x) && std::isfinite(value.y) &&
               std::isfinite(value.z) && glm::all(glm::greaterThanEqual(value, glm::vec3(0.0F)));
    };
    if (!finiteNonNegative(settings.rayleighScatteringPerMeter) ||
        !finiteNonNegative(settings.mieScatteringPerMeter) ||
        !finiteNonNegative(settings.mieExtinctionPerMeter) ||
        !finiteNonNegative(settings.ozoneExtinctionPerMeter) ||
        !finiteNonNegative(settings.solarIrradiance) ||
        !finiteNonNegative(settings.groundAlbedo)) {
        throw std::runtime_error("Atmosphere spectra must be finite and non-negative");
    }
    const CloudSettings& clouds = settings.clouds;
    if (clouds.enabled &&
        (!std::isfinite(clouds.coverage) || clouds.coverage < 0.0F ||
         clouds.coverage > 1.0F || !finitePositive(clouds.density) ||
         !std::isfinite(clouds.baseAltitudeMeters) ||
         clouds.baseAltitudeMeters < 0.0F ||
         !finitePositive(clouds.thicknessMeters) ||
         clouds.baseAltitudeMeters + clouds.thicknessMeters >=
             settings.topRadiusMeters - settings.bottomRadiusMeters ||
         !finitePositive(clouds.horizontalScaleMeters) ||
         !std::isfinite(clouds.windSpeedMetersPerSecond) ||
         clouds.windSpeedMetersPerSecond < 0.0F ||
         !std::isfinite(glm::length(glm::vec2(clouds.windDirection.x,
                                               clouds.windDirection.z))) ||
         glm::length(glm::vec2(clouds.windDirection.x,
                               clouds.windDirection.z)) < 1.0e-5F)) {
        throw std::runtime_error("Atmosphere cloud settings are invalid or out of range");
    }
}

DensitySample densities(float altitude, const AtmosphereSettings& settings) {
    altitude = std::max(0.0F, altitude);
    const float ozone = altitude < 10'000.0F || altitude > 40'000.0F
                            ? 0.0F
                            : altitude < 25'000.0F
                                  ? (altitude - 10'000.0F) / 15'000.0F
                                  : (40'000.0F - altitude) / 15'000.0F;
    return {
        .rayleigh = std::exp(-altitude / settings.rayleighScaleHeightMeters),
        .mie = std::exp(-altitude / settings.mieScaleHeightMeters),
        .ozone = settings.ozoneEnabled ? glm::clamp(ozone, 0.0F, 1.0F) : 0.0F,
    };
}

float distanceToSphere(float radius, float cosine, float sphereRadius) {
    const float discriminant =
        radius * radius * (cosine * cosine - 1.0F) + sphereRadius * sphereRadius;
    if (discriminant < 0.0F) {
        return -1.0F;
    }
    return -radius * cosine + std::sqrt(discriminant);
}

float distanceToGround(float radius, float cosine,
                       const AtmosphereSettings& settings) {
    const float discriminant = radius * radius * (cosine * cosine - 1.0F) +
                               settings.bottomRadiusMeters *
                                   settings.bottomRadiusMeters;
    return discriminant >= 0.0F
               ? -radius * cosine - std::sqrt(discriminant)
               : -1.0F;
}

bool rayIntersectsGround(float radius, float cosine,
                         const AtmosphereSettings& settings) {
    if (cosine >= 0.0F) {
        return false;
    }
    const float discriminant = radius * radius * (cosine * cosine - 1.0F) +
                               settings.bottomRadiusMeters *
                                   settings.bottomRadiusMeters;
    return discriminant >= 0.0F;
}

glm::vec3 extinction(const DensitySample& density,
                     const AtmosphereSettings& settings) {
    return settings.rayleighScatteringPerMeter * density.rayleigh +
           settings.mieExtinctionPerMeter * density.mie +
           settings.ozoneExtinctionPerMeter * density.ozone;
}

glm::vec3 integrateTransmittance(float radius, float cosine,
                                const AtmosphereSettings& settings) {
    if (rayIntersectsGround(radius, cosine, settings)) {
        return glm::vec3(0.0F);
    }
    const float distance = distanceToSphere(radius, cosine, settings.topRadiusMeters);
    if (!(distance > 0.0F)) {
        return glm::vec3(1.0F);
    }
    const float stepLength = distance / static_cast<float>(transmittanceSteps);
    glm::vec3 opticalDepth(0.0F);
    for (std::uint32_t step = 0; step < transmittanceSteps; ++step) {
        const float sampleDistance = (static_cast<float>(step) + 0.5F) * stepLength;
        const float sampleRadius = std::sqrt(
            sampleDistance * sampleDistance + 2.0F * radius * cosine * sampleDistance +
            radius * radius);
        opticalDepth += extinction(
                            densities(sampleRadius - settings.bottomRadiusMeters, settings),
                            settings) *
                        stepLength;
    }
    return glm::exp(-opticalDepth);
}

TransmittanceLut precomputeTransmittance(const AtmosphereSettings& settings) {
    TransmittanceLut result{
        .texels = std::vector<glm::vec3>(
            static_cast<std::size_t>(transmittanceWidth) * transmittanceHeight),
    };
    const float atmosphereThickness =
        settings.topRadiusMeters - settings.bottomRadiusMeters;
    for (std::uint32_t y = 0; y < transmittanceHeight; ++y) {
        const float altitudeCoordinate =
            static_cast<float>(y) / static_cast<float>(transmittanceHeight - 1U);
        // Concentrate LUT rows near the ground, where density changes fastest.
        const float altitude = atmosphereThickness * altitudeCoordinate * altitudeCoordinate;
        const float radius = settings.bottomRadiusMeters + altitude;
        for (std::uint32_t x = 0; x < transmittanceWidth; ++x) {
            const float cosine =
                -1.0F + 2.0F * static_cast<float>(x) /
                            static_cast<float>(transmittanceWidth - 1U);
            result.texels[static_cast<std::size_t>(y) * transmittanceWidth + x] =
                integrateTransmittance(radius, cosine, settings);
        }
    }
    return result;
}

glm::vec3 sampleTransmittance(const TransmittanceLut& lut, float radius,
                              float cosine, const AtmosphereSettings& settings) {
    if (rayIntersectsGround(radius, cosine, settings)) {
        return glm::vec3(0.0F);
    }
    const float atmosphereThickness =
        settings.topRadiusMeters - settings.bottomRadiusMeters;
    const float altitude = glm::clamp(radius - settings.bottomRadiusMeters, 0.0F,
                                      atmosphereThickness);
    const float x = (glm::clamp(cosine, -1.0F, 1.0F) * 0.5F + 0.5F) *
                    static_cast<float>(transmittanceWidth - 1U);
    const float y = std::sqrt(altitude / atmosphereThickness) *
                    static_cast<float>(transmittanceHeight - 1U);
    const std::uint32_t x0 = static_cast<std::uint32_t>(std::floor(x));
    const std::uint32_t y0 = static_cast<std::uint32_t>(std::floor(y));
    const std::uint32_t x1 = std::min(x0 + 1U, transmittanceWidth - 1U);
    const std::uint32_t y1 = std::min(y0 + 1U, transmittanceHeight - 1U);
    const float tx = x - std::floor(x);
    const float ty = y - std::floor(y);
    const auto texel = [&lut](std::uint32_t sampleX, std::uint32_t sampleY) {
        return lut.texels[static_cast<std::size_t>(sampleY) * transmittanceWidth +
                          sampleX];
    };
    return glm::mix(glm::mix(texel(x0, y0), texel(x1, y0), tx),
                    glm::mix(texel(x0, y1), texel(x1, y1), tx), ty);
}

float rayleighPhase(float cosine) {
    return 3.0F * (1.0F + cosine * cosine) / (16.0F * pi);
}

float miePhase(float cosine, float g) {
    const float denominator = std::max(
        1.0e-4F, 1.0F + g * g - 2.0F * g * cosine);
    return 3.0F * (1.0F - g * g) * (1.0F + cosine * cosine) /
           (8.0F * pi * (2.0F + g * g) *
            denominator * std::sqrt(denominator));
}

glm::vec3 integrateSky(const glm::vec3& direction, const glm::vec3& sunDirection,
                       const TransmittanceLut& transmittance,
                       const AtmosphereSettings& settings) {
    const float observerRadius =
        settings.bottomRadiusMeters + settings.observerAltitudeMeters;
    const glm::vec3 observer(0.0F, observerRadius, 0.0F);
    const float viewCosine = direction.y;
    const bool hitsGround = rayIntersectsGround(observerRadius, viewCosine, settings);
    const float atmosphereDistance =
        distanceToSphere(observerRadius, viewCosine, settings.topRadiusMeters);
    float segmentDistance = atmosphereDistance;
    if (hitsGround) {
        segmentDistance = distanceToGround(observerRadius, viewCosine, settings);
    }
    if (!(segmentDistance > 0.0F)) {
        return glm::vec3(0.0F);
    }

    const float phaseCosine = glm::dot(direction, sunDirection);
    const float phaseRayleigh = rayleighPhase(phaseCosine);
    const float phaseMie = miePhase(phaseCosine, settings.miePhaseFunctionG);
    const float stepLength = segmentDistance / static_cast<float>(skySteps);
    glm::vec3 viewOpticalDepth(0.0F);
    glm::vec3 singleScattering(0.0F);
    for (std::uint32_t step = 0; step < skySteps; ++step) {
        const float sampleDistance = (static_cast<float>(step) + 0.5F) * stepLength;
        const glm::vec3 samplePosition = observer + direction * sampleDistance;
        const float sampleRadius = glm::length(samplePosition);
        const DensitySample density =
            densities(sampleRadius - settings.bottomRadiusMeters, settings);
        const glm::vec3 localUp = samplePosition / sampleRadius;
        const glm::vec3 viewTransmittance = glm::exp(-viewOpticalDepth);
        const glm::vec3 sunTransmittance = sampleTransmittance(
            transmittance, sampleRadius, glm::dot(localUp, sunDirection), settings);
        const glm::vec3 scattering =
            settings.rayleighScatteringPerMeter * (density.rayleigh * phaseRayleigh) +
            settings.mieScatteringPerMeter * (density.mie * phaseMie);
        singleScattering += viewTransmittance * sunTransmittance * scattering *
                            settings.solarIrradiance * stepLength;
        viewOpticalDepth += extinction(density, settings) * stepLength;
    }

    // Higher scattering orders predominantly return energy to long, dense
    // paths near the horizon. Bruneton's full model precomputes four orders;
    // this bounded energy compensation preserves that behavior in Genesis's
    // environment bake without a persistent 3D scattering texture.
    const glm::vec3 segmentTransmittance = glm::exp(-viewOpticalDepth);
    const glm::vec3 multipleScattering = singleScattering *
        glm::clamp((glm::vec3(1.0F) - segmentTransmittance) * 0.42F,
                   glm::vec3(0.0F), glm::vec3(0.42F));
    glm::vec3 radiance = singleScattering + multipleScattering;

    if (hitsGround) {
        const glm::vec3 groundPoint = observer + direction * segmentDistance;
        const glm::vec3 groundNormal = glm::normalize(groundPoint);
        const float sunCosine = std::max(0.0F, glm::dot(groundNormal, sunDirection));
        const glm::vec3 sunTransmittance = sampleTransmittance(
            transmittance, settings.bottomRadiusMeters + 1.0F, sunCosine, settings);
        const glm::vec3 groundIrradiance = settings.solarIrradiance *
            (sunTransmittance * sunCosine + glm::vec3(0.035F));
        radiance += segmentTransmittance * settings.groundAlbedo *
                    groundIrradiance / pi;
    }
    return glm::max(radiance * settings.radianceScale, glm::vec3(0.0F));
}

} // namespace

AtmosphereDemoView atmosphereDemoPreset(std::uint32_t preset) {
    // Exact SetView values from atmosphere/demo/demo.cc in Bruneton's
    // reference implementation.
    static constexpr std::array<AtmosphereDemoView, 9> presets{
        AtmosphereDemoView{9'000.0F, 1.47F, 0.0F, 1.3F, 3.0F, 10.0F},
        AtmosphereDemoView{9'000.0F, 1.47F, 0.0F, 1.564F, -3.0F, 10.0F},
        AtmosphereDemoView{7'000.0F, 1.57F, 0.0F, 1.54F, -2.96F, 10.0F},
        AtmosphereDemoView{7'000.0F, 1.57F, 0.0F, 1.328F, -3.044F, 10.0F},
        AtmosphereDemoView{9'000.0F, 1.39F, 0.0F, 1.2F, 0.7F, 10.0F},
        AtmosphereDemoView{9'000.0F, 1.5F, 0.0F, 1.628F, 1.05F, 200.0F},
        AtmosphereDemoView{7'000.0F, 1.43F, 0.0F, 1.57F, 1.34F, 40.0F},
        AtmosphereDemoView{2.7e6F, 0.81F, 0.0F, 1.57F, 2.0F, 10.0F},
        AtmosphereDemoView{1.2e7F, 0.0F, 0.0F, 0.93F, -2.0F, 10.0F},
    };
    if (preset == 0U || preset > presets.size()) {
        throw std::invalid_argument("Atmosphere demo preset must be in [1, 9]");
    }
    return presets[preset - 1U];
}

glm::vec3 atmosphereDemoCameraFromEarthCenterKm(const AtmosphereDemoView& view,
                                                 float bottomRadiusMeters) {
    if (!std::isfinite(view.viewDistanceMeters) || view.viewDistanceMeters < 0.0F ||
        !std::isfinite(view.viewZenithAngleRadians) ||
        !std::isfinite(view.viewAzimuthAngleRadians) ||
        !std::isfinite(bottomRadiusMeters) || bottomRadiusMeters <= 0.0F) {
        throw std::invalid_argument("Atmosphere demo camera parameters are invalid");
    }
    const float sinZenith = std::sin(view.viewZenithAngleRadians);
    const float distanceKm = view.viewDistanceMeters / 1'000.0F;
    const glm::vec3 cameraFromLocalOriginKm{
        sinZenith * std::cos(view.viewAzimuthAngleRadians) * distanceKm,
        sinZenith * std::sin(view.viewAzimuthAngleRadians) * distanceKm,
        std::cos(view.viewZenithAngleRadians) * distanceKm,
    };
    return cameraFromLocalOriginKm + glm::vec3(0.0F, 0.0F, bottomRadiusMeters / 1'000.0F);
}

glm::vec3 atmosphereDemoSunDirection(const AtmosphereDemoView& view) {
    const float sinZenith = std::sin(view.sunZenithAngleRadians);
    return glm::normalize(glm::vec3{
        std::cos(view.sunAzimuthAngleRadians) * sinZenith,
        std::sin(view.sunAzimuthAngleRadians) * sinZenith,
        std::cos(view.sunZenithAngleRadians),
    });
}

glm::vec3 timeOfDaySunDirection(float localSolarHour,
                                const TimeOfDaySunSettings& settings) {
    if (!std::isfinite(localSolarHour) || !std::isfinite(settings.latitudeDegrees) ||
        settings.latitudeDegrees < -89.9F || settings.latitudeDegrees > 89.9F ||
        settings.dayOfYear == 0U || settings.dayOfYear > 365U) {
        throw std::invalid_argument("Time-of-day Sun parameters are invalid");
    }
    constexpr float degreesToRadians = std::numbers::pi_v<float> / 180.0F;
    const float latitude = settings.latitudeDegrees * degreesToRadians;
    const float dayAngle = 2.0F * std::numbers::pi_v<float> *
                           (static_cast<float>(settings.dayOfYear) - 81.0F) / 365.0F;
    const float declination = 23.44F * degreesToRadians * std::sin(dayAngle);
    const float wrappedHour = localSolarHour - 24.0F * std::floor(localSolarHour / 24.0F);
    const float hourAngle = (wrappedHour - 12.0F) * 15.0F * degreesToRadians;

    // Local tangent frame: +X east, +Y north, +Z zenith. This produces a
    // continuous below-horizon path at night instead of clamping at sunset.
    const float cosDeclination = std::cos(declination);
    const glm::vec3 direction{
        -cosDeclination * std::sin(hourAngle),
        std::sin(declination) * std::cos(latitude) -
            cosDeclination * std::cos(hourAngle) * std::sin(latitude),
        std::sin(declination) * std::sin(latitude) +
            cosDeclination * std::cos(hourAngle) * std::cos(latitude),
    };
    return glm::normalize(direction);
}

float timeOfDayHourFromSunDirection(const glm::vec3& direction,
                                    const TimeOfDaySunSettings& settings) {
    if (!std::isfinite(direction.x) || !std::isfinite(direction.y) ||
        !std::isfinite(direction.z) || glm::dot(direction, direction) < 1.0e-8F ||
        !std::isfinite(settings.latitudeDegrees) ||
        settings.latitudeDegrees < -89.9F || settings.latitudeDegrees > 89.9F ||
        settings.dayOfYear == 0U || settings.dayOfYear > 365U) {
        throw std::invalid_argument("Time-of-day inverse Sun parameters are invalid");
    }
    constexpr float degreesToRadians = std::numbers::pi_v<float> / 180.0F;
    const float latitude = settings.latitudeDegrees * degreesToRadians;
    const float dayAngle = 2.0F * std::numbers::pi_v<float> *
                           (static_cast<float>(settings.dayOfYear) - 81.0F) / 365.0F;
    const float declination = 23.44F * degreesToRadians * std::sin(dayAngle);
    const glm::vec3 sun = glm::normalize(direction);
    const float cosDeclination = std::cos(declination);
    const float sinHourAngle = -sun.x / cosDeclination;
    const float cosHourAngle =
        (sun.z - std::sin(declination) * std::sin(latitude)) /
        (cosDeclination * std::cos(latitude));
    const float hourAngle = std::atan2(sinHourAngle, cosHourAngle);
    float hour = 12.0F + hourAngle * 12.0F / std::numbers::pi_v<float>;
    hour -= 24.0F * std::floor(hour / 24.0F);
    return hour;
}

void applyTimeOfDaySun(AtmosphereDemoView& view, float localSolarHour,
                       const TimeOfDaySunSettings& settings) {
    const glm::vec3 direction = timeOfDaySunDirection(localSolarHour, settings);
    view.sunZenithAngleRadians =
        std::acos(glm::clamp(direction.z, -1.0F, 1.0F));
    view.sunAzimuthAngleRadians = std::atan2(direction.y, direction.x);
}

void advanceTimeOfDaySun(AtmosphereDemoView& view, float elapsedSolarHours,
                         const TimeOfDaySunSettings& settings) {
    if (!std::isfinite(elapsedSolarHours) || !std::isfinite(settings.latitudeDegrees) ||
        settings.latitudeDegrees < -89.9F || settings.latitudeDegrees > 89.9F ||
        settings.dayOfYear == 0U || settings.dayOfYear > 365U) {
        throw std::invalid_argument("Time-of-day Sun advance parameters are invalid");
    }
    constexpr float degreesToRadians = std::numbers::pi_v<float> / 180.0F;
    const float latitude = settings.latitudeDegrees * degreesToRadians;
    const glm::vec3 celestialNorth =
        glm::normalize(glm::vec3(0.0F, std::cos(latitude), std::sin(latitude)));
    const float angle = -elapsedSolarHours * 15.0F * degreesToRadians;
    const float cosine = std::cos(angle);
    const float sine = std::sin(angle);
    const glm::vec3 current = atmosphereDemoSunDirection(view);
    const glm::vec3 advanced = glm::normalize(
        current * cosine + glm::cross(celestialNorth, current) * sine +
        celestialNorth * glm::dot(celestialNorth, current) * (1.0F - cosine));
    view.sunZenithAngleRadians = std::acos(glm::clamp(advanced.z, -1.0F, 1.0F));
    view.sunAzimuthAngleRadians = std::atan2(advanced.y, advanced.x);
}

bool atmosphereHistoryResetRequired(const AtmosphereDemoView& previous,
                                    const AtmosphereDemoView& current) {
    const auto validView = [](const AtmosphereDemoView& view) {
        return std::isfinite(view.viewDistanceMeters) && view.viewDistanceMeters >= 0.0F &&
               std::isfinite(view.viewZenithAngleRadians) &&
               std::isfinite(view.viewAzimuthAngleRadians) &&
               std::isfinite(view.sunZenithAngleRadians) &&
               std::isfinite(view.sunAzimuthAngleRadians) &&
               std::isfinite(view.exposure) && view.exposure > 0.0F;
    };
    if (!validView(previous) || !validView(current)) {
        return true;
    }

    constexpr float degreesToRadians = std::numbers::pi_v<float> / 180.0F;
    constexpr float maximumSunStep = 8.0F * degreesToRadians;
    constexpr float maximumViewDirectionStep = 12.0F * degreesToRadians;
    constexpr float maximumExposureStepEv = 1.0F;
    constexpr float maximumDistanceRatio = 1.5F;

    const glm::vec3 previousSun = atmosphereDemoSunDirection(previous);
    const glm::vec3 currentSun = atmosphereDemoSunDirection(current);
    const float sunAgreement = glm::clamp(glm::dot(previousSun, currentSun), -1.0F, 1.0F);
    if (sunAgreement < std::cos(maximumSunStep)) {
        return true;
    }

    const auto viewDirection = [](const AtmosphereDemoView& view) {
        const float sine = std::sin(view.viewZenithAngleRadians);
        return glm::normalize(glm::vec3{
            sine * std::cos(view.viewAzimuthAngleRadians),
            sine * std::sin(view.viewAzimuthAngleRadians),
            std::cos(view.viewZenithAngleRadians),
        });
    };
    const float viewAgreement = glm::clamp(
        glm::dot(viewDirection(previous), viewDirection(current)), -1.0F, 1.0F);
    if (viewAgreement < std::cos(maximumViewDirectionStep)) {
        return true;
    }

    const float smallerDistance =
        std::max(std::min(previous.viewDistanceMeters, current.viewDistanceMeters), 1.0F);
    const float largerDistance =
        std::max(previous.viewDistanceMeters, current.viewDistanceMeters);
    if (largerDistance / smallerDistance > maximumDistanceRatio) {
        return true;
    }

    return std::abs(std::log2(current.exposure / previous.exposure)) >
           maximumExposureStepEv;
}

bool AtmospherePrecomputedLuts::valid() const {
    return transmittance.size() ==
               static_cast<std::size_t>(brunetonTransmittanceWidth) *
                   brunetonTransmittanceHeight * 4U &&
           scattering.size() == static_cast<std::size_t>(brunetonScatteringWidth) *
                                    brunetonScatteringHeight * brunetonScatteringDepth * 4U &&
           irradiance.size() == static_cast<std::size_t>(brunetonIrradianceWidth) *
                                   brunetonIrradianceHeight * 4U;
}

AtmospherePrecomputedLuts loadAtmospherePrecomputedLuts(
    const std::filesystem::path& directory) {
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error(
            "Bruneton atmosphere lookup tables require little-endian float decoding");
    }
#if defined(GENESIS_BRUNETON_ATMOSPHERE_DIR)
    const std::filesystem::path resolvedDirectory =
        directory.empty() ? std::filesystem::path(GENESIS_BRUNETON_ATMOSPHERE_DIR) : directory;
#else
    const std::filesystem::path resolvedDirectory = directory;
#endif
    if (resolvedDirectory.empty()) {
        throw std::runtime_error("No Bruneton atmosphere lookup-table directory was configured");
    }
    const auto readFloatFile = [&resolvedDirectory](const char* name,
                                                    std::size_t expectedFloatCount) {
        const std::filesystem::path path = resolvedDirectory / name;
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream) {
            throw std::runtime_error("Failed to open atmosphere lookup table: " + path.string());
        }
        const std::streamsize expectedBytes =
            static_cast<std::streamsize>(expectedFloatCount * sizeof(float));
        if (stream.tellg() != expectedBytes) {
            throw std::runtime_error("Atmosphere lookup table has an unexpected size: " +
                                     path.string());
        }
        stream.seekg(0, std::ios::beg);
        std::vector<float> values(expectedFloatCount);
        if (!stream.read(reinterpret_cast<char*>(values.data()), expectedBytes)) {
            throw std::runtime_error("Failed to read atmosphere lookup table: " + path.string());
        }
        return values;
    };
    AtmospherePrecomputedLuts result{
        .transmittance = readFloatFile(
            "transmittance.dat", static_cast<std::size_t>(brunetonTransmittanceWidth) *
                                     brunetonTransmittanceHeight * 4U),
        .scattering = readFloatFile(
            "scattering.dat", static_cast<std::size_t>(brunetonScatteringWidth) *
                                  brunetonScatteringHeight * brunetonScatteringDepth * 4U),
        .irradiance = readFloatFile(
            "irradiance.dat", static_cast<std::size_t>(brunetonIrradianceWidth) *
                                 brunetonIrradianceHeight * 4U),
    };
    if (!result.valid()) {
        throw std::runtime_error("Loaded Bruneton atmosphere lookup tables are invalid");
    }
    return result;
}

FloatImage makePrecomputedAtmosphereEnvironment(
    const AtmosphereSettings& settings) {
    validate(settings);
    const glm::vec3 sunDirection = glm::normalize(settings.sunDirection);
    const TransmittanceLut transmittance = precomputeTransmittance(settings);
    FloatImage result{
        .width = settings.environmentWidth,
        .height = settings.environmentHeight,
        .channels = 4,
        .texels = std::vector<float>(
            static_cast<std::size_t>(settings.environmentWidth) *
            settings.environmentHeight * 4U),
    };
    for (std::uint32_t y = 0; y < settings.environmentHeight; ++y) {
        for (std::uint32_t x = 0; x < settings.environmentWidth; ++x) {
            const glm::vec3 direction = uvToDirection(
                (static_cast<float>(x) + 0.5F) /
                    static_cast<float>(settings.environmentWidth),
                (static_cast<float>(y) + 0.5F) /
                    static_cast<float>(settings.environmentHeight));
            setTexel(result, x, y,
                     integrateSky(direction, sunDirection, transmittance, settings));
        }
    }
    return result;
}

} // namespace renderer::lighting
