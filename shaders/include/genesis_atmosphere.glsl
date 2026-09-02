#ifndef GENESIS_ATMOSPHERE_GLSL
#define GENESIS_ATMOSPHERE_GLSL

#include "include/bruneton_atmosphere.glsl"
#include "include/genesis_clouds.glsl"

// Bruneton's reference demo is Z-up. Genesis is right-handed Y-up, so this
// fixed basis change lets the renderer's camera rays address the unmodified
// reference lookup parameterization.
vec3 genesisToBrunetonDirection(vec3 direction) {
    return vec3(direction.x, -direction.z, direction.y);
}

float genesisSolarDiskCoverage(vec3 viewDirection, vec3 sunDirection) {
    float angularDistance = acos(clamp(dot(viewDirection, sunDirection), -1.0, 1.0));
    // Primary path-traced atmosphere is evaluated below output resolution.
    // Feathering by roughly half of an internal pixel keeps the physical disk
    // size while avoiding the square two-to-four-pixel silhouette seen after
    // reconstruction/upscaling.
    const float angularFilterRadius = 0.0020;
    return 1.0 - smoothstep(
        ATMOSPHERE.sun_angular_radius - angularFilterRadius,
        ATMOSPHERE.sun_angular_radius + angularFilterRadius,
        angularDistance);
}

vec3 genesisAtmosphereRadianceInternal(vec3 genesisDirection,
                                       vec3 cameraFromEarthCenterKm,
                                       vec3 sunDirection,
                                       bool includeSolarDisk) {
    const vec3 viewDirection = normalize(genesisToBrunetonDirection(genesisDirection));
    const vec3 sun = normalize(sunDirection);

    vec3 transmittance;
    vec3 radiance = GetSkyRadiance(cameraFromEarthCenterKm, viewDirection, 0.0, sun,
                                   transmittance);

    // The reference demo shades the spherical planet separately from the sky
    // lookup. Preserve that behavior so presets 8 and 9 show the orbital limb.
    const float cameraAlongRay = dot(cameraFromEarthCenterKm, viewDirection);
    const float discriminant = cameraAlongRay * cameraAlongRay -
        dot(cameraFromEarthCenterKm, cameraFromEarthCenterKm) +
        ATMOSPHERE.bottom_radius * ATMOSPHERE.bottom_radius;
    bool groundHit = false;
    if (discriminant >= 0.0) {
        const float distanceToGround = -cameraAlongRay - sqrt(discriminant);
        if (distanceToGround > 0.0) {
            groundHit = true;
            const vec3 point = cameraFromEarthCenterKm + viewDirection * distanceToGround;
            const vec3 normal = normalize(point);
            vec3 skyIrradiance;
            const vec3 sunIrradiance =
                GetSunAndSkyIrradiance(point, normal, sun, skyIrradiance);
            const vec3 referenceGroundAlbedo = vec3(0.0, 0.0, 0.04);
            vec3 groundRadiance = referenceGroundAlbedo * (1.0 / PI) *
                                  (sunIrradiance + skyIrradiance);
            vec3 segmentTransmittance;
            const vec3 inScattering = GetSkyRadianceToPoint(
                cameraFromEarthCenterKm, point, 0.0, sun, segmentTransmittance);
            const vec3 groundSceneRadiance =
                groundRadiance * segmentTransmittance + inScattering;

            // Analytically soften the planet limb on its ground-facing side.
            // A binary sphere hit left a stair-stepped, ruler-straight seam at
            // the exact horizon, especially at the lower RR render extent.
            float cameraRadius = length(cameraFromEarthCenterKm);
            vec3 observerUp = cameraFromEarthCenterKm / cameraRadius;
            float horizonMu = -sqrt(max(
                1.0 - (ATMOSPHERE.bottom_radius * ATMOSPHERE.bottom_radius) /
                          (cameraRadius * cameraRadius),
                0.0));
            float angularFilter = 0.00125;
            float cosineFilter = angularFilter *
                                 sqrt(max(1.0 - horizonMu * horizonMu, 1.0e-4));
            float groundCoverage = smoothstep(
                0.0, max(cosineFilter, 1.0e-5),
                horizonMu - dot(observerUp, viewDirection));
            radiance = mix(radiance, groundSceneRadiance, groundCoverage);
        }
    }

    if (includeSolarDisk) {
        radiance += transmittance * GetSolarRadiance() *
                    genesisSolarDiskCoverage(viewDirection, sun);
    }

    // The published Earth LUT is parameterized only down to a solar elevation
    // of -12 degrees (ATMOSPHERE.mu_s_min). Sampling it below that limit clamps
    // to permanent nautical twilight, which produced Genesis' uniform blue
    // "night". Fade that final supported twilight state through astronomical
    // twilight (-18 degrees), then retain only a dim, view-dependent airglow.
    const float astronomicalTwilightSunMu = -0.30901699; // sin(-18 degrees)
    const float nightFloorBlendStartSunMu = -0.08715574; // sin(-5 degrees)
    vec3 observerUp = normalize(cameraFromEarthCenterKm);
    float localSunMu = dot(observerUp, sun);
    float lookupVisibility = smoothstep(
        astronomicalTwilightSunMu, ATMOSPHERE.mu_s_min, localSunMu);
    float nightFloorAmount = 1.0 - smoothstep(
        ATMOSPHERE.mu_s_min, nightFloorBlendStartSunMu, localSunMu);
    radiance *= lookupVisibility;
    if (nightFloorAmount > 0.0) {
        float viewElevationMu = clamp(dot(observerUp, viewDirection), 0.0, 1.0);
        float horizonAirMass = mix(
            0.55, 1.45, pow(1.0 - viewElevationMu, 3.0));
        // Keep the floor strongest while the LUT is fading, then settle to the
        // approved deep-night level. This prevents a black luminance dip in
        // nautical twilight when looking away from the sunset.
        float twilightFloorBoost = mix(1.0, 2.5, lookupVisibility);
        vec3 airglow = vec3(0.00028, 0.00046, 0.00092) *
                       horizonAirMass * twilightFloorBoost;
        vec3 groundNightFloor = vec3(0.00006, 0.00008, 0.00013);
        radiance += mix(airglow, groundNightFloor, groundHit ? 1.0 : 0.0) *
                    nightFloorAmount;
    }

    // The reference demo's baseline exposure is 10. Genesis tone maps later,
    // so preserve that baseline in scene-linear HDR and express +/- as EV.
    return max(radiance * 10.0, vec3(0.0));
}

vec3 genesisAtmosphereRadiance(vec3 genesisDirection,
                               vec3 cameraFromEarthCenterKm,
                               vec3 sunDirection, bool includeSolarDisk) {
    return genesisAtmosphereRadianceInternal(
        genesisDirection, cameraFromEarthCenterKm, sunDirection,
        includeSolarDisk);
}

// Return the view direction immediately above the observer's geometric
// horizon. Ground fog uses this to inherit the same clouded sky color as the
// visible background instead of converging toward a separate constant color.
vec3 genesisGroundFogSourceDirection(vec3 genesisDirection,
                                     vec3 cameraFromEarthCenterKm) {
    vec3 viewDirection = normalize(genesisToBrunetonDirection(genesisDirection));
    vec3 observerUp = normalize(cameraFromEarthCenterKm);
    float viewMu = dot(observerUp, viewDirection);
    vec3 tangentDirection = viewDirection - observerUp * viewMu;
    float tangentLength = length(tangentDirection);
    if (tangentLength <= 1.0e-6) {
        vec3 tangentAxis = abs(observerUp.z) < 0.99
                               ? vec3(0.0, 0.0, 1.0)
                               : vec3(1.0, 0.0, 0.0);
        tangentDirection = normalize(cross(tangentAxis, observerUp));
    } else {
        tangentDirection /= tangentLength;
    }
    float cameraRadius = length(cameraFromEarthCenterKm);
    float horizonMu = -sqrt(max(
        1.0 - (ATMOSPHERE.bottom_radius * ATMOSPHERE.bottom_radius) /
                  max(cameraRadius * cameraRadius, 1.0e-6),
        0.0));
    float sourceMu = clamp(max(viewMu, horizonMu + 0.00075), -0.9999, 0.9999);
    vec3 sourceDirection = normalize(
        tangentDirection * sqrt(max(1.0 - sourceMu * sourceMu, 0.0)) +
        observerUp * sourceMu);
    // Bruneton Z-up -> Genesis Y-up.
    return normalize(vec3(sourceDirection.x, sourceDirection.z,
                          -sourceDirection.y));
}

// Apply the same Bruneton atmosphere used by the sky to a finite primary-ray
// surface. This is aerial perspective rather than a screen-space blur: distant
// radiance is extinguished along the actual camera segment and replaced by the
// matching in-scattered sky light. The dense low-altitude Mie profile naturally
// produces the ground haze that softens a terrestrial horizon.
vec3 genesisApplyAerialPerspective(vec3 surfaceRadiance,
                                   vec3 genesisDirection,
                                   float hitDistanceMeters,
                                   vec3 cameraFromEarthCenterKm,
                                   vec3 sunDirection,
                                   vec3 groundFogSourceRadiance) {
    if (hitDistanceMeters <= 0.0) {
        return max(surfaceRadiance, vec3(0.0));
    }

    vec3 viewDirection = normalize(genesisToBrunetonDirection(genesisDirection));
    vec3 point = cameraFromEarthCenterKm +
                 viewDirection * (hitDistanceMeters * 0.001);
    // The demo ground is a local tangent plane and can sit a few metres below
    // the analytic planet radius near the origin. Keep the lookup endpoint in
    // the valid atmosphere domain without changing the rendered geometry.
    float pointRadius = length(point);
    float minimumRadius = ATMOSPHERE.bottom_radius + 0.001;
    if (pointRadius < minimumRadius && pointRadius > 1.0e-6) {
        point *= minimumRadius / pointRadius;
    }

    vec3 transmittance;
    vec3 inScattering = GetSkyRadianceToPoint(
        cameraFromEarthCenterKm, point, 0.0, normalize(sunDirection),
        transmittance);
    if (any(isnan(transmittance)) || any(isinf(transmittance)) ||
        any(isnan(inScattering)) || any(isinf(inScattering))) {
        return max(surfaceRadiance, vec3(0.0));
    }

    // genesisAtmosphereRadianceInternal uses the reference demo's 10x
    // scene-linear radiance scale; keep finite and infinite rays in one energy
    // convention so the horizon converges toward the sky instead of a seam.
    vec3 aerialRadiance = max(
        surfaceRadiance * clamp(transmittance, vec3(0.0), vec3(1.0)) +
            max(inScattering, vec3(0.0)) * 10.0,
        vec3(0.0));

    // Add a modest weather-scale aerosol layer close to the ground. The clear
    // Bruneton atmosphere intentionally models excellent visibility and can
    // leave the finite demo plane visible at its 50 km boundary. Integrating an
    // exponential layer along the real curved segment makes that boundary
    // disappear while leaving nearby objects essentially untouched.
    const float groundFogScaleHeightKm = 0.25;
    const float groundFogExtinctionPerKm = 0.18;
    const int groundFogSamples = 4;
    float distanceKm = hitDistanceMeters * 0.001;
    float integratedDensity = 0.0;
    for (int sampleIndex = 0; sampleIndex < groundFogSamples; ++sampleIndex) {
        float sampleT = (float(sampleIndex) + 0.5) / float(groundFogSamples);
        vec3 samplePoint = cameraFromEarthCenterKm +
                           viewDirection * (distanceKm * sampleT);
        float sampleHeightKm = max(
            length(samplePoint) - ATMOSPHERE.bottom_radius, 0.0);
        integratedDensity += exp(-sampleHeightKm / groundFogScaleHeightKm);
    }
    float fogOpticalDepth = groundFogExtinctionPerKm * distanceKm *
                            integratedDensity / float(groundFogSamples);
    float groundFogAmount = 1.0 - exp(-max(fogOpticalDepth, 0.0));
    if (groundFogAmount <= 1.0e-4) {
        return aerialRadiance;
    }

    if (any(isnan(groundFogSourceRadiance)) ||
        any(isinf(groundFogSourceRadiance))) {
        return aerialRadiance;
    }
    return mix(aerialRadiance, max(groundFogSourceRadiance, vec3(0.0)),
               clamp(groundFogAmount, 0.0, 1.0));
}

bool genesisCloudSphereRoots(vec3 origin, vec3 direction, float radius,
                             out float nearDistance, out float farDistance) {
    float b = dot(origin, direction);
    float c = dot(origin, origin) - radius * radius;
    float discriminant = b * b - c;
    if (discriminant < 0.0) {
        nearDistance = 0.0;
        farDistance = 0.0;
        return false;
    }
    float root = sqrt(discriminant);
    nearDistance = -b - root;
    farDistance = -b + root;
    return farDistance > 0.0;
}

// Returns the first visible section of a spherical cloud shell. The same
// function handles a surface observer, a camera inside the layer, and the
// orbital Bruneton demo views without flattening the horizon.
bool genesisCloudRayInterval(vec3 origin, vec3 direction, float innerRadius,
                             float outerRadius, out float startDistance,
                             out float endDistance) {
    float outerNear;
    float outerFar;
    if (!genesisCloudSphereRoots(origin, direction, outerRadius,
                                 outerNear, outerFar)) {
        return false;
    }

    float innerNear;
    float innerFar;
    bool intersectsInner = genesisCloudSphereRoots(
        origin, direction, innerRadius, innerNear, innerFar);
    float cameraRadius = length(origin);
    if (cameraRadius < innerRadius) {
        if (!intersectsInner) {
            return false;
        }
        startDistance = max(innerFar, 0.0);
        endDistance = outerFar;
    } else if (cameraRadius <= outerRadius) {
        startDistance = 0.0;
        endDistance = outerFar;
        if (intersectsInner && innerNear > 0.0) {
            endDistance = min(endDistance, innerNear);
        }
    } else {
        startDistance = max(outerNear, 0.0);
        endDistance = outerFar;
        if (intersectsInner && innerNear > startDistance) {
            endDistance = min(endDistance, innerNear);
        }
    }

    // The planet is opaque. Without this test a surface observer looking down
    // enters the cloud shell again on the far side of Earth, which makes the
    // cloud deck appear below the camera and leaves the zenith incorrectly
    // clear. Clip an interval that starts before the ground and reject one that
    // is entirely hidden behind it.
    float groundNear;
    float groundFar;
    if (genesisCloudSphereRoots(origin, direction, ATMOSPHERE.bottom_radius,
                                groundNear, groundFar)) {
        float groundDistance = groundNear > 0.0 ? groundNear : groundFar;
        if (groundDistance > 0.0) {
            if (groundDistance <= startDistance + 1.0e-4) {
                return false;
            }
            endDistance = min(endDistance, groundDistance);
        }
    }
    return endDistance > startDistance + 1.0e-4;
}

float genesisCloudPhase(float cosineTheta, float anisotropy) {
    float g2 = anisotropy * anisotropy;
    return (1.0 - g2) /
           max(4.0 * PI * pow(1.0 + g2 - 2.0 * anisotropy * cosineTheta, 1.5),
               1.0e-4);
}

float genesisCloudLightTransmittance(vec3 point, vec3 sunDirection,
                                     float density, vec4 cloudLayer) {
    vec3 localUp = normalize(point);
    float shellThickness = cloudLayer.y - cloudLayer.x;
    float sunElevation = max(dot(localUp, sunDirection), 0.12);
    float normalizedHeight = clamp(
        (length(point) - cloudLayer.x) / max(shellThickness, 1.0e-3), 0.0, 1.0);
    float distanceToTop = shellThickness * (1.0 - normalizedHeight) / sunElevation;
    // A local density multiplied by the entire slant distance turns every
    // density contour into a different opaque shell.  At low Sun elevations
    // that error is magnified into the horizontal/radial ribs visible in the
    // beauty pass even though the opacity pass is smooth.  A cloud normally
    // exits laterally before traversing the full spherical layer, and multiple
    // scattering supplies a non-zero lighting floor.  Bound the representative
    // path and vary extinction only mildly with local density.
    float representativePath = min(distanceToTop, 7.0);
    float extinction = mix(0.22, 0.28,
                           smoothstep(0.08, 0.95, density));
    float directTransmittance = exp(-extinction * representativePath * 0.55);
    return mix(0.08, 1.0, directTransmittance);
}

// Low-frequency cloud shadow sampled from a point in the lower atmosphere.
// Marching these projected shadows along the view ray creates crepuscular
// rays that remain registered to the actual procedural cloud openings.
float genesisCloudShaftSunVisibility(vec3 atmospherePoint, vec3 sunDirection,
                                     vec4 cloudLayer, vec4 cloudWeather) {
    float shadowRadius = mix(cloudLayer.x, cloudLayer.y, 0.48);
    float nearDistance;
    float farDistance;
    if (!genesisCloudSphereRoots(atmospherePoint, sunDirection, shadowRadius,
                                 nearDistance, farDistance)) {
        return 1.0;
    }
    float shadowDistance = nearDistance > 0.0 ? nearDistance : farDistance;
    if (shadowDistance <= 0.0) {
        return 1.0;
    }
    vec3 shadowPoint = atmospherePoint + sunDirection * shadowDistance;
    // Preserve enough of the weather-map footprint to distinguish adjacent
    // openings without importing the high-frequency 3D erosion field into
    // the shaft march (which would shimmer as the camera moves).
    GenesisCloudColumn column = genesisCloudColumn(
        shadowPoint.xy, cloudWeather, 0.32);
    float threshold = genesisCloudCoverageThreshold(cloudLayer.z);
    // Project a soft, low-frequency optical column.  A nearly binary column
    // makes neighboring integration samples flip between full Sun and full
    // shadow, which turns the shaft march into visible bright/dark bands.
    float opacity = smoothstep(threshold - 0.140, threshold + 0.160,
                               column.field);
    float sunElevation = max(dot(normalize(shadowPoint), sunDirection), 0.10);
    float slantDepth = (cloudLayer.y - cloudLayer.x) / sunElevation;
    return clamp(exp(-opacity * max(cloudLayer.w, 0.0) * slantDepth * 0.58),
                 0.0, 1.0);
}

// Filtered angular occlusion used only by the additive aerial-light pass.
// Visible cloud density still comes from the full three-dimensional field;
// this lower-frequency column prevents its high-frequency erosion from being
// stretched into noisy screen-space spokes.
float genesisCloudShaftDirectionVisibility(vec3 cameraFromEarthCenterKm,
                                            vec3 direction,
                                            vec4 cloudLayer,
                                            vec4 cloudWeather) {
    float startDistance;
    float endDistance;
    if (!genesisCloudRayInterval(cameraFromEarthCenterKm, direction,
                                 cloudLayer.x, cloudLayer.y,
                                 startDistance, endDistance)) {
        return 1.0;
    }
    endDistance = min(endDistance, startDistance + 72.0);
    float span = endDistance - startDistance;
    if (span <= 0.0) {
        return 1.0;
    }
    // A handful of filtered density samples preserves real 3D openings in the
    // shaft occluder. Sampling only the 2D weather column reduced every cloud
    // near the Sun to one enormous sector and could not produce distinct rays.
    const int occlusionSteps = 5;
    float opticalDepth = 0.0;
    float stepLength = span / float(occlusionSteps);
    for (int stepIndex = 0; stepIndex < occlusionSteps; ++stepIndex) {
        float distance = startDistance +
                         (float(stepIndex) + 0.5) * stepLength;
        float density = genesisCloudDensity(
            cameraFromEarthCenterKm + direction * distance,
            cloudLayer, cloudWeather, 0.34, 0.0);
        opticalDepth += density * stepLength * 1.35;
    }
    return clamp(exp(-opticalDepth), 0.0, 1.0);
}

vec3 genesisCloudShaftRadianceCorrection(vec3 cameraFromEarthCenterKm,
                                         vec3 viewDirection,
                                         vec3 sunDirection,
                                         float distanceToCloud,
                                         vec4 cloudLayer,
                                         vec4 cloudWeather,
                                         vec3 clearRadiance,
                                         float cloudTransmittance,
                                         float rayJitter) {
    vec3 observerUp = normalize(cameraFromEarthCenterKm);
    float sunElevation = dot(observerUp, sunDirection);
    float viewElevation = dot(observerUp, viewDirection);
    if (sunElevation <= 0.01 || viewElevation < -0.04 ||
        length(cameraFromEarthCenterKm) >= cloudLayer.x) {
        return vec3(0.0);
    }

    float horizonWeight = 1.0 - smoothstep(0.02, 0.42, viewElevation);
    float alignment = clamp(dot(viewDirection, sunDirection), -1.0, 1.0);
    float angularSupport = smoothstep(0.72, 0.997, alignment);
    if (angularSupport <= 1.0e-4 || distanceToCloud <= 0.05) {
        return vec3(0.0);
    }

    // Reconstruct a filtered angular source from the real 3D cloud
    // transmittance in two tight rings around the Sun. Extending that source
    // along the current radial tangent creates a crepuscular ray, while the
    // azimuth filter avoids the hard pinwheel sectors of a one-sample ring.
    vec3 radialVector = viewDirection - sunDirection * alignment;
    float radialLength = length(radialVector);
    vec3 fallbackTangent = cross(sunDirection, observerUp);
    if (length(fallbackTangent) < 1.0e-4) {
        fallbackTangent = cross(sunDirection, vec3(0.0, 1.0, 0.0));
    }
    vec3 radialTangent = radialLength > 1.0e-4
                             ? radialVector / radialLength
                             : normalize(fallbackTangent);
    vec3 azimuthPerpendicular = normalize(cross(sunDirection,
                                                radialTangent));
    float weightedVisibility = 0.0;
    float weightedVisibilitySquared = 0.0;
    float visibilityWeight = 0.0;
    for (int radiusIndex = 0; radiusIndex < 2; ++radiusIndex) {
        float probeAngle = radiusIndex == 0 ? 0.024 : 0.055;
        float radiusWeight = radiusIndex == 0 ? 1.0 : 0.68;
        for (int azimuthIndex = 0; azimuthIndex < 5; ++azimuthIndex) {
            float signedIndex = float(azimuthIndex - 2);
            float azimuthOffset = signedIndex * 0.14;
            vec3 filteredTangent = normalize(
                radialTangent * cos(azimuthOffset) +
                azimuthPerpendicular * sin(azimuthOffset));
            vec3 sampleDirection = normalize(
                sunDirection * cos(probeAngle) +
                filteredTangent * sin(probeAngle));
            float sampleVisibility = genesisCloudShaftDirectionVisibility(
                cameraFromEarthCenterKm, sampleDirection,
                cloudLayer, cloudWeather);
            float azimuthWeight = azimuthIndex == 2
                                      ? 1.0
                                      : (abs(signedIndex) < 1.5 ? 0.72 : 0.34);
            float sampleWeight = radiusWeight * azimuthWeight;
            weightedVisibility += sampleVisibility * sampleWeight;
            weightedVisibilitySquared += sampleVisibility * sampleVisibility *
                                         sampleWeight;
            visibilityWeight += sampleWeight;
        }
    }
    float illuminatedAir = weightedVisibility / max(visibilityWeight, 1.0e-5);
    float meanSquared = weightedVisibilitySquared /
                        max(visibilityWeight, 1.0e-5);
    float pathVariation = sqrt(max(meanSquared -
                                   illuminatedAir * illuminatedAir, 0.0));
    float forwardFocus = pow(max(alignment, 0.0), 5.0);
    float brokenWeather = smoothstep(0.08, 0.34, cloudLayer.z) *
                          (1.0 - smoothstep(0.82, 0.99, cloudLayer.z));
    float angularDistance = acos(alignment);
    float transitionContrast = smoothstep(0.028, 0.34, pathVariation);
    float sunCoreFade = smoothstep(0.020, 0.050, angularDistance);
    float radialEnvelope = 1.0 - smoothstep(0.10, 0.34, angularDistance);
    float beamFraction = pow(clamp(illuminatedAir, 0.0, 1.0), 1.35) *
                         transitionContrast *
                         sunCoreFade *
                         radialEnvelope *
                         brokenWeather *
                         mix(0.42, 1.12, horizonWeight) *
                         mix(0.62, 1.0, forwardFocus) *
                         pow(angularSupport, 0.72);
    float cameraRadius = length(cameraFromEarthCenterKm);
    vec3 sunTransmittance = GetTransmittanceToSun(
        ATMOSPHERE, transmittance_texture, cameraRadius,
        dot(observerUp, sunDirection));
    vec3 shaftSource = clearRadiance * 0.24 +
                       ATMOSPHERE.solar_irradiance * sunTransmittance * 0.050;
    // Restrict the additive scattering to visible clear air so the correction
    // cannot darken or alter the silhouette of the cloud body itself.
    return shaftSource * min(beamFraction * 1.55, 0.85) *
           clamp(cloudTransmittance, 0.0, 1.0);
}

// Composites single-scattered procedural clouds over the exact Bruneton sky.
// Radii and distances are kilometers, matching the published lookup tables.
vec3 genesisAtmosphereCloudRadiance(vec3 genesisDirection,
                                    vec3 cameraFromEarthCenterKm,
                                    vec3 sunDirection, vec4 cloudLayer,
                                    vec4 cloudWeather, bool includeSolarDisk,
                                    float rayJitter,
                                    out float cloudTransmittance) {
    vec3 clearRadiance = genesisAtmosphereRadianceInternal(
        genesisDirection, cameraFromEarthCenterKm, sunDirection,
        includeSolarDisk);
    cloudTransmittance = 1.0;
    if (cloudWeather.w < 0.5 || cloudLayer.w <= 0.0 || cloudLayer.z <= 0.0) {
        return clearRadiance;
    }

    vec3 viewDirection = normalize(genesisToBrunetonDirection(genesisDirection));
    vec3 sun = normalize(sunDirection);
    float startDistance;
    float endDistance;
    if (!genesisCloudRayInterval(cameraFromEarthCenterKm, viewDirection,
                                 cloudLayer.x, cloudLayer.y,
                                 startDistance, endDistance)) {
        return clearRadiance;
    }

    // Surface views do not benefit from marching hundreds of kilometers of
    // near-horizon cloud shell. Bounding that interval keeps each stratum
    // smaller than the density features; orbital presets retain the full shell.
    if (length(cameraFromEarthCenterKm) < cloudLayer.y + 50.0) {
        endDistance = min(endDistance, startDistance + 96.0);
    }

    float grazingBlend = 1.0 - smoothstep(
        0.15, 0.70,
        abs(dot(normalize(cameraFromEarthCenterKm), viewDirection)));
    // Local kilometre-scale detail is appropriate near and inside the cloud
    // layer.  Fade to the spherical macro-weather field while climbing away
    // from it so an orbital view does not expose the repeating noise volume.
    float altitudeAboveClouds = length(cameraFromEarthCenterKm) - cloudLayer.y;
    float planetaryCloudBlend = smoothstep(80.0, 520.0,
                                           altitudeAboveClouds);
    // A near-normal orbital ray only crosses a few kilometres of cloud. A
    // 620-metre cell left as few as nine samples through the entire layer and
    // exposed optical-depth contours as nested rings. Keep normal-incidence
    // cells below a quarter kilometre while retaining the denser grazing path.
    float targetCellLength = mix(0.22, 0.12, grazingBlend);
    float intervalLength = endDistance - startDistance;
    // Anchor cells in physical distance. Normalizing every cell by an integer
    // per-pixel step count shifts the entire sample lattice whenever ceil()
    // changes by one, which appears as nested horizontal cloud scallops.
    float physicalCellLength = max(targetCellLength,
                                   intervalLength / 768.0);
    int viewSteps = int(clamp(ceil(intervalLength / physicalCellLength),
                               1.0, 768.0));
    vec3 accumulated = vec3(0.0);
    float transmittance = 1.0;
    float opacityWeightedDistance = 0.0;
    float opacityWeight = 0.0;
    float phase = mix(genesisCloudPhase(dot(viewDirection, sun), -0.22),
                      genesisCloudPhase(dot(viewDirection, sun), 0.68), 0.82) *
                  (4.0 * PI);
    phase = min(phase, 1.8);
    float opticalSurfaceDistance = -1.0;
    // Two-point Gauss-Legendre quadrature integrates every world-space cell.
    for (int stepIndex = 0; stepIndex < 768; ++stepIndex) {
        if (stepIndex >= viewSteps) {
            break;
        }
        float cellStart = startDistance +
                          float(stepIndex) * physicalCellLength;
        float cellEnd = min(startDistance +
                            float(stepIndex + 1) * physicalCellLength,
                            endDistance);
        float stepLength = cellEnd - cellStart;
        const float innerOffset = 0.2113248654;
        float subCellShift = (rayJitter - 0.5) * 0.34;
        float nodeA = clamp(innerOffset + subCellShift, 0.035, 0.465);
        float nodeB = clamp(1.0 - innerOffset + subCellShift, 0.535, 0.965);
        float distanceA = mix(cellStart, cellEnd, nodeA);
        float distanceB = mix(cellStart, cellEnd, nodeB);
        float detailWeightA = 1.0 - smoothstep(28.0, 110.0, distanceA);
        float detailWeightB = 1.0 - smoothstep(28.0, 110.0, distanceB);
        float densityA = genesisCloudDensity(
            cameraFromEarthCenterKm + viewDirection * distanceA,
            cloudLayer, cloudWeather, detailWeightA,
            planetaryCloudBlend);
        float densityB = genesisCloudDensity(
            cameraFromEarthCenterKm + viewDirection * distanceB,
            cloudLayer, cloudWeather, detailWeightB,
            planetaryCloudBlend);
        float density = 0.5 * (densityA + densityB);
        float distance = 0.5 * (distanceA + distanceB);
        vec3 point = cameraFromEarthCenterKm + viewDirection * distance;
        if (density > 1.0e-3) {
            vec3 localUp = normalize(point);
            float muSun = dot(localUp, sun);
            vec3 atmosphereToSun = GetTransmittanceToSun(
                ATMOSPHERE, transmittance_texture, length(point), muSun);
            float selfShadow = genesisCloudLightTransmittance(
                point, sun, density, cloudLayer);
            float localSunVisibility = smoothstep(-0.02, 0.03, muSun);
            float cloudHeight = clamp(
                (length(point) - cloudLayer.x) /
                    max(cloudLayer.y - cloudLayer.x, 1.0e-3),
                0.0, 1.0);
            float powder = 1.0 - exp(-density * stepLength * 1.6);
            float forwardLining = pow(max(dot(viewDirection, sun), 0.0), 6.0) *
                                  (1.0 - powder);
            float overcastBlend = smoothstep(
                0.72, 1.0, clamp(cloudLayer.z, 0.0, 1.0));
            // Bruneton sky radiance is mapped into the renderer's display
            // scale by a factor of ten. Keep cloud single scattering in the
            // same range; the previous unscaled source allowed blue
            // foreground air to visually replace even an opaque cloud.
            vec3 directSource = ATMOSPHERE.solar_irradiance * atmosphereToSun *
                                selfShadow * phase * 0.32 * localSunVisibility;
            // With the Sun high in the sky, a surface observer sees the
            // optically shaded underside rather than the illuminated cloud
            // top.  Applying this to the direct lobe (not just the ambient
            // term) prevents dense overhead systems from collapsing into a
            // uniformly bright sheet.  At low solar elevations, side light
            // remains available for warm sunrise and sunset silhouettes.
            float cloudTopLighting = mix(
                0.28, 1.0, smoothstep(0.08, 0.76, cloudHeight));
            float overheadSunWeight = smoothstep(0.08, 0.52, muSun);
            directSource *= mix(1.0, cloudTopLighting, overheadSunWeight);
            // Low-density rims scatter more of the direct beam toward the
            // viewer while compact interiors stay shaded. This uses the
            // density already evaluated for the view march, avoiding another
            // expensive procedural sample or a noisy secondary light march.
            float densityLighting = mix(
                1.08, 0.94, smoothstep(0.10, 0.88, density));
            directSource *= densityLighting;
            directSource *= mix(1.0, 1.25, forwardLining);
            directSource *= mix(1.0, 0.08, overcastBlend);
            // From orbit the resolved shell samples are illuminated cloud
            // tops. Restore their high albedo after the local underside/core
            // shading terms so macro weather reads as cloud, not gray land.
            directSource *= mix(1.0, 1.65, planetaryCloudBlend);
            float daylightAmbient = smoothstep(-0.24, -0.02, muSun);
            vec3 ambientFloor = mix(vec3(0.00018, 0.00028, 0.00055),
                                    vec3(0.008, 0.011, 0.018),
                                    daylightAmbient);
            float clearLuminance = dot(
                clearRadiance, vec3(0.2126, 0.7152, 0.0722));
            vec3 diffuseSky = mix(vec3(clearLuminance), clearRadiance, 0.38);
            vec3 ambientSource = diffuseSky * 0.10 +
                                 ATMOSPHERE.solar_irradiance * atmosphereToSun *
                                     (0.032 * daylightAmbient) +
                                 ambientFloor * 2.0;
            // Cloud bases and dense cores receive less diffuse sky light.
            // This restores readable volume without extra shadow-ray noise or
            // the cost of another procedural march toward the Sun.
            float baseOcclusion = mix(
                0.84, 1.0, smoothstep(0.06, 0.58, cloudHeight));
            float coreOcclusion = mix(
                1.0, 0.86, smoothstep(0.18, 0.92, density));
            ambientSource *= baseOcclusion * coreOcclusion;
            ambientSource *= mix(1.0, 1.22, planetaryCloudBlend);
            // In a closed deck the emergent radiance should converge to a
            // stable diffuse source. If this retains per-sample height and
            // density factors, the first opaque march cell becomes visible as
            // horizontal contour bands. Broken clouds keep the richer source;
            // only the deep-overcast endpoint approaches this neutral value.
            vec3 overcastAmbient = diffuseSky * 0.015 +
                                   mix(vec3(0.0015, 0.0017, 0.0022),
                                       vec3(0.255, 0.260, 0.270),
                                       daylightAmbient);
            ambientSource = mix(ambientSource, overcastAmbient,
                                overcastBlend);
            // Density is an optical-depth control, not merely a shape scale.
            // This extinction range keeps fair-weather wisps transmissive but
            // lets high-density overcast become genuinely opaque over only a
            // few kilometers instead of retaining a blue-sky wash.
            const float cloudExtinction = 1.55;
            float stepOpticalDepth = density * stepLength * cloudExtinction;
            float stepTransmittance = exp(-stepOpticalDepth);
            float scatteredFraction = 1.0 - stepTransmittance;
            float scatterWeight = transmittance * scatteredFraction;
            accumulated += (directSource + ambientSource) * scatterWeight;
            // Place each cell's scattering at the analytic centroid of its
            // homogeneous optical segment. A midpoint jumps by one full cell
            // as an opaque surface crosses a march boundary, turning smooth
            // cloud growth into visible contour bands.
            float scatterCentroid = 0.5;
            if (stepOpticalDepth > 1.0e-3) {
                scatterCentroid = clamp(
                    1.0 / stepOpticalDepth -
                        1.0 / max(exp(stepOpticalDepth) - 1.0, 1.0e-4),
                    0.0, 1.0);
            }
            float scatterDistance = mix(cellStart, cellEnd, scatterCentroid);
            opacityWeightedDistance += scatterDistance * scatterWeight;
            opacityWeight += scatterWeight;
            float previousTransmittance = transmittance;
            transmittance = previousTransmittance * stepTransmittance;
            if (opticalSurfaceDistance < 0.0 && transmittance < 0.78) {
                // Solve the threshold crossing inside the cell instead of
                // snapping the optical surface to its midpoint. The latter
                // exposed the 220 m view-march spacing as nested rings from
                // orbit and as horizontal steps inside the cloud layer.
                float crossingOpticalDepth =
                    log(max(previousTransmittance, 0.78) / 0.78);
                float crossingFraction = clamp(
                    crossingOpticalDepth / max(stepOpticalDepth, 1.0e-5),
                    0.0, 1.0);
                opticalSurfaceDistance = mix(
                    cellStart, cellEnd, crossingFraction);
            }
            if (transmittance < 0.008) {
                transmittance = 0.0;
                break;
            }
        }
    }
    cloudTransmittance = clamp(transmittance, 0.0, 1.0);
    vec3 cloudComposite = accumulated + clearRadiance * cloudTransmittance;
    float shaftOccluderDistance = endDistance;
    if (opacityWeight > 1.0e-4) {
        // Split the clear atmosphere at the opacity-weighted cloud distance.
        // Air between the observer and the cloud remains visible, while cloud
        // scattering and the background behind it receive the correct
        // view-path extinction. This is one Bruneton segment lookup per sky
        // pixel rather than one lookup per march step.
        float representativeDistance =
            clamp(opacityWeightedDistance / opacityWeight,
                  startDistance, endDistance);
        shaftOccluderDistance = representativeDistance;
        vec3 representativePoint = cameraFromEarthCenterKm +
                                   viewDirection * representativeDistance;
        vec3 viewTransmittance;
        vec3 foregroundAir = GetSkyRadianceToPoint(
            cameraFromEarthCenterKm, representativePoint, 0.0, sun,
            viewTransmittance) * 10.0;
        vec3 backgroundClear = max(clearRadiance - foregroundAir, vec3(0.0));
        // The volume integral owns opacity. Evaluate one short Sun march at
        // the first optical surface and reuse it for the whole composite. This
        // is far cheaper and more stable than a shadow march at every view
        // sample, yet it follows real 3D openings instead of exposing nested
        // local-density contours as lit shells.
        float surfaceDistance = opticalSurfaceDistance > 0.0
                                    ? opticalSurfaceDistance
                                    : representativeDistance;
        vec3 surfacePoint = cameraFromEarthCenterKm +
                            viewDirection * surfaceDistance;
        float surfaceDetailWeight =
            1.0 - smoothstep(28.0, 110.0, surfaceDistance);
        float sunMarchStep = mix(1.25, 0.72, surfaceDetailWeight);
        float sunOpticalDepth = 0.0;
        for (int lightStep = 0; lightStep < 6; ++lightStep) {
            float lightDistance = (float(lightStep) + 0.65) * sunMarchStep;
            float lightDensity = genesisCloudDensity(
                surfacePoint + sun * lightDistance,
                cloudLayer, cloudWeather,
                min(surfaceDetailWeight, 0.62), planetaryCloudBlend);
            sunOpticalDepth += lightDensity * sunMarchStep;
        }
        float surfaceSunVisibility = exp(-sunOpticalDepth * 0.62);
        float surfaceLighting = mix(
            0.70, 1.12, smoothstep(0.05, 0.95,
                sqrt(clamp(surfaceSunVisibility, 0.0, 1.0))));
        // Preserve broad weather structure in a closed deck without making
        // the march cells visible again. This post-integration modulation is
        // horizontal and filtered, so it adds storm-scale tonal variation but
        // cannot reopen blue-sky holes in an optically opaque volume.
        float overcastCompositeBlend = smoothstep(
            0.72, 1.0, clamp(cloudLayer.z, 0.0, 1.0));
        GenesisCloudColumn overcastColumn = genesisCloudColumnForView(
            representativePoint, cloudWeather, 0.0,
            planetaryCloudBlend);
        float overcastTexture = mix(
            0.58, 1.24,
            smoothstep(0.22, 0.78, overcastColumn.field));
        float fairWeatherTone = mix(
            0.94, 1.05,
            smoothstep(0.24, 0.76, overcastColumn.field));
        float fairSurfaceModulation = mix(
            mix(1.0, fairWeatherTone, 0.35),
            surfaceLighting * fairWeatherTone,
            0.35 * (1.0 - overcastCompositeBlend));
        vec3 cloudMedia = viewTransmittance * accumulated *
                          fairSurfaceModulation *
                          mix(1.0, overcastTexture,
                              overcastCompositeBlend);
        cloudComposite = foregroundAir + cloudMedia +
                         backgroundClear * cloudTransmittance;
    }
    vec3 shaftCorrection = genesisCloudShaftRadianceCorrection(
        cameraFromEarthCenterKm, viewDirection, sun, shaftOccluderDistance,
        cloudLayer, cloudWeather, clearRadiance, cloudTransmittance,
        rayJitter);
    if (cloudWeather.w > 2.5) {
        const vec3 debugLuminanceWeights = vec3(0.2126, 0.7152, 0.0722);
        float clearLuminance = max(dot(clearRadiance, debugLuminanceWeights),
                                   1.0e-4);
        float shaftIllumination = dot(max(shaftCorrection, vec3(0.0)),
                                      debugLuminanceWeights) /
                                  clearLuminance;
        // White marks the strongest additive aerial illumination. Cloud bodies
        // remain black because the production correction is masked there.
        return vec3(clamp(shaftIllumination / 1.5, 0.0, 1.0));
    }
    cloudComposite += shaftCorrection;
    cloudComposite = max(cloudComposite, vec3(0.0));
    return cloudComposite;
}

// The dynamic environment cache deliberately excludes the solar disk so it is
// not double counted by secondary path rays. Primary sky rays add it back with
// the cached cloud transmittance, allowing clouds to cover the Sun.
vec3 genesisAtmosphereSolarDiskRadiance(vec3 genesisDirection,
                                        vec3 cameraFromEarthCenterKm,
                                        vec3 sunDirection,
                                        vec4 cloudLayer,
                                        vec4 cloudWeather) {
    vec3 viewDirection = normalize(genesisToBrunetonDirection(genesisDirection));
    vec3 sun = normalize(sunDirection);
    float angularDistance = acos(clamp(dot(viewDirection, sun), -1.0, 1.0));
    float diskCoverage = genesisSolarDiskCoverage(viewDirection, sun);
    // A solar disk is several orders of magnitude brighter than the sky. A
    // finite-support screen-space blur turns that tiny HDR source into a
    // clipped square before tone mapping. Model the camera/eye aureole in
    // angular space instead: two radial Gaussian lobes give the disk a round
    // core and a faint broad halo while preserving the exact physical disk.
    // Three angular lobes preserve HDR energy around the solar disk before
    // bloom extraction. This produces a round, eye-searing source without
    // asking the generic bloom pass to overexpose every bright cloud.
    float coreGlare = exp(-0.5 * pow(angularDistance / 0.015, 2.0)) * 1.8e-4;
    float wideGlare = exp(-0.5 * pow(angularDistance / 0.035, 2.0)) * 1.4e-6;
    float veilGlare = exp(-0.5 * pow(angularDistance / 0.080, 2.0)) * 1.5e-8;
    float glareCoverage = coreGlare + wideGlare + veilGlare;
    if (diskCoverage <= 0.0 && glareCoverage < 1.0e-10) {
        return vec3(0.0);
    }
    vec3 atmosphereTransmittance;
    GetSkyRadiance(cameraFromEarthCenterKm, viewDirection, 0.0, sun,
                   atmosphereTransmittance);
    // Use one stable, low-frequency optical column for the tiny solar source.
    // Reusing the view-ray march's high-frequency transmittance made each
    // distance stratum modulate a different part of the aureole, revealing
    // concentric rings as the Sun passed behind particular clouds.
    float solarCloudVisibility = 1.0;
    if (cloudWeather.w > 0.5 && cloudLayer.z > 0.0 && cloudLayer.w > 0.0) {
        solarCloudVisibility = genesisCloudShaftSunVisibility(
            cameraFromEarthCenterKm, sun, cloudLayer, cloudWeather);
    }
    return atmosphereTransmittance * GetSolarRadiance() *
           solarCloudVisibility *
           (diskCoverage + glareCoverage) * 10.0;
}

#endif // GENESIS_ATMOSPHERE_GLSL
