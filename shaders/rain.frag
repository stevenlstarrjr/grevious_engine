#version 460

#include "include/genesis_noise.glsl"
#include "include/weather_wetness.glsl"

layout(set = 0, binding = 0) uniform sampler2D sceneDepth;
layout(set = 0, binding = 1) uniform FrameData
{
    mat4 viewProjection;
    mat4 inverseViewProjection;
    mat4 unjitteredViewProjection;
    mat4 previousViewProjection;
    mat4 previousUnjitteredViewProjection;
    vec4 cameraPositionExposure;
    vec4 directionalDirectionIntensity;
    vec4 directionalColor;
    vec4 pointPositionIntensity;
    vec4 pointColorRange;
    vec4 environmentParameters;
    uvec4 cullingParameters;
    uvec4 visibilityMask;
    vec4 temporalJitter;
    uvec4 temporalFlags;
    vec4 shadowSceneCenterRadius;
    vec4 shadowCascadeCenterRadius1;
    vec4 shadowCascadeCenterRadius2;
    vec4 shadowCascadeSplits;
    vec4 shadowCascadeDepthRadii;
    vec4 previousCameraPosition;
    vec4 atmosphereCamera;
    vec4 atmosphereSunDirection;
    vec4 cloudLayerParameters;
    vec4 cloudWeatherParameters;
    vec4 weatherParameters;
} frameData;
layout(set = 0, binding = 2) uniform sampler2D sceneNormal;

layout(push_constant) uniform RainControls
{
    // intensity, simulation time, terminal speed, physical streak length
    vec4 amountTimeSpeedLength;
    // world-space wind velocity XYZ and camera-local field distance
    vec4 windMaximumDistance;
} rain;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

uint rainHash(uint value)
{
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

float rainRandom(uvec3 cell, uint stream)
{
    uint seed = rainHash(cell.x ^ rainHash(cell.y + 0x9e3779b9u) ^
                         rainHash(cell.z + stream * 0x85ebca6bu));
    return float(seed & 0x00ffffffu) / 16777215.0;
}

vec3 reconstructWorldPosition(vec2 coordinates, float depth)
{
    vec4 world = frameData.inverseViewProjection *
                 vec4(coordinates * 2.0 - 1.0, depth, 1.0);
    return world.xyz / max(abs(world.w), 1.0e-6);
}

vec3 decodeOctahedralNormal(vec2 encoded)
{
    vec2 value = encoded * 2.0 - 1.0;
    vec3 normal = vec3(value, 1.0 - abs(value.x) - abs(value.y));
    float fold = clamp(-normal.z, 0.0, 1.0);
    normal.xy += vec2(normal.x >= 0.0 ? -fold : fold,
                      normal.y >= 0.0 ? -fold : fold);
    return normalize(normal);
}

vec3 sampleWorldNormal(vec2 coordinates)
{
    vec4 packedNormal = texture(sceneNormal, coordinates);
    return frameData.environmentParameters.z > 0.5
        ? decodeOctahedralNormal(packedNormal.xy)
        : normalize(packedNormal.xyz * 2.0 - 1.0);
}

vec2 projectToPixels(vec3 worldPosition, vec2 outputExtent)
{
    vec4 clip = frameData.unjitteredViewProjection * vec4(worldPosition, 1.0);
    vec2 ndc = clip.xy / max(abs(clip.w), 1.0e-6);
    return (ndc * 0.5 + 0.5) * outputExtent;
}

float steamField(vec3 cameraPosition, vec3 viewDirection,
                 float sceneDistance, float groundHeight,
                 float activity, float simulationTime,
                 float steamAgeSeconds)
{
    // Vapor is born over several seconds after precipitation stops. Opacity
    // lags slightly behind spatial growth so a plume begins invisibly rather
    // than popping in as a small opaque shape.
    float sizeGrowth = smoothstep(0.0, 5.0, steamAgeSeconds);
    float opacityGrowth = smoothstep(0.30, 4.5, steamAgeSeconds);
    if (activity <= 0.001 || opacityGrowth <= 0.0001 ||
        abs(viewDirection.y) < 1.0e-5) {
        return 0.0;
    }

    const float steamBottom = 0.025;
    // The demo camera is a little over three metres above the receiver plane.
    // A three-metre layer never crossed the horizon, so it could only read as
    // a faint change in the ground colour.  Real visible post-rain mist forms
    // taller, separated convection plumes.
    const float steamHeight = 2.8;
    float activeSteamHeight = mix(steamBottom, steamHeight, sizeGrowth);
    float bottomDistance =
        (groundHeight + steamBottom - cameraPosition.y) / viewDirection.y;
    float topDistance =
        (groundHeight + activeSteamHeight - cameraPosition.y) / viewDirection.y;
    float entryDistance = max(min(bottomDistance, topDistance), 0.0);
    float exitDistance = min(max(bottomDistance, topDistance), sceneDistance);
    if (exitDistance <= entryDistance) {
        return 0.0;
    }

    // Integrate a continuous world-space vapor field.  Earlier versions used
    // one analytic cylinder per ground cell; their regular silhouettes and
    // first march samples read as vertical tubes and dotted floor decals.
    const int sampleCount = 48;
    // Keep the visible condensation local. Integrating many unrelated source
    // islands all the way to the horizon converges to a uniform fog curtain.
    float segmentLength = min(exitDistance - entryDistance, 23.0);
    float density = 0.0;
    // A stable per-pixel offset breaks up equal-distance ray-march strata.
    // It is intentionally independent of simulation time, so it dithers
    // bands without making stationary vapor sparkle.
    float rayJitter = mix(
        0.12, 0.88, genesisNoiseInterleavedGradient(gl_FragCoord.xy));
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        float sampleFraction =
            (float(sampleIndex) + rayJitter) / float(sampleCount);
        vec3 position = cameraPosition + viewDirection *
            (entryDistance + segmentLength * sampleFraction);
        float normalizedHeight = clamp(
            (position.y - groundHeight) / activeSteamHeight, 0.0, 1.0);
        float heightMeters = position.y - groundHeight;

        // Domain warping bends only the upper vapor.  Its amplitude is zero
        // at the surface, so the wet source pattern remains ground-anchored.
        vec3 warpCoordinate = vec3(
            position.x * 0.17,
            heightMeters * 0.24 - simulationTime * 0.075,
            position.z * 0.17);
        vec2 curl = vec2(
            genesisNoiseSimplex(warpCoordinate + vec3(7.2, 19.1, -3.4)),
            genesisNoiseSimplex(warpCoordinate + vec3(-13.7, 4.8, 21.5))) -
            0.5;
        vec2 sourcePosition = position.xz - curl *
            (1.15 * normalizedHeight * normalizedHeight);

        // Continuous low-frequency islands replace the visible emitter grid.
        // Two smaller octaves erode otherwise broad, uniform source lobes.
        float sourceBroad = genesisNoiseSimplex(
            vec3(sourcePosition * 0.27, 13.6));
        float sourceDetail = genesisNoiseSimplex(
            vec3(sourcePosition * 0.61 + vec2(9.1, -5.7), 31.2));
        float sourceMicro = genesisNoiseSimplex(
            vec3(sourcePosition * 1.34 + vec2(-17.2, 11.6), 47.9));
        // Wide transitions preserve a real opacity gradient instead of
        // turning the source islands into binary cut-out silhouettes.
        float sourceSignal = sourceBroad * 0.68 + sourceDetail * 0.24 +
                             sourceMicro * 0.08;
        // Lowering the isosurface threshold as the vapor matures expands each
        // source island laterally instead of revealing it at its final size.
        float source = smoothstep(mix(0.74, 0.43, sizeGrowth),
                                  mix(0.89, 0.79, sizeGrowth), sourceSignal);

        // Only the vertical coordinate advances with time. The broad scale
        // forms the plume, while two smaller, slower scales carve wisps out of
        // it without introducing fast temporal sparkle.
        vec3 risingCoordinate = vec3(
            sourcePosition.x * 0.46,
            heightMeters * 0.62 - simulationTime * 0.38,
            sourcePosition.y * 0.46);
        float broadVapor = genesisNoiseSimplex(risingCoordinate);
        float fineVapor = genesisNoiseSimplex(
            risingCoordinate * 2.13 + vec3(17.4, -8.2, 5.9));
        float microVapor = genesisNoiseSimplex(vec3(
            sourcePosition.x * 1.57 + 29.3,
            heightMeters * 1.18 - simulationTime * 0.24 - 14.7,
            sourcePosition.y * 1.57 + 6.4));
        float billow = smoothstep(0.33, 0.78,
            broadVapor * 0.63 + fineVapor * 0.25 + microVapor * 0.12);
        float strand = smoothstep(0.34, 0.78,
            genesisNoiseSimplex(risingCoordinate *
                                vec3(1.65, 1.28, 1.65) +
                                vec3(-4.3, 11.8, 23.1)));

        // Condensation appears above the receiver, then thins progressively;
        // no density is deposited directly onto the floor.
        float verticalFade = smoothstep(0.025, 0.17, normalizedHeight) *
            (1.0 - smoothstep(0.48, 1.0, normalizedHeight));
        // Keep a dry visual bubble around the viewer. Steam begins beyond the
        // camera footprint, so nearby vapor cannot veil the scene ahead.
        float cameraDistance = length(position.xz - cameraPosition.xz);
        float cameraClearance = smoothstep(7.0, 10.0, cameraDistance);
        float distanceFade = 1.0 - smoothstep(18.0, 23.0, cameraDistance);
        float wispyErosion = mix(0.68, 1.18, microVapor);
        density += source * billow * mix(0.42, 1.0, strand) * wispyErosion *
                   verticalFade * cameraClearance * distanceFade * 1.2;
    }

    float opticalDepth = activity * opacityGrowth * density /
                         float(sampleCount) * segmentLength * 0.34;
    float coverage = clamp(1.0 - exp(-opticalDepth), 0.0, 0.065);
    // The late composite ultimately resolves to a display target with sparse
    // alpha precision at this low opacity. Sub-LSB triangular dithering turns
    // visible contour steps into a soft gradient without temporal shimmer.
    float alphaDither = genesisNoiseInterleavedGradient(
                            gl_FragCoord.xy + vec2(19.0, 47.0)) -
                        genesisNoiseInterleavedGradient(
                            gl_FragCoord.yx + vec2(71.0, 13.0));
    coverage += alphaDither * min(1.0 / 255.0, coverage * 0.16);
    return clamp(coverage, 0.0, 0.065);
}

float rainLayer(vec2 pixel, vec2 outputExtent, float sceneDistance,
                vec3 cameraForward, vec3 velocity, uint layerIndex,
                float intensity)
{
    const float layerCount = 6.0;
    float layerFraction = (float(layerIndex) + 0.5) / layerCount;
    float maximumDistance = max(rain.windMaximumDistance.w, 1.0);
    float distanceFromCamera = mix(1.75, maximumDistance,
                                   pow(layerFraction, 1.65));
    if (distanceFromCamera >= sceneDistance) {
        return 0.0;
    }
    // The streak field represents airborne rain, not decals on the first-hit
    // surface. Fade it out before the depth boundary and let the separate,
    // world-anchored impact field represent contact with the ground.
    float surfaceClearance = sceneDistance - distanceFromCamera;
    float airborneFade = sceneDistance < 1.0e19
        ? smoothstep(0.30, 1.10, surfaceClearance)
        : 1.0;

    vec3 anchor = frameData.cameraPositionExposure.xyz +
                  cameraForward * distanceFromCamera;
    float speed = max(length(velocity), 1.0);
    float physicalStreakLength = max(rain.amountTimeSpeedLength.w, 0.05);
    vec3 streakVector = velocity / speed * physicalStreakLength;
    vec2 anchorPixels = projectToPixels(anchor, outputExtent);
    vec2 tipPixels = projectToPixels(anchor + streakVector, outputExtent);
    vec2 projectedStreak = tipPixels - anchorPixels;
    float physicalPixelLength = length(projectedStreak);
    if (physicalPixelLength < 0.5) {
        projectedStreak = vec2(0.0, -1.0);
        physicalPixelLength = 0.5;
    }

    // Drop positions advance in a wind-independent vertical basis. Wind only
    // changes the instantaneous streak angle, so gust or direction changes
    // cannot move the field sideways or make its fall phase run backward.
    vec2 verticalTipPixels = projectToPixels(
        anchor + vec3(0.0, -physicalStreakLength, 0.0), outputExtent);
    vec2 projectedVertical = verticalTipPixels - anchorPixels;
    float verticalPixelLength = length(projectedVertical);
    if (verticalPixelLength < 0.5) {
        projectedVertical = vec2(0.0, -1.0);
        verticalPixelLength = 0.5;
    }
    vec2 fallAxis = projectedVertical / verticalPixelLength;
    vec2 sideAxis = vec2(-fallAxis.y, fallAxis.x);
    float streakLength = clamp(physicalPixelLength, 1.25, 10.0);
    float unboundedFallSpeedPixels =
        verticalPixelLength * max(rain.amountTimeSpeedLength.z, 1.0) /
        physicalStreakLength;
    float speedPixels = clamp(unboundedFallSpeedPixels, 80.0, 1800.0);

    // Each depth slice uses a different lattice scale and phase. This avoids
    // the vertical columns that make a regular screen-space rain field visible.
    float layerSeed = rainRandom(uvec3(layerIndex, 17u, 91u), 11u);
    float cellWidth = mix(52.0, 17.0, layerFraction) *
                      mix(0.82, 1.18, layerSeed);
    float cellHeight = max(streakLength * 3.0, mix(148.0, 58.0, layerFraction)) *
                       mix(0.86, 1.14, rainRandom(uvec3(layerIndex, 43u, 7u), 12u));
    vec2 basisPosition = vec2(dot(pixel, sideAxis), dot(pixel, fallAxis));
    basisPosition += vec2(layerSeed * 173.0,
                          rainRandom(uvec3(layerIndex, 5u, 29u), 13u) * 251.0);
    float advectedY = basisPosition.y - speedPixels * rain.amountTimeSpeedLength.y;
    ivec2 cell = ivec2(floor(basisPosition.x / cellWidth),
                       floor(advectedY / cellHeight));
    uvec3 hashCell = uvec3(uint(cell.x), uint(cell.y), layerIndex) ^
                     uvec3(0x68bc21ebu, 0x02e5be93u,
                           layerIndex * 0x9e3779b9u);

    float activation = rainRandom(hashCell, 0u);
    float activeThreshold = mix(0.95, 0.30, intensity);
    float activationFade = smoothstep(
        activeThreshold, min(activeThreshold + 0.08, 1.0), activation);

    float lateralCenter = (float(cell.x) + rainRandom(hashCell, 1u)) * cellWidth;
    float verticalCenter = (float(cell.y) + 0.20 +
                            0.60 * rainRandom(hashCell, 2u)) * cellHeight +
                           speedPixels * rain.amountTimeSpeedLength.y;
    vec2 dropOffset = vec2(basisPosition.x - lateralCenter,
                           basisPosition.y - verticalCenter);
    vec2 projectedAxis = normalize(vec2(dot(projectedStreak, sideAxis),
                                        dot(projectedStreak, fallAxis)));
    float slant = (rainRandom(hashCell, 3u) - 0.5) * 0.12;
    vec2 localAxis = normalize(projectedAxis + vec2(slant, 0.0));
    vec2 localSide = vec2(localAxis.y, -localAxis.x);
    float lateralDistance = abs(dot(dropOffset, localSide));
    float signedAlong = dot(dropOffset, localAxis);

    float width = mix(0.54, 0.22, layerFraction) *
                  mix(0.68, 1.22, rainRandom(hashCell, 4u));
    float lengthVariation = pow(rainRandom(hashCell, 5u), 1.35);
    float randomLength = streakLength * mix(0.30, 0.92, lengthVariation);
    float halfLength = max(0.65, randomLength * 0.5);
    float antialias = max(fwidth(lateralDistance), 0.72);

    // A low-opacity soft sheath around a narrow core approximates motion blur
    // from a photographic exposure instead of drawing an opaque white rod.
    float core = 1.0 - smoothstep(width, width + antialias, lateralDistance);
    float sheath = 1.0 - smoothstep(width + 0.35,
                                    width + 1.35 + 0.35 * layerFraction,
                                    lateralDistance);
    float cap = 1.0 - smoothstep(halfLength, halfLength + 1.15,
                                 abs(signedAlong));
    float normalizedAlong = clamp(signedAlong / (2.0 * halfLength) + 0.5,
                                  0.0, 1.0);
    float exposureTaper = smoothstep(0.0, 0.22, normalizedAlong) *
                          (1.0 - smoothstep(0.76, 1.0, normalizedAlong));
    exposureTaper = mix(0.52, 1.0, exposureTaper);

    float opacityVariation = mix(0.28, 0.86,
                                 rainRandom(hashCell, 6u));
    float distanceFade = mix(1.0, 0.45, layerFraction);
    return (0.78 * core + 0.22 * sheath) * cap * exposureTaper *
           opacityVariation * distanceFade * activationFade * airborneFade;
}

float rainImpactField(vec3 worldPosition, vec3 surfaceNormal,
                      float sceneDistance, float intensity,
                      float puddleRippleMask)
{
    float upwardFacing = smoothstep(0.52, 0.90,
                                    dot(surfaceNormal, vec3(0.0, 1.0, 0.0)));
    if (upwardFacing <= 0.0) {
        return 0.0;
    }

    // Keep the impact lattice at one world-space scale. Resizing its cells as
    // intensity changed moved every hashed center and made contacts visibly
    // slide across the floor during the weather cycle. Intensity now changes
    // only how many fixed cells are active.
    const float cellSize = 0.54;
    ivec2 baseCell = ivec2(floor(worldPosition.xz / cellSize));
    float surfaceFootprint = clamp(length(fwidth(worldPosition.xz)), 0.0015, 0.12);
    float impacts = 0.0;
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            ivec2 cell = baseCell + ivec2(offsetX, offsetY);
            uvec3 hashCell = uvec3(uint(cell.x), uint(cell.y), 0x51f2u);
            float activation = rainRandom(hashCell, 21u);
            if (activation > mix(0.07, 0.52, intensity)) {
                continue;
            }

            vec2 centerOffset = vec2(rainRandom(hashCell, 22u),
                                     rainRandom(hashCell, 23u));
            vec2 center = (vec2(cell) + mix(vec2(0.14), vec2(0.86), centerOffset)) *
                          cellSize;
            float distanceToImpact = length(worldPosition.xz - center);
            float cycle = mix(0.48, 1.05, rainRandom(hashCell, 24u));
            float age = fract(rain.amountTimeSpeedLength.y / cycle +
                              rainRandom(hashCell, 25u));
            // The hashed center remains fixed in world space while a capillary
            // wave expands away from it. Linear-to-slightly-decelerating
            // growth and a widening edge read as a real water ripple without
            // translating the impact across the surface.
            float growth = pow(age, 0.78);
            float maximumRadius = mix(0.060, 0.095,
                                      rainRandom(hashCell, 26u));
            float radius = mix(0.005, maximumRadius, growth);
            float ringWidth = mix(0.0022, 0.0080, growth);
            float antialias = max(surfaceFootprint * 0.35, 0.0015);
            float ring = 1.0 - smoothstep(ringWidth, ringWidth + antialias,
                                          abs(distanceToImpact - radius));
            float innerRing = 1.0 - smoothstep(
                ringWidth * 0.75, ringWidth * 0.75 + antialias,
                abs(distanceToImpact - radius * 0.58));
            float rippleLife = smoothstep(0.0, 0.025, age) *
                               (1.0 - smoothstep(0.24, 0.88, age));
            float contact = 1.0 - smoothstep(0.006,
                                             0.022 + antialias,
                                             distanceToImpact);
            float contactLife = 1.0 - smoothstep(0.0, 0.14, age);
            // Capillary rings need a standing water film. The initial contact
            // splash can still occur on dry or merely damp material, but its
            // expanding ripple is restricted to the shared puddle mask.
            impacts += (ring + innerRing * 0.22) * rippleLife *
                           puddleRippleMask +
                       contact * contactLife * 0.75;
        }
    }

    float distanceFade = 1.0 - smoothstep(
        rain.windMaximumDistance.w * 0.42,
        rain.windMaximumDistance.w * 0.95,
        sceneDistance);
    float footprintFade = 1.0 - smoothstep(0.045, 0.12, surfaceFootprint);
    return clamp(impacts * upwardFacing * distanceFade * footprintFade,
                 0.0, 1.0);
}

void main()
{
    float intensity = clamp(rain.amountTimeSpeedLength.x, 0.0, 1.0);
    float accumulatedWetness = clamp(frameData.weatherParameters.w, 0.0, 1.0);
    float steamActivity =
        (1.0 - smoothstep(0.0, 0.015, intensity)) *
        smoothstep(0.03, 0.35, accumulatedWetness);
    if (intensity <= 0.0 && steamActivity <= 0.0) {
        outColor = vec4(0.0);
        return;
    }

    vec2 depthExtent = vec2(textureSize(sceneDepth, 0));
    vec2 depthTexel = 1.0 / max(depthExtent, vec2(1.0));
    // First-hit AOVs are written by a jittered primary ray but carry
    // unjittered device depth. Follow the jitter to the texel representing
    // this stable output pixel; otherwise world-space impact cells crawl over
    // a stationary floor as the reconstruction jitter advances each frame.
    vec2 stableUv = clamp(
        uv + frameData.temporalJitter.xy * depthTexel,
        depthTexel * 0.5, vec2(1.0) - depthTexel * 0.5);
    float depth = texture(sceneDepth, stableUv).r;
    vec3 cameraPosition = frameData.cameraPositionExposure.xyz;
    float sceneDistance = 1.0e20;
    vec3 surfacePosition = vec3(0.0);
    if (depth < 0.999999) {
        surfacePosition = reconstructWorldPosition(stableUv, depth);
        sceneDistance = length(surfacePosition - cameraPosition);
    }

    vec2 stableCenterUv = vec2(0.5) +
                          frameData.temporalJitter.xy * depthTexel;
    vec3 centerNear = reconstructWorldPosition(stableCenterUv, 0.0);
    vec3 centerFar = reconstructWorldPosition(stableCenterUv, 1.0);
    vec3 cameraForward = normalize(centerFar - centerNear);
    vec3 windVelocity = rain.windMaximumDistance.xyz;
    // Two low-frequency waves keep the wind from looking mechanically
    // constant without introducing the high-frequency direction changes that
    // would make thin streaks shimmer. Clouds use the steady large-scale wind.
    float gust = clamp(0.82 +
                       0.12 * sin(rain.amountTimeSpeedLength.y * 0.55) +
                       0.06 * sin(rain.amountTimeSpeedLength.y * 1.31 + 2.0),
                       0.64, 1.0);
    vec2 outputExtent = vec2(1.0) / max(fwidth(uv), vec2(1.0e-6));

    float coverage = 0.0;
    if (intensity > 0.0) {
        for (uint layer = 0u; layer < 6u; ++layer) {
            float layerWindVariation = mix(
                0.88, 1.12, rainRandom(uvec3(layer, 73u, 19u), 31u));
            vec3 velocity = vec3(windVelocity.x * gust * layerWindVariation,
                                 -max(rain.amountTimeSpeedLength.z, 1.0),
                                 windVelocity.z * gust * layerWindVariation);
            coverage += rainLayer(gl_FragCoord.xy, outputExtent, sceneDistance,
                                  cameraForward, velocity, layer, intensity);
        }
    }
    float geometryBehindDrop = depth < 0.999999 ? 1.0 : 0.0;
    coverage *= mix(1.35, 1.0, geometryBehindDrop);
    coverage = clamp(coverage * mix(0.25, 0.58, intensity), 0.0, 0.56);

    float impactCoverage = 0.0;
    if (geometryBehindDrop > 0.5 && intensity > 0.0) {
        vec3 surfaceNormal = sampleWorldNormal(stableUv);
        GenesisWetnessLayers surfaceWetness = genesisRainWetnessLayers(
            surfacePosition, surfaceNormal, accumulatedWetness,
            frameData.weatherParameters.z);
        float puddleRippleMask = smoothstep(
            0.05, 0.35, surfaceWetness.puddles);
        impactCoverage = rainImpactField(surfacePosition, surfaceNormal,
                                         sceneDistance, intensity,
                                         puddleRippleMask);
        impactCoverage = clamp(impactCoverage * mix(0.20, 0.46, intensity),
                               0.0, 0.34);
    }

    float daylight = frameData.atmosphereSunDirection.w > 0.5
        ? smoothstep(-0.16, 0.18, frameData.atmosphereSunDirection.z)
        : 0.75;
    // This pass runs after tone mapping, so rain color must already match the
    // display-referred scene illumination. A nontrivial constant multiplier
    // makes drops look self-lit at night. Keep only a faint cool reflection in
    // darkness, then rise smoothly through twilight to the daylight response.
    float ambientIllumination = pow(daylight, 0.65);
    vec3 skyRain = mix(vec3(0.010, 0.014, 0.022),
                       vec3(0.32, 0.38, 0.42), ambientIllumination);
    vec3 geometryRain = mix(vec3(0.016, 0.021, 0.028),
                            vec3(0.66, 0.70, 0.72), ambientIllumination);
    vec3 rainColor = mix(skyRain, geometryRain, geometryBehindDrop);
    vec3 impactColor = mix(vec3(0.022, 0.030, 0.040),
                           vec3(0.80, 0.87, 0.90), ambientIllumination);
    float precipitationCoverage =
        1.0 - (1.0 - coverage) * (1.0 - impactCoverage);
    vec3 precipitationColor =
        (rainColor * coverage +
         impactColor * impactCoverage * (1.0 - coverage)) /
        max(precipitationCoverage, 1.0e-5);

    float steamCoverage = 0.0;
    if (steamActivity > 0.0) {
        // Fullscreen UV y=0 is the top of the Vulkan target.  Probe a clear
        // bottom corner of the demo rather than the sky so the steam layer
        // receives a valid ground height and normal.
        vec2 groundProbeUv = clamp(
            vec2(0.08, 0.94) + frameData.temporalJitter.xy * depthTexel,
            depthTexel * 0.5, vec2(1.0) - depthTexel * 0.5);
        float groundProbeDepth = texture(sceneDepth, groundProbeUv).r;
        if (groundProbeDepth < 0.999999 &&
            dot(sampleWorldNormal(groundProbeUv), vec3(0.0, 1.0, 0.0)) > 0.72) {
            float groundHeight =
                reconstructWorldPosition(groundProbeUv, groundProbeDepth).y;
            vec3 viewDirection = normalize(
                reconstructWorldPosition(stableUv, 1.0) - cameraPosition);
            steamCoverage = steamField(
                cameraPosition, viewDirection, sceneDistance, groundHeight,
                steamActivity, rain.amountTimeSpeedLength.y,
                frameData.weatherParameters.z);
        }
    }
    float sunElevation = frameData.atmosphereSunDirection.z;
    float lowSun = smoothstep(-0.10, 0.02, sunElevation) *
                   (1.0 - smoothstep(0.08, 0.30, sunElevation));
    vec3 daylightSteam = mix(vec3(0.70, 0.74, 0.75),
                             vec3(0.82, 0.67, 0.56), lowSun * 0.75);
    float steamLighting = max(ambientIllumination, lowSun * 0.92);
    vec3 steamColor = mix(vec3(0.024, 0.030, 0.038),
                          daylightSteam, steamLighting);
    float combinedCoverage = 1.0 -
        (1.0 - precipitationCoverage) * (1.0 - steamCoverage);
    vec3 combinedColor =
        (precipitationColor * precipitationCoverage +
         steamColor * steamCoverage * (1.0 - precipitationCoverage)) /
        max(combinedCoverage, 1.0e-5);
    outColor = vec4(combinedColor, combinedCoverage);
}
