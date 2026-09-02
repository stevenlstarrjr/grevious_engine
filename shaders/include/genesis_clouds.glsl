#ifndef GENESIS_CLOUDS_GLSL
#define GENESIS_CLOUDS_GLSL

#include "include/genesis_noise.glsl"

#ifndef GENESIS_CLOUD_BASE_NOISE_BINDING
#error GENESIS_CLOUD_BASE_NOISE_BINDING must be defined before including genesis_clouds.glsl
#endif
#ifndef GENESIS_CLOUD_DETAIL_NOISE_BINDING
#error GENESIS_CLOUD_DETAIL_NOISE_BINDING must be defined before including genesis_clouds.glsl
#endif

layout(set = 0, binding = GENESIS_CLOUD_BASE_NOISE_BINDING) uniform sampler3D
    genesisCloudBaseNoiseVolume;
layout(set = 0, binding = GENESIS_CLOUD_DETAIL_NOISE_BINDING) uniform sampler3D
    genesisCloudDetailNoiseVolume;

// Shared procedural cloud field. Sky integration, raster direct lighting, and
// path-traced direct lighting all use these coordinates so weather motion does
// not drift between the visible cloud layer and its shadows on the world.
struct GenesisCloudColumn {
    vec2 warpedHorizontal;
    float field;
    float baseVariation;
};

float genesisCloudCoverageThreshold(float coverage) {
    // Coverage is a weather-control percentage. At 1.0 the threshold must
    // approach a closed deck rather than preserving the large holes of the
    // fair-weather noise distribution.
    float normalizedCoverage = clamp(coverage, 0.0, 1.0);
    return mix(0.78, 0.06, pow(normalizedCoverage, 1.35));
}

GenesisCloudColumn genesisCloudColumn(vec2 pointHorizontalKm,
                                      vec4 cloudWeather,
                                      float detailWeight) {
    float scaleKm = max(cloudWeather.z, 0.25);
    vec2 horizontal = (pointHorizontalKm - cloudWeather.xy) / scaleKm;
    float weatherLod = mix(2.5, 0.5, clamp(detailWeight, 0.0, 1.0));
    vec4 weatherNoise = genesisNoiseVolume(
        genesisCloudBaseNoiseVolume,
        vec3(horizontal * vec2(0.055, 0.047), 0.31), weatherLod);
    vec2 weatherWarp = vec2(weatherNoise.r, weatherNoise.a) - vec2(0.5);
    vec2 warpedHorizontal = horizontal + weatherWarp * 2.4;
    // Build broad weather fronts from two differently oriented, anisotropic
    // fields. A single isotropic simplex octave reads as a collection of
    // circular islands when viewed from below; crossed fronts retain large
    // coherent systems but give them shoulders, bays, and branching gaps.
    const mat2 frontRotation = mat2(0.82, 0.57, -0.57, 0.82);
    vec2 frontHorizontal = frontRotation * warpedHorizontal;
    float weatherPrimary = weatherNoise.r;
    float weatherCross = mix(weatherNoise.b, weatherNoise.a, 0.45);
    float weather = clamp(mix(weatherPrimary, weatherCross, 0.18), 0.0, 1.0);
    vec4 footprintNoise = genesisNoiseVolume(
        genesisCloudBaseNoiseVolume,
        vec3(frontHorizontal * vec2(0.52, 0.44), 0.73) +
            vec3(0.37, 0.19, 0.0),
        mix(2.0, 0.0, clamp(detailWeight, 0.0, 1.0)));
    float footprint = mix(footprintNoise.r, footprintNoise.g, 0.24);
    float baseVariation = mix(footprintNoise.b, footprintNoise.a, 0.35);
    float field = weather * 0.50 + footprint * 0.46 +
                  weatherCross * 0.04 +
                  (baseVariation - 0.5) * mix(0.025, 0.10, detailWeight);
    return GenesisCloudColumn(warpedHorizontal, field, baseVariation);
}

// The local weather field is intentionally measured in kilometres so it can
// cast stable shadows onto the ground.  Reusing those coordinates from orbit
// wraps the tileable 3D volumes hundreds of times across the visible planet,
// reducing the clouds to a repeated cellular carpet.  At planetary distance,
// sample a low-frequency field on the sphere instead.  The two differently
// oriented volumes avoid latitude/longitude seams and keep weather animation
// continuous through the same offset control.
GenesisCloudColumn genesisCloudColumnForView(vec3 point,
                                             vec4 cloudWeather,
                                             float detailWeight,
                                             float planetaryBlend) {
    GenesisCloudColumn localColumn = genesisCloudColumn(
        point.xy, cloudWeather, detailWeight);
    float blend = clamp(planetaryBlend, 0.0, 1.0);
    if (blend <= 0.0) {
        return localColumn;
    }

    vec3 planetNormal = normalize(point);
    vec3 weatherPhase = vec3(cloudWeather.xy * 0.00035, 0.0);
    vec3 primaryCoordinate =
        vec3(planetNormal.z, planetNormal.x, planetNormal.y) *
            vec3(0.96, 1.12, 0.88) +
        weatherPhase + vec3(0.17, 0.41, 0.73);
    vec3 crossCoordinate =
        vec3(planetNormal.y + planetNormal.z * 0.31,
             planetNormal.z - planetNormal.x * 0.27,
             planetNormal.x + planetNormal.y * 0.19) *
            vec3(2.18, 1.86, 2.42) -
        weatherPhase.zxy * 1.37 + vec3(0.61, 0.09, 0.34);
    float planetaryFilter = mix(1.35, 0.70,
                                clamp(detailWeight, 0.0, 1.0));
    vec4 primaryNoise = genesisNoiseVolume(
        genesisCloudBaseNoiseVolume, primaryCoordinate, planetaryFilter);
    vec4 crossNoise = genesisNoiseVolume(
        genesisCloudBaseNoiseVolume, crossCoordinate,
        planetaryFilter + 0.65);
    float planetaryWeather = mix(primaryNoise.r, primaryNoise.b, 0.12);
    float planetaryFootprint = mix(crossNoise.g, crossNoise.a, 0.38);
    float planetaryVariation = mix(primaryNoise.a, crossNoise.b, 0.42);
    float planetaryDetail = mix(primaryNoise.g, crossNoise.r, 0.48);
    // The primary field owns the synoptic system; the cross field only bends
    // and breaks its boundary. Equal weighting produced an evenly distributed
    // collection of round islands instead of recognisable fronts.
    float planetaryField = planetaryWeather * 0.76 +
                            planetaryFootprint * 0.24 +
                            (planetaryVariation - 0.5) * 0.045 +
                            (planetaryDetail - 0.5) * 0.085;
    planetaryField = clamp(0.5 + (planetaryField - 0.5) * 1.65,
                           0.0, 1.0);

    // This coordinate is only used while fading out the local billow field.
    // Its very large scale prevents a fine texture from reappearing during
    // the transition between aircraft and orbital views.
    float planetaryScaleKm = max(cloudWeather.z, 0.25) * 48.0;
    vec2 planetaryHorizontal = point.xy / planetaryScaleKm;
    return GenesisCloudColumn(
        mix(localColumn.warpedHorizontal, planetaryHorizontal, blend),
        mix(localColumn.field, planetaryField, blend),
        mix(localColumn.baseVariation, planetaryVariation, blend));
}

float genesisCloudDensity(vec3 point, vec4 cloudLayer, vec4 cloudWeather,
                          float detailWeight, float planetaryBlend) {
    float radius = length(point);
    float thickness = max(cloudLayer.y - cloudLayer.x, 1.0e-3);
    float height = (radius - cloudLayer.x) / thickness;
    if (height <= 0.0 || height >= 1.0) {
        return 0.0;
    }

    GenesisCloudColumn column = genesisCloudColumnForView(
        point, cloudWeather, detailWeight, planetaryBlend);
    float coverage = clamp(cloudLayer.z, 0.0, 1.0);
    float threshold = genesisCloudCoverageThreshold(coverage);
    // Most cells in a weather volume are empty. Reject columns that cannot
    // reach the final threshold even after maximum local erosion before
    // evaluating the more expensive three-dimensional billow field.
    float maximumEdgeLift = mix(0.025, 0.105,
                                1.0 - clamp(detailWeight, 0.0, 1.0)) +
                            mix(0.01, 0.16, detailWeight) * 0.5;
    if (column.field < threshold - maximumEdgeLift - 0.02) {
        return 0.0;
    }
    // Keep the shape field strongly coherent through height so cumulus growth
    // reads as one vertical body. The old near-isotropic height frequency
    // crossed several independent noise lobes from base to top, exposing the
    // cloud as a stack of detached horizontal pancakes. A gentle wind shear
    // bends the rising mass while preserving continuous towers.
    // Keep the large bodies upright. Strong normalized shear was turning a
    // tall cloud into a long diagonal brush stroke when viewed near sunset.
    vec2 verticalShear = vec2(0.36, -0.21) * height;
    vec3 shapeCoordinate =
        vec3(column.warpedHorizontal * vec2(2.30, 2.05) + verticalShear,
             height * 1.15) +
        vec3(19.1, 3.7, 7.4);
    float filterLod = mix(3.0, 0.0, clamp(detailWeight, 0.0, 1.0));
    // One filtered base lookup replaces several analytic octaves. Horizontal
    // frequencies create kilometer-scale lobes while the reduced vertical
    // frequency keeps them connected into towers instead of stacked slices.
    const mat2 billowRotation = mat2(0.80, 0.60, -0.60, 0.80);
    vec3 baseCoordinate =
        vec3(billowRotation * shapeCoordinate.xy * vec2(0.50, 0.43),
             shapeCoordinate.z * 0.10) +
        vec3(0.17, 0.43, 0.29);
    vec4 baseNoise = genesisNoiseVolume(
        genesisCloudBaseNoiseVolume, baseCoordinate, filterLod);
    // Reconstruct a Perlin-Worley body rather than thresholding Perlin alone.
    // The cellular channels round the large mass into cauliflower lobes while
    // Perlin joins those lobes into one cloud instead of a field of spheres.
    float worleyFbm = dot(baseNoise.gba, vec3(0.625, 0.250, 0.125));
    float perlinWorley = clamp(
        (baseNoise.r - (1.0 - worleyFbm) * 0.52) /
            max(0.32 + worleyFbm * 0.68, 1.0e-3),
        0.0, 1.0);
    float shape = clamp(mix(baseNoise.r, perlinWorley, 0.72), 0.0, 1.0);
    vec3 detailCoordinate =
        vec3(billowRotation * shapeCoordinate.xy * vec2(1.68, 1.42),
             shapeCoordinate.z * 0.14) +
        vec3(0.61, 0.11, 0.77);
    vec4 detailNoise = genesisNoiseVolume(
        genesisCloudDetailNoiseVolume, detailCoordinate, filterLod * 0.65);
    float erosion = dot(detailNoise.rgb, vec3(0.55, 0.30, 0.15));

    float columnSignal = column.field - threshold;
    float overcastBlend = smoothstep(0.72, 1.0, coverage);
    // Establish one stable vertical envelope per weather column.  The weather
    // signal controls a column's development, while a low vertical frequency
    // in the 3D shape field bends the boundary continuously through height.
    // This avoids both failure modes of the earlier field: a 2D footprint
    // extruded into a curtain and independent noise slices stacked as ribbons.
    float growthSeed = clamp(
        (columnSignal + 0.12) / 0.40,
        0.0, 1.0);
    float growth = smoothstep(0.0, 1.0, growthSeed) *
                   mix(0.88, 1.05, column.baseVariation);
    growth = pow(clamp(growth, 0.0, 1.0), 0.72);
    float cloudTop = mix(0.34, 0.99, growth);
    float cloudBase = mix(0.01, 0.13, column.baseVariation);
    float bodyHeight = clamp(
        (height - cloudBase) / max(cloudTop - cloudBase, 0.08),
        0.0, 1.0);
    float lower = smoothstep(cloudBase, cloudBase + 0.10, height);
    float upper = 1.0 - smoothstep(max(cloudTop - 0.20, 0.12),
                                   cloudTop, height);
    float verticalEnvelope = lower * upper;
    float middleLift = sin(bodyHeight * 3.14159265359) *
                       mix(0.052, 0.018, overcastBlend);
    float topCarve = pow(bodyHeight, 1.70) *
                     mix(0.110, 0.018, overcastBlend);
    float shapeSignal = (shape - 0.45) *
                        mix(0.56, 0.09, overcastBlend);
    float formationSignal = columnSignal + shapeSignal +
                            middleLift - topCarve;
    // Broad signal-space ramps create genuine fractional density at the
    // silhouette.  The 3D signal moves the boundary itself; it is not an
    // opacity texture painted onto a vertically extruded weather mask.
    float fairBody = smoothstep(-0.160, 0.210, formationSignal);
    float overcastSignal = columnSignal + (shape - 0.5) * 0.07;
    float overcastBody = smoothstep(-0.080, 0.060, overcastSignal);
    float localBody = mix(fairBody, overcastBody, overcastBlend) *
                      verticalEnvelope;
    // Individual kilometre-scale billows are below a pixel from orbit.  Their
    // repeated texture is not useful signal there, so the planetary endpoint
    // resolves broad weather coverage and lets the shell integration provide
    // opacity instead of exposing tiled cells.
    float planetaryMask = smoothstep(threshold - 0.045,
                                     threshold + 0.065, column.field);
    float planetaryBody = planetaryMask * verticalEnvelope * mix(
        0.90, 1.0, smoothstep(0.22, 0.78, column.baseVariation));
    float formed = mix(localBody, planetaryBody,
                       clamp(planetaryBlend, 0.0, 1.0));
    float bodyVariation = mix(0.72, 1.0, smoothstep(0.34, 0.76, shape));
    bodyVariation = mix(bodyVariation,
                        mix(0.90, 1.0, smoothstep(0.32, 0.76, shape)),
                        overcastBlend);
    bodyVariation = mix(
        bodyVariation,
        mix(0.82, 1.0,
            smoothstep(0.18, 0.82, column.baseVariation)),
        clamp(planetaryBlend, 0.0, 1.0));
    float edgeWeight = 1.0 - smoothstep(0.28, 0.82, formed);
    float erosionStrength = mix(0.020, 0.18, clamp(detailWeight, 0.0, 1.0)) *
                            mix(1.0, 0.18, overcastBlend) *
                            (1.0 - clamp(planetaryBlend, 0.0, 1.0));
    formed = clamp(formed - (1.0 - erosion) * erosionStrength * edgeWeight,
                   0.0, 1.0);
    return clamp(formed * bodyVariation *
                     max(cloudLayer.w, 0.0),
                 0.0, 2.0);
}

// Projects a Genesis Y-up world position into the cloud shell along the Sun.
// atmosphereCamera.w carries the planet bottom radius in kilometers. Genesis XZ
// maps to Bruneton XY, while Genesis Y maps to the reference Z-up altitude.
float genesisCloudSurfaceSunVisibility(vec3 worldPosition,
                                       vec4 atmosphereCamera,
                                       vec3 referenceSunDirection,
                                       vec4 cloudLayer,
                                       vec4 cloudWeather) {
    if (cloudWeather.w < 0.5 || cloudLayer.z <= 0.0 || cloudLayer.w <= 0.0 ||
        atmosphereCamera.w <= 0.0) {
        return 1.0;
    }

    vec3 sunDirection = normalize(referenceSunDirection);
    float sunElevation = sunDirection.z;
    if (sunElevation <= 0.015) {
        return 1.0;
    }

    float sampleRadius = mix(cloudLayer.x, cloudLayer.y, 0.48);
    float sampleHeightKm = max(sampleRadius - atmosphereCamera.w -
                                   worldPosition.y * 0.001,
                               0.0);
    vec2 localHorizontalKm = vec2(worldPosition.x, -worldPosition.z) * 0.001;
    vec2 cloudHorizontalKm = atmosphereCamera.xy + localHorizontalKm +
                             sunDirection.xy * (sampleHeightKm / sunElevation);
    // The column footprint is deliberately lower-frequency than visible cloud
    // erosion. That is equivalent to filtering a conventional shadow map and
    // avoids crawling or flashing at kilometer-scale distances.
    GenesisCloudColumn column = genesisCloudColumn(cloudHorizontalKm, cloudWeather, 0.0);
    float threshold = genesisCloudCoverageThreshold(cloudLayer.z);
    float opacity = smoothstep(threshold - 0.045, threshold + 0.070,
                               column.field);
    float slantDepthKm = (cloudLayer.y - cloudLayer.x) /
                         max(sunElevation, 0.16);
    float opticalDepth = opacity * max(cloudLayer.w, 0.0) *
                         slantDepthKm * 0.72;
    return clamp(exp(-opticalDepth), 0.0, 1.0);
}

vec3 genesisCloudShadowTint(float visibility) {
    float clearAmount = smoothstep(0.08, 0.82, visibility);
    return mix(vec3(0.52, 0.63, 0.82), vec3(1.0), clearAmount);
}

#endif
