#version 460

#include "include/ibl.glsl"

#define LOOK_FRAME_BINDING 0
#include "include/look_frame.glsl"

layout(set = 0, binding = 5) uniform sampler2D environmentMap;

#define BRUNETON_TRANSMITTANCE_BINDING 15
#define BRUNETON_SCATTERING_BINDING 16
#define BRUNETON_IRRADIANCE_BINDING 17
#define GENESIS_CLOUD_BASE_NOISE_BINDING 18
#define GENESIS_CLOUD_DETAIL_NOISE_BINDING 19
#include "include/genesis_atmosphere.glsl"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outMotion;
layout(location = 2) out vec4 outNormal;

void main()
{
    vec2 ndc = uv * 2.0 - 1.0;
    // Geometry is rendered with projection jitter, but an infinite procedural
    // sky has no surface samples for that jitter to improve. Reconstruct its
    // ray at the unjittered pixel center so cloud integration does not switch
    // strata every frame. TAA still reprojects actual camera rotation through
    // the unjittered matrices written below.
    vec2 extent = vec2(frameData.cullingParameters.zw);
    vec2 jitterNdc = 2.0 * frameData.temporalJitter.xy /
                     max(extent, vec2(1.0));
    vec4 world = frameData.inverseViewProjection *
                 vec4(ndc + jitterNdc, 1.0, 1.0);
    vec3 worldPosition = world.xyz / world.w;
    vec3 direction = normalize(worldPosition - frameData.cameraPositionExposure.xyz);
    vec3 radiance;
    if (frameData.atmosphereSunDirection.w > 0.5) {
        if (frameData.cloudWeatherParameters.w > 0.5) {
            float cloudTransmittance;
            // Dither volume strata spatially but never temporally. A single
            // coherent phase exposes the shell samples as contour bands, while
            // changing that phase every frame makes the entire sky vibrate.
            float cloudRayJitter =
                genesisNoiseInterleavedGradient(floor(gl_FragCoord.xy));
            radiance = genesisAtmosphereCloudRadiance(
                direction, frameData.atmosphereCamera.xyz,
                frameData.atmosphereSunDirection.xyz,
                frameData.cloudLayerParameters, frameData.cloudWeatherParameters,
                false, cloudRayJitter,
                cloudTransmittance);
            if (frameData.cloudWeatherParameters.w > 1.5 &&
                frameData.cloudWeatherParameters.w < 2.5) {
                radiance = vec3(1.0 - cloudTransmittance);
            } else if (frameData.cloudWeatherParameters.w < 2.5) {
                radiance += genesisAtmosphereSolarDiskRadiance(
                    direction, frameData.atmosphereCamera.xyz,
                    frameData.atmosphereSunDirection.xyz,
                    frameData.cloudLayerParameters,
                    frameData.cloudWeatherParameters);
            }
        } else {
            radiance = genesisAtmosphereRadiance(
                direction, frameData.atmosphereCamera.xyz,
                frameData.atmosphereSunDirection.xyz, false);
            radiance += genesisAtmosphereSolarDiskRadiance(
                direction, frameData.atmosphereCamera.xyz,
                frameData.atmosphereSunDirection.xyz,
                frameData.cloudLayerParameters,
                frameData.cloudWeatherParameters);
        }
    } else {
        radiance = texture(environmentMap,
                           rotatedDirectionToLatLongUv(
                               direction, frameData.environmentParameters.x)).rgb;
    }
    outColor = vec4(radiance * frameData.lookColor.w, 1.0);
    // Reproject the infinite environment as a direction (w = 0). Camera
    // translation must not introduce parallax into sky motion vectors.
    vec4 currentClip = frameData.unjitteredViewProjection * vec4(direction, 0.0);
    vec4 previousClip = frameData.previousUnjitteredViewProjection * vec4(direction, 0.0);
    bool currentValid = currentClip.w > 1.0e-6 &&
                        !any(isnan(currentClip)) && !any(isinf(currentClip));
    bool previousValid = currentValid && previousClip.w > 1.0e-6 &&
                         !any(isnan(previousClip)) && !any(isinf(previousClip));
    vec2 currentUv = currentValid ? currentClip.xy / currentClip.w * 0.5 + 0.5 : uv;
    vec2 previousUv = previousValid ? previousClip.xy / previousClip.w * 0.5 + 0.5 : uv;
    outMotion = vec4(currentUv - previousUv, 1.0, previousValid ? 1.0 : 0.0);
    outNormal = vec4(direction * 0.5 + 0.5, 0.0);
}
