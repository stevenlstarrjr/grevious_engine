#version 460
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_ray_tracing : require

#include "include/ibl.glsl"
#include "include/path_trace.glsl"
#include "include/path_trace_environment.glsl"

#define LOOK_FRAME_BINDING 2
#include "include/look_frame.glsl"
#include "include/first_hit_ids.glsl"

layout(location = 0) rayPayloadInEXT RadiancePayload payload;
layout(set = 0, binding = 12, rgba32f) uniform image2D directRadianceImage;
layout(set = 0, binding = 13, rgba16f) uniform image2D surfaceAovImage;
layout(set = 0, binding = 14, rgba16f) uniform image2D normalAovImage;
layout(set = 0, binding = 15, rgba32f) uniform image2D motionDepthAovImage;
layout(set = 0, binding = 16, r32f) uniform image2D depthAovImage;
layout(set = 0, binding = 17, rgba16f) uniform image2D nrdDiffuseInputImage;
layout(set = 0, binding = 18, rgba16f) uniform image2D nrdSpecularInputImage;
layout(set = 0, binding = 19, rgba16f) uniform image2D nrdMaterialImage;
layout(set = 0, binding = 20, rgba16f) uniform image2D rrDiffuseAlbedoImage;
layout(set = 0, binding = 21, rgba16f) uniform image2D rrSpecularAlbedoImage;
layout(set = 0, binding = 22, rgba16f) uniform image2D rrNormalRoughnessImage;
layout(set = 0, binding = 23, r32f) uniform image2D rrSpecularHitDistanceImage;

#define BRUNETON_TRANSMITTANCE_BINDING 24
#define BRUNETON_SCATTERING_BINDING 25
#define BRUNETON_IRRADIANCE_BINDING 26
#define GENESIS_CLOUD_BASE_NOISE_BINDING 27
#define GENESIS_CLOUD_DETAIL_NOISE_BINDING 28
#include "include/genesis_atmosphere.glsl"

layout(push_constant) uniform PathTracePushConstants
{
    uint sampleIndex;
    uint sampleInFrame;
    uint samplesThisFrame;
    uint fixedPrimarySample;
} controls;

void main()
{
    if (payload.depth == ID_ONLY_RAY_DEPTH) {
        writeFirstHitIds(uvec2(0u));
        return;
    }
    bool atmosphereEnabled = frameData.atmosphereSunDirection.w > 0.5;
    vec3 backgroundDirection = normalize(gl_WorldRayDirectionEXT);
    if (atmosphereEnabled && payload.depth == 0u &&
        controls.fixedPrimarySample == 2u) {
        // Ray Reconstruction needs a jittered primary ray for finite geometry,
        // but Genesis composites primary atmosphere misses after RR because
        // they have no surface guides. Reconstruct that background ray at the
        // unjittered pixel center; otherwise the exact sky/planet/cloud sample
        // shifts by the projection jitter every frame and appears to vibrate.
        vec2 extent = vec2(gl_LaunchSizeEXT.xy);
        vec2 pixelCenterNdc =
            ((vec2(gl_LaunchIDEXT.xy) + vec2(0.5)) / extent) * 2.0 - 1.0;
        vec2 jitterNdc = 2.0 * frameData.temporalJitter.xy / extent;
        vec4 stableWorld = frameData.inverseViewProjection *
                           vec4(pixelCenterNdc + jitterNdc, 1.0, 1.0);
        if (abs(stableWorld.w) > 1.0e-7 &&
            !any(isnan(stableWorld)) && !any(isinf(stableWorld))) {
            backgroundDirection = normalize(
                stableWorld.xyz / stableWorld.w -
                frameData.cameraPositionExposure.xyz);
        }
    }
    vec2 uv = atmosphereEnabled
                  ? directionToDualParaboloidUv(backgroundDirection)
                  : rotatedDirectionToLatLongUv(
                        backgroundDirection,
                        frameData.environmentParameters.x);
    // Keep the visible sky exact. Secondary rays use the small GPU-refreshed
    // lat-long cache, avoiding the full Bruneton integration at every bounce.
    vec3 radiance;
    if (atmosphereEnabled && payload.depth == 0u) {
        if (frameData.cloudWeatherParameters.w > 0.5) {
            float cloudTransmittance;
            // Primary atmosphere is an exact background composite, not a
            // Monte Carlo path sample. Keep its march phase fixed per pixel;
            // advancing one coherent phase with sampleIndex made the entire
            // cloud image pulse under Ray Reconstruction. Spatial variation
            // hides march strata without introducing temporal noise.
            float cloudRayJitter = genesisNoiseInterleavedGradient(
                vec2(gl_LaunchIDEXT.xy));
            radiance = genesisAtmosphereCloudRadiance(
                backgroundDirection, frameData.atmosphereCamera.xyz,
                frameData.atmosphereSunDirection.xyz,
                frameData.cloudLayerParameters, frameData.cloudWeatherParameters,
                false, cloudRayJitter,
                cloudTransmittance);
            if (frameData.cloudWeatherParameters.w > 1.5 &&
                frameData.cloudWeatherParameters.w < 2.5) {
                radiance = vec3(1.0 - cloudTransmittance);
            } else if (frameData.cloudWeatherParameters.w < 2.5) {
                radiance += genesisAtmosphereSolarDiskRadiance(
                    backgroundDirection, frameData.atmosphereCamera.xyz,
                    frameData.atmosphereSunDirection.xyz,
                    frameData.cloudLayerParameters,
                    frameData.cloudWeatherParameters);
            }
        } else {
            radiance = genesisAtmosphereRadiance(
                backgroundDirection, frameData.atmosphereCamera.xyz,
                frameData.atmosphereSunDirection.xyz, false);
            radiance += genesisAtmosphereSolarDiskRadiance(
                backgroundDirection, frameData.atmosphereCamera.xyz,
                frameData.atmosphereSunDirection.xyz,
                frameData.cloudLayerParameters,
                frameData.cloudWeatherParameters);
        }
    } else {
        radiance = max(textureLod(environmentMap, uv, 0.0).rgb, vec3(0.0));
    }
    if (payload.depth == 0u) {
        radiance *= frameData.lookColor.w;
    }
    if (payload.depth > 0u) {
        float environmentPdf = pathEnvironmentPdf(
            backgroundDirection, frameData.environmentParameters.x,
            frameData.environmentParameters.y > 0.5 && !atmosphereEnabled);
        radiance *= pathPowerHeuristic(payload.bouncePdf, environmentPdf);
    }
    if (payload.depth == 1u) {
        payload.firstLobeHitDistance = 0.0;
    }
    if (payload.depth == 0u) {
        payload.primaryHitDistance = 0.0;
    }
    if (payload.depth == 0u) {
        ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);
        vec4 currentClip = frameData.unjitteredViewProjection *
                           vec4(gl_WorldRayDirectionEXT, 0.0);
        vec4 previousClip = frameData.previousUnjitteredViewProjection *
                            vec4(gl_WorldRayDirectionEXT, 0.0);
        vec2 motion = vec2(0.0);
        float previousDepth = -1.0;
        if (abs(currentClip.w) > 1.0e-7 && abs(previousClip.w) > 1.0e-7 &&
            !any(isnan(currentClip)) && !any(isinf(currentClip)) &&
            !any(isnan(previousClip)) && !any(isinf(previousClip))) {
            vec2 currentNdc = currentClip.xy / currentClip.w;
            vec2 previousNdc = previousClip.xy / previousClip.w;
            motion = (currentNdc - previousNdc) * 0.5;
            previousDepth = 1.0;
        }
        vec3 directRadiance = radiance;
        uint sampleIndex = controls.sampleIndex;
        // Match the fresh-frame beauty signal used by Ray Reconstruction.
        if (controls.fixedPrimarySample < 2u && sampleIndex > 0u) {
            float count = float(sampleIndex);
            directRadiance =
                (imageLoad(directRadianceImage, pixel).rgb * count + directRadiance) /
                (count + 1.0);
        }
        imageStore(directRadianceImage, pixel,
                   vec4(max(directRadiance, vec3(0.0)), 1.0));
        imageStore(nrdDiffuseInputImage, pixel, vec4(0.0));
        imageStore(nrdSpecularInputImage, pixel, vec4(0.0));
        imageStore(nrdMaterialImage, pixel, vec4(0.0));
        imageStore(rrDiffuseAlbedoImage, pixel, vec4(0.0));
        imageStore(rrSpecularAlbedoImage, pixel, vec4(0.0));
        imageStore(rrNormalRoughnessImage, pixel, vec4(0.0, 0.0, 1.0, 1.0));
        imageStore(rrSpecularHitDistanceImage, pixel, vec4(0.0));
        if (controls.sampleInFrame + 1u == controls.samplesThisFrame) {
            imageStore(surfaceAovImage, pixel, vec4(0.0));
            imageStore(normalAovImage, pixel, vec4(0.0));
            imageStore(motionDepthAovImage, pixel, vec4(motion, previousDepth, 0.0));
            imageStore(depthAovImage, pixel, vec4(1.0));
        }
    }
    payload.radiance = radiance;
}
