#version 460
#extension GL_EXT_ray_tracing : require

#include "include/ibl.glsl"
#include "include/path_trace.glsl"
#include "include/path_trace_environment.glsl"

layout(set = 0, binding = 2) uniform FrameData
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
} frameData;

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

layout(push_constant) uniform PathTracePushConstants
{
    uint sampleIndex;
    uint sampleInFrame;
    uint samplesThisFrame;
    uint fixedPrimarySample;
} controls;

void main()
{
    vec2 uv = rotatedDirectionToLatLongUv(gl_WorldRayDirectionEXT,
                                           frameData.environmentParameters.x);
    vec3 radiance = max(textureLod(environmentMap, uv, 0.0).rgb, vec3(0.0));
    if (payload.depth > 0u) {
        float environmentPdf = pathEnvironmentPdf(
            gl_WorldRayDirectionEXT, frameData.environmentParameters.x,
            frameData.environmentParameters.y > 0.5);
        radiance *= pathPowerHeuristic(payload.bouncePdf, environmentPdf);
    }
    if (payload.depth == 1u) {
        payload.firstLobeHitDistance = 0.0;
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
        if (sampleIndex > 0u) {
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
