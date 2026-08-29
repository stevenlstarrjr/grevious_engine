#version 460

#include "include/post.glsl"

layout(set = 0, binding = 0) uniform sampler2D hdrColor;
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
} frameData;
layout(set = 0, binding = 2) uniform sampler2D bloomColor;
layout(set = 0, binding = 3) uniform sampler2D motionDepth;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

vec3 decodeOctahedralNormal(vec2 encoded)
{
    vec2 value = encoded * 2.0 - 1.0;
    vec3 normal = vec3(value, 1.0 - abs(value.x) - abs(value.y));
    if (normal.z < 0.0) {
        vec2 signs = vec2(normal.x >= 0.0 ? 1.0 : -1.0,
                          normal.y >= 0.0 ? 1.0 : -1.0);
        normal.xy = (vec2(1.0) - abs(normal.yx)) * signs;
    }
    return normalize(normal);
}

void main()
{
    vec4 resolved = texture(hdrColor, uv);
    if (frameData.temporalFlags.y == 1u) {
        vec2 motion = texture(motionDepth, uv).xy;
        float magnitude = clamp(length(motion) * 64.0, 0.0, 1.0);
        outColor = vec4(clamp(vec3(0.5) + vec3(motion * 32.0, 0.0), 0.0, 1.0) +
                            vec3(0.0, 0.0, magnitude),
                        1.0);
        return;
    }
    if (frameData.temporalFlags.y == 2u) {
        float confidence = clamp(resolved.a, 0.0, 1.0);
        outColor = vec4(1.0 - confidence, confidence, 0.0, 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 3u) {
        outColor = vec4(rendererAcesFitted(texture(bloomColor, uv).rgb), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 4u) {
        outColor = vec4(rendererAcesFitted(resolved.rgb), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 5u || frameData.temporalFlags.y == 6u) {
        vec3 radiance = resolved.rgb;
        if (frameData.temporalFlags.y == 6u) {
            radiance = max(radiance - texture(bloomColor, uv).rgb, vec3(0.0));
        }
        radiance *= max(frameData.cameraPositionExposure.w, 0.0);
        outColor = vec4(rendererAcesFitted(radiance), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 7u) {
        outColor = vec4(clamp(resolved.rgb, 0.0, 1.0), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 8u) {
        float surface = texture(motionDepth, uv).w > 0.0 ? 1.0 : 0.0;
        outColor = vec4(vec3(clamp(resolved.a, 0.0, 1.0) * surface), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 9u || frameData.temporalFlags.y == 10u) {
        bool shadingNormal = frameData.temporalFlags.y == 10u;
        vec2 encoded = shadingNormal ? resolved.zw : resolved.xy;
        float surface = texture(motionDepth, uv).w > 0.0 ? 1.0 : 0.0;
        outColor = vec4((decodeOctahedralNormal(encoded) * 0.5 + 0.5) * surface, 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 11u) {
        float magnitude = clamp(length(resolved.xy) * 64.0, 0.0, 1.0);
        outColor = vec4(clamp(vec3(0.5) + vec3(resolved.xy * 32.0, 0.0), 0.0, 1.0) +
                            vec3(0.0, 0.0, magnitude),
                        1.0);
        return;
    }
    if (frameData.temporalFlags.y == 12u) {
        float surface = texture(motionDepth, uv).w > 0.0 ? 1.0 : 0.0;
        outColor = vec4(vec3((1.0 - clamp(resolved.r, 0.0, 1.0)) * surface), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 13u) {
        float sceneScale = max(frameData.shadowCascadeDepthRadii.w, 1.0e-6);
        float distance = max(resolved.w, 0.0);
        outColor = vec4(vec3(1.0 - exp(-distance / sceneScale)), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 14u) {
        vec3 directRadiance = texture(bloomColor, uv).rgb;
        vec3 indirectRadiance = max(resolved.rgb - directRadiance, vec3(0.0));
        vec3 residual = abs(resolved.rgb - directRadiance - indirectRadiance);
        outColor = vec4(clamp(residual * 256.0, 0.0, 1.0), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 15u) {
        float confidence = clamp(resolved.r, 0.0, 1.0);
        outColor = vec4(1.0 - confidence, confidence, 0.0, 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 16u) {
        vec3 radiance = resolved.rgb * max(frameData.cameraPositionExposure.w, 0.0);
        outColor = vec4(rendererAcesFitted(radiance), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 17u) {
        float weight = clamp(resolved.a, 0.0, 1.0);
        outColor = vec4(1.0 - weight, weight, 0.0, 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 18u) {
        vec3 radiance = resolved.rgb * max(frameData.cameraPositionExposure.w, 0.0);
        outColor = vec4(rendererAcesFitted(radiance), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 19u) {
        float variance = max(resolved.a, 0.0);
        float mapped = 1.0 - exp(-sqrt(variance) * 4.0);
        outColor = vec4(mapped, mapped * mapped, 0.0, 1.0);
        return;
    }

    vec3 hdr = resolved.rgb;
    if (frameData.temporalFlags.w != 0u) {
        float bloomIntensity = max(uintBitsToFloat(frameData.visibilityMask.w), 0.0);
        hdr += texture(bloomColor, uv).rgb * bloomIntensity;
    }
    hdr *= max(frameData.cameraPositionExposure.w, 0.0);
    outColor = vec4(rendererAcesFitted(hdr), 1.0);
}
