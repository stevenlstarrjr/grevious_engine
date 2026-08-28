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

    vec3 hdr = resolved.rgb;
    if (frameData.temporalFlags.w != 0u) {
        float bloomIntensity = max(uintBitsToFloat(frameData.visibilityMask.w), 0.0);
        hdr += texture(bloomColor, uv).rgb * bloomIntensity;
    }
    hdr *= max(frameData.cameraPositionExposure.w, 0.0);
    outColor = vec4(rendererAcesFitted(hdr), 1.0);
}
