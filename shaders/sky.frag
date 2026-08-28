#version 460

#include "include/ibl.glsl"

layout(set = 0, binding = 0) uniform FrameData
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

layout(set = 0, binding = 5) uniform sampler2D environmentMap;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outMotion;
layout(location = 2) out vec4 outNormal;

void main()
{
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 world = frameData.inverseViewProjection * vec4(ndc, 1.0, 1.0);
    vec3 worldPosition = world.xyz / world.w;
    vec3 direction = normalize(worldPosition - frameData.cameraPositionExposure.xyz);
    outColor = vec4(texture(
        environmentMap,
        rotatedDirectionToLatLongUv(direction, frameData.environmentParameters.x)).rgb, 1.0);
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
