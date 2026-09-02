#version 460

#include "include/shadows.glsl"

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
    vec4 atmosphereCamera;
    vec4 atmosphereSunDirection;
    vec4 cloudLayerParameters;
    vec4 cloudWeatherParameters;
} frameData;

layout(push_constant) uniform ShadowPushConstants
{
    uint cascadeIndex;
} shadowPush;

layout(location = 0) in vec3 inPosition;
layout(location = 2) in vec2 inUv;
layout(location = 4) in vec2 inUv1;
layout(location = 5) in vec4 inColor;

struct InstanceRecord
{
    vec4 centerRadius;
    uvec4 meshMaterial;
    uvec4 visibilityLayer;
};

layout(std430, set = 0, binding = 6) readonly buffer Instances
{
    InstanceRecord instances[];
};

layout(std430, set = 0, binding = 7) readonly buffer VisibleInstances
{
    uint visibleInstanceIndices[];
};

struct TransformRecord
{
    mat4 world;
    mat4 normalToWorld;
    mat4 previousWorld;
};

layout(std430, set = 0, binding = 9) readonly buffer Transforms
{
    TransformRecord transforms[];
};

layout(location = 0) out vec2 uv;
layout(location = 1) out vec2 uv1;
layout(location = 2) out flat uint materialIndex;
layout(location = 3) out vec4 vertexColor;

void main()
{
    uint sceneInstanceIndex = visibleInstanceIndices[uint(gl_InstanceIndex)];
    InstanceRecord instance = instances[sceneInstanceIndex];
    TransformRecord transform = transforms[instance.meshMaterial.z];
    uv = inUv;
    uv1 = inUv1;
    vertexColor = inColor;
    materialIndex = instance.meshMaterial.y;
    vec3 worldPosition = vec3(transform.world * vec4(inPosition, 1.0));
    vec4 cascadeCenterRadius = directionalShadowCascadeCenterRadius(
        shadowPush.cascadeIndex, frameData.shadowSceneCenterRadius,
        frameData.shadowCascadeCenterRadius1, frameData.shadowCascadeCenterRadius2);
    float cascadeDepthRadius = directionalShadowCascadeDepthRadius(
        shadowPush.cascadeIndex, frameData.shadowCascadeDepthRadii);
    vec3 shadow = directionalShadowCoordinate(
        worldPosition, frameData.directionalDirectionIntensity.xyz, cascadeCenterRadius,
        cascadeDepthRadius);
    gl_Position = vec4(shadow.xy * 2.0 - 1.0, shadow.z, 1.0);
}
