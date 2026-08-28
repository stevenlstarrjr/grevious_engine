#version 460

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

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;
layout(location = 3) in vec4 inTangent;
layout(location = 4) in vec2 inUv1;

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

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec2 uv;
layout(location = 3) out vec4 worldTangent;
layout(location = 4) out vec2 uv1;
layout(location = 5) out flat uint materialIndex;
layout(location = 6) out vec4 currentClipPosition;
layout(location = 7) out vec4 previousClipPosition;

void main()
{
    uint sceneInstanceIndex = visibleInstanceIndices[uint(gl_InstanceIndex)];
    InstanceRecord instance = instances[sceneInstanceIndex];
    TransformRecord transform = transforms[instance.meshMaterial.z];
    worldPosition = vec3(transform.world * vec4(inPosition, 1.0));
    worldNormal = normalize(mat3(transform.normalToWorld) * inNormal);
    uv = inUv;
    vec3 transformedTangent = mat3(transform.world) * inTangent.xyz;
    transformedTangent = normalize(transformedTangent -
                                   worldNormal * dot(worldNormal, transformedTangent));
    float mirroredHandedness = (instance.meshMaterial.w & 1u) != 0u ? -1.0 : 1.0;
    worldTangent = vec4(transformedTangent, inTangent.w * mirroredHandedness);
    uv1 = inUv1;
    materialIndex = instance.meshMaterial.y;
    gl_Position = frameData.viewProjection * vec4(worldPosition, 1.0);
    currentClipPosition = frameData.unjitteredViewProjection * vec4(worldPosition, 1.0);
    vec3 previousWorldPosition = vec3(transform.previousWorld * vec4(inPosition, 1.0));
    previousClipPosition = frameData.previousUnjitteredViewProjection *
                           vec4(previousWorldPosition, 1.0);
}
