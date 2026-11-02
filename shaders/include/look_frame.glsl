#ifndef GENESIS_LOOK_FRAME_GLSL
#define GENESIS_LOOK_FRAME_GLSL

layout(set = 0, binding = LOOK_FRAME_BINDING) uniform FrameData
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
    vec4 lookColor;
    uvec4 lookOptions;
} frameData;

// Linear reflectance and perceptual roughness for the neutral clay override.
const float clayAlbedo = 0.18;
const float clayRoughness = 0.85;
#endif
