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
    // x: environment rotation; y: test radiance; z/w: current/previous test center X.
    vec4 environmentParameters;
    uvec4 cullingParameters;
    // xy: layer mask; z: procedural post-test pattern.
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

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outMotion;
layout(location = 2) out vec4 outNormal;

void main()
{
    ivec2 size = ivec2(frameData.cullingParameters.zw);
    ivec2 pixel = clamp(ivec2(gl_FragCoord.xy), ivec2(0), size - 1);
    int centerX = clamp(int(frameData.environmentParameters.z * float(size.x - 1)), 0, size.x - 1);
    int centerY = clamp(int(frameData.directionalColor.w * float(size.y - 1)), 0, size.y - 1);
    uint pattern = frameData.visibilityMask.z;
    bool bright = false;
    if (pattern == 2u) {
        bright = true;
    } else if (pattern == 3u) {
        bright = pixel == ivec2(centerX, centerY);
    } else if (pattern == 4u || pattern == 9u) {
        bright = abs(pixel.x - centerX) <= 2 && abs(pixel.y - centerY) <= 2;
    } else if (pattern == 5u) {
        bright = pixel.y == centerY;
    } else if (pattern == 6u) {
        bright = pixel.x == centerX;
    } else if (pattern == 7u) {
        bright = pixel == ivec2(size.x / 4, size.y / 4) ||
                 pixel == ivec2(centerX, centerY) ||
                 pixel == ivec2((size.x * 3) / 4, (size.y * 3) / 4);
    } else if (pattern == 8u) {
        bright = ((pixel.x + pixel.y) & 1) == 0;
    }

    vec3 radiance = bright ? vec3(frameData.environmentParameters.y) : vec3(0.0);
    vec2 motion = vec2(0.0);
    if (pattern == 9u && bright) {
        motion.x = frameData.environmentParameters.z - frameData.environmentParameters.w;
    }
    outColor = vec4(radiance, 1.0);
    outMotion = vec4(motion, 1.0, 1.0);
    outNormal = vec4(0.5, 0.5, 1.0, 0.0);
}
