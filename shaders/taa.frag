#version 460

layout(set = 0, binding = 0) uniform sampler2D currentColor;
layout(set = 0, binding = 1) uniform sampler2D historyColor;
// xy: current UV minus previous UV; z: expected previous device depth;
// w: previous projection validity.
layout(set = 0, binding = 2) uniform sampler2D motionDepth;
layout(set = 0, binding = 3) uniform sampler2D currentDepth;
layout(set = 0, binding = 4) uniform sampler2D historyDepth;
// RGB: packed world-space surface normal or sky direction; alpha: surface.
layout(set = 0, binding = 5) uniform sampler2D currentNormal;
layout(set = 0, binding = 6) uniform sampler2D historyNormal;
layout(set = 0, binding = 7) uniform FrameData
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
    // xy: current jitter; zw: previous jitter, all in pixel units.
    vec4 temporalJitter;
    // x: history is valid; y: debug-view selector; z: TAA enabled.
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

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outHistoryColor;
layout(location = 1) out float outHistoryDepth;
layout(location = 2) out vec4 outHistoryNormal;

bool finiteVector(vec4 value)
{
    return !any(isnan(value)) && !any(isinf(value));
}

vec3 decodeNormal(vec3 packed)
{
    vec3 decoded = packed * 2.0 - 1.0;
    float lengthSquared = dot(decoded, decoded);
    return lengthSquared > 1.0e-8 ? decoded * inversesqrt(lengthSquared)
                                 : vec3(0.0, 0.0, 1.0);
}

float luminance(vec3 color)
{
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

void main()
{
    ivec2 size = textureSize(currentColor, 0);
    ivec2 coordinate = clamp(ivec2(gl_FragCoord.xy), ivec2(0), size - 1);
    vec3 current = texelFetch(currentColor, coordinate, 0).rgb;
    float depth = texelFetch(currentDepth, coordinate, 0).r;
    vec4 motion = texelFetch(motionDepth, coordinate, 0);
    vec4 normal = texelFetch(currentNormal, coordinate, 0);
    if (any(isnan(current)) || any(isinf(current))) {
        current = vec3(0.0);
    }
    if (isnan(depth) || isinf(depth)) {
        depth = 1.0;
    }
    if (!finiteVector(motion)) {
        motion = vec4(0.0, 0.0, depth, 0.0);
    }
    if (!finiteVector(normal)) {
        normal = vec4(0.5, 0.5, 1.0, 0.0);
    }
    // Geometry is rasterized with projection jitter; the procedural sky is
    // deliberately evaluated at unjittered pixel centers. Applying the camera
    // correction to both shifts sky history a second time and produces a
    // one-pixel repeating wobble even when the camera is stationary.
    vec2 jitterCorrection = normal.a > 0.5
        ? (frameData.temporalJitter.zw - frameData.temporalJitter.xy) /
              vec2(size)
        : vec2(0.0);
    vec2 previousUv = uv - motion.xy + jitterCorrection;

    bool insideHistory = all(greaterThanEqual(previousUv, vec2(0.0))) &&
                         all(lessThanEqual(previousUv, vec2(1.0)));
    bool acceptHistory = frameData.temporalFlags.x != 0u &&
                         motion.w > 0.5 && insideHistory;

    vec3 neighborhoodMinimum = current;
    vec3 neighborhoodMaximum = current;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 neighbor = clamp(coordinate + ivec2(x, y), ivec2(0), size - 1);
            vec3 sampleColor = texelFetch(currentColor, neighbor, 0).rgb;
            if (any(isnan(sampleColor)) || any(isinf(sampleColor))) {
                sampleColor = current;
            }
            neighborhoodMinimum = min(neighborhoodMinimum, sampleColor);
            neighborhoodMaximum = max(neighborhoodMaximum, sampleColor);
        }
    }

    vec3 accumulated = current;
    if (acceptHistory) {
        float previousDepth = texture(historyDepth, previousUv).r;
        float depthThreshold = max(0.00075, fwidth(depth) * 2.0);
        acceptHistory = abs(previousDepth - motion.z) <= depthThreshold;
        vec4 reprojectedNormal = texture(historyNormal, previousUv);
        bool currentSurface = normal.a > 0.5;
        bool previousSurface = reprojectedNormal.a > 0.5;
        acceptHistory = acceptHistory && finiteVector(reprojectedNormal) &&
                        currentSurface == previousSurface &&
                        dot(decodeNormal(normal.rgb),
                            decodeNormal(reprojectedNormal.rgb)) >= 0.85;
        if (acceptHistory) {
            vec3 history = texture(historyColor, previousUv).rgb;
            if (any(isnan(history)) || any(isinf(history))) {
                acceptHistory = false;
            } else {
                history = clamp(history, neighborhoodMinimum, neighborhoodMaximum);
                float motionPixels = length(motion.xy * vec2(size));
                float currentLuminance = luminance(current);
                float historyLuminance = luminance(history);
                float relativeDifference =
                    abs(currentLuminance - historyLuminance) /
                    max(max(currentLuminance, historyLuminance), 0.05);
                float response = mix(1.0, 0.55,
                                     smoothstep(0.04, 0.35, relativeDifference));
                // A 0.92 stationary history weight visibly softened the native
                // image. Keep enough history for sub-pixel stability while
                // responding faster to real shading and silhouette changes.
                float historyWeight =
                    clamp((0.86 - motionPixels * 0.10) * response, 0.08, 0.86);
                accumulated = mix(current, history, historyWeight);
            }
        }
    }

    outHistoryColor = vec4(accumulated, acceptHistory ? 1.0 : 0.0);
    outHistoryDepth = depth;
    outHistoryNormal = normal;
}
