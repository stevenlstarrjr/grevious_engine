#version 460

layout(set = 0, binding = 0) uniform sampler2D currentColor;
layout(set = 0, binding = 1) uniform sampler2D currentDepth;
layout(set = 0, binding = 2) uniform sampler2D currentMotion;
layout(set = 0, binding = 3) uniform sampler2D previousColor;
layout(set = 0, binding = 4) uniform sampler2D previousDepth;

layout(push_constant) uniform TemporalUpscalingControls
{
    // xy: current jitter; zw: previous jitter, all in render pixels.
    vec4 temporalJitter;
    // x: output history is valid; y: input is stochastic path-traced radiance;
    // z: the camera moved this frame.
    uvec4 flags;
} controls;

layout(location = 0) in vec2 uv;
layout(location = 0) out float outReactiveMask;

bool finiteVector(vec4 value)
{
    return !any(isnan(value)) && !any(isinf(value));
}

float luminance(vec3 color)
{
    return dot(max(color, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
}

vec3 boundedColor(vec3 color)
{
    color = max(color, vec3(0.0));
    return color / (vec3(1.0) + color);
}

void main()
{
    ivec2 renderSize = textureSize(currentColor, 0);
    ivec2 sourceCoordinate = clamp(ivec2(uv * vec2(renderSize)),
                                   ivec2(0), renderSize - 1);
    vec3 current = texture(currentColor, uv).rgb;
    float depth = texelFetch(currentDepth, sourceCoordinate, 0).r;
    vec4 motion = texelFetch(currentMotion, sourceCoordinate, 0);
    if (controls.flags.x == 0u || !finiteVector(vec4(current, depth)) ||
        !finiteVector(motion)) {
        outReactiveMask = 1.0;
        return;
    }

    vec2 jitterCorrection = (controls.temporalJitter.zw - controls.temporalJitter.xy) /
                            vec2(renderSize);
    vec2 previousUv = uv - motion.xy + jitterCorrection;
    bool insideHistory = all(greaterThanEqual(previousUv, vec2(0.0))) &&
                         all(lessThanEqual(previousUv, vec2(1.0)));
    if (!insideHistory) {
        outReactiveMask = 1.0;
        return;
    }

    float depthVariation = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 neighbor = clamp(sourceCoordinate + ivec2(x, y),
                                   ivec2(0), renderSize - 1);
            float sampleDepth = texelFetch(currentDepth, neighbor, 0).r;
            if (!isnan(sampleDepth) && !isinf(sampleDepth)) {
                depthVariation = max(depthVariation, abs(sampleDepth - depth));
            }
        }
    }

    // Motion Z is the depth at which this surface is expected to appear in
    // the previous frame. Comparing history against current depth makes
    // acceptance oscillate whenever the camera moves toward or away from a
    // surface, which presents as large-scale temporal wobble.
    float expectedPreviousDepth = motion.z;
    ivec2 historySize = textureSize(previousDepth, 0);
    ivec2 historyCoordinate = clamp(ivec2(previousUv * vec2(historySize)),
                                    ivec2(0), historySize - 1);
    float historyDepth = texelFetch(previousDepth, historyCoordinate, 0).r;
    vec4 history = texture(previousColor, previousUv);
    float depthThreshold = max(0.0005, depthVariation * 0.75);
    if (!finiteVector(history) || isnan(historyDepth) || isinf(historyDepth) ||
        expectedPreviousDepth < 0.0 ||
        abs(historyDepth - expectedPreviousDepth) > depthThreshold) {
        outReactiveMask = 1.0;
        return;
    }

    // Stochastic path radiance changes even when the underlying surface does
    // not. Treating that Monte Carlo variance as reactive content discards the
    // temporal samples needed to make a one-sample moving frame stable. Depth
    // and motion checks above still reject actual disocclusions.
    if (controls.flags.y != 0u) {
        float motionPixels = length(motion.xy * vec2(historySize));
        // Preserve enough valid history to suppress one-sample path noise, but
        // increase rejection as reprojection distance grows.
        float motionReactive = smoothstep(1.0, 8.0, motionPixels);
        if (controls.flags.z != 0u) {
            motionReactive = max(motionReactive, 0.35);
        }
        outReactiveMask = max(1.0 - clamp(history.a, 0.0, 1.0), motionReactive);
        return;
    }

    float currentLogLuminance = log2(1.0 + luminance(current));
    float historyLogLuminance = log2(1.0 + luminance(history.rgb));
    float luminanceChange = abs(currentLogLuminance - historyLogLuminance);
    vec3 colorChange = abs(boundedColor(current) - boundedColor(history.rgb));
    float chromaticChange = max(colorChange.r, max(colorChange.g, colorChange.b));
    float reactive = max(smoothstep(0.04, 0.30, luminanceChange),
                         smoothstep(0.03, 0.20, chromaticChange));
    reactive = max(reactive, 1.0 - clamp(history.a, 0.0, 1.0));
    outReactiveMask = clamp(reactive, 0.0, 1.0);
}
