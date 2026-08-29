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
    // x: output history is valid.
    uvec4 flags;
} controls;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outHistoryColor;
layout(location = 1) out float outHistoryDepth;

bool finiteVector(vec4 value)
{
    return !any(isnan(value)) && !any(isinf(value));
}

vec3 finiteColor(vec3 color)
{
    return any(isnan(color)) || any(isinf(color)) ? vec3(0.0) : color;
}

void main()
{
    ivec2 renderSize = textureSize(currentColor, 0);
    ivec2 outputSize = textureSize(previousColor, 0);
    ivec2 sourceCoordinate = clamp(ivec2(uv * vec2(renderSize)), ivec2(0), renderSize - 1);

    vec3 current = finiteColor(texture(currentColor, uv).rgb);
    float depth = texelFetch(currentDepth, sourceCoordinate, 0).r;
    vec4 motion = texelFetch(currentMotion, sourceCoordinate, 0);
    if (isnan(depth) || isinf(depth)) {
        depth = 1.0;
    }
    if (!finiteVector(motion)) {
        motion = vec4(0.0);
    }

    vec3 neighborhoodMinimum = current;
    vec3 neighborhoodMaximum = current;
    float depthVariation = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            ivec2 neighbor = clamp(sourceCoordinate + ivec2(x, y), ivec2(0), renderSize - 1);
            vec3 sampleColor = finiteColor(texelFetch(currentColor, neighbor, 0).rgb);
            float sampleDepth = texelFetch(currentDepth, neighbor, 0).r;
            neighborhoodMinimum = min(neighborhoodMinimum, sampleColor);
            neighborhoodMaximum = max(neighborhoodMaximum, sampleColor);
            if (!isnan(sampleDepth) && !isinf(sampleDepth)) {
                depthVariation = max(depthVariation, abs(sampleDepth - depth));
            }
        }
    }

    vec2 jitterCorrection = (controls.temporalJitter.zw - controls.temporalJitter.xy) /
                            vec2(renderSize);
    vec2 previousUv = uv - motion.xy + jitterCorrection;
    bool insideHistory = all(greaterThanEqual(previousUv, vec2(0.0))) &&
                         all(lessThanEqual(previousUv, vec2(1.0)));
    bool acceptHistory = controls.flags.x != 0u && insideHistory;

    vec3 accumulated = current;
    if (acceptHistory) {
        float historyDepth = texture(previousDepth, previousUv).r;
        float depthThreshold = max(0.0015, depthVariation * 2.0);
        acceptHistory = !isnan(historyDepth) && !isinf(historyDepth) &&
                        abs(historyDepth - depth) <= depthThreshold;
        if (acceptHistory) {
            vec3 history = texture(previousColor, previousUv).rgb;
            if (any(isnan(history)) || any(isinf(history))) {
                acceptHistory = false;
            } else {
                history = clamp(history, neighborhoodMinimum, neighborhoodMaximum);
                float motionPixels = length(motion.xy * vec2(outputSize));
                float historyWeight = clamp(0.90 - motionPixels * 0.075, 0.10, 0.90);
                accumulated = mix(current, history, historyWeight);
            }
        }
    }

    outHistoryColor = vec4(accumulated, acceptHistory ? 1.0 : 0.0);
    outHistoryDepth = depth;
}
