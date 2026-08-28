#version 460

#include "include/post.glsl"

layout(set = 0, binding = 0) uniform sampler2D sourceImage;
layout(set = 0, binding = 1) uniform BloomControls
{
    vec4 intensityFilterScaleThresholdKnee;
} bloomControls;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

void main()
{
    vec2 texel = 1.0 / vec2(textureSize(sourceImage, 0));
    vec3 color = texture(sourceImage, uv).rgb * 0.5;
    color += texture(sourceImage, uv + texel * vec2(-1.0, -1.0)).rgb * 0.125;
    color += texture(sourceImage, uv + texel * vec2( 1.0, -1.0)).rgb * 0.125;
    color += texture(sourceImage, uv + texel * vec2(-1.0,  1.0)).rgb * 0.125;
    color += texture(sourceImage, uv + texel * vec2( 1.0,  1.0)).rgb * 0.125;

    float brightness = max(max(color.r, color.g), color.b);
    float threshold = max(bloomControls.intensityFilterScaleThresholdKnee.z, 0.0);
    float knee = max(bloomControls.intensityFilterScaleThresholdKnee.w, 1.0e-5);
    float soft = clamp((brightness - threshold + knee) / (2.0 * knee), 0.0, 1.0);
    soft = soft * soft * knee;
    float contribution = max(brightness - threshold, soft) /
                         max(brightness, 1.0e-5);
    outColor = vec4(color * contribution, 1.0);
}
