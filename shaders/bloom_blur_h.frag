#version 460

layout(set = 0, binding = 0) uniform sampler2D sourceImage;
layout(set = 0, binding = 1) uniform BloomControls
{
    vec4 intensityFilterScaleThresholdKnee;
} bloomControls;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

void main()
{
    float filterScale = max(bloomControls.intensityFilterScaleThresholdKnee.y, 0.0);
    vec2 offset = vec2(filterScale / float(textureSize(sourceImage, 0).x), 0.0);
    vec3 color = texture(sourceImage, uv).rgb * 0.227027;
    color += texture(sourceImage, uv + offset * 1.384615).rgb * 0.316216;
    color += texture(sourceImage, uv - offset * 1.384615).rgb * 0.316216;
    color += texture(sourceImage, uv + offset * 3.230769).rgb * 0.070270;
    color += texture(sourceImage, uv - offset * 3.230769).rgb * 0.070270;
    outColor = vec4(color, 1.0);
}
