#ifndef RENDERER_POST_GLSL
#define RENDERER_POST_GLSL

const float BLOOM_THRESHOLD = 1.2;
const float BLOOM_KNEE = 0.6;
const float BLOOM_INTENSITY = 0.08;

vec3 rendererAcesFitted(vec3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

#endif
