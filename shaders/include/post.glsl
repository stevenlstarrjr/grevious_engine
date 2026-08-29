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

vec3 rendererPathTracedDisplayFitted(vec3 color)
{
    // The compact ACES fit has a deliberately deep toe. That is attractive for
    // ordinary raster lighting, but it crushes physically valid multi-bounce
    // radiance in interiors when a sunlit opening is also visible. Preserve
    // additional scene-referred shadow latitude before applying the unchanged
    // ACES highlight shoulder. The boost falls away continuously with luminance,
    // so direct sun and emissive highlights retain their existing white point.
    float sceneLuminance = max(dot(color, vec3(0.2126, 0.7152, 0.0722)), 0.0);
    // Deep interiors need roughly three additional stops to coexist with a
    // daylight opening on an SDR display. The exponential makes this a local
    // shadow compression rather than a second global exposure: it is 8x at
    // black, about 3.6x at middle-dark values, and effectively 1x in sunlight.
    float shadowLatitude = 1.0 + 7.0 * exp(-12.0 * sceneLuminance);
    return rendererAcesFitted(color * shadowLatitude);
}

#endif
