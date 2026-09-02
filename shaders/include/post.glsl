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

vec3 rendererAgxContrastApprox(vec3 x)
{
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 -
           6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

vec3 rendererAgx(vec3 color)
{
    // Minimal real-time AgX display transform. The inset/outset matrices place
    // linear sRGB in AgX's working space, while the logarithmic 16.5-stop domain
    // and sigmoid preserve separation in bright skies and reflections instead
    // of driving each RGB channel independently into a hard white shoulder.
    const mat3 agxInset = mat3(
        0.842479062253094,  0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772,  0.0784336000000000,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    const mat3 agxOutset = mat3(
         1.19687900512017,   -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368,  1.15190312990417,   -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433,  1.15107367264116);
    const float minEv = -12.47393;
    const float maxEv = 4.026069;

    vec3 working = agxInset * max(color, vec3(0.0));
    working = clamp(log2(max(working, vec3(1.0e-10))), minEv, maxEv);
    working = (working - minEv) / (maxEv - minEv);
    working = rendererAgxContrastApprox(working);

    // AgX's sigmoid is display encoded. Convert it back to linear light before
    // writing to the sRGB swapchain so Vulkan performs the final sRGB encoding
    // exactly once.
    vec3 displayLinear = pow(max(agxOutset * working, vec3(0.0)), vec3(2.2));
    return clamp(displayLinear, 0.0, 1.0);
}

#endif
