#ifndef GENESIS_NOISE_GLSL
#define GENESIS_NOISE_GLSL

// Shared analytic noise primitives. These remain useful for low-frequency
// control fields; expensive volumetric detail should come from generated,
// filtered textures so ray marching does not recalculate octaves per sample.
const mat3 GENESIS_NOISE_OCTAVE_ROTATION = mat3(
     0.00,  0.80,  0.60,
    -0.80,  0.36, -0.48,
    -0.60, -0.48,  0.64);

vec3 genesisNoiseGradient(vec3 cell) {
    cell = fract(cell * vec3(0.1031, 0.1030, 0.0973));
    cell += dot(cell, cell.yxz + 33.33);
    vec3 gradient = fract((cell.xxy + cell.yxx) * cell.zyx) * 2.0 - 1.0;
    return gradient * inversesqrt(max(dot(gradient, gradient), 1.0e-4));
}

float genesisNoiseSimplex(vec3 p) {
    const float skew = 1.0 / 3.0;
    const float unskew = 1.0 / 6.0;
    vec3 cell = floor(p + dot(p, vec3(skew)));
    vec3 x0 = p - cell + dot(cell, vec3(unskew));
    vec3 order = step(x0.yzx, x0.xyz);
    vec3 inverseOrder = 1.0 - order;
    vec3 i1 = min(order, inverseOrder.zxy);
    vec3 i2 = max(order, inverseOrder.zxy);
    vec3 x1 = x0 - i1 + unskew;
    vec3 x2 = x0 - i2 + 2.0 * unskew;
    vec3 x3 = x0 - 1.0 + 3.0 * unskew;

    vec4 attenuation = max(0.6 - vec4(
        dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    attenuation *= attenuation;
    attenuation *= attenuation;
    vec4 contribution = vec4(
        dot(genesisNoiseGradient(cell), x0),
        dot(genesisNoiseGradient(cell + i1), x1),
        dot(genesisNoiseGradient(cell + i2), x2),
        dot(genesisNoiseGradient(cell + vec3(1.0)), x3));
    return clamp(0.5 + 21.0 * dot(attenuation, contribution), 0.0, 1.0);
}

float genesisNoiseFbm(vec3 p, float detailWeight) {
    float detail = clamp(detailWeight, 0.0, 1.0);
    float value = 0.5;
    value += (genesisNoiseSimplex(p) - 0.5) * 0.62;
    p = GENESIS_NOISE_OCTAVE_ROTATION * p * 2.03 + vec3(17.1, 9.2, 13.7);
    value += (genesisNoiseSimplex(p) - 0.5) *
             0.25 * mix(0.35, 1.0, detail);
    p = GENESIS_NOISE_OCTAVE_ROTATION * p * 2.03 + vec3(17.1, 9.2, 13.7);
    value += (genesisNoiseSimplex(p) - 0.5) * 0.13 * detail;
    return clamp(value, 0.0, 1.0);
}

float genesisNoiseRidged(float value) {
    float ridge = 1.0 - abs(value * 2.0 - 1.0);
    return ridge * ridge;
}

vec4 genesisNoiseVolume(sampler3D volume, vec3 coordinate, float lod) {
    return textureLod(volume, fract(coordinate), max(lod, 0.0));
}

// Jimenez-style interleaved gradient noise. This is deterministic for a
// screen/cache pixel and gives neighboring cloud rays decorrelated strata
// without requiring another texture or mutable RNG state.
float genesisNoiseInterleavedGradient(vec2 pixel) {
    return fract(52.9829189 *
                 fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

#endif
