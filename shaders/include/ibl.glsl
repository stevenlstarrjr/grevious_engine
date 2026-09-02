#ifndef GENESIS_RENDER_IBL_GLSL
#define GENESIS_RENDER_IBL_GLSL

const float IBL_PI = 3.14159265358979323846;

vec2 directionToLatLongUv(vec3 direction)
{
    vec3 unitDirection = normalize(direction);
    float u = atan(unitDirection.z, unitDirection.x) / (2.0 * IBL_PI) + 0.5;
    float v = acos(clamp(unitDirection.y, -1.0, 1.0)) / IBL_PI;
    return vec2(fract(u), v);
}

vec3 rotateEnvironmentDirection(vec3 direction, float radians)
{
    float cosine = cos(radians);
    float sine = sin(radians);
    return vec3(cosine * direction.x - sine * direction.z,
                direction.y,
                sine * direction.x + cosine * direction.z);
}

vec2 rotatedDirectionToLatLongUv(vec3 direction, float radians)
{
    return directionToLatLongUv(rotateEnvironmentDirection(direction, radians));
}

// Dynamic atmosphere cache mapping. Each hemisphere gets a guarded square in
// a 2:1 atlas. Unlike lat-long it has no pole singularity, and unlike an
// octahedral map it has no seam through the downward orbital view.
const float GENESIS_DUAL_PARABOLOID_GUARD = 0.985;

vec2 directionToDualParaboloidUv(vec3 direction)
{
    vec3 unitDirection = normalize(direction);
    bool upperHemisphere = unitDirection.y >= 0.0;
    vec2 projected = unitDirection.xz /
                     max(1.0 + abs(unitDirection.y), 1.0e-7);
    vec2 localUv = projected * (0.5 * GENESIS_DUAL_PARABOLOID_GUARD) + 0.5;
    return vec2((localUv.x + (upperHemisphere ? 0.0 : 1.0)) * 0.5,
                localUv.y);
}

vec3 dualParaboloidUvToDirection(vec2 uv)
{
    bool upperHemisphere = uv.x < 0.5;
    float localU = upperHemisphere ? uv.x * 2.0 : uv.x * 2.0 - 1.0;
    vec2 projected = (vec2(localU, uv.y) - 0.5) *
                     (2.0 / GENESIS_DUAL_PARABOLOID_GUARD);
    float radiusSquared = dot(projected, projected);
    if (radiusSquared > 1.0) {
        projected *= inversesqrt(radiusSquared);
        radiusSquared = 1.0;
    }
    float inverseDenominator = 1.0 / (1.0 + radiusSquared);
    float vertical = (1.0 - radiusSquared) * inverseDenominator;
    return normalize(vec3(2.0 * projected.x * inverseDenominator,
                          upperHemisphere ? vertical : -vertical,
                          2.0 * projected.y * inverseDenominator));
}

vec3 fresnelSchlickRoughness(float normalDotView, vec3 f0, float roughness)
{
    vec3 grazing = max(vec3(1.0 - roughness), f0);
    float weight = 1.0 - clamp(normalDotView, 0.0, 1.0);
    float weightSquared = weight * weight;
    weight = weightSquared * weightSquared * weight;
    return f0 + (grazing - f0) * weight;
}

#endif
