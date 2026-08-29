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

vec3 fresnelSchlickRoughness(float normalDotView, vec3 f0, float roughness)
{
    vec3 grazing = max(vec3(1.0 - roughness), f0);
    float weight = 1.0 - clamp(normalDotView, 0.0, 1.0);
    float weightSquared = weight * weight;
    weight = weightSquared * weightSquared * weight;
    return f0 + (grazing - f0) * weight;
}

#endif
