#ifndef GENESIS_RENDER_SHADOWS_GLSL
#define GENESIS_RENDER_SHADOWS_GLSL

vec3 directionalShadowCoordinate(vec3 worldPosition, vec3 lightTravelDirection,
                                 vec4 sceneCenterRadius, float depthRadius)
{
    vec3 direction = normalize(lightTravelDirection);
    vec3 referenceUp = abs(direction.y) > 0.98 ? vec3(1.0, 0.0, 0.0)
                                               : vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(referenceUp, direction));
    vec3 up = cross(direction, right);
    vec3 relative = worldPosition - sceneCenterRadius.xyz;
    float inverseDiameter = 0.5 / max(sceneCenterRadius.w, 1.0e-5);
    float inverseDepth = 0.5 / max(depthRadius, 1.0e-5);
    return vec3(dot(relative, right) * inverseDiameter + 0.5,
                dot(relative, up) * inverseDiameter + 0.5,
                dot(relative, direction) * inverseDepth + 0.5);
}

float directionalShadowCascadeDepthRadius(uint cascadeIndex, vec4 depthRadii)
{
    return cascadeIndex == 0u ? depthRadii.x
                              : (cascadeIndex == 1u ? depthRadii.y : depthRadii.z);
}

vec4 directionalShadowCascadeCenterRadius(uint cascadeIndex, vec4 cascade0,
                                          vec4 cascade1, vec4 cascade2)
{
    return cascadeIndex == 0u ? cascade0 : (cascadeIndex == 1u ? cascade1 : cascade2);
}

#endif
