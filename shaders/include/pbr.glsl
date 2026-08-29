#ifndef GENESIS_RENDER_PBR_GLSL
#define GENESIS_RENDER_PBR_GLSL

// Portions derived from pbrt-v4, Copyright (c) 1998-2020 Matt Pharr,
// Wenzel Jakob, and Greg Humphreys; Apache-2.0.
// Isotropic GPU port of pbrt-v4 TrowbridgeReitzDistribution D/Lambda/G and
// rough microfacet reflection at revision 5f7a606806a4ac7b939131ded9d7a30ebd02416e.
// PBRT_PROVENANCE.md records deliberate glTF/realtime adaptations.
const float PBR_PI = 3.14159265358979323846;
const float PBR_MIN_ALPHA = 1.0e-4;

float pow5(float value)
{
    float squared = value * value;
    return squared * squared * value;
}

float perceptualRoughnessToAlpha(float roughness)
{
    float clamped = clamp(roughness, 0.0, 1.0);
    return max(clamped * clamped, PBR_MIN_ALPHA);
}

float trowbridgeReitzD(float normalDotHalf, float alpha)
{
    float noH = clamp(normalDotHalf, 0.0, 1.0);
    if (noH <= 0.0) {
        return 0.0;
    }
    float safeAlpha = max(alpha, PBR_MIN_ALPHA);
    float alphaSquared = safeAlpha * safeAlpha;
    float cosineSquared = noH * noH;
    float tangentSquared = (1.0 - cosineSquared) / cosineSquared;
    float cosineFourth = cosineSquared * cosineSquared;
    if (isinf(tangentSquared) || cosineFourth < 1.0e-16) {
        return 0.0;
    }
    float exponent = tangentSquared / alphaSquared;
    float onePlusExponent = 1.0 + exponent;
    return 1.0 / (PBR_PI * alphaSquared * cosineFourth *
                  onePlusExponent * onePlusExponent);
}

float smithLambda(float normalDotDirection, float alpha)
{
    float noW = clamp(normalDotDirection, 0.0, 1.0);
    if (noW <= 0.0) {
        return 1.0e20;
    }
    float tanThetaSquared = max(0.0, 1.0 - noW * noW) / (noW * noW);
    return (sqrt(1.0 + alpha * alpha * tanThetaSquared) - 1.0) * 0.5;
}

float smithMaskingShadowing(float normalDotView, float normalDotLight, float alpha)
{
    if (normalDotView <= 0.0 || normalDotLight <= 0.0) {
        return 0.0;
    }
    return 1.0 / (1.0 + smithLambda(normalDotView, alpha) +
                  smithLambda(normalDotLight, alpha));
}

vec3 fresnelSchlick(float viewDotHalf, vec3 f0)
{
    float weight = pow5(1.0 - clamp(viewDotHalf, 0.0, 1.0));
    return f0 + (vec3(1.0) - f0) * weight;
}

void evaluateMetallicRoughnessBrdfLobes(vec3 baseColor, float metallic,
                                        float perceptualRoughness, vec3 normal,
                                        vec3 viewDirection, vec3 lightDirection,
                                        out vec3 diffuse, out vec3 specular)
{
    float noV = clamp(dot(normal, viewDirection), 0.0, 1.0);
    float noL = clamp(dot(normal, lightDirection), 0.0, 1.0);
    if (noV <= 0.0 || noL <= 0.0) {
        diffuse = vec3(0.0);
        specular = vec3(0.0);
        return;
    }
    vec3 halfVector = normalize(viewDirection + lightDirection);
    float noH = clamp(dot(normal, halfVector), 0.0, 1.0);
    float voH = clamp(dot(viewDirection, halfVector), 0.0, 1.0);
    float alpha = perceptualRoughnessToAlpha(perceptualRoughness);
    vec3 f0 = mix(vec3(0.04), clamp(baseColor, 0.0, 1.0), clamp(metallic, 0.0, 1.0));
    vec3 fresnel = fresnelSchlick(voH, f0);
    float distribution = trowbridgeReitzD(noH, alpha);
    float geometry = smithMaskingShadowing(noV, noL, alpha);
    specular = fresnel * distribution * geometry / max(4.0 * noV * noL, 1.0e-7);
    vec3 diffuseWeight = (vec3(1.0) - fresnel) * (1.0 - clamp(metallic, 0.0, 1.0));
    diffuse = diffuseWeight * clamp(baseColor, 0.0, 1.0) / PBR_PI;
}

vec3 evaluateMetallicRoughnessBrdf(vec3 baseColor, float metallic,
                                   float perceptualRoughness, vec3 normal,
                                   vec3 viewDirection, vec3 lightDirection)
{
    vec3 diffuse;
    vec3 specular;
    evaluateMetallicRoughnessBrdfLobes(baseColor, metallic, perceptualRoughness,
                                       normal, viewDirection, lightDirection,
                                       diffuse, specular);
    return diffuse + specular;
}

vec3 evaluateDirectLight(vec3 baseColor, float metallic, float perceptualRoughness,
                         vec3 normal, vec3 viewDirection, vec3 lightDirection,
                         vec3 incidentRadiance)
{
    float noL = clamp(dot(normal, lightDirection), 0.0, 1.0);
    return evaluateMetallicRoughnessBrdf(baseColor, metallic, perceptualRoughness,
                                        normal, viewDirection, lightDirection) *
           max(incidentRadiance, vec3(0.0)) * noL;
}

#endif
