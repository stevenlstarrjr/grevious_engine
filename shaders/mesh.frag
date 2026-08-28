#version 460
#extension GL_EXT_nonuniform_qualifier : require

#include "include/pbr.glsl"
#include "include/ibl.glsl"
#include "include/shadows.glsl"

layout(set = 0, binding = 0) uniform FrameData
{
    mat4 viewProjection;
    mat4 inverseViewProjection;
    mat4 unjitteredViewProjection;
    mat4 previousViewProjection;
    mat4 previousUnjitteredViewProjection;
    vec4 cameraPositionExposure;
    vec4 directionalDirectionIntensity;
    vec4 directionalColor;
    vec4 pointPositionIntensity;
    vec4 pointColorRange;
    vec4 environmentParameters;
    uvec4 cullingParameters;
    uvec4 visibilityMask;
    vec4 temporalJitter;
    uvec4 temporalFlags;
    vec4 shadowSceneCenterRadius;
    vec4 shadowCascadeCenterRadius1;
    vec4 shadowCascadeCenterRadius2;
    vec4 shadowCascadeSplits;
    vec4 shadowCascadeDepthRadii;
    vec4 previousCameraPosition;
} frameData;

layout(set = 0, binding = 2) uniform sampler2D diffuseIrradianceMap;
layout(set = 0, binding = 3) uniform sampler2D prefilteredSpecularMap;
layout(set = 0, binding = 4) uniform sampler2D brdfIntegrationLut;
layout(set = 0, binding = 10) uniform sampler2DShadow directionalShadowMaps[3];
layout(set = 1, binding = 0) uniform sampler2D materialTextures[];

struct MaterialRecord
{
    vec4 baseColorFactor;
    vec4 emissiveAlphaCutoff;
    vec4 metallicRoughnessOcclusionNormal;
    uvec4 textureIndices;
    uvec4 textureState;
};

layout(std430, set = 0, binding = 8) readonly buffer Materials
{
    MaterialRecord materials[];
};

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec4 worldTangent;
layout(location = 4) in vec2 uv1;
layout(location = 5) in flat uint materialIndex;
layout(location = 6) in vec4 currentClipPosition;
layout(location = 7) in vec4 previousClipPosition;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outMotion;
layout(location = 2) out vec4 outNormal;

const uint BASE_COLOR_TEXTURE_BIT = 1u << 0u;
const uint METALLIC_ROUGHNESS_TEXTURE_BIT = 1u << 1u;
const uint NORMAL_TEXTURE_BIT = 1u << 2u;
const uint OCCLUSION_TEXTURE_BIT = 1u << 3u;
const uint EMISSIVE_TEXTURE_BIT = 1u << 4u;
const uint ALPHA_MASK_MATERIAL_BIT = 1u << 0u;

vec2 materialUv(MaterialRecord material, uint textureBit)
{
    return (material.textureState.w & textureBit) != 0u ? uv1 : uv;
}

float directionalShadowVisibility(vec3 position, vec3 normal, vec3 lightDirection)
{
    if (frameData.shadowCascadeSplits.w <= 0.0) {
        return 1.0;
    }

    float viewDistance =
        max((frameData.unjitteredViewProjection * vec4(position, 1.0)).w, 0.0);
    uint firstCascade = viewDistance <= frameData.shadowCascadeSplits.x
                            ? 0u
                            : (viewDistance <= frameData.shadowCascadeSplits.y ? 1u : 2u);
    uint cascadeIndex = firstCascade;
    vec3 coordinate = vec3(-1.0);
    for (uint candidate = firstCascade; candidate < 3u; ++candidate) {
        vec4 centerRadius = directionalShadowCascadeCenterRadius(
            candidate, frameData.shadowSceneCenterRadius,
            frameData.shadowCascadeCenterRadius1, frameData.shadowCascadeCenterRadius2);
        float depthRadius = directionalShadowCascadeDepthRadius(
            candidate, frameData.shadowCascadeDepthRadii);
        vec3 candidateCoordinate = directionalShadowCoordinate(
            position, frameData.directionalDirectionIntensity.xyz, centerRadius, depthRadius);
        if (all(greaterThanEqual(candidateCoordinate, vec3(0.0))) &&
            all(lessThanEqual(candidateCoordinate, vec3(1.0)))) {
            cascadeIndex = candidate;
            coordinate = candidateCoordinate;
            break;
        }
    }
    if (coordinate.x < 0.0) {
        return 1.0;
    }
    float normalDotLight = clamp(dot(normal, lightDirection), 0.0, 1.0);
    float bias = max(0.0015 * (1.0 - normalDotLight), 0.00025);
    vec2 texel = 1.0 /
                 vec2(textureSize(directionalShadowMaps[nonuniformEXT(cascadeIndex)], 0));
    float visibility = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            visibility += texture(directionalShadowMaps[nonuniformEXT(cascadeIndex)],
                                  vec3(coordinate.xy + vec2(x, y) * texel,
                                       coordinate.z - bias));
        }
    }
    return visibility / 9.0;
}

void main()
{
    MaterialRecord material = materials[materialIndex];
    uint textureMask = material.textureState.y;
    vec4 sampledBaseColor = vec4(1.0);
    if ((textureMask & BASE_COLOR_TEXTURE_BIT) != 0u) {
        sampledBaseColor = texture(
            materialTextures[nonuniformEXT(material.textureIndices.x)],
            materialUv(material, BASE_COLOR_TEXTURE_BIT));
    }
    vec4 baseColor = material.baseColorFactor * sampledBaseColor;
    if ((material.textureState.z & ALPHA_MASK_MATERIAL_BIT) != 0u &&
        baseColor.a < material.emissiveAlphaCutoff.a) {
        discard;
    }
    float metallic = material.metallicRoughnessOcclusionNormal.x;
    float perceptualRoughness = material.metallicRoughnessOcclusionNormal.y;
    if ((textureMask & METALLIC_ROUGHNESS_TEXTURE_BIT) != 0u) {
        vec4 metallicRoughness = texture(
            materialTextures[nonuniformEXT(material.textureIndices.y)],
            materialUv(material, METALLIC_ROUGHNESS_TEXTURE_BIT));
        metallic *= metallicRoughness.b;
        perceptualRoughness *= metallicRoughness.g;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    perceptualRoughness = clamp(perceptualRoughness, 0.0, 1.0);

    vec3 normal = normalize(worldNormal);
    if ((textureMask & NORMAL_TEXTURE_BIT) != 0u) {
        vec3 tangent = normalize(worldTangent.xyz -
                                 normal * dot(normal, worldTangent.xyz));
        vec3 bitangent = cross(normal, tangent) * worldTangent.w;
        vec3 tangentNormal = texture(
            materialTextures[nonuniformEXT(material.textureIndices.z)],
            materialUv(material, NORMAL_TEXTURE_BIT)).xyz * 2.0 - 1.0;
        tangentNormal.xy *= material.metallicRoughnessOcclusionNormal.w;
        normal = normalize(mat3(tangent, bitangent, normal) * tangentNormal);
    }
    vec3 viewDirection = normalize(frameData.cameraPositionExposure.xyz - worldPosition);
    vec3 materialColor = baseColor.rgb;

    vec3 directionalLight = normalize(-frameData.directionalDirectionIntensity.xyz);
    vec3 directionalRadiance = frameData.directionalColor.rgb *
                               frameData.directionalDirectionIntensity.w;
    float sunVisibility = directionalShadowVisibility(worldPosition, normal, directionalLight);
    vec3 radiance = evaluateDirectLight(materialColor, metallic, perceptualRoughness,
                                        normal, viewDirection, directionalLight,
                                        directionalRadiance) * sunVisibility;

    vec3 toPoint = frameData.pointPositionIntensity.xyz - worldPosition;
    float distanceSquared = max(dot(toPoint, toPoint), 1.0e-6);
    float distanceToPoint = sqrt(distanceSquared);
    vec3 pointDirection = toPoint / distanceToPoint;
    float range = frameData.pointColorRange.w;
    float rangeWindow = 1.0;
    if (range > 0.0) {
        float ratio = distanceToPoint / range;
        rangeWindow = pow(clamp(1.0 - pow(ratio, 4.0), 0.0, 1.0), 2.0);
    }
    vec3 pointRadiance = frameData.pointColorRange.rgb * frameData.pointPositionIntensity.w *
                         rangeWindow / distanceSquared;
    radiance += evaluateDirectLight(materialColor, metallic, perceptualRoughness,
                                    normal, viewDirection, pointDirection, pointRadiance);

    float normalDotView = clamp(dot(normal, viewDirection), 0.0, 1.0);
    vec3 f0 = mix(vec3(0.04), materialColor, metallic);
    vec3 fresnel = fresnelSchlickRoughness(normalDotView, f0, perceptualRoughness);
    vec3 diffuseWeight = (vec3(1.0) - fresnel) * (1.0 - metallic);
    float environmentRotation = frameData.environmentParameters.x;
    vec3 irradiance = texture(
        diffuseIrradianceMap,
        rotatedDirectionToLatLongUv(normal, environmentRotation)).rgb;
    vec3 diffuseIndirect = diffuseWeight * materialColor * irradiance / PBR_PI;

    vec3 reflection = reflect(-viewDirection, normal);
    float maximumLod = float(textureQueryLevels(prefilteredSpecularMap) - 1);
    vec3 prefiltered = textureLod(prefilteredSpecularMap,
                                  rotatedDirectionToLatLongUv(reflection,
                                                              environmentRotation),
                                  perceptualRoughness * maximumLod).rgb;
    vec2 brdf = texture(brdfIntegrationLut,
                        vec2(normalDotView, perceptualRoughness)).rg;
    vec3 specularIndirect = prefiltered * (f0 * brdf.x + brdf.y);
    float ambientOcclusion = 1.0;
    if ((textureMask & OCCLUSION_TEXTURE_BIT) != 0u) {
        float sampledOcclusion = texture(
            materialTextures[nonuniformEXT(material.textureIndices.w)],
            materialUv(material, OCCLUSION_TEXTURE_BIT)).r;
        ambientOcclusion = mix(1.0, sampledOcclusion,
                               clamp(material.metallicRoughnessOcclusionNormal.z, 0.0, 1.0));
    }
    radiance += (diffuseIndirect + specularIndirect) * ambientOcclusion;

    vec3 emissive = material.emissiveAlphaCutoff.rgb;
    if ((textureMask & EMISSIVE_TEXTURE_BIT) != 0u) {
        emissive *= texture(
            materialTextures[nonuniformEXT(material.textureState.x)],
            materialUv(material, EMISSIVE_TEXTURE_BIT)).rgb;
    }
    radiance += emissive;

    outColor = vec4(radiance, baseColor.a);
    bool currentValid = currentClipPosition.w > 1.0e-6 &&
                        !any(isnan(currentClipPosition)) &&
                        !any(isinf(currentClipPosition));
    vec2 currentUv = currentValid
        ? currentClipPosition.xy / currentClipPosition.w * 0.5 + 0.5
        : gl_FragCoord.xy / vec2(frameData.cullingParameters.zw);
    bool previousValid = currentValid && previousClipPosition.w > 1.0e-6 &&
                         !any(isnan(previousClipPosition)) &&
                         !any(isinf(previousClipPosition));
    vec2 previousUv = previousValid
        ? previousClipPosition.xy / previousClipPosition.w * 0.5 + 0.5
        : currentUv;
    float previousDepth = previousValid
        ? previousClipPosition.z / previousClipPosition.w
        : gl_FragCoord.z;
    outMotion = vec4(currentUv - previousUv, previousDepth,
                     previousValid ? 1.0 : 0.0);
    // World-space normals make the history independent of camera rotation.
    // Alpha distinguishes surface samples from the infinite environment.
    outNormal = vec4(normal * 0.5 + 0.5, 1.0);
}
