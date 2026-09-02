#version 460
#extension GL_EXT_nonuniform_qualifier : require
#ifdef RENDERER_RAY_QUERY_SHADOWS
#extension GL_EXT_ray_query : require
#endif

#include "include/pbr.glsl"
#include "include/ibl.glsl"
#define GENESIS_CLOUD_BASE_NOISE_BINDING 18
#define GENESIS_CLOUD_DETAIL_NOISE_BINDING 19
#include "include/genesis_clouds.glsl"
#include "include/weather_wetness.glsl"
#ifndef RENDERER_RAY_QUERY_SHADOWS
#include "include/shadows.glsl"
#endif

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
    vec4 atmosphereCamera;
    vec4 atmosphereSunDirection;
    vec4 cloudLayerParameters;
    vec4 cloudWeatherParameters;
    vec4 weatherParameters;
} frameData;

layout(set = 0, binding = 2) uniform sampler2D diffuseIrradianceMap;
layout(set = 0, binding = 3) uniform sampler2D prefilteredSpecularMap;
layout(set = 0, binding = 4) uniform sampler2D brdfIntegrationLut;
layout(set = 0, binding = 5) uniform sampler2D environmentMap;
#ifdef RENDERER_RAY_QUERY_SHADOWS
layout(set = 0, binding = 11) uniform accelerationStructureEXT sceneTlas;
#else
layout(set = 0, binding = 10) uniform sampler2DShadow directionalShadowMaps[3];
#endif
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

#ifdef RENDERER_RAY_QUERY_SHADOWS
struct RayInstanceRecord
{
    vec4 centerRadius;
    uvec4 meshMaterial;
    uvec4 visibilityLayer;
};

layout(std430, set = 0, binding = 6) readonly buffer RayInstances
{
    RayInstanceRecord rayInstances[];
};

layout(std430, set = 0, binding = 12) readonly buffer RayVertices
{
    float rayVertexData[];
};

layout(std430, set = 0, binding = 13) readonly buffer RayIndices
{
    uint rayIndices[];
};

layout(std430, set = 0, binding = 14) readonly buffer RayGeometries
{
    uvec4 rayGeometryRanges[];
};
#endif

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec4 worldTangent;
layout(location = 4) in vec2 uv1;
layout(location = 5) in flat uint materialIndex;
layout(location = 6) in vec4 currentClipPosition;
layout(location = 7) in vec4 previousClipPosition;
layout(location = 8) in flat uint sceneInstanceIndex;
layout(location = 9) in vec4 vertexColor;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outMotion;
layout(location = 2) out vec4 outNormal;

const uint BASE_COLOR_TEXTURE_BIT = 1u << 0u;
const uint METALLIC_ROUGHNESS_TEXTURE_BIT = 1u << 1u;
const uint NORMAL_TEXTURE_BIT = 1u << 2u;
const uint OCCLUSION_TEXTURE_BIT = 1u << 3u;
const uint EMISSIVE_TEXTURE_BIT = 1u << 4u;
const uint ALPHA_MASK_MATERIAL_BIT = 1u << 0u;
const uint ALPHA_BLEND_MATERIAL_BIT = 1u << 1u;
const uint ADDITIVE_MATERIAL_BIT = 1u << 3u;
const uint RAIN_RECEIVER_MATERIAL_BIT = 1u << 4u;
const uint WET_COAT_MATERIAL_BIT = 1u << 5u;

vec2 materialUv(MaterialRecord material, uint textureBit)
{
    return (material.textureState.w & textureBit) != 0u ? uv1 : uv;
}

#ifdef RENDERER_RAY_QUERY_SHADOWS
vec2 rayVertexUv(uint vertexIndex, bool useUv1)
{
    uint firstComponent = vertexIndex * 18u + (useUv1 ? 12u : 6u);
    return vec2(rayVertexData[firstComponent], rayVertexData[firstComponent + 1u]);
}

vec4 rayVertexColor(uint vertexIndex)
{
    uint firstComponent = vertexIndex * 18u + 14u;
    return vec4(rayVertexData[firstComponent], rayVertexData[firstComponent + 1u],
                rayVertexData[firstComponent + 2u], rayVertexData[firstComponent + 3u]);
}

bool rayCandidateBlocksLight(uint instanceIndex, uint primitiveIndex, vec2 barycentrics)
{
    RayInstanceRecord instance = rayInstances[instanceIndex];
    MaterialRecord material = materials[instance.meshMaterial.y];
    uint materialFlags = material.textureState.z;
    if ((materialFlags & (ALPHA_BLEND_MATERIAL_BIT | ADDITIVE_MATERIAL_BIT)) != 0u) {
        return false;
    }
    if ((materialFlags & ALPHA_MASK_MATERIAL_BIT) == 0u) {
        return true;
    }

    uvec4 geometry = rayGeometryRanges[instance.meshMaterial.x];
    uint triangleFirstIndex = geometry.x + primitiveIndex * 3u;
    uvec3 triangle =
        uvec3(rayIndices[triangleFirstIndex], rayIndices[triangleFirstIndex + 1u],
              rayIndices[triangleFirstIndex + 2u]) + geometry.y;
    vec3 weights = vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics);
    vec4 candidateColor = rayVertexColor(triangle.x) * weights.x +
                          rayVertexColor(triangle.y) * weights.y +
                          rayVertexColor(triangle.z) * weights.z;
    float alpha = material.baseColorFactor.a * candidateColor.a;
    if ((material.textureState.y & BASE_COLOR_TEXTURE_BIT) != 0u) {
        bool useUv1 = (material.textureState.w & BASE_COLOR_TEXTURE_BIT) != 0u;
        vec2 candidateUv = rayVertexUv(triangle.x, useUv1) * weights.x +
                           rayVertexUv(triangle.y, useUv1) * weights.y +
                           rayVertexUv(triangle.z, useUv1) * weights.z;

        // Candidate rays have no useful screen-space texture derivatives. Base-level
        // alpha is deterministic; ray-cone LOD selection belongs in a later quality pass.
        alpha *= textureLod(materialTextures[nonuniformEXT(material.textureIndices.x)],
                            candidateUv, 0.0).a;
    }
    return alpha >= material.emissiveAlphaCutoff.a;
}

float offsetRayComponent(float position, float normal)
{
    // Move the origin by a fixed number of representable floats. Unlike a
    // world-space epsilon, this remains effective both near and far from the
    // origin and avoids precision-dependent self intersections.
    const float originThreshold = 1.0 / 32.0;
    const float nearOriginOffset = 1.0 / 65536.0;
    const float integerOffsetScale = 256.0;
    int integerOffset = int(integerOffsetScale * normal);
    float integerPosition = intBitsToFloat(floatBitsToInt(position) +
                                            (position < 0.0 ? -integerOffset
                                                            : integerOffset));
    return abs(position) < originThreshold
               ? position + nearOriginOffset * normal
               : integerPosition;
}

vec3 offsetRayOrigin(vec3 position, vec3 normal, float minimumOffset)
{
    vec3 representableOffset = vec3(offsetRayComponent(position.x, normal.x),
                                    offsetRayComponent(position.y, normal.y),
                                    offsetRayComponent(position.z, normal.z));
    return representableOffset + normal * minimumOffset;
}
#endif

float directionalShadowVisibility(vec3 position, vec3 normal, vec3 lightDirection)
{
    if (frameData.shadowCascadeSplits.w <= 0.0) {
        return 1.0;
    }

#ifdef RENDERER_RAY_QUERY_SHADOWS
    float sceneScale = max(frameData.shadowCascadeDepthRadii.w, 1.0);
    float normalDotLight = clamp(dot(normal, lightDirection), 0.0, 1.0);
    float originBias = sceneScale * mix(1.0e-3, 5.0e-4, normalDotLight);
    vec3 geometricNormal = cross(dFdx(position), dFdy(position));
    float geometricLengthSquared = dot(geometricNormal, geometricNormal);
    geometricNormal = geometricLengthSquared > 1.0e-12
                          ? geometricNormal * inversesqrt(geometricLengthSquared)
                          : normal;
    geometricNormal = dot(geometricNormal, normal) < 0.0 ? -geometricNormal : geometricNormal;
    // Always offset to the side of the triangle into which this shadow ray
    // travels. The previous normal-only offset could remain below a neighboring
    // triangle at grazing angles and produce the dotted self-shadow pattern.
    vec3 outgoingNormal = dot(geometricNormal, lightDirection) >= 0.0
                              ? geometricNormal
                              : -geometricNormal;
    vec3 origin = offsetRayOrigin(position, outgoingNormal, originBias);
    float selfIntersectionLimit = originBias * 8.0;
    rayQueryEXT query;
    // Rays travel from the receiver toward the light, reversing the raster
    // shadow-map traversal direction. Front-face culling therefore matches the
    // raster pipeline's back-face culling convention.
    rayQueryInitializeEXT(query, sceneTlas,
                          gl_RayFlagsTerminateOnFirstHitEXT |
                              gl_RayFlagsNoOpaqueEXT |
                              gl_RayFlagsCullFrontFacingTrianglesEXT,
                          0xff, origin, originBias, lightDirection, 1.0e27);
    while (rayQueryProceedEXT(query)) {
        uint candidateInstance = rayQueryGetIntersectionInstanceCustomIndexEXT(query, false);
        bool nearSelfIntersection = candidateInstance == sceneInstanceIndex &&
                                    rayQueryGetIntersectionTEXT(query, false) <=
                                        selfIntersectionLimit;
        if (!nearSelfIntersection &&
            rayQueryGetIntersectionTypeEXT(query, false) ==
                gl_RayQueryCandidateIntersectionTriangleEXT &&
            rayCandidateBlocksLight(candidateInstance,
                                    rayQueryGetIntersectionPrimitiveIndexEXT(query, false),
                                    rayQueryGetIntersectionBarycentricsEXT(query, false))) {
            rayQueryConfirmIntersectionEXT(query);
        }
    }
    return rayQueryGetIntersectionTypeEXT(query, true) ==
                   gl_RayQueryCommittedIntersectionNoneEXT
               ? 1.0
               : 0.0;
#else
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
#endif
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
    vec4 baseColor = effectiveGltfBaseColor(
        material.baseColorFactor, sampledBaseColor, vertexColor);
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
    // Shared wetness staging darkens rain receivers before forming puddles.
    uint weatherMaterialFlags = material.textureState.z &
        (RAIN_RECEIVER_MATERIAL_BIT | WET_COAT_MATERIAL_BIT);
    if (weatherMaterialFlags != 0u && frameData.weatherParameters.w > 0.0) {
        GenesisWetnessLayers wetness = genesisRainWetnessLayers(
            worldPosition, normal, frameData.weatherParameters.w,
            frameData.weatherParameters.z);
        if ((weatherMaterialFlags & RAIN_RECEIVER_MATERIAL_BIT) != 0u) {
            baseColor.rgb *= mix(1.0, 0.48, wetness.dampness);
            baseColor.rgb *= mix(1.0, 0.72, wetness.poolingDampness);
            baseColor.rgb *= mix(1.0, 0.86, wetness.puddles);
            perceptualRoughness = mix(
                perceptualRoughness, 0.12, wetness.puddles);
        }
        if ((weatherMaterialFlags & WET_COAT_MATERIAL_BIT) != 0u) {
            baseColor.rgb *= mix(1.0, 0.92, wetness.coating);
            perceptualRoughness = mix(
                perceptualRoughness, 0.055, wetness.coating);
        }
    }
    vec3 viewDirection = normalize(frameData.cameraPositionExposure.xyz - worldPosition);
    vec3 materialColor = baseColor.rgb;

    vec3 directionalLight = normalize(-frameData.directionalDirectionIntensity.xyz);
    float cloudSunVisibility = genesisCloudSurfaceSunVisibility(
        worldPosition, frameData.atmosphereCamera,
        frameData.atmosphereSunDirection.xyz,
        frameData.cloudLayerParameters, frameData.cloudWeatherParameters);
    float cachedCloudEnergy = frameData.cloudWeatherParameters.w > 0.5
                                  ? textureLod(
                                        environmentMap,
                                        directionToDualParaboloidUv(directionalLight),
                                        0.0).a
                                  : 1.0;
    // Retain a subtle whole-sky energy response while the projected field owns
    // the visible shadow pattern. This keeps overcast illumination coherent
    // without recreating the previous uniform on/off shadow.
    cloudSunVisibility *= mix(0.90, 1.0, cachedCloudEnergy);
    vec3 directionalRadiance = frameData.directionalColor.rgb *
                                frameData.directionalDirectionIntensity.w *
                                cloudSunVisibility *
                                genesisCloudShadowTint(cloudSunVisibility);
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
    float atmosphereAmbientScale = frameData.atmosphereSunDirection.w > 0.5
                                       ? mix(0.025, 1.0,
                                             smoothstep(-0.16, 0.08,
                                                        frameData.atmosphereSunDirection.z))
                                       : 1.0;
    if (frameData.cloudWeatherParameters.w > 0.5) {
        atmosphereAmbientScale *= mix(
            1.0, 0.55,
            clamp(frameData.cloudLayerParameters.z *
                  frameData.cloudLayerParameters.w, 0.0, 1.0));
    }
    vec3 irradiance = texture(
        diffuseIrradianceMap,
        rotatedDirectionToLatLongUv(normal, environmentRotation)).rgb *
        atmosphereAmbientScale;
    vec3 diffuseIndirect = diffuseWeight * materialColor * irradiance / PBR_PI;

    vec3 reflection = reflect(-viewDirection, normal);
    float maximumLod = float(textureQueryLevels(prefilteredSpecularMap) - 1);
    vec3 prefiltered = textureLod(prefilteredSpecularMap,
                                  rotatedDirectionToLatLongUv(reflection,
                                                              environmentRotation),
                                  perceptualRoughness * maximumLod).rgb *
                       atmosphereAmbientScale;
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
    // UINT_MAX is never a valid resident scene instance (Vulkan custom indices
    // are 24-bit). Keeping this invariant in the surface marker also preserves
    // the vertex/fragment interface in the non-ray-query pipeline.
    float surfaceMarker = sceneInstanceIndex != 0xffffffffu ? 1.0 : 0.0;
    outNormal = vec4(normal * 0.5 + 0.5, surfaceMarker);
}
