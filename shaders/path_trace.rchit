#version 460
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require

// Transport structure and cosine-hemisphere sampling are adapted from pbrt-v4,
// Copyright (c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys;
// Apache-2.0. PBRT_PROVENANCE.md records the bounded realtime differences.

#include "include/pbr.glsl"
#include "include/ibl.glsl"
#include "include/path_trace.glsl"
#include "include/path_trace_bsdf.glsl"
#include "include/path_trace_environment.glsl"
#define GENESIS_CLOUD_BASE_NOISE_BINDING 27
#define GENESIS_CLOUD_DETAIL_NOISE_BINDING 28
#include "include/genesis_clouds.glsl"
#include "include/weather_wetness.glsl"

layout(set = 0, binding = 0) uniform accelerationStructureEXT sceneTlas;
#define LOOK_FRAME_BINDING 2
#include "include/look_frame.glsl"
#include "include/first_hit_ids.glsl"

layout(std430, set = 0, binding = 3) readonly buffer Vertices
{
    float vertexData[];
};
layout(std430, set = 0, binding = 4) readonly buffer Indices
{
    uint indices[];
};

struct InstanceRecord
{
    vec4 centerRadius;
    uvec4 meshMaterial;
    uvec4 visibilityLayer;
};
layout(std430, set = 0, binding = 5) readonly buffer Instances
{
    InstanceRecord instances[];
};

struct MaterialRecord
{
    vec4 baseColorFactor;
    vec4 emissiveAlphaCutoff;
    vec4 metallicRoughnessOcclusionNormal;
    uvec4 textureIndices;
    uvec4 textureState;
};
layout(std430, set = 0, binding = 6) readonly buffer Materials
{
    MaterialRecord materials[];
};

struct TransformRecord
{
    mat4 world;
    mat4 normalToWorld;
    mat4 previousWorld;
};
layout(std430, set = 0, binding = 7) readonly buffer Transforms
{
    TransformRecord transforms[];
};
layout(std430, set = 0, binding = 8) readonly buffer RayGeometries
{
    uvec4 rayGeometryRanges[];
};

struct EmissiveTriangleRecord
{
    uvec4 identity;
    vec4 sampling;
};
layout(std430, set = 0, binding = 11) readonly buffer EmissiveTriangles
{
    uvec4 emissiveTriangleHeader;
    EmissiveTriangleRecord emissiveTriangles[];
};
layout(set = 0, binding = 12, rgba32f) uniform image2D directRadianceImage;
layout(set = 0, binding = 13, rgba16f) uniform image2D surfaceAovImage;
layout(set = 0, binding = 14, rgba16f) uniform image2D normalAovImage;
layout(set = 0, binding = 15, rgba32f) uniform image2D motionDepthAovImage;
layout(set = 0, binding = 16, r32f) uniform image2D depthAovImage;
layout(set = 0, binding = 17, rgba16f) uniform image2D nrdDiffuseInputImage;
layout(set = 0, binding = 18, rgba16f) uniform image2D nrdSpecularInputImage;
layout(set = 0, binding = 19, rgba16f) uniform image2D nrdMaterialImage;
layout(set = 0, binding = 20, rgba16f) uniform image2D rrDiffuseAlbedoImage;
layout(set = 0, binding = 21, rgba16f) uniform image2D rrSpecularAlbedoImage;
layout(set = 0, binding = 22, rgba16f) uniform image2D rrNormalRoughnessImage;
layout(set = 0, binding = 23, r32f) uniform image2D rrSpecularHitDistanceImage;
layout(set = 1, binding = 0) uniform sampler2D materialTextures[];

layout(location = 0) rayPayloadInEXT RadiancePayload payload;
layout(location = 1) rayPayloadEXT float shadowVisibility;
hitAttributeEXT vec2 hitBarycentrics;

layout(push_constant) uniform PathTracePushConstants
{
    uint sampleIndex;
    uint sampleInFrame;
    uint samplesThisFrame;
    uint fixedPrimarySample;
    uvec2 virtualState;
    uvec2 virtualPixels;
} controls;

#include "include/virtual_texture.glsl"

const uint BASE_COLOR_TEXTURE_BIT = 1u << 0u;
const uint METALLIC_ROUGHNESS_TEXTURE_BIT = 1u << 1u;
const uint NORMAL_TEXTURE_BIT = 1u << 2u;
const uint OCCLUSION_TEXTURE_BIT = 1u << 3u;
const uint EMISSIVE_TEXTURE_BIT = 1u << 4u;
const uint MIRRORED_TRANSFORM_INSTANCE_BIT = 1u << 0u;
const uint RAIN_RECEIVER_MATERIAL_BIT = 1u << 4u;
const uint WET_COAT_MATERIAL_BIT = 1u << 5u;

struct PathTriangleSurface
{
    vec3 worldPositions[3];
    vec2 uv0[3];
    vec2 uv1[3];
};

vec3 vertexPosition(uint vertexIndex)
{
    uint firstComponent = vertexIndex * 18u;
    return vec3(vertexData[firstComponent], vertexData[firstComponent + 1u],
                vertexData[firstComponent + 2u]);
}

vec3 vertexNormal(uint vertexIndex)
{
    uint firstComponent = vertexIndex * 18u + 3u;
    return vec3(vertexData[firstComponent], vertexData[firstComponent + 1u],
                vertexData[firstComponent + 2u]);
}

vec2 vertexUv(uint vertexIndex, bool useUv1)
{
    uint firstComponent = vertexIndex * 18u + (useUv1 ? 12u : 6u);
    return vec2(vertexData[firstComponent], vertexData[firstComponent + 1u]);
}

vec4 vertexTangent(uint vertexIndex)
{
    uint firstComponent = vertexIndex * 18u + 8u;
    return vec4(vertexData[firstComponent], vertexData[firstComponent + 1u],
                vertexData[firstComponent + 2u], vertexData[firstComponent + 3u]);
}

vec4 vertexColor(uint vertexIndex)
{
    uint firstComponent = vertexIndex * 18u + 14u;
    return vec4(vertexData[firstComponent], vertexData[firstComponent + 1u],
                vertexData[firstComponent + 2u], vertexData[firstComponent + 3u]);
}

vec2 pathTriangleUv(PathTriangleSurface surface, int vertexIndex, bool useUv1)
{
    return useUv1 ? surface.uv1[vertexIndex] : surface.uv0[vertexIndex];
}

float pathTextureLod(uint textureIndex, PathTriangleSurface surface,
                     bool useUv1, float coneWidth)
{
    vec2 uv0 = pathTriangleUv(surface, 0, useUv1);
    vec2 uv1 = pathTriangleUv(surface, 1, useUv1);
    vec2 uv2 = pathTriangleUv(surface, 2, useUv1);
    float ratio01 = length(uv1 - uv0) /
                    max(length(surface.worldPositions[1] - surface.worldPositions[0]), 1.0e-7);
    float ratio12 = length(uv2 - uv1) /
                    max(length(surface.worldPositions[2] - surface.worldPositions[1]), 1.0e-7);
    float ratio20 = length(uv0 - uv2) /
                    max(length(surface.worldPositions[0] - surface.worldPositions[2]), 1.0e-7);
    float uvPerWorldUnit = max(ratio01, max(ratio12, ratio20));
    ivec2 dimensions = textureSize(
        materialTextures[nonuniformEXT(textureIndex)], 0);
    float textureExtent = float(max(dimensions.x, dimensions.y));
    float footprintInTexels = max(coneWidth * uvPerWorldUnit * textureExtent, 1.0);
    float maximumLod = float(max(
        textureQueryLevels(materialTextures[nonuniformEXT(textureIndex)]) - 1, 0));
    return clamp(log2(footprintInTexels), 0.0, maximumLod);
}

vec4 samplePathMaterialTexture(uint textureIndex, uint textureBit,
                               MaterialRecord material, PathTriangleSurface surface,
                               vec3 weights, float coneWidth)
{
    bool useUv1 = (material.textureState.w & textureBit) != 0u;
    vec2 uv = pathTriangleUv(surface, 0, useUv1) * weights.x +
              pathTriangleUv(surface, 1, useUv1) * weights.y +
              pathTriangleUv(surface, 2, useUv1) * weights.z;
    if (textureBit == BASE_COLOR_TEXTURE_BIT && any(notEqual(controls.virtualState,uvec2(0)))) {
        VirtualState vt = VirtualState(controls.virtualState);
        vec2 dimensions = vec2(vt.words[0], vt.words[1]);
        float footprint = 0.0;
        for (int i=0;i<3;++i) {
            int j=(i+1)%3;
            vec2 delta = pathTriangleUv(surface,i,useUv1)-pathTriangleUv(surface,j,useUv1);
            float lengthWorld = max(length(surface.worldPositions[i]-surface.worldPositions[j]),1e-7);
            footprint=max(footprint,coneWidth*length(delta*dimensions)/lengthWorld);
        }
        return sampleVirtual(vt, uv, log2(max(1.0,footprint)));
    }
    float lod = pathTextureLod(textureIndex, surface, useUv1, coneWidth);
    return textureLod(materialTextures[nonuniformEXT(textureIndex)], uv, lod);
}

PathTriangleSurface loadPathTriangleSurface(uint instanceIndex, uint triangleIndex,
                                            out InstanceRecord instance,
                                            out MaterialRecord material)
{
    instance = instances[instanceIndex];
    material = materials[instance.meshMaterial.y];
    uvec4 geometry = rayGeometryRanges[instance.meshMaterial.x];
    uint triangleFirstIndex = geometry.x + triangleIndex * 3u;
    uvec3 triangle = uvec3(indices[triangleFirstIndex], indices[triangleFirstIndex + 1u],
                           indices[triangleFirstIndex + 2u]) + geometry.y;
    TransformRecord transform = transforms[instance.meshMaterial.z];
    PathTriangleSurface surface;
    surface.worldPositions[0] = vec3(transform.world * vec4(vertexPosition(triangle.x), 1.0));
    surface.worldPositions[1] = vec3(transform.world * vec4(vertexPosition(triangle.y), 1.0));
    surface.worldPositions[2] = vec3(transform.world * vec4(vertexPosition(triangle.z), 1.0));
    surface.uv0[0] = vertexUv(triangle.x, false);
    surface.uv0[1] = vertexUv(triangle.y, false);
    surface.uv0[2] = vertexUv(triangle.z, false);
    surface.uv1[0] = vertexUv(triangle.x, true);
    surface.uv1[1] = vertexUv(triangle.y, true);
    surface.uv1[2] = vertexUv(triangle.z, true);
    return surface;
}

vec3 pathEmissiveRadiance(MaterialRecord material, PathTriangleSurface surface,
                          vec3 weights, float coneWidth)
{
    vec3 emissive = max(material.emissiveAlphaCutoff.rgb, vec3(0.0));
    if ((material.textureState.y & EMISSIVE_TEXTURE_BIT) != 0u) {
        emissive *= samplePathMaterialTexture(
            material.textureState.x, EMISSIVE_TEXTURE_BIT, material,
            surface, weights, coneWidth).rgb;
    }
    return pathTraceFinite(emissive) ? max(emissive, vec3(0.0)) : vec3(0.0);
}

struct PathEmissiveSample
{
    vec3 direction;
    vec3 radiance;
    float distance;
    float pdf;
};

PathEmissiveSample emptyPathEmissiveSample()
{
    PathEmissiveSample result;
    result.direction = vec3(0.0);
    result.radiance = vec3(0.0);
    result.distance = 0.0;
    result.pdf = 0.0;
    return result;
}

PathEmissiveSample samplePathEmissiveTriangle(vec3 position, float coneWidth,
                                              float coneSpread, inout uint state)
{
    uint lightCount = emissiveTriangleHeader.x;
    if (lightCount == 0u) {
        return emptyPathEmissiveSample();
    }

    float selectionSample = pathTraceRandom(state);
    uint lower = 0u;
    uint upper = lightCount;
    while (lower < upper) {
        uint middle = lower + (upper - lower) / 2u;
        if (selectionSample < emissiveTriangles[middle].sampling.x) {
            upper = middle;
        } else {
            lower = middle + 1u;
        }
    }
    uint lightIndex = min(lower, lightCount - 1u);
    EmissiveTriangleRecord light = emissiveTriangles[lightIndex];
    if (light.sampling.y <= 0.0 || light.sampling.z <= 0.0) {
        return emptyPathEmissiveSample();
    }

    InstanceRecord lightInstance;
    MaterialRecord lightMaterial;
    PathTriangleSurface lightSurface = loadPathTriangleSurface(
        light.identity.x, light.identity.y, lightInstance, lightMaterial);
    float root = sqrt(pathTraceRandom(state));
    float edgeSample = pathTraceRandom(state);
    vec3 weights = vec3(1.0 - root, root * (1.0 - edgeSample), root * edgeSample);
    vec3 lightPosition = lightSurface.worldPositions[0] * weights.x +
                         lightSurface.worldPositions[1] * weights.y +
                         lightSurface.worldPositions[2] * weights.z;
    vec3 toLight = lightPosition - position;
    float distanceSquared = dot(toLight, toLight);
    if (distanceSquared <= 1.0e-12) {
        return emptyPathEmissiveSample();
    }
    float distance = sqrt(distanceSquared);
    vec3 direction = toLight / distance;
    vec3 lightNormal = cross(lightSurface.worldPositions[1] - lightSurface.worldPositions[0],
                             lightSurface.worldPositions[2] - lightSurface.worldPositions[0]);
    float normalLengthSquared = dot(lightNormal, lightNormal);
    if (normalLengthSquared <= 1.0e-12) {
        return emptyPathEmissiveSample();
    }
    lightNormal *= inversesqrt(normalLengthSquared);
    float lightCosine = abs(dot(lightNormal, -direction));
    float pdf = pathAreaLightSolidAnglePdf(
        light.sampling.y, light.sampling.z, distanceSquared, lightCosine);
    if (pdf <= 0.0) {
        return emptyPathEmissiveSample();
    }

    PathEmissiveSample result;
    result.direction = direction;
    result.radiance = pathEmissiveRadiance(
        lightMaterial, lightSurface, weights,
        max(coneWidth + distance * coneSpread, 1.0e-7));
    result.distance = distance;
    result.pdf = pdf;
    return result;
}

vec2 pathEncodeOctahedralNormal(vec3 value)
{
    vec3 normal = value / max(abs(value.x) + abs(value.y) + abs(value.z), 1.0e-7);
    if (normal.z < 0.0) {
        vec2 signs = vec2(normal.x >= 0.0 ? 1.0 : -1.0,
                          normal.y >= 0.0 ? 1.0 : -1.0);
        normal.xy = (vec2(1.0) - abs(normal.yx)) * signs;
    }
    return normal.xy * 0.5 + 0.5;
}

void writePathFirstHitAovs(vec3 objectPosition, TransformRecord transform,
                           vec3 baseColor, float roughness, vec3 geometricNormal,
                           vec3 shadingNormal)
{
    ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);
    imageStore(surfaceAovImage, pixel,
               vec4(clamp(baseColor, vec3(0.0), vec3(1.0)),
                    clamp(roughness, 0.0, 1.0)));
    imageStore(normalAovImage, pixel,
               vec4(pathEncodeOctahedralNormal(geometricNormal),
                    pathEncodeOctahedralNormal(shadingNormal)));

    vec3 worldPosition = vec3(transform.world * vec4(objectPosition, 1.0));
    vec3 previousWorldPosition =
        vec3(transform.previousWorld * vec4(objectPosition, 1.0));
    vec4 currentClip = frameData.unjitteredViewProjection * vec4(worldPosition, 1.0);
    vec4 previousClip =
        frameData.previousUnjitteredViewProjection * vec4(previousWorldPosition, 1.0);
    vec2 motion = vec2(0.0);
    float currentDepth = 1.0;
    float previousDepth = -1.0;
    if (abs(currentClip.w) > 1.0e-7 && abs(previousClip.w) > 1.0e-7 &&
        !any(isnan(currentClip)) && !any(isinf(currentClip)) &&
        !any(isnan(previousClip)) && !any(isinf(previousClip))) {
        vec3 currentNdc = currentClip.xyz / currentClip.w;
        vec3 previousNdc = previousClip.xyz / previousClip.w;
        motion = (currentNdc.xy - previousNdc.xy) * 0.5;
        currentDepth = clamp(currentNdc.z, 0.0, 1.0);
        previousDepth = clamp(previousNdc.z, 0.0, 1.0);
    }
    imageStore(motionDepthAovImage, pixel,
               vec4(motion, previousDepth, max(gl_HitTEXT, 0.0)));
    imageStore(depthAovImage, pixel, vec4(currentDepth));
}

void writePathDirectRadianceAov(vec3 directRadiance)
{
    ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);
    uint sampleIndex = controls.sampleIndex;
    // Ray Reconstruction consumes a fresh noisy signal every frame. Keep its
    // direct component equally fresh so total - direct remains a valid
    // indirect-light candidate for the reuse pass.
    if (controls.fixedPrimarySample < 2u && sampleIndex > 0u) {
        float count = float(sampleIndex);
        directRadiance =
            (imageLoad(directRadianceImage, pixel).rgb * count + directRadiance) /
            (count + 1.0);
    }
    imageStore(directRadianceImage, pixel,
               vec4(max(directRadiance, vec3(0.0)), 1.0));
}

void main()
{
    uint currentDepth = payload.depth;
    if (currentDepth == ID_ONLY_RAY_DEPTH) {
        writeFirstHitIds(instances[gl_InstanceCustomIndexEXT].meshMaterial.zy + uvec2(1u));
        return;
    }
    if (currentDepth == 0u) {
        payload.primaryHitDistance = max(gl_HitTEXT, 0.0);
    }
    if (currentDepth == 1u) {
        payload.firstLobeHitDistance = max(gl_HitTEXT, 0.0);
    }
    InstanceRecord instance = instances[gl_InstanceCustomIndexEXT];
    MaterialRecord material = materials[instance.meshMaterial.y];
    uvec4 geometry = rayGeometryRanges[instance.meshMaterial.x];
    uint triangleFirstIndex = geometry.x + gl_PrimitiveID * 3u;
    uvec3 triangle = uvec3(indices[triangleFirstIndex], indices[triangleFirstIndex + 1u],
                           indices[triangleFirstIndex + 2u]) + geometry.y;
    vec3 weights = vec3(1.0 - hitBarycentrics.x - hitBarycentrics.y, hitBarycentrics);
    TransformRecord transform = transforms[instance.meshMaterial.z];
    PathTriangleSurface surface;
    surface.worldPositions[0] = vec3(transform.world * vec4(vertexPosition(triangle.x), 1.0));
    surface.worldPositions[1] = vec3(transform.world * vec4(vertexPosition(triangle.y), 1.0));
    surface.worldPositions[2] = vec3(transform.world * vec4(vertexPosition(triangle.z), 1.0));
    surface.uv0[0] = vertexUv(triangle.x, false);
    surface.uv0[1] = vertexUv(triangle.y, false);
    surface.uv0[2] = vertexUv(triangle.z, false);
    surface.uv1[0] = vertexUv(triangle.x, true);
    surface.uv1[1] = vertexUv(triangle.y, true);
    surface.uv1[2] = vertexUv(triangle.z, true);

    vec3 objectNormal = vertexNormal(triangle.x) * weights.x +
                        vertexNormal(triangle.y) * weights.y +
                        vertexNormal(triangle.z) * weights.z;
    vec3 objectPosition = vertexPosition(triangle.x) * weights.x +
                          vertexPosition(triangle.y) * weights.y +
                          vertexPosition(triangle.z) * weights.z;
    vec3 normal = normalize(mat3(transform.normalToWorld) * objectNormal);
    vec3 viewDirection = normalize(-gl_WorldRayDirectionEXT);
    normal = dot(normal, viewDirection) < 0.0 ? -normal : normal;
    vec3 position = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    vec3 triangleNormal = cross(surface.worldPositions[1] - surface.worldPositions[0],
                                surface.worldPositions[2] - surface.worldPositions[0]);
    float geometricLengthSquared = dot(triangleNormal, triangleNormal);
    triangleNormal = geometricLengthSquared > 1.0e-12
                         ? triangleNormal * inversesqrt(geometricLengthSquared)
                         : normal;
    vec3 geometricNormal = triangleNormal;
    geometricNormal = dot(geometricNormal, viewDirection) < 0.0
                          ? -geometricNormal
                          : geometricNormal;

    float sceneScale = max(frameData.shadowCascadeDepthRadii.w, 1.0);
    float rayConeWidth = max(payload.coneWidth + gl_HitTEXT * payload.coneSpread,
                             sceneScale * 1.0e-7);
    uint textureMask = material.textureState.y;

    vec4 sampledBaseColor = vec4(1.0);
    if ((textureMask & BASE_COLOR_TEXTURE_BIT) != 0u) {
        sampledBaseColor = samplePathMaterialTexture(
            material.textureIndices.x, BASE_COLOR_TEXTURE_BIT, material,
            surface, weights, rayConeWidth);
    }
    vec4 interpolatedVertexColor = vertexColor(triangle.x) * weights.x +
                                   vertexColor(triangle.y) * weights.y +
                                   vertexColor(triangle.z) * weights.z;
    vec3 baseColor = effectiveGltfBaseColor(
        material.baseColorFactor, sampledBaseColor, interpolatedVertexColor).rgb;

    float metallic = material.metallicRoughnessOcclusionNormal.x;
    float roughness = material.metallicRoughnessOcclusionNormal.y;
    if ((textureMask & METALLIC_ROUGHNESS_TEXTURE_BIT) != 0u) {
        vec4 sampledMetallicRoughness = samplePathMaterialTexture(
            material.textureIndices.y, METALLIC_ROUGHNESS_TEXTURE_BIT, material,
            surface, weights, rayConeWidth);
        metallic *= sampledMetallicRoughness.b;
        roughness *= sampledMetallicRoughness.g;
    }
    metallic = clamp(metallic, 0.0, 1.0);
    roughness = clamp(roughness, 0.0, 1.0);

    vec4 objectTangent = vertexTangent(triangle.x) * weights.x +
                         vertexTangent(triangle.y) * weights.y +
                         vertexTangent(triangle.z) * weights.z;
    vec3 tangent = mat3(transform.world) * objectTangent.xyz;
    tangent -= normal * dot(normal, tangent);
    float tangentLengthSquared = dot(tangent, tangent);
    vec3 fallbackBitangent;
    if (tangentLengthSquared > 1.0e-12) {
        tangent *= inversesqrt(tangentLengthSquared);
    } else {
        pathTraceBasis(normal, tangent, fallbackBitangent);
    }
    float tangentHandedness = objectTangent.w < 0.0 ? -1.0 : 1.0;
    if ((instance.meshMaterial.w & MIRRORED_TRANSFORM_INSTANCE_BIT) != 0u) {
        tangentHandedness = -tangentHandedness;
    }
    if ((textureMask & NORMAL_TEXTURE_BIT) != 0u) {
        vec3 tangentNormal = samplePathMaterialTexture(
            material.textureIndices.z, NORMAL_TEXTURE_BIT, material,
            surface, weights, rayConeWidth).xyz * 2.0 - 1.0;
        tangentNormal.xy *= material.metallicRoughnessOcclusionNormal.w;
        vec3 bitangent = cross(normal, tangent) * tangentHandedness;
        vec3 mappedNormal = normalize(mat3(tangent, bitangent, normal) * tangentNormal);
        normal = dot(mappedNormal, geometricNormal) > 0.0 ? mappedNormal : geometricNormal;
    }

    // Shared wetness staging darkens rain receivers before forming puddles.
    uint weatherMaterialFlags = material.textureState.z &
        (RAIN_RECEIVER_MATERIAL_BIT | WET_COAT_MATERIAL_BIT);
    if (weatherMaterialFlags != 0u && frameData.weatherParameters.w > 0.0) {
        GenesisWetnessLayers wetness = genesisRainWetnessLayers(
            position, normal, frameData.weatherParameters.w,
            frameData.weatherParameters.z);
        if ((weatherMaterialFlags & RAIN_RECEIVER_MATERIAL_BIT) != 0u) {
            baseColor *= mix(1.0, 0.48, wetness.dampness);
            baseColor *= mix(1.0, 0.72, wetness.poolingDampness);
            baseColor *= mix(1.0, 0.86, wetness.puddles);
            roughness = mix(roughness, 0.12, wetness.puddles);
        }
        if ((weatherMaterialFlags & WET_COAT_MATERIAL_BIT) != 0u) {
            baseColor *= mix(1.0, 0.92, wetness.coating);
            roughness = mix(roughness, 0.055, wetness.coating);
        }
    }

    if (frameData.lookOptions.x != 0u) {
        baseColor = vec3(clayAlbedo);
        metallic = 0.0;
        roughness = clayRoughness;
    }
    if (currentDepth == 0u && controls.sampleInFrame + 1u == controls.samplesThisFrame) {
        writePathFirstHitAovs(objectPosition, transform, baseColor, roughness,
                              geometricNormal, normal);
    }

    vec3 emissive = pathEmissiveRadiance(material, surface, weights, rayConeWidth);
    float emissiveMisWeight = 1.0;
    if (currentDepth > 0u && gl_PrimitiveID < instance.visibilityLayer.w &&
        instance.visibilityLayer.z < emissiveTriangleHeader.x) {
        uint lightIndex = instance.visibilityLayer.z + gl_PrimitiveID;
        if (lightIndex < emissiveTriangleHeader.x) {
            EmissiveTriangleRecord light = emissiveTriangles[lightIndex];
            float lightPdf = pathAreaLightSolidAnglePdf(
                light.sampling.y, light.sampling.z, gl_HitTEXT * gl_HitTEXT,
                abs(dot(triangleNormal, -gl_WorldRayDirectionEXT)));
            emissiveMisWeight = pathPowerHeuristic(payload.bouncePdf, lightPdf);
        }
    }
    vec3 directDiffuseRadiance = emissive * emissiveMisWeight;
    vec3 directSpecularRadiance = vec3(0.0);
    vec3 indirectDiffuseRadiance = vec3(0.0);
    vec3 indirectSpecularRadiance = vec3(0.0);
    float rayBias = sceneScale * 5.0e-4;
    uint state = payload.seed;
    uint maxBounces = max(1u, uint(frameData.environmentParameters.z + 0.5));
    uint russianRouletteStartDepth =
        max(1u, uint(frameData.environmentParameters.w + 0.5));
    bool canContinue = currentDepth < maxBounces;
    uint sampledContinuationLobe = 2u;

    vec3 lightDirection = normalize(-frameData.directionalDirectionIntensity.xyz);
    float cloudSunVisibility = genesisCloudSurfaceSunVisibility(
        position, frameData.atmosphereCamera,
        frameData.atmosphereSunDirection.xyz,
        frameData.cloudLayerParameters, frameData.cloudWeatherParameters);
    float directionalIntensity = max(frameData.directionalDirectionIntensity.w, 0.0) *
                                 cloudSunVisibility;
    float directionalNoL = max(dot(normal, lightDirection), 0.0);
    if (directionalIntensity > 0.0 && directionalNoL > 0.0) {
        shadowVisibility = 1.0;
        if (frameData.shadowCascadeSplits.w > 0.0) {
            shadowVisibility = 0.0;
            traceRayEXT(sceneTlas,
                        gl_RayFlagsTerminateOnFirstHitEXT |
                            gl_RayFlagsSkipClosestHitShaderEXT,
                        0xff, 1, 2, 1, position + geometricNormal * rayBias,
                        rayBias, lightDirection, 1.0e27, 1);
        }
        vec3 incident = frameData.directionalColor.rgb * directionalIntensity *
                        genesisCloudShadowTint(cloudSunVisibility);
        vec3 diffuseBrdf;
        vec3 specularBrdf;
        evaluateMetallicRoughnessBrdfLobes(
            baseColor, metallic, roughness, normal, viewDirection, lightDirection,
            diffuseBrdf, specularBrdf);
        vec3 visibleIncident =
            shadowVisibility * max(incident, vec3(0.0)) * directionalNoL;
        directDiffuseRadiance += visibleIncident * diffuseBrdf;
        directSpecularRadiance += visibleIncident * specularBrdf;
    }

    PathEmissiveSample emissiveSample = samplePathEmissiveTriangle(
        position, rayConeWidth, payload.coneSpread, state);
    float emissiveNoL = dot(normal, emissiveSample.direction);
    if (emissiveSample.pdf > 0.0 && emissiveNoL > 0.0 &&
        dot(geometricNormal, emissiveSample.direction) > 0.0 &&
        emissiveSample.distance > 2.0 * rayBias &&
        any(greaterThan(emissiveSample.radiance, vec3(0.0)))) {
        shadowVisibility = 0.0;
        traceRayEXT(sceneTlas,
                    gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                    0xff, 1, 2, 1, position + geometricNormal * rayBias,
                    rayBias, emissiveSample.direction,
                    emissiveSample.distance - rayBias, 1);
        float misWeight = 1.0;
        if (canContinue) {
            float bsdfPdf = pathBsdfPdf(
                baseColor, metallic, roughness, normal, viewDirection,
                emissiveSample.direction);
            misWeight = pathPowerHeuristic(emissiveSample.pdf, bsdfPdf);
        }
        vec3 emissiveDiffuseBrdf;
        vec3 emissiveSpecularBrdf;
        evaluateMetallicRoughnessBrdfLobes(
            baseColor, metallic, roughness, normal, viewDirection,
            emissiveSample.direction, emissiveDiffuseBrdf, emissiveSpecularBrdf);
        vec3 emissiveScale = shadowVisibility * emissiveSample.radiance *
                             (emissiveNoL * misWeight / emissiveSample.pdf);
        directDiffuseRadiance += emissiveScale * emissiveDiffuseBrdf;
        directSpecularRadiance += emissiveScale * emissiveSpecularBrdf;
    }

    PathEnvironmentSample environmentSample = samplePathEnvironment(
        state, frameData.environmentParameters.x,
        frameData.environmentParameters.y > 0.5,
        max(frameData.pointColorRange.w, 0.0),
        frameData.atmosphereSunDirection.w > 0.5);
    float environmentNoL = dot(normal, environmentSample.direction);
    if (environmentSample.pdf > 0.0 && environmentNoL > 0.0 &&
        dot(geometricNormal, environmentSample.direction) > 0.0) {
        shadowVisibility = 0.0;
        traceRayEXT(sceneTlas,
                    gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                    0xff, 1, 2, 1, position + geometricNormal * rayBias,
                    rayBias, environmentSample.direction, 1.0e27, 1);
        float misWeight = 1.0;
        if (canContinue) {
            float bsdfPdf = pathBsdfPdf(
                baseColor, metallic, roughness, normal, viewDirection,
                environmentSample.direction);
            misWeight = pathPowerHeuristic(environmentSample.pdf, bsdfPdf);
        }
        vec3 environmentDiffuseBrdf;
        vec3 environmentSpecularBrdf;
        evaluateMetallicRoughnessBrdfLobes(
            baseColor, metallic, roughness, normal, viewDirection,
            environmentSample.direction, environmentDiffuseBrdf, environmentSpecularBrdf);
        vec3 environmentScale = shadowVisibility * environmentSample.radiance *
                                (environmentNoL * misWeight / environmentSample.pdf);
        directDiffuseRadiance += environmentScale * environmentDiffuseBrdf;
        directSpecularRadiance += environmentScale * environmentSpecularBrdf;
    }
    payload.seed = state;

    if (canContinue) {
        PathBsdfSample bounce = samplePathBsdf(
            baseColor, metallic, roughness, normal, viewDirection, state);
        payload.seed = state;
        if (bounce.pdf > 0.0 && dot(bounce.direction, geometricNormal) > 0.0) {
            sampledContinuationLobe = bounce.sampledSpecular;
            uint nextDepth = currentDepth + 1u;
            // Ray visibility already accounts for occlusion. Multiplying a glTF
            // baked AO approximation here would attenuate real light again at
            // every bounce and bias the transport progressively darker.
            vec3 continuationThroughput = bounce.throughput;
            vec3 nextPathThroughput = payload.pathThroughput * continuationThroughput;
            float survivalProbability = 1.0;
            bool survived = true;
            if (nextDepth > russianRouletteStartDepth) {
                survivalProbability =
                    pathRussianRouletteSurvivalProbability(nextPathThroughput);
                survived = survivalProbability > 0.0 &&
                           pathTraceRandom(state) < survivalProbability;
            }
            if (survived) {
                continuationThroughput /= survivalProbability;
                nextPathThroughput /= survivalProbability;
                payload.depth = nextDepth;
                payload.seed = state;
                payload.radiance = vec3(0.0);
                payload.pathThroughput = nextPathThroughput;
                payload.coneWidth = rayConeWidth;
                payload.coneSpread = max(payload.coneSpread, bounce.coneSpread);
                payload.bouncePdf = bounce.pdf;
                traceRayEXT(sceneTlas, gl_RayFlagsNoneEXT, 0xff, 0, 2, 0,
                            position + geometricNormal * rayBias, rayBias,
                            bounce.direction, 1.0e27, 0);
                if (pathTraceFinite(payload.radiance)) {
                    vec3 returnedRadiance = continuationThroughput * payload.radiance;
                    if (bounce.sampledSpecular != 0u) {
                        indirectSpecularRadiance += returnedRadiance;
                    } else {
                        indirectDiffuseRadiance += returnedRadiance;
                    }
                }
            }
        }
    }
    // Use one energy regime during motion and stationary accumulation. The old
    // 0.2 -> 12.0 sample-index switch made sunlit shadow boundaries pulse as
    // the camera changed state. This stable realtime bound still rejects rare
    // low-PDF HDR fireflies without changing the lighting estimator over time.
    const float maximumSurfaceContributionLuminance = 4.0;
    vec3 directRadiance = directDiffuseRadiance + directSpecularRadiance;
    vec3 clampedDirectRadiance = clampPathRadianceContribution(
        directRadiance, maximumSurfaceContributionLuminance);
    float directLuminance = pathTraceLuminance(max(directRadiance, vec3(0.0)));
    float directScale = directLuminance > 1.0e-7
        ? pathTraceLuminance(clampedDirectRadiance) / directLuminance
        : 0.0;
    directDiffuseRadiance *= directScale;
    directSpecularRadiance *= directScale;
    directRadiance = clampedDirectRadiance;

    vec3 indirectRadiance = indirectDiffuseRadiance + indirectSpecularRadiance;
    vec3 clampedIndirectRadiance = clampPathRadianceContribution(
        indirectRadiance, maximumSurfaceContributionLuminance);
    float indirectLuminance = pathTraceLuminance(max(indirectRadiance, vec3(0.0)));
    float indirectScale = indirectLuminance > 1.0e-7
        ? pathTraceLuminance(clampedIndirectRadiance) / indirectLuminance
        : 0.0;
    indirectDiffuseRadiance *= indirectScale;
    indirectSpecularRadiance *= indirectScale;
    indirectRadiance = clampedIndirectRadiance;
    if (currentDepth == 0u) {
        writePathDirectRadianceAov(directRadiance);
        vec3 diffuseFactor = max(baseColor * (1.0 - metallic), vec3(0.04));
        vec3 specularFactor = max(mix(vec3(0.04), baseColor, metallic), vec3(0.04));
        vec3 diffuseSignal = clamp(
            (directDiffuseRadiance + indirectDiffuseRadiance) / diffuseFactor,
            vec3(0.0), vec3(65504.0));
        vec3 specularSignal = clamp(
            (directSpecularRadiance + indirectSpecularRadiance) / specularFactor,
            vec3(0.0), vec3(65504.0));
        float lobeHitDistance = clamp(payload.firstLobeHitDistance, 0.0, 65504.0);
        float diffuseHitDistance = sampledContinuationLobe == 0u ? lobeHitDistance : 0.0;
        float specularHitDistance = sampledContinuationLobe == 1u ? lobeHitDistance : 0.0;
        ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);
        if (controls.sampleInFrame > 0u) {
            float count = float(controls.sampleInFrame);
            vec4 previousDiffuse = imageLoad(nrdDiffuseInputImage, pixel);
            vec4 previousSpecular = imageLoad(nrdSpecularInputImage, pixel);
            diffuseSignal = (previousDiffuse.rgb * count + diffuseSignal) / (count + 1.0);
            specularSignal = (previousSpecular.rgb * count + specularSignal) / (count + 1.0);
            diffuseHitDistance = max(previousDiffuse.a, diffuseHitDistance);
            specularHitDistance = max(previousSpecular.a, specularHitDistance);
        }
        imageStore(nrdDiffuseInputImage, pixel,
                   vec4(diffuseSignal, diffuseHitDistance));
        imageStore(nrdSpecularInputImage, pixel,
                   vec4(specularSignal, specularHitDistance));
        imageStore(nrdMaterialImage, pixel, vec4(baseColor, metallic));
        imageStore(rrDiffuseAlbedoImage, pixel,
                   vec4(clamp(baseColor * (1.0 - metallic), vec3(0.0), vec3(1.0)), 1.0));
        imageStore(rrSpecularAlbedoImage, pixel,
                   vec4(clamp(mix(vec3(0.04), baseColor, metallic),
                              vec3(0.0), vec3(1.0)), 1.0));
        imageStore(rrNormalRoughnessImage, pixel,
                   vec4(normalize(normal), roughness));
        imageStore(rrSpecularHitDistanceImage, pixel, vec4(specularHitDistance));
    }
    payload.radiance = directRadiance + indirectRadiance;
}
