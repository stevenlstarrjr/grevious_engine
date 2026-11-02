#version 460

#include "include/post.glsl"

layout(set = 0, binding = 0) uniform sampler2D hdrColor;
#define LOOK_FRAME_BINDING 1
#include "include/look_frame.glsl"
layout(set = 0, binding = 2) uniform sampler2D bloomColor;
layout(set = 0, binding = 3) uniform sampler2D motionDepth;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

vec3 decodeOctahedralNormal(vec2 encoded)
{
    vec2 value = encoded * 2.0 - 1.0;
    vec3 normal = vec3(value, 1.0 - abs(value.x) - abs(value.y));
    if (normal.z < 0.0) {
        vec2 signs = vec2(normal.x >= 0.0 ? 1.0 : -1.0,
                          normal.y >= 0.0 ? 1.0 : -1.0);
        normal.xy = (vec2(1.0) - abs(normal.yx)) * signs;
    }
    return normalize(normal);
}

float luminance(vec3 color)
{
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 sharpenTemporalColor(vec2 coordinates, vec3 center)
{
    vec3 north = textureOffset(hdrColor, coordinates, ivec2(0, -1)).rgb;
    vec3 south = textureOffset(hdrColor, coordinates, ivec2(0, 1)).rgb;
    vec3 west = textureOffset(hdrColor, coordinates, ivec2(-1, 0)).rgb;
    vec3 east = textureOffset(hdrColor, coordinates, ivec2(1, 0)).rgb;
    vec3 neighborhoodMinimum = min(center, min(min(north, south), min(west, east)));
    vec3 neighborhoodMaximum = max(center, max(max(north, south), max(west, east)));
    vec3 neighborhoodAverage = (north + south + west + east) * 0.25;

    float minimumLuminance = min(luminance(center),
                                 min(min(luminance(north), luminance(south)),
                                     min(luminance(west), luminance(east))));
    float maximumLuminance = max(luminance(center),
                                 max(max(luminance(north), luminance(south)),
                                     max(luminance(west), luminance(east))));
    float relativeContrast = (maximumLuminance - minimumLuminance) /
                             max(maximumLuminance, 0.10);
    float strength = mix(0.08, 0.22, smoothstep(0.02, 0.30, relativeContrast));
    return clamp(center + (center - neighborhoodAverage) * strength,
                 neighborhoodMinimum, neighborhoodMaximum);
}

uint filmGrainHash(uint value)
{
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

float filmGrainRandom(ivec2 cell)
{
    uint seed = filmGrainHash(uint(cell.x) ^ filmGrainHash(uint(cell.y)));
    float first = float(filmGrainHash(seed) & 0xffffu) / 65535.0;
    float second = float(filmGrainHash(seed ^ 0x68bc21ebu) & 0xffffu) / 65535.0;
    return first + second - 1.0;
}

float filmGrainNoise(vec2 pixel)
{
    // Interpolate a slightly enlarged random lattice so neighboring pixels
    // form organic grains instead of independent television-static speckles.
    vec2 grainPosition = pixel * 0.68;
    ivec2 base = ivec2(floor(grainPosition));
    vec2 blend = fract(grainPosition);
    blend = blend * blend * (3.0 - 2.0 * blend);
    float north = mix(filmGrainRandom(base), filmGrainRandom(base + ivec2(1, 0)), blend.x);
    float south = mix(filmGrainRandom(base + ivec2(0, 1)),
                      filmGrainRandom(base + ivec2(1, 1)), blend.x);
    float clustered = mix(north, south, blend.y);
    float fine = filmGrainRandom(ivec2(pixel));
    return clustered * 0.85 + fine * 0.15;
}

vec3 applyFilmGrain(vec3 mapped)
{
    float intensity = max(frameData.directionalColor.w, 0.0);
    if (intensity <= 0.0 || frameData.visibilityMask.z != 0u) {
        return mapped;
    }

    // Keep presentation grain fixed in screen space. Animated grain competes
    // with temporal denoising and reads as shimmer in an interactive renderer.
    float grain = filmGrainNoise(gl_FragCoord.xy);
    float value = luminance(mapped);
    float visibleAboveBlack = smoothstep(0.01, 0.12, value);
    float highlightAttenuation = 1.0 - 0.65 * smoothstep(0.55, 1.0, value);
    float amplitude = intensity * visibleAboveBlack * highlightAttenuation;
    return clamp(mapped + vec3(grain * amplitude), 0.0, 1.0);
}

void main()
{
    vec4 resolved = texture(hdrColor, uv);
    // Ray Reconstruction is designed for surfaces with meaningful depth,
    // normal, albedo, and hit-distance guides. Primary atmosphere misses have
    // no surface guides, so retaining them in RR history produces large bright
    // shapes that slowly fade as the Sun moves. For the final atmosphere view,
    // composite the current exact path-traced background after reconstruction
    // while leaving all finite scene surfaces in the reconstructed image.
    if (frameData.temporalFlags.y == 0u &&
        frameData.atmosphereSunDirection.w > 0.5 &&
        frameData.environmentParameters.z >= 1.0) {
        ivec2 motionSize = textureSize(motionDepth, 0);
        ivec2 motionCoordinate = clamp(ivec2(uv * vec2(motionSize)),
                                       ivec2(0), motionSize - 1);
        bool infiniteBackground =
            texelFetch(motionDepth, motionCoordinate, 0).w <= 0.0;
        // A current-frame binary hit/miss mask is intentionally jittered at
        // silhouettes. Replacing RR with the exact sky on those boundary
        // samples makes the horizon alternate between two pipelines every
        // frame. Only replace pixels safely inside the background; leave a
        // one-texel guard band for RR to reconstruct and antialias.
        bool backgroundInterior = infiniteBackground;
        for (int y = -1; y <= 1 && backgroundInterior; ++y) {
            for (int x = -1; x <= 1; ++x) {
                ivec2 neighbor = clamp(motionCoordinate + ivec2(x, y),
                                       ivec2(0), motionSize - 1);
                backgroundInterior = backgroundInterior &&
                    texelFetch(motionDepth, neighbor, 0).w <= 0.0;
            }
        }
        if (backgroundInterior) {
            resolved = vec4(texture(bloomColor, uv).rgb, 1.0);
        }
    }
    if (frameData.temporalFlags.y == 1u) {
        vec2 motion = texture(motionDepth, uv).xy;
        float magnitude = clamp(length(motion) * 64.0, 0.0, 1.0);
        outColor = vec4(clamp(vec3(0.5) + vec3(motion * 32.0, 0.0), 0.0, 1.0) +
                            vec3(0.0, 0.0, magnitude),
                        1.0);
        return;
    }
    if (frameData.temporalFlags.y == 2u) {
        float confidence = clamp(resolved.a, 0.0, 1.0);
        outColor = vec4(1.0 - confidence, confidence, 0.0, 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 3u) {
        outColor = vec4(rendererAcesFitted(texture(bloomColor, uv).rgb), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 4u) {
        outColor = vec4(rendererAcesFitted(resolved.rgb), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 20u) {
        float reactive = clamp(texture(bloomColor, uv).r, 0.0, 1.0);
        outColor = vec4(vec3(reactive), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 5u || frameData.temporalFlags.y == 6u) {
        vec3 radiance = resolved.rgb;
        if (frameData.temporalFlags.y == 6u) {
            radiance = max(radiance - texture(bloomColor, uv).rgb, vec3(0.0));
        }
        radiance *= max(frameData.cameraPositionExposure.w, 0.0);
        outColor = vec4(rendererAcesFitted(radiance), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 7u) {
        outColor = vec4(clamp(resolved.rgb, 0.0, 1.0), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 8u) {
        float surface = texture(motionDepth, uv).w > 0.0 ? 1.0 : 0.0;
        outColor = vec4(vec3(clamp(resolved.a, 0.0, 1.0) * surface), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 9u || frameData.temporalFlags.y == 10u) {
        bool shadingNormal = frameData.temporalFlags.y == 10u;
        vec2 encoded = shadingNormal ? resolved.zw : resolved.xy;
        float surface = texture(motionDepth, uv).w > 0.0 ? 1.0 : 0.0;
        outColor = vec4((decodeOctahedralNormal(encoded) * 0.5 + 0.5) * surface, 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 11u) {
        float magnitude = clamp(length(resolved.xy) * 64.0, 0.0, 1.0);
        outColor = vec4(clamp(vec3(0.5) + vec3(resolved.xy * 32.0, 0.0), 0.0, 1.0) +
                            vec3(0.0, 0.0, magnitude),
                        1.0);
        return;
    }
    if (frameData.temporalFlags.y == 12u) {
        float surface = texture(motionDepth, uv).w > 0.0 ? 1.0 : 0.0;
        outColor = vec4(vec3((1.0 - clamp(resolved.r, 0.0, 1.0)) * surface), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 13u) {
        float sceneScale = max(frameData.shadowCascadeDepthRadii.w, 1.0e-6);
        float distance = max(resolved.w, 0.0);
        outColor = vec4(vec3(1.0 - exp(-distance / sceneScale)), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 14u) {
        vec3 directRadiance = texture(bloomColor, uv).rgb;
        vec3 indirectRadiance = max(resolved.rgb - directRadiance, vec3(0.0));
        vec3 residual = abs(resolved.rgb - directRadiance - indirectRadiance);
        outColor = vec4(clamp(residual * 256.0, 0.0, 1.0), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 15u) {
        float confidence = clamp(resolved.r, 0.0, 1.0);
        outColor = vec4(1.0 - confidence, confidence, 0.0, 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 16u) {
        vec3 radiance = resolved.rgb * max(frameData.cameraPositionExposure.w, 0.0);
        outColor = vec4(rendererAcesFitted(radiance), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 17u) {
        float weight = clamp(resolved.a, 0.0, 1.0);
        outColor = vec4(1.0 - weight, weight, 0.0, 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 18u) {
        vec3 radiance = resolved.rgb * max(frameData.cameraPositionExposure.w, 0.0);
        outColor = vec4(rendererAcesFitted(radiance), 1.0);
        return;
    }
    if (frameData.temporalFlags.y == 19u) {
        float variance = max(resolved.a, 0.0);
        float mapped = 1.0 - exp(-sqrt(variance) * 4.0);
        outColor = vec4(mapped, mapped * mapped, 0.0, 1.0);
        return;
    }

    vec3 hdr = resolved.rgb;
    if (frameData.temporalFlags.z != 0u) {
        // Sharpen only the presented image. Temporal history remains unsharpened,
        // preventing feedback ringing while restoring detail lost in the resolve.
        hdr = sharpenTemporalColor(uv, hdr);
    }
    if (frameData.temporalFlags.w != 0u) {
        float bloomIntensity = max(uintBitsToFloat(frameData.visibilityMask.w), 0.0);
        hdr += texture(bloomColor, uv).rgb * bloomIntensity;
    }
    hdr *= max(frameData.cameraPositionExposure.w, 0.0);
    vec3 mapped = rendererAgx(hdr * frameData.lookColor.rgb);
    outColor = vec4(applyFilmGrain(mapped), 1.0);
}
