#ifndef GENESIS_RENDER_PATH_TRACE_ENVIRONMENT_GLSL
#define GENESIS_RENDER_PATH_TRACE_ENVIRONMENT_GLSL

struct PathEnvironmentAliasEntry
{
    float threshold;
    uint aliasIndex;
    float probabilityMass;
    float reserved;
};

layout(set = 0, binding = 9) uniform sampler2D environmentMap;
layout(std430, set = 0, binding = 10) readonly buffer EnvironmentSampling
{
    PathEnvironmentAliasEntry environmentAliasEntries[];
};

struct PathEnvironmentSample
{
    vec3 direction;
    vec3 radiance;
    float pdf;
};

float pathEnvironmentTexelSolidAngle(uint row, uint width, uint height)
{
    float theta0 = PATH_TRACE_PI * float(row) / float(height);
    float theta1 = PATH_TRACE_PI * float(row + 1u) / float(height);
    return (2.0 * PATH_TRACE_PI / float(width)) *
           max(cos(theta0) - cos(theta1), 0.0);
}

float pathEnvironmentPdf(vec3 worldDirection, float rotation, bool importanceSampling)
{
    if (!importanceSampling) {
        return 1.0 / (4.0 * PATH_TRACE_PI);
    }
    ivec2 dimensions = textureSize(environmentMap, 0);
    vec2 uv = rotatedDirectionToLatLongUv(worldDirection, rotation);
    uint width = uint(dimensions.x);
    uint height = uint(dimensions.y);
    uint x = min(uint(uv.x * float(width)), width - 1u);
    uint y = min(uint(uv.y * float(height)), height - 1u);
    uint index = y * width + x;
    float solidAngle = pathEnvironmentTexelSolidAngle(y, width, height);
    return solidAngle > 0.0
               ? max(environmentAliasEntries[index].probabilityMass / solidAngle, 0.0)
               : 0.0;
}

PathEnvironmentSample samplePathEnvironment(inout uint state, float rotation,
                                            bool importanceSampling)
{
    ivec2 dimensions = textureSize(environmentMap, 0);
    uint width = uint(dimensions.x);
    uint height = uint(dimensions.y);
    uint entryCount = width * height;
    float tableSample = pathTraceRandom(state) * float(entryCount);
    uint column = min(uint(tableSample), entryCount - 1u);
    PathEnvironmentAliasEntry columnEntry = environmentAliasEntries[column];
    uint selected = fract(tableSample) < columnEntry.threshold
                        ? column
                        : columnEntry.aliasIndex;
    uint x = selected % width;
    uint y = selected / width;

    float uSample = pathTraceRandom(state);
    float thetaSample = pathTraceRandom(state);
    float u = importanceSampling
                  ? (float(x) + uSample) / float(width)
                  : uSample;
    float theta0 = PATH_TRACE_PI * float(y) / float(height);
    float theta1 = PATH_TRACE_PI * float(y + 1u) / float(height);
    float cosineTheta = importanceSampling
                            ? mix(cos(theta0), cos(theta1), thetaSample)
                            : mix(1.0, -1.0, thetaSample);
    float theta = acos(clamp(cosineTheta, -1.0, 1.0));
    float phi = (u - 0.5) * 2.0 * PATH_TRACE_PI;
    float sineTheta = sin(theta);
    vec3 textureDirection = vec3(sineTheta * cos(phi), cosineTheta,
                                 sineTheta * sin(phi));
    vec2 uv = vec2(u, theta / PATH_TRACE_PI);
    float solidAngle = pathEnvironmentTexelSolidAngle(y, width, height);

    PathEnvironmentSample result;
    result.direction = normalize(rotateEnvironmentDirection(textureDirection, -rotation));
    result.radiance = max(textureLod(environmentMap, uv, 0.0).rgb, vec3(0.0));
    result.pdf = importanceSampling
                     ? (solidAngle > 0.0
                            ? max(environmentAliasEntries[selected].probabilityMass /
                                      solidAngle,
                                  0.0)
                            : 0.0)
                     : 1.0 / (4.0 * PATH_TRACE_PI);
    return result;
}

#endif
