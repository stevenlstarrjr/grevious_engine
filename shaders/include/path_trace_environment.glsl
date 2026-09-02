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

vec3 samplePathEnvironmentCone(inout uint state, vec3 axis, float angularRadius)
{
    if (angularRadius <= 0.0) {
        return axis;
    }
    float cosineTheta = mix(1.0, cos(angularRadius), pathTraceRandom(state));
    float sineTheta = sqrt(max(1.0 - cosineTheta * cosineTheta, 0.0));
    float phi = 2.0 * PATH_TRACE_PI * pathTraceRandom(state);
    vec3 reference = abs(axis.y) < 0.999 ? vec3(0.0, 1.0, 0.0)
                                         : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(reference, axis));
    vec3 bitangent = cross(axis, tangent);
    return normalize(axis * cosineTheta +
                     (tangent * cos(phi) + bitangent * sin(phi)) * sineTheta);
}

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
                                            bool importanceSampling,
                                            float shadowAngularRadius,
                                            bool dynamicAtmosphereEnvironment)
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
    vec3 sourceDirection = normalize(rotateEnvironmentDirection(textureDirection, -rotation));
    // Sampling a finite cone around the selected HDR texel convolves the lighting source while
    // retaining its total energy. The unblurred environment remains available to camera and
    // reflection rays, so this changes penumbra width rather than background sharpness.
    result.direction = samplePathEnvironmentCone(state, sourceDirection, shadowAngularRadius);
    vec2 radianceUv = dynamicAtmosphereEnvironment
                          ? directionToDualParaboloidUv(sourceDirection)
                          : uv;
    result.radiance = max(textureLod(environmentMap, radianceUv, 0.0).rgb,
                          vec3(0.0));
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
