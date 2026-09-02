#ifndef GENESIS_RENDER_PATH_TRACE_GLSL
#define GENESIS_RENDER_PATH_TRACE_GLSL

const float PATH_TRACE_PI = 3.14159265358979323846;

struct RadiancePayload
{
    vec3 radiance;
    vec3 pathThroughput;
    uint depth;
    uint seed;
    float coneWidth;
    float coneSpread;
    float bouncePdf;
    float firstLobeHitDistance;
    float primaryHitDistance;
};

float pathRussianRouletteSurvivalProbability(vec3 throughput)
{
    if (any(isnan(throughput)) || any(isinf(throughput))) {
        return 0.0;
    }
    float maximumComponent = max(throughput.x, max(throughput.y, throughput.z));
    if (maximumComponent <= 0.0) {
        return 0.0;
    }
    if (maximumComponent >= 1.0) {
        return 1.0;
    }
    return clamp(maximumComponent, 0.05, 0.95);
}

float pathPowerHeuristic(float firstPdf, float secondPdf)
{
    float firstSquared = firstPdf * firstPdf;
    float secondSquared = secondPdf * secondPdf;
    float denominator = firstSquared + secondSquared;
    return denominator > 0.0 ? firstSquared / denominator : 0.0;
}

float pathAreaLightSolidAnglePdf(float selectionProbability, float area,
                                 float distanceSquared, float absoluteCosine)
{
    if (selectionProbability <= 0.0 || area <= 0.0 || distanceSquared <= 0.0 ||
        absoluteCosine <= 0.0 || isnan(selectionProbability) || isnan(area) ||
        isnan(distanceSquared) || isnan(absoluteCosine) || isinf(selectionProbability) ||
        isinf(area) || isinf(distanceSquared) || isinf(absoluteCosine)) {
        return 0.0;
    }
    float pdf = selectionProbability * distanceSquared / (area * absoluteCosine);
    return isnan(pdf) || isinf(pdf) ? 0.0 : pdf;
}

uint pathTraceHash(uint value)
{
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

float pathTraceRandom(inout uint state)
{
    state = pathTraceHash(state + 0x9e3779b9u);
    return float(state >> 8u) * (1.0 / 16777216.0);
}

vec3 cosineSampleHemisphere(vec3 normal, inout uint state)
{
    float radius = sqrt(pathTraceRandom(state));
    float angle = 2.0 * PATH_TRACE_PI * pathTraceRandom(state);
    vec3 local = vec3(radius * cos(angle), radius * sin(angle),
                      sqrt(max(0.0, 1.0 - radius * radius)));
    vec3 reference = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0)
                                          : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(reference, normal));
    vec3 bitangent = cross(normal, tangent);
    return normalize(tangent * local.x + bitangent * local.y + normal * local.z);
}

#endif
