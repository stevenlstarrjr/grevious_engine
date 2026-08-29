#ifndef GENESIS_RENDER_PATH_TRACE_BSDF_GLSL
#define GENESIS_RENDER_PATH_TRACE_BSDF_GLSL

struct PathBsdfSample
{
    vec3 direction;
    vec3 throughput;
    float pdf;
    float coneSpread;
};

bool pathTraceFinite(vec3 value)
{
    return !any(isnan(value)) && !any(isinf(value));
}

float pathTraceLuminance(vec3 value)
{
    return dot(value, vec3(0.2126, 0.7152, 0.0722));
}

void pathTraceBasis(vec3 normal, out vec3 tangent, out vec3 bitangent)
{
    vec3 reference = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0)
                                          : vec3(1.0, 0.0, 0.0);
    tangent = normalize(cross(reference, normal));
    bitangent = cross(normal, tangent);
}

vec3 sampleVisibleGgxNormal(vec3 localView, float alpha, vec2 sampleValue)
{
    vec3 stretched = normalize(vec3(alpha * localView.x, alpha * localView.y,
                                    localView.z));
    if (stretched.z < 0.0) {
        stretched = -stretched;
    }
    vec3 tangent = stretched.z < 0.99999
                       ? normalize(cross(vec3(0.0, 0.0, 1.0), stretched))
                       : vec3(1.0, 0.0, 0.0);
    vec3 bitangent = cross(stretched, tangent);
    float radius = sqrt(sampleValue.x);
    float angle = 2.0 * PATH_TRACE_PI * sampleValue.y;
    vec2 disk = radius * vec2(cos(angle), sin(angle));
    float horizon = sqrt(max(0.0, 1.0 - disk.x * disk.x));
    float blend = (1.0 + stretched.z) * 0.5;
    disk.y = mix(horizon, disk.y, blend);
    float projectedZ = sqrt(max(0.0, 1.0 - dot(disk, disk)));
    vec3 hemisphere = disk.x * tangent + disk.y * bitangent + projectedZ * stretched;
    return normalize(vec3(alpha * hemisphere.x, alpha * hemisphere.y,
                          max(1.0e-6, hemisphere.z)));
}

float pathSpecularSampleProbability(vec3 baseColor, float metallic,
                                    vec3 normal, vec3 viewDirection)
{
    float noV = clamp(dot(normal, viewDirection), 0.0, 1.0);
    vec3 f0 = mix(vec3(0.04), clamp(baseColor, 0.0, 1.0),
                  clamp(metallic, 0.0, 1.0));
    float specularWeight = pathTraceLuminance(fresnelSchlick(noV, f0));
    float diffuseWeight = pathTraceLuminance(clamp(baseColor, 0.0, 1.0)) *
                          (1.0 - clamp(metallic, 0.0, 1.0)) *
                          max(0.0, 1.0 - specularWeight);
    float totalWeight = specularWeight + diffuseWeight;
    if (totalWeight <= 1.0e-7) {
        return 0.5;
    }
    return clamp(specularWeight / totalWeight, 0.05, 0.95);
}

float visibleGgxReflectionPdf(float roughness, vec3 normal, vec3 viewDirection,
                              vec3 lightDirection)
{
    float noV = clamp(dot(normal, viewDirection), 0.0, 1.0);
    float noL = clamp(dot(normal, lightDirection), 0.0, 1.0);
    vec3 halfUnnormalized = viewDirection + lightDirection;
    float halfLengthSquared = dot(halfUnnormalized, halfUnnormalized);
    if (noV <= 0.0 || noL <= 0.0 || halfLengthSquared <= 1.0e-7) {
        return 0.0;
    }
    vec3 halfVector = halfUnnormalized * inversesqrt(halfLengthSquared);
    float noH = clamp(dot(normal, halfVector), 0.0, 1.0);
    float voH = clamp(dot(viewDirection, halfVector), 0.0, 1.0);
    if (noH <= 0.0 || voH <= 0.0) {
        return 0.0;
    }
    float alpha = perceptualRoughnessToAlpha(roughness);
    float masking = 1.0 / (1.0 + smithLambda(noV, alpha));
    return max(trowbridgeReitzD(noH, alpha) * masking / (4.0 * noV), 0.0);
}

float pathBsdfPdf(vec3 baseColor, float metallic, float roughness,
                  vec3 normal, vec3 viewDirection, vec3 lightDirection)
{
    float noV = clamp(dot(normal, viewDirection), 0.0, 1.0);
    float noL = clamp(dot(normal, lightDirection), 0.0, 1.0);
    if (noV <= 0.0 || noL <= 0.0) {
        return 0.0;
    }
    float specularProbability = pathSpecularSampleProbability(
        baseColor, metallic, normal, viewDirection);
    float diffusePdf = noL / PATH_TRACE_PI;
    float specularPdf = visibleGgxReflectionPdf(
        roughness, normal, viewDirection, lightDirection);
    return mix(diffusePdf, specularPdf, specularProbability);
}

PathBsdfSample emptyPathBsdfSample()
{
    PathBsdfSample result;
    result.direction = vec3(0.0);
    result.throughput = vec3(0.0);
    result.pdf = 0.0;
    result.coneSpread = 0.0;
    return result;
}

PathBsdfSample samplePathBsdf(vec3 baseColor, float metallic, float roughness,
                              vec3 normal, vec3 viewDirection, inout uint state)
{
    float specularProbability = pathSpecularSampleProbability(
        baseColor, metallic, normal, viewDirection);
    vec3 direction;
    bool sampledSpecular = pathTraceRandom(state) < specularProbability;
    if (sampledSpecular) {
        vec3 tangent;
        vec3 bitangent;
        pathTraceBasis(normal, tangent, bitangent);
        vec3 localView = vec3(dot(viewDirection, tangent),
                              dot(viewDirection, bitangent),
                              dot(viewDirection, normal));
        vec2 sampleValue = vec2(pathTraceRandom(state), pathTraceRandom(state));
        vec3 localHalf = sampleVisibleGgxNormal(
            localView, perceptualRoughnessToAlpha(roughness), sampleValue);
        vec3 halfVector = normalize(tangent * localHalf.x +
                                    bitangent * localHalf.y +
                                    normal * localHalf.z);
        direction = reflect(-viewDirection, halfVector);
    } else {
        direction = cosineSampleHemisphere(normal, state);
    }

    float noL = clamp(dot(normal, direction), 0.0, 1.0);
    if (noL <= 0.0 || !pathTraceFinite(direction)) {
        return emptyPathBsdfSample();
    }
    float pdf = pathBsdfPdf(baseColor, metallic, roughness, normal,
                            viewDirection, direction);
    if (pdf <= 0.0 || isnan(pdf) || isinf(pdf)) {
        return emptyPathBsdfSample();
    }
    vec3 throughput = evaluateMetallicRoughnessBrdf(
        baseColor, metallic, roughness, normal, viewDirection, direction) *
        (noL / pdf);
    if (!pathTraceFinite(throughput)) {
        return emptyPathBsdfSample();
    }

    PathBsdfSample result;
    result.direction = normalize(direction);
    result.throughput = max(throughput, vec3(0.0));
    result.pdf = pdf;
    // A diffuse sample spans the hemisphere. Visible-GGX spread follows alpha;
    // the floor preserves the incoming camera footprint for near-mirror rays.
    result.coneSpread = sampledSpecular
                            ? max(1.0e-4, 0.5 * perceptualRoughnessToAlpha(roughness))
                            : 0.5;
    return result;
}

#endif
