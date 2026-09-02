#ifndef GENESIS_WEATHER_WETNESS_GLSL
#define GENESIS_WEATHER_WETNESS_GLSL

float genesisWetnessHash(vec2 cell)
{
    vec3 value = fract(vec3(cell.xyx) * 0.1031);
    value += dot(value, value.yzx + 33.33);
    return fract((value.x + value.y) * value.z);
}

float genesisWetnessNoise(vec2 position)
{
    vec2 cell = floor(position);
    vec2 fraction = fract(position);
    fraction = fraction * fraction * (3.0 - 2.0 * fraction);
    float a = genesisWetnessHash(cell);
    float b = genesisWetnessHash(cell + vec2(1.0, 0.0));
    float c = genesisWetnessHash(cell + vec2(0.0, 1.0));
    float d = genesisWetnessHash(cell + vec2(1.0, 1.0));
    return mix(mix(a, b, fraction.x), mix(c, d, fraction.x), fraction.y);
}

// Builds the first wetness layer from many stable, splash-sized impact marks.
// Each mark is anchored to a world-space cell so changing rain intensity or
// accumulation cannot make the damp pattern slide across the surface.
float genesisDampImpactSpots(vec2 worldPosition, float growth)
{
    // The cells are intentionally larger than a single raindrop. Each mark
    // represents the locally overlapping impacts visible as one dark splash
    // spot at the camera distances used by the weather demo.
    // A dense impact lattice lets newly arriving marks close dry gaps. Mature
    // marks never receive a scene-wide radius multiplier.
    const float cellSize = 0.32;
    vec2 baseCell = floor(worldPosition / cellSize);
    float depositedDampness = 0.0;

    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 cell = baseCell + vec2(float(x), float(y));
            float centerX = genesisWetnessHash(cell + vec2(13.7, 41.3));
            float centerY = genesisWetnessHash(cell + vec2(79.1, 17.9));
            vec2 center = (cell + mix(vec2(0.38), vec2(0.62),
                                      vec2(centerX, centerY))) * cellSize;

            // Every cell has its own arrival time. Successive hits are short
            // local events instead of one long opacity ramp; a large set of
            // slowly changing marks would average into a whole-ground fade.
            float birth = genesisWetnessHash(cell + vec2(31.4, 67.2)) * 0.72;
            if (growth <= birth) {
                continue;
            }

            float firstDeposit = smoothstep(birth, birth + 0.025, growth);
            float secondDeposit = smoothstep(
                birth + 0.085, birth + 0.110, growth);
            float laterDeposits = smoothstep(
                birth + 0.185, birth + 0.210, growth);

            float shape = genesisWetnessHash(cell + vec2(53.8, 92.6));
            vec2 offset = worldPosition - center;

            // The first hit stays splash-sized. Repeated hits ultimately make
            // neighbouring cells overlap, so there are no permanently dry
            // pixels left immediately before pooling begins.
            float targetRadius = mix(0.320, 0.380, shape);
            float radiusLayer = 0.35 + 0.25 * secondDeposit +
                                0.40 * laterDeposits;
            float radius = targetRadius * radiusLayer;

            float edgeNoise = genesisWetnessNoise(
                worldPosition * 22.0 + cell * 2.7);
            float signedDistance = length(offset) -
                radius * mix(0.90, 1.10, edgeNoise);
            float spot = 1.0 - smoothstep(-0.012, 0.018, signedDistance);

            // Treat the hit sequence as repeated deposition, not a binary reveal.
            // A first hit creates a light damp mark; later hits deepen it.
            // Contributions add, so independently placed marks become darker
            // exactly where their soft, irregular edges overlap.
            float depositedAmount = 0.27 * firstDeposit +
                                    0.40 * secondDeposit +
                                    0.75 * laterDeposits;
            depositedDampness += spot * depositedAmount;
        }
    }

    // Convert accumulated deposits with an asymptotic saturation response.
    // A hard clamp made broad regions hit exactly 1.0 together, which looked
    // like the entire ground fading to a uniform damp value in one moment.
    // This retains visible differences between one, two, and many overlaps.
    return 1.0 - exp(-1.45 * max(depositedDampness, 0.0));
}

struct GenesisWetnessLayers
{
    float dampness;
    float poolingDampness;
    float puddles;
    float coating;
};

// Four persistent responses share one reservoir but retain separate spatial
// behavior. Impact dampness advances across the exposed surface, the future
// puddle footprint darkens while matte, and glossy water then grows from basin
// peaks to a capped threshold. The optional coating remains independent for
// explicitly tagged non-porous materials.
GenesisWetnessLayers genesisRainWetnessLayers(
    vec3 worldPosition, vec3 surfaceNormal, float accumulatedWetness,
    float postRainAgeSeconds)
{
    float accumulation = clamp(accumulatedWetness, 0.0, 1.0);
    float upward = dot(normalize(surfaceNormal), vec3(0.0, 1.0, 0.0));
    float rainExposure = smoothstep(0.15, 0.88, upward);
    float puddleExposure = smoothstep(0.62, 0.94, upward);
    float broad = genesisWetnessNoise(worldPosition.xz * 0.075);
    float detail = genesisWetnessNoise(worldPosition.xz * 0.31 + vec2(17.0, -9.0));
    float basin = broad * 0.78 + detail * 0.22;

    // Stage one: ordinary impacts nearly cover the exposed porous receiver.
    // Stage two darkens the complete future puddle footprint while it remains
    // matte. Stage three then grows standing glossy water from basin peaks.
    const float dampCompletion = 0.72;
    const float poolingDampStart = 0.72;
    const float poolingDampEnd = 0.84;
    const float puddleStart = 0.86;
    // Linear time distribution keeps independent impact arrivals evenly
    // spread through the storm. smoothstep accelerated the middle of this
    // range and compressed most visible ground coverage into about a second.
    float dampGrowth = clamp(accumulation / dampCompletion, 0.0, 1.0);
    float dampCoverage = genesisDampImpactSpots(worldPosition.xz, dampGrowth);
    float dampness = rainExposure * dampCoverage;

    // Gate collected water from the final damp value at this exact material
    // point. Both raster fragments and RTX closest hits therefore require a
    // mature damp layer here before the basin can darken or turn glossy.
    float puddleDampGate = smoothstep(0.80, 0.86, dampCoverage);

    float poolingFootprint = smoothstep(0.57, 0.69, basin);
    float poolingGrowth = smoothstep(
        poolingDampStart, poolingDampEnd, accumulation);
    float poolingDampness = puddleExposure * poolingFootprint * poolingGrowth *
                            puddleDampGate;

    // Stage three: start with a tiny glossy pool at the deepest basin peaks,
    // then expand across only the already-dark pooling footprint.
    float puddleGrowth = smoothstep(puddleStart, 1.0, accumulation);
    float spatialPuddleGrowth = pow(puddleGrowth, 1.35);
    float puddleFront = mix(1.02, 0.66, spatialPuddleGrowth);
    float puddleCoverage = smoothstep(puddleFront - 0.045,
                                      puddleFront + 0.045, basin);
    float puddles = puddleExposure * puddleCoverage *
                    smoothstep(0.0, 0.18, puddleGrowth) * puddleDampGate;

    if (postRainAgeSeconds > 0.0) {
        // The CPU reservoir evaporates at 0.033 units/second. Reconstruct its
        // level when precipitation stopped so drying can be sequenced instead
        // of merely playing the wetting thresholds backward.
        float wetnessAtRainStop = clamp(
            accumulation + postRainAgeSeconds * 0.033, 0.0, 1.0);
        float dryProgress = wetnessAtRainStop > 1.0e-4
            ? clamp(1.0 - accumulation / wetnessAtRainStop, 0.0, 1.0)
            : 1.0;

        float initialDampGrowth = clamp(
            wetnessAtRainStop / dampCompletion, 0.0, 1.0);
        float initialDampCoverage = genesisDampImpactSpots(
            worldPosition.xz, initialDampGrowth);
        float initialDampness = rainExposure * initialDampCoverage;
        float initialPuddleDampGate = smoothstep(
            0.80, 0.86, initialDampCoverage);

        float initialPuddleGrowth = smoothstep(
            puddleStart, 1.0, wetnessAtRainStop);
        float initialPoolingGrowth = smoothstep(
            poolingDampStart, poolingDampEnd, wetnessAtRainStop);

        // Drying stage one: exposed damp ground clears while the collected
        // water and the saturated material beneath it remain unchanged.
        float exposedDampRemaining =
            1.0 - smoothstep(0.0, 0.28, dryProgress);

        // Drying stage two: after the exposed ground has dried, move the
        // puddle isosurface back toward basin peaks so pools visibly contract.
        float puddleRemaining =
            1.0 - smoothstep(0.30, 0.72, dryProgress);
        float dryingPuddleGrowth = initialPuddleGrowth * puddleRemaining;
        float dryingSpatialPuddleGrowth = pow(dryingPuddleGrowth, 1.35);
        float dryingPuddleFront = mix(
            1.02, 0.66, dryingSpatialPuddleGrowth);
        float dryingPuddleCoverage = smoothstep(
            dryingPuddleFront - 0.045,
            dryingPuddleFront + 0.045, basin);
        puddles = puddleExposure * dryingPuddleCoverage *
            smoothstep(0.0, 0.18, dryingPuddleGrowth) *
            initialPuddleDampGate;

        // Drying stage three: a dark footprint survives under the former
        // puddle. Only after the visible water is gone does this last damp
        // isosurface contract and fade into the dry receiver.
        float residualRemaining =
            1.0 - smoothstep(0.74, 1.0, dryProgress);
        float residualGrowth = initialPuddleGrowth * residualRemaining;
        float residualFront = mix(1.02, 0.66, residualGrowth);
        float residualCoverage = smoothstep(
            residualFront - 0.055, residualFront + 0.055, basin);
        float residualDampness = rainExposure * residualCoverage *
            smoothstep(0.0, 0.16, residualGrowth);
        dampness = max(initialDampness * exposedDampRemaining,
                       residualDampness);

        // The saturated matte footprint under collected water survives until
        // the glossy pool has contracted, then clears during the final stage.
        float poolingRemaining =
            1.0 - smoothstep(0.72, 1.0, dryProgress);
        poolingDampness = puddleExposure * poolingFootprint *
                          initialPoolingGrowth * poolingRemaining *
                          initialPuddleDampGate;
    }

    float coatingField = broad * 0.35 + detail * 0.65;
    float coatingFront = mix(1.04, 0.30, accumulation);
    float patchyCoating = smoothstep(coatingFront - 0.08,
                                     coatingFront + 0.08, coatingField);
    float continuousCoating = smoothstep(0.58, 0.96, accumulation);
    float coating = mix(patchyCoating, 1.0, continuousCoating) *
                    smoothstep(0.02, 0.22, accumulation);

    GenesisWetnessLayers layers;
    layers.dampness = clamp(dampness, 0.0, 1.0);
    layers.poolingDampness = clamp(poolingDampness, 0.0, 1.0);
    layers.puddles = clamp(puddles, 0.0, 1.0);
    layers.coating = clamp(coating, 0.0, 1.0);
    return layers;
}

#endif
