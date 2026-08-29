# PBRT-v4 provenance

PBRT-v4 is Genesis Render's mathematical reference, not a linked runtime
dependency or an architecture template. It is pinned under
`reference/pbrt-v4`; the reviewed revision is
`5f7a606806a4ac7b939131ded9d7a30ebd02416e`. PBRT-v4 is distributed under the
Apache License 2.0; this project preserves attribution whenever implementation
code is ported. Derived source carries PBRT's copyright and Apache-2.0
attribution alongside this ledger.

## Current source map

| Renderer implementation | PBRT-v4 source | Relationship |
|---|---|---|
| `PbrMath.cpp`: `trowbridgeReitzD` | `src/pbrt/util/scattering.h`: `TrowbridgeReitzDistribution::D` | Direct isotropic port that preserves PBRT's operation order. |
| `PbrMath.cpp`: `smithLambda` and `smithMaskingShadowing` | `src/pbrt/util/scattering.h`: `Lambda` and `G` | Isotropic port of PBRT's joint Smith form. |
| `PbrMath.cpp`: `fresnelDielectric` | `src/pbrt/util/scattering.h`: `FrDielectric` | Direct scalar port with guarded realtime inputs. |
| CPU/shader microfacet specular evaluation | `src/pbrt/bxdfs.h`: `ConductorBxDF::f` | Same `D F G / (4 cos(theta_i) cos(theta_o))` structure, evaluated in linear RGB. |
| `IblBaker.cpp`: `sampleVisibleGgx` | `src/pbrt/util/scattering.h`: `TrowbridgeReitzDistribution::Sample_wm` | Direct isotropic visible-normal sampling port. |
| `TemporalSampling.cpp`: `radicalInverse` | `src/pbrt/util/lowdiscrepancy.h`: `RadicalInverse` | Direct digit-reversal/overflow-guard port with the prime base supplied explicitly; bases two and three feed a centered finite TAA sequence. |
| `path_trace.rchit`, `path_trace_bsdf.glsl`, and `PbrMath.cpp`: bounded BSDF continuations and throughput-based Russian roulette | `src/pbrt/cpu/integrators.cpp`: path-integrator direct-light/path loop and `rrBeta` termination; `src/pbrt/util/sampling.h`: cosine-hemisphere sampling; `src/pbrt/util/scattering.h`: `TrowbridgeReitzDistribution::Sample_wm` and `PDF` | Bounded realtime adaptation: selected directional, environment, and emissive-triangle visibility samples at every hit plus one-to-four diffuse/visible-GGX mixture continuations (default two). The renderer chooses a lobe with a clamped RGB-energy heuristic, evaluates the full metallic/roughness BRDF, carries explicit solid-angle PDF and `f * absDot / PDF` throughput, and applies compensated Russian roulette after continuation two. The renderer adds a 0.05 survival floor and fixed four-bounce cap. |
| `IblBaker.cpp`: `buildEnvironmentSamplingDistribution`; `path_trace_environment.glsl`; environment NEE/MIS in `path_trace.rchit` and `path_trace.rmiss` | `src/pbrt/lights.cpp`: `ImageInfiniteLight::SampleLi` and `PDF_Li`; `src/pbrt/util/sampling.h`: `PiecewiseConstant2D` and `PowerHeuristic` | Same luminance-weighted spherical environment distribution and complementary power-heuristic light/BSDF weighting. The realtime renderer uses exact lat-long texel solid angles and a flat Vose alias table instead of PBRT's piecewise-constant 2D sampler. CPU tests reconstruct every alias probability; the RTX gate compares uniform and importance RMSE at a fixed four-sample budget against a 64-sample reference. |
| `GpuScene.cpp`: `buildEmissiveTriangleDistribution`; `PbrMath.cpp`: `areaLightSolidAnglePdf`; emissive NEE/MIS in `path_trace.rchit` | `src/pbrt/lights.cpp`: `DiffuseAreaLight::SampleLi` and `PDF_Li`; `src/pbrt/util/sampling.h`: uniform triangle sampling and `PowerHeuristic` | Same uniform-area triangle sampling, area-to-solid-angle density conversion, visibility test, and complementary light/BSDF power heuristic. The realtime adaptation flattens visible finest-resident emitter triangles into a static world-area/emissive-factor-luminance CDF, evaluates emissive textures only after selection, and uses reserved instance metadata for O(1) hit-PDF lookup. CPU tests verify distribution normalization, transform-aware area weighting, visibility filtering, singular PDFs, and the RTX gate compares explicit sampling with BSDF-only transport at a fixed four-sample budget. |
| Generated tangent cleanup/fallback | `src/pbrt/util/vecmath.h`, `GramSchmidt()` and `CoordinateSystem()` | `src/assets/TangentGenerator.cpp`; MikkTSpace remains authoritative for UV derivatives, while PBRT's frame utilities orthogonalize inherited degenerate-island results and prevent undefined shader normalization where a tangent cannot be inferred. |

The matching fourth-edition book sections are `Reflection Models / Roughness
Using Microfacet Theory`, `Conductor BRDF`, and `Dielectric BSDF`. Tests evaluate
the renderer's closed-form GPU-friendly equations against PBRT's original-form
isotropic equations over roughness and angle sweeps, in addition to reciprocity
and white-furnace checks.

## Deliberate realtime adaptations

- glTF metallic/roughness supplies perceptual roughness, so the renderer uses
  `alpha = roughness^2`. PBRT-v4's own `RoughnessToAlpha` comment notes that this
  square is the more perceptually uniform mapping, while retaining its shipped
  mapping for PBRT scene compatibility.
- RGB Fresnel-Schlick and metallic `F0` blending implement the glTF realtime
  material contract. Exact PBRT dielectric Fresnel remains in the CPU reference
  layer, and exact conductor Fresnel will be used for measured `eta/k` materials.
- Split-sum IBL, Hi-Z, GPU culling, the Vulkan render graph, temporal reuse, and
  denoising are realtime architecture. They may reuse PBRT sampling and
  transport math, but they are not represented as PBRT code.
- Path reconstruction outputs are renderer-owned architecture rather than a
  PBRT port. Beauty and direct radiance are accumulated separately (indirect is
  derived), while first-hit albedo/roughness, octahedral geometric/shading
  normals, unjittered motion, device depth, and world hit distance provide one
  shared contract for later real-time denoising and converged VFX output.
- Path temporal radiance reconstruction is also renderer-owned realtime
  architecture, not a PBRT port. It uses motion-reprojected validated history,
  current-neighborhood clipping, and bounded motion/convergence weighting while
  preserving the raw progressive accumulator. PBRT remains relevant to the
  radiance samples being reconstructed, not to this temporal reuse policy.
- Frame-local animated GPU Scene transforms, synchronized TLAS updates,
  `previousWorld` motion, per-moving-frame progressive resets, and selective
  temporal reuse are renderer-owned animation/history policy rather than PBRT
  code. PBRT remains relevant to each fresh radiance sample, not to the decision
  to retain or reject realtime history around moving rigid geometry.
- Temporal luminance moments and the bounded AOV-guided 3x3 spatial filter are
  renderer-owned realtime reconstruction, not PBRT ports. Their variance,
  surface-class, depth, normal, albedo, roughness, and hit-distance weighting is
  deliberately measured against low-sample temporal output and fades as the
  progressive accumulator converges.
- The Vulkan-independent `AovFrameSink`, packed GPU readback, semantic plane
  expansion, PFM writer, and manifest schema are renderer/output architecture,
  not PBRT ports. They expose the same live reconstruction data for converged
  VFX jobs without creating a separate transport implementation. PBRT remains a
  reference for future offline-quality sampling and output comparisons.
- Closest-hit texture LOD uses a renderer-specific ray cone: adjacent primary
  camera rays define angular spread, triangle UV/world edge ratios convert cone
  width to texel footprint, and a bounded diffuse/GGX heuristic widens the
  secondary cone. This is not a port of PBRT's ray differentials.
- Transport begins in linear RGB rather than PBRT-v4's sampled spectrum, as
  required by the renderer plan.
- The path slice uses deterministic integer hashing rather than PBRT's samplers
  and caps transport at one to four sampled metallic/roughness bounces, default
  two. Environment and emissive-triangle NEE run at every hit; nonterminal hits
  have a complementary BSDF technique while the configured terminal hit uses
  unit light-sample weight. The emitter CDF is static and weights textured
  emitters by their factor rather than integrating texture power. Russian
  roulette starts after continuation two, uses compensated
  cumulative RGB throughput, and adds a 0.05 survival floor for bounded realtime
  variance. The clamped lobe-selection heuristic and flat alias representation
  are renderer-specific. This is a validation baseline, not a claim of
  PBRT-equivalent variance or feature coverage.

Every later PBRT-derived implementation must extend this map with its reviewed
source file, symbol, relationship, and validation method before the associated
milestone is considered complete.
