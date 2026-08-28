# PBRT-v4 provenance

PBRT-v4 is the renderer's mathematical reference, not a linked runtime
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
| Future direct-light/path sampling | `src/pbrt/cpu/integrators.cpp`, `src/pbrt/util/sampling.h` | Not implemented yet; NEE, MIS, throughput, and Russian roulette will be ported when their milestones begin. |
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
- Transport begins in linear RGB rather than PBRT-v4's sampled spectrum, as
  required by the renderer plan.

Every later PBRT-derived implementation must extend this map with its reviewed
source file, symbol, relationship, and validation method before the associated
milestone is considered complete.
