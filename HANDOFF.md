# Genesis Render Handoff

Last updated: 2026-09-02

Repository: <https://github.com/stevenlstarrjr/grevious_engine>

Branch: `main`

## Scope

This is a native Windows/Linux, real-time-first C++20 Vulkan renderer, not a
complete game engine. Its shared rendering core is intended to support both a
frame-budgeted game profile and a later converged VFX/animation profile; do not
fork separate scene/material/lighting implementations. PBRT-v4 and the PBRT
book are rendering references, never runtime dependencies. Do not add unrelated
physics, audio, scripting, networking, ECS, or editor systems here.

The planned host-engine physics direction is documented in
`ARCHITECTURE.md` under **Host physics and simulation boundary**. PhysX is the
preferred first host backend behind `IPhysicsWorld`; Blast and Flow are optional
later host integrations. Genesis Render consumes renderer-neutral scene/transform/
geometry/volume updates and must never expose or link PhysX, Blast, Flow, Jolt,
or CUDA types through its public renderer API.

The supported product hardware target is **NVIDIA RTX-capable graphics cards
only**. Required devices must provide Vulkan 1.3, acceleration structures, and
the ray-tracing pipeline. CUDA, DLSS/Ray Reconstruction, and other NVIDIA-specific
techniques are valid implementation choices. Existing portable/raster/reference
paths may remain useful for diagnostics and blended-material rendering, but AMD,
Intel, integrated, and software Vulkan devices are not supported product targets.

## What Works Now

- The renderer core is available as the SDL-independent `Genesis::Renderer`
  static library. `RenderSurface` separates presentation from SDL, the Win32
  implementation accepts an existing HWND, and Genisis Studio embeds it inside
  a Qt window without linking SDL into the viewport host.
- The editor integration has explicit W/M/R ownership: W and M use the Genesis
  raster renderer with a host-supplied post-tone-map overlay pass; W replaces
  the scene presentation with workbench triangles, while M composites the grid,
  selection, component, and gizmo geometry over PBR. R is the clean path-traced
  view with no editor overlays. The legacy Studio Vulkan viewport source remains
  compiled and preserved but is not an automatic runtime fallback.
- Material Preview supports live in-memory scene replacement. Reserved device-
  local scene buffers receive asynchronous geometry, instance, material,
  transform, and meshlet-work uploads without rebuilding pipelines, swapchain,
  or IBL; capacity/texture/RT changes take the deliberate full-rebuild path.
  W/M are event-driven while idle, and M disables unused RT acceleration
  structures in favor of raster shadows. Empty editor scenes are valid.

- Vulkan 1.3 device/swapchain, synchronization2, dynamic rendering, timeline
  semaphores, asynchronous uploads, VMA resources, and validation.
- Render Graph-managed passes, barriers, layouts, and resource ownership.
- Metallic/roughness PBR, GGX, normal mapping, emissive materials, HDR IBL,
  exposure, a shared real-time AgX display transform, and certified single-level bloom.
- GPU-driven projected-size LOD, meshlets, frustum culling, previous-frame Hi-Z
  occlusion, and indirect-count rendering.
- TAA, motion vectors, contrast-adaptive history weighting, presentation-only
  sharpening, history rejection/reset, resize handling, and scripted temporal
  regressions.
- One shadow-casting sun with three GPU-culled 2048x2048 cascades, 3x3 PCF,
  stable centers, and alpha-cutout shadows.
- Bruneton's precomputed Earth atmosphere now drives the shared raster/path sky
  and Sun. It supports the original nine demo views, interactive Sun placement,
  astronomical day/night playback, and automatic procedural stars/Milky Way.
  A 24-step spherical-shell cloud marcher with four-step self-shadowing updates
  a shared 512x256 RGBA32F environment cache; alpha carries Sun transmittance so
  clouds cover the disk and attenuate raster/path direct sunlight. The cache
  follows wind in real time and costs about 1.7-2.0 ms on the RTX 2060 SUPER.
  NanoVDB remains the planned imported/local sparse-volume representation, not
  the planet-scale weather-cloud implementation.
- Camera-local rain is available through `--rain on` and
  `--rain-intensity 0..1`. `--wind-speed 0..75` and
  `--wind-direction DEGREES` drive one horizontal wind shared with the clouds;
  direction 0 is world +X and 90 is world +Z. Six irregular distance layers produce depth-tested analytic
  streaks after temporal reconstruction and tone mapping, keeping rain out of
  TAA/DLSS/Ray Reconstruction history. Rain falls at 24 m/s in a
  wind-independent monotonic phase; smooth low-frequency gusts and per-layer
  variation change streak slant without ever making drops move upward.
  `--demo weather` runs a continuous 90-second drizzle/downpour/storm/clearing
  showcase with intensity 0-1, wind 3-32 m/s, and a smoothly rotating direction;
  the zero-rain interval raises low analytic steam while accumulated wetness
  evaporates. `--weather-cycle-seconds 12..300` changes its duration.
  Rain, ground-contact, and ripple colors also follow Sun elevation: nighttime
  values are faint cool reflections rather than a fixed bright multiplier, while
  the established daylight response is unchanged.
  Weather materials share three persistent responses between raster and RTX
  closest-hit shading. `rainReceiver` grows stable splash-sized dark spots that
  deepen through repeated analytic deposits and add darkness where they overlap,
  forming a damp surface before bounded glossy puddles appear. No uniform
  completion blend or synchronized spot expansion is used; newly arriving
  rain-impact marks close the last dry gaps through short staggered deposit
  events, with no independent whole-ground damp-opacity ramp or hard saturation
  clamp. Mature neighbouring cells overlap so dry pixels cannot survive the
  impact phase. Impact dampness completes by reservoir 0.72; each raster fragment
  and RTX closest hit then gates pooling from its final local damp value. The full
  future puddle footprint darkens while matte through 0.84. Glossy water begins
  at basin peaks at 0.86 and slowly expands through 1.0. Accumulation is capped
  at 0.025 per second so the late matte and glossy stages each remain visible for
  several seconds in the 90-second demo. Evaporation is 0.033 per second, making
  the full drying sequence about 30 seconds at maximum saturation. The opt-in
  `wetCoat` role grows from patches into a continuous glossy film for roads,
  vehicle paint, grills, and similar surfaces. A session reservoir drives each
  fixed world-space layer; the sphere matrix previews wet coat while the ground
  previews dampness plus puddles.
  The composite samples depth plus world normals
  for short-lived, perspective-correct contact ripples on upward-facing surfaces.
  The combined streak/impact pass measured about 0.33 ms at 1280x720 on the RTX
  2060 SUPER with DLSS Ray Reconstruction. The damp spots are an analytic
  impact pattern; simulated per-drop deposition, drainage, and saved wetness
  state remain deferred.
  The dirt ground's 4K displacement source currently drives a stronger derived
  normal (38.4 gradient scale, 1.15 material normal strength). The 100 km ground
  remains two triangles, so true geometric/RT displacement is still a separate
  near-field geometry/intersection task.
- Honest Vulkan RT capability/device policy, a deterministic RT scene planner,
  static BLAS/TLAS ownership/build code, and synchronized per-frame TLAS
  updates validated on RTX hardware.
- Opaque and alpha-masked directional-light ray-query shadows with reflected
  TLAS/geometry descriptors, explicit update-to-fragment barriers, full-width
  renderer-layer filtering, scale-stable origin offsets, near-source candidate
  rejection, and raster fallback for blended/additive scenes.
- A separate progressive path-traced render mode with ray-pipeline/SBT
  ownership, direct-light visibility at every hit, one-to-four sampled
  diffuse/visible-GGX bounces (default two), throughput-based Russian roulette
  after the second bounce, explicit throughput/PDF accounting, solid-angle
  luminance environment sampling, world-area/luminance emissive-triangle
  selection with power-heuristic MIS, persistent HDR accumulation,
  deterministic resets, renderer-owned path reconstruction outputs, and the
  existing exposure/tone-mapping handoff. Outputs include accumulated beauty
  and direct radiance with derived indirect, albedo/roughness, geometric and
  shading normals, motion, current/expected-previous device depth, and world hit
  distance. A separate compute pass now ping-pongs first-hit geometry metadata,
  validates motion-reprojected history, rejects disocclusions, and emits a
  renderer-owned confidence mask. A second compute pass reprojects ping-ponged
  `RGBA16F` radiance, clamps it to the current 3x3 neighborhood, applies bounded
  motion/convergence weighting, and tracks ping-ponged `RG16F` luminance
  moments. A third pass applies one variance- and AOV-guided 3x3 spatial filter
  whose strength fades with progressive convergence before tone mapping. An
  optional export pass copies nine packed targets once at a selected frame and
  expands them behind the Vulkan-independent `AovFrameSink` into 15 semantic
  linear float planes. The built-in v1 sink writes deterministic PFM files plus
  a JSON manifest; a future EXR sink can implement the same boundary. Path mode
  also supports the deterministic rigid-instance transform script: matching
  frame-local GPU Scene and TLAS transforms carry current/previous matrices,
  moving frames reset progressive accumulation to one sample, and validated
  temporal history remains reusable on matching surfaces while disocclusions
  are rejected.
- A renderer-owned `IUpscaler` contract with strict 1:1 `native`, portable
  `spatial`, and renderer-owned `temporal` reference backends, separate
  render/output extents, canonical
  color/depth/motion/jitter/exposure/reset inputs, and one render-graph boundary
  shared by raster and path modes before tone mapping. Spatial accepts a
  configurable 0.5-1.0 scale and hardware-linearly reconstructs a distinct
  output-resolution linear-HDR target. Temporal adds output-resolution
  color/depth history with jitter-corrected, clipped, depth/motion-aware reuse.
  Quality/balanced/performance modes map to 2/3, 10/17, and 1/2 scale. The ABI
  explicitly declares reconstruction type, history/output ownership, optional
  features, and canonical temporal input conventions without vendor or Vulkan
  types.
- Native Windows NVIDIA support through the signed Streamline 2.12 runtime:
  DLSS Super Resolution owns temporal reconstruction/output at the selected
  quality extent, while DLSS Ray Reconstruction consumes one fresh noisy path
  sample per frame plus renderer-authored depth, motion, diffuse/specular
  albedo, world-normal/roughness, and specular-hit-distance guides. The RTX 2060
  SUPER reports both features supported and executes the graph as `TLAS update
  -> noisy path tracing -> NVIDIA DLSS Ray Reconstruction -> tone mapping ->
  Present`.
- NVIDIA NRD RELAX remains the renderer-owned diagnostic path-denoiser mode. Ray Reconstruction
  bypasses the local temporal, spatial, and NRD stack so histories have one
  owner. Raster and path-traced presentation now share AgX for highlight rolloff,
  hue retention, and shadow latitude without changing linear transport buffers;
  recursive transport no longer multiplies baked glTF AO into traced visibility.

## Concrete Progress Numbers

These are tested examples, not final engine maximums:

| Measurement | Current evidence |
|---|---:|
| Directional-shadow demo objects | 36 |
| Fine-LOD triangles after instancing | 80,642 |
| Unique triangles in its RT plan | 2,306 |
| Reusable RT mesh structures planned | 2 |
| RTX update-capable persistent RT resources | 188,446 bytes |
| RTX path-animation persistent RT resources | 202,270 bytes |
| Path-animation frame-local transform buffers | 2 x 6,912 bytes |
| RTX temporary startup build scratch | 0.0545645 MiB |
| RTX TLAS update scratch per frame slot | 2,432 bytes |
| GPU scene scale-test instances | 32,768 |
| GPU LOD/meshlet work candidates | 262,144 |
| Shadow-casting lights | 1 directional sun |
| Dynamic atmosphere/cloud environment | 512x256 RGBA32F |
| Cloud view/light march | 24 / 4 steps |
| RTX 2060 dynamic cloud cache | 1.7-2.0 ms |
| Raster shadow maps when selected | 3 x 2048x2048 D32 |
| Ray-query directional-shadow on/off RMSE | 0.0467753439165 |
| Ray-query alpha-shadow on/off RMSE | 0.0175814992413 |
| Path beauty + direct accumulation at 1280x720 | 28.125 MiB RGBA32F |
| Path current first-hit AOVs at 1280x720 | 31.6406 MiB |
| Path geometry history + confidence at 1280x720 | 24.6094 MiB |
| Path temporal radiance history at 1280x720 | 14.0625 MiB RGBA16F |
| Path temporal luminance moments at 1280x720 | 7.03125 MiB RG16F |
| Path spatial radiance output at 1280x720 | 7.03125 MiB RGBA16F |
| Total persistent path targets at 1280x720 | 112.5 MiB |
| Optional AOV staging at 1280x720 | 77.34375 MiB host-visible |
| Expanded PFM v1 frame at 1280x720 | 116.0156 MiB plus manifest |
| Path environment alias table at 256x128 | 524,288 bytes |
| Path-tracing one-sample/32-sample RMSE | 0.0395091618743 |
| Path-tracing deterministic repeat RMSE | 0 |
| Depth-1/depth-4 32-sample RMSE | 0.0163448365271 |
| Depth-4 deterministic repeat RMSE | 0 |
| Four-sample importance/uniform environment RMSE ratio | 0.769239 |
| Emissive validation triangles | 2,304 |
| Emissive triangle header/CDF buffer | 73,744 bytes |
| Four-sample emissive sampled/BSDF-only RMSE ratio | 0.723552 |
| Emissive deterministic repeat RMSE | 0 |
| AOV albedo deterministic repeat RMSE | 0 |
| AOV geometric/shading normal RMSE | 0.00323482300546 |
| AOV moving/static motion RMSE | 0.0273575809056 |
| AOV beauty decomposition residual maximum | 0 |
| History invalid/static validity RMSE | 0.814431464956 |
| History invalid/teleport validity RMSE | 0 |
| RTX 1280x720 path-history validation, 32-frame mean | 0.120671 ms |
| RTX 1280x720 path temporal reconstruction, 32-frame mean | 0.18209 ms |
| RTX 1280x720 previous single-pass spatial denoising, 32-frame mean | 0.333075 ms |
| RTX 1280x720 three-scale denoising, 16 moving frames | 0.530902 ms |
| RTX 1280x720 one-sample interactive path stack, 16 moving frames | 3.022354 ms |
| RTX 1280x720 three-sample path tracing, 32 moving frames | 7.78731 ms |
| RTX 1280x720 three-sample interactive path stack, 32 moving frames | 8.722215 ms |
| Same-size minimize/restore | 1421 accumulated samples, 0 path resets |
| RTX 640x360 one-frame path AOV readback | 1.73222-2.88858 ms |
| Path-animation moving pixels at 320x180 | 125, maximum motion 0.00283263 UV |
| Path-animation accepted moving-history mean | 0.952 |
| Path-animation rejected surface pixels | 17, including 6 moving pixels |
| Path-animation frame-slot stress | 120 synchronized updates, 119 resets |
| RTX 320x180 animated TLAS update | 0.0177707 ms |
| RTX 320x180 animated history validation | 0.0105173 ms |
| Temporal/raw adjacent moving-frame RMSE ratio | 0.806044 |
| Spatial/temporal adjacent moving-frame RMSE ratio | 0.876577 |
| Spatial/temporal one-sample reference RMSE ratio | 0.737758 |
| RTX 640x360 emissive path dispatch, 64-frame mean | 0.371332 ms |
| RTX 1280x720 default-two-bounce path dispatch, 32-frame mean | 4.06463 ms |
| RTX 1280x720 textured glTF path dispatch, 32-frame mean | 2.11111 ms |
| Native implicit/explicit raster output RMSE | 0 |
| Spatial repeated raster output RMSE | 0 |
| Spatial quality dispatch, 213x120 to 320x180 | 0.00728 ms |
| Spatial balanced dispatch, 188x106 to 320x180 | 0.006824 ms |
| Spatial performance dispatch, 160x90 to 320x180 | 0.006848 ms |
| Spatial custom 0.75 dispatch, 240x135 to 320x180 | 0.007184 ms |
| Spatial animated-path performance dispatch | 0.007584 ms |
| Native/spatial short animated path dispatch | 0.338204 / 0.218904 ms |
| Temporal raster performance dispatch, 160x90 to 320x180 | 0.0157333 ms |
| Temporal animated-path performance dispatch | 0.017624 ms |
| Temporal output history at 320x180 | 1.31836 MiB |
| Bindless texture descriptors | 64 current fixed capacity |
| Debug tests | 18/18 passing |
| Release tests | 18/18 passing |

No game-ready maximum animated-object count, polygon budget, clustered-light
count, streaming texture budget, or layered-material count has been measured.

## Hardware Validation

Validated on Intel Iris Xe:

- Debug and Release pass all 18 tests.
- Recorded Vulkan core, synchronization, and best-practices validation runs
  report no renderer errors.
- `--ray-tracing auto` and `--ray-tracing off` raster captures match at RMSE 0.
- Intel exposes no AS/ray-query/RT-pipeline support, allocates zero RT memory,
  and uses the certified raster path.

The host loader may report that it skipped its unrelated unusable `dzn` driver.

Validated on NVIDIA RTX 2060 Super 8 GB with driver 595.84:

- Debug and Release pass all 18 tests.
- The device enables acceleration structures, ray queries, and the ray-tracing
  pipeline. Opaque and alpha-masked scenes select the live ray-query shadow
  backend.
- `--ray-tracing auto` builds two BLAS and one TLAS for 2,306 unique triangles
  and 36 instances. The update-capable path uses 188,446 persistent bytes,
  0.0545645 MiB of temporary startup scratch, and 2,432 update scratch bytes per
  frame slot.
- The Debug and Release TLAS gates each record eight static and eight changing
  in-place updates. With shadows disabled to isolate update behavior, all
  captures match RT-off at RMSE 0; the changing path is driven by the explicit
  `--rt-transform-script oscillate` test mode.
- In path mode the same script updates two frame-local transform buffers as
  well as the TLAS. Frame three retains one accumulated sample after two
  deliberate resets, reports 125 moving pixels with maximum motion 0.00283263
  UV, accepts moving history at mean confidence 0.952, and rejects 17 changed
  surface pixels. All 15 animated AOV planes repeat exactly. TLAS update and
  history validation average 0.0177707 ms and 0.0105173 ms at 320x180. A 120-frame
  stress run records 120 synchronized scripted updates and 119 resets through
  repeated frame-slot reuse, with clean core, synchronization, and
  best-practices validation.
- An alpha-enabled 240-frame Release run records 240 updates through repeated
  frame-slot reuse and scripted swapchain recreation without a validation
  warning or error.
- Debug and Release ray-query gates produce finite, deterministic output:
  repeat and automatic/required RMSE are 0, shadows-on/off RMSE is
  0.0467753439165, and ray-query/raster reference RMSE is 0.00547396798572.
- The ray-query graph skips raster shadow culling/drawing. Core,
  synchronization, and best-practices validation report no warning or error.
- Alpha-shadow scenes build one BLAS and one TLAS using 8,206 persistent RT
  bytes and select ray query in automatic and required modes. Repeat and
  automatic/required RMSE are 0, shadows-on/off RMSE is 0.0175814992413, and
  ray-query/raster reference RMSE is 0.00562910967074. Opaque checker cells
  darken while transparent holes remain unshadowed.
- `--render-mode path-traced` builds a reflected ray pipeline with one raygen,
  two miss, and two triangle-hit SBT records. `--path-max-bounces 1|2|3|4`
  selects a pipeline recursion depth of bounce count plus two; the default is
  two bounces and recursion depth 4. A 32-sample 1280x720 Release run is finite
  and bit-identical across repeats, differs from one sample by RMSE
  0.0395091618743. The full validation ray dispatch averages 4.06463 ms against
  a 5 ms gate, including reconstruction-output writes.
- Explicit depth-1 and depth-4 32-sample captures differ by RMSE
  0.0163448365271. Repeated depth-4 output is bit-identical, exercising the
  throughput-based Russian-roulette path under recursion depth 6.
- The path-only 256x128 environment alias table occupies 524,288 bytes. At a
  fixed four-sample 640x360 budget, solid-angle luminance sampling reduces RMSE
  against a 64-sample reference from 0.0391032915832 for uniform-sphere sampling
  to 0.0300797832782, an importance/uniform ratio of 0.769239.
- The emissive-sphere validation scene exposes 2,304 emitter triangles above a
  diffuse receiver. At a fixed four-sample budget, explicit triangle sampling
  reduces RMSE against a 64-sample reference from 0.0444489341774 for BSDF-only
  transport to 0.032161095846, a ratio of 0.723552. Repeated 64-sample output is
  bit-identical, and the 640x360 dispatch averages 0.371332 ms.
- The same dispatch writes accumulated beauty/direct radiance, packed first-hit
  albedo/roughness and geometric/shading normals, plus motion/expected-previous-
  depth/world-distance data and a dedicated current-device-depth image. Indirect
  is derived from beauty minus direct, saving one
  redundant `RGBA32F` target. Albedo repeats at RMSE 0, geometric/shading normals
  differ by RMSE 0.00323482300546, moving/static motion differs by RMSE
  0.0273575809056, and the high-gain lighting residual has maximum 0.
- The path-history compute pass reprojects with `previousUv = currentUv - motion`,
  stores ping-ponged current device depth plus geometric-normal/surface class,
  and rejects invalid history, non-finite or out-of-bounds projections,
  surface/sky transitions, surface depth discontinuities, and geometric-normal
  changes below a 0.85 dot product. Stationary and translating-camera validity
  captures have luminance 0.71266082292 and 0.711792616323. First-frame and
  scripted-teleport rejection captures are bit-identical at RMSE 0. The current
  reconstruction run averages 0.120671 ms at 1280x720 and has a 0.25 ms gate.
- The temporal-radiance pass retains raw `RGBA32F` accumulation as the reference
  and writes two ping-ponged `RGBA16F` reconstructed images plus `RG16F`
  luminance moments. Accepted history is 3x3-neighborhood-clipped and weighted
  by motion plus a four-equivalent-sample convergence cap; first-frame or
  rejected pixels copy raw radiance. First-frame raw/temporal output differs by
  0.000558511296533 RMSE. Adjacent translating frames change by
  0.0603450265089 RMSE raw and 0.0486407735702 temporal, a ratio of 0.806044.
  The pass averages 0.18209 ms at 1280x720 against a 0.25 ms gate.
- The spatial pass uses temporal variance plus depth, geometric normal, albedo,
  roughness, hit distance, and surface class for a bounded 3x3 filter. Against
  the 64-sample temporal reference, one-sample temporal and spatial RMSE are
  0.0591471802356 and 0.0436362789205, a ratio of 0.737758. Adjacent translating
  spatial frames measure 0.0426373774477 RMSE, ratio 0.876577 against temporal.
  The pass averages 0.333075 ms at 1280x720 against a 0.50 ms gate.
  `--path-debug raw-radiance|temporal-radiance|reconstruction-weight|variance`
  exposes each reconstruction stage.
- Damaged Helmet validates five imported material textures through the bindless
  closest-hit set and ray-cone mip selection. Its finite 32-sample output has
  mean luminance 0.535638235354 and averages 2.11111 ms for the ray dispatch.
- The path-tracing validator runs core, synchronization, and best-practices
  validation without warning/error output. Live camera input is disabled for
  automated captures. Two scripted swapchain recreations rebuild all radiance
  and first-hit targets, produce two explicit resets, and finish with 13 valid
  samples after the last reset.
- The upscaling gate proves implicit and explicit native selection are exact at
  RMSE 0 and spatial repeats are exact at RMSE 0. It reconstructs 160x90 HDR to
  320x180, recreates 640x360 render resources for an odd 1279x719 output and
  restores 320x180 render resources for 640x360, and carries eight animated
  path frames through matching GPU Scene/TLAS updates. Quality (213x120),
  balanced (188x106), performance (160x90), and custom 0.75 (240x135) modes all
  produce the 320x180 output under their timing budgets. TAA, bloom, core,
  synchronization, and best-practices validation are clean.

## Milestone 8 Status

Implemented and tested:

- `--ray-tracing off|auto|required` and complete internal capability policy;
  the supported product target itself remains RTX-only.
- Whole primitive/LOD ranges retained alongside raster meshlets.
- One static BLAS plan per reusable primitive group and one TLAS instance per
  rendered object.
- Shared-mesh reuse, finest-resident-LOD selection, index validation,
  alpha/double-sided/mirrored classification, and Vulkan 3x4 transforms.
- Vulkan acceleration-structure ownership, exact size queries, aligned
  non-overlapping scratch, upload dependency, BLAS-to-TLAS barrier, exact fence
  wait, and requested persistent-byte reporting.
- Two frame-local host instance inputs and two frame-local update scratch
  buffers, guarded by frame timeline waits.
- Explicit host-write and prior-AS dependencies followed by an in-place TLAS
  `UPDATE` every submitted frame.
- CPU-tested changed-transform resolution and a deterministic raster TLAS-only
  transform script for RTX validation; path mode additionally updates its
  frame-local GPU Scene transforms as described above.
- A conditional PBR fragment variant using `GL_EXT_ray_query`, set 0 binding 11
  TLAS reflection/update, resident geometry at bindings 12-14, Vulkan-opaque
  fast traversal, and terminate-on-first-hit alpha candidate confirmation.
- Candidate UV reconstruction from triangle indices and barycentrics, bindless
  base-color alpha testing at deterministic mip 0, reversed-direction
  front-face culling, and a geometric-normal scene-scaled origin offset.
- An acceleration-structure-build write to fragment-shader read barrier after
  every TLAS update.
- Full 64-bit renderer-layer propagation; CPU intersection resolves each
  Vulkan 8-bit instance mask to a binary `0xff` or `0` without truncation.
- Opaque/alpha-masked automatic and required selection with explicit raster
  fallback for blended or additive geometry.
- Stage-specific GPU timestamps around the in-place TLAS `UPDATE` and PBR
  geometry draw, plus a matched shadows-off/on Release gate that estimates the
  fragment shader's ray-query traversal cost without changing pipelines.

Not working yet:

- Ray-cone or equivalent alpha-texture LOD selection beyond deterministic mip 0.
- Soft/area-light sampling, local-light ray shadows, and ray-shadow denoising.
- Production animation input at the validated TLAS transform-update boundary.
- Startup BLAS/TLAS build timing and practical RTX object/triangle limits.

The RTX 2060 Super timing gate ran two matched 240-frame Release captures at
1280x720 with required core, synchronization, and best-practices validation.
The 36-instance TLAS update measured 0.0206669 ms with traversal disabled and
0.0186416 ms with traversal enabled. The PBR geometry draw measured 0.0690935 ms
without traversal and 0.172482 ms with traversal, a 0.1033885 ms (149.636%)
incremental estimate. Because traversal is embedded in the PBR fragment shader,
the delta is the defensible effect measurement; it is not presented as a
standalone traversal timestamp. `tools/validate_ray_query_timings.sh` reproduces
the gate and writes both raw reports plus a summary.

Raster shadows remain the certified reference and blended-material backend;
they are not a non-RTX product-support path.

## Progressive Path-Tracing Slice

Implemented and hardware-validated:

- `--render-mode rasterized|path-traced`, with rasterized as the default and an
  explicit AS + ray-pipeline capability gate. `--path-max-bounces 1|2|3|4`
  defaults to two and requires pipeline recursion depth `bounces + 2`.
- A reusable `RayTracingPipeline` owner that builds general/triangle shader
  groups and packs an aligned host-written SBT within the device handle,
  base-alignment, and maximum-stride limits.
- A dedicated graph path: per-frame TLAS update, progressive ray dispatch into
  persistent radiance and reconstruction targets, tone mapping, optional
  capture, and present. The certified raster graph is unchanged when the mode
  is not selected.
- Deterministic per-pixel/per-sample random sequences, incremental averaging,
  finite-value guards, and accumulation resets for camera/view/environment and
  swapchain changes.
- `--demo hdri-backdrop` adds the sphere-grid receiver plane while disabling the
  procedural sun, point light, and directional shadows, isolating HDRI lighting
  and reflections in both raster and path modes.
- Path surface contributions use a bounded scene-linear luminance outlier clamp
  before accumulation. Primary camera rays retain the full HDR environment
  range; the biased realtime bound suppresses persistent low-PDF fireflies on
  glossy materials without altering the visible HDRI background.
- Opaque factor and texture evaluation using the resident instance, material,
  transform, vertex, index, geometry-range, and bindless texture records. Hit
  shaders reconstruct UV0/UV1 and tangent frames, sample base-color,
  metallic/roughness, normal, occlusion, and emissive maps, and select mips from
  a camera ray cone propagated through the sampled bounce. Alpha still fails
  explicitly rather than rendering with incomplete traversal semantics.
- Direct directional-light sampling with a shadow ray at every hit, one-to-four
  diffuse/visible-GGX metallic-roughness mixture bounces, explicit solid-angle
  PDF and `f * absDot / PDF` throughput, throughput-based Russian roulette after
  the second bounce, emissive factors, and HDR lat-long environment misses.
- A path-only alias table weighted by nonnegative environment luminance and
  exact lat-long texel solid angle, uniform-solid-angle sampling within each
  selected texel, direct environment visibility at every hit, and
  power-heuristic MIS between light and BSDF techniques at nonterminal hits.
  Terminal hits use unit light-sample weight. Black maps fall back to
  uniform-sphere sampling.
- A static path-only CDF over visible emissive instances' finest resident RT
  triangles, weighted by world area times emissive-factor luminance. Selection
  uses uniform triangle-area sampling, evaluates emissive textures at the sample,
  converts the PDF to solid angle, traces visibility, and uses complementary MIS
  when BSDF paths hit emitters. Reserved instance words provide O(1) hit lookup;
  `--path-emissive-sampling on|off` retains the measured BSDF-only comparison.
- A reconstruction contract containing accumulated beauty/direct radiance with
  derived indirect, first-hit albedo/roughness, octahedral geometric/shading
  normals, unjittered current-minus-previous motion, expected previous device
  depth, current device depth, and world hit distance. A renderer-owned compute
  pass ping-pongs depth/geometric-normal/surface metadata and emits a validated
  confidence mask. A second pass reprojects, clips, and blends temporal radiance.
  It also tracks first/second luminance moments. Moving frames trace three fresh
  samples per pixel by default and feed back the previous validated denoised
  radiance at a motion-aware weight; `--path-interactive-samples 1..8` exposes
  the quality/performance control. Progressive stationary samples return to one
  fresh sample and reduce that reuse so the accumulator takes over. Three atrous
  passes split and independently filter direct and
  indirect light at step widths 1, 2, and 4.
  They use variance, depth, normal, albedo, roughness, hit distance, and
  luminance edge weights; direct filtering is deliberately tighter, indirect
  filtering is stronger, and environment pixels remain untouched. `--path-debug`
  makes every channel, the lighting residual,
  history validity, raw/temporal/final radiance, applied reconstruction weight,
  and variance capturable without creating a second rendering path. Same-size
  swapchain recovery preserves these offscreen images and history across a
  minimize/restore cycle. The replacement passes shader reflection, SPIR-V,
  render-graph, scripted-motion, core, synchronization, and best-practices
  validation on the RTX 2060 SUPER.
- `tools/validate_path_tracing.sh`, including convergence, bit-exact repeat,
  depth-1/depth-4 transport and depth-4 repeat checks, fixed-budget
  environment and emissive RMSE comparisons, emissive repeat/timing checks, a
  deterministic albedo, normal/motion separation, every inspectable AOV, a
  beauty/direct/indirect residual check, stationary/moving/teleport history
  rejection checks, raw/temporal/spatial stability and reference checks, 5 ms
  stationary path, 10 ms three-sample interactive path, 0.25 ms
  history/reconstruction, and 1.25 ms multi-scale RTX 2060 Super budgets,
  resize resets, sample/TLAS counts, ignored live camera input, and clean Vulkan
  validation.
- `tools/validate_aov_export.sh` writes all 15 linear float32 PFM planes and the
  v1 manifest at frame 8 while simultaneously capturing a PNG. The binary
  checker validates dimensions, channels, finite/range rules, surface/sky
  normals, motion convention, and exact beauty decomposition. Vulkan core,
  synchronization, and best-practices validation are clean; repeated one-frame
  640x360 readbacks measured 1.73222-2.88858 ms against a deliberately loose
  4 ms regression ceiling.
- `tools/validate_path_animation.sh` certifies a static control, fully rejected
  first frame, animated third frame, and bit-exact animated repeat. It checks
  frame-local GPU Scene/TLAS agreement, object motion, one-sample accumulation
  resets, accepted moving interiors, rejected disocclusions, TLAS/history
  timing gates, 120-frame frame-slot reuse, and clean Vulkan validation.

Not working yet:

- Alpha/transmission-aware denoising.
- Adaptive kernels, vendor denoiser backends, production EXR/color-management
  output, and render-job scheduling. Geometry validation, disocclusion
  rejection, bounded temporal reuse, moments/variance, direct/indirect-specific
  multi-scale filtering, and the neutral float AOV sink are implemented.
- Host-facing animation ingestion, runtime instance handles, skinning/morph
  deformation, and deforming BLAS updates. One deterministic rigid-instance
  path now resets accumulation and reprojects validated history correctly.
- Incremental emissive-distribution updates for animated transforms, changing
  materials, and RT LOD/residency changes; the current CDF is rebuilt with the
  static scene.

## Upscaling Baseline and Temporal Reference

Implemented and hardware-validated:

- `--upscaler native` selects the strict 1:1 identity backend and remains the
  default. `--upscaler spatial` selects the portable scaled backend and
  `--upscaler temporal` selects the portable temporal reference;
  `--upscaler-quality quality|balanced|performance` selects 2/3, 10/17, or 1/2
  scale. Quality is the scaled-backend default. `--render-scale` accepts
  0.5-1.0 as a mutually exclusive custom mode. Backend parsing and construction
  do not expose Vulkan or vendor SDK types.
- `FrameInput` carries render/output extents, color, depth, motion vectors,
  current/previous jitter, exposure, optional reactive-mask availability, and
  history reset. Invalid or incomplete native inputs fail deterministically.
- `BackendCapabilities` declares reconstruction type, history/output ownership,
  reactive-mask, sharpening, dynamic-resolution support, and supported scale.
  `InputConventions` fixes scene-linear HDR before exposure, Vulkan `[0,1]`
  device depth, jitter-free current-minus-previous UV motion, render-pixel
  jitter, and a positive linear exposure multiplier.
- Extent-bound raster, temporal, path, bloom, and AOV work now uses the render
  extent, while presentation and capture retain the output extent.
- Raster and path graphs schedule the same upscaling boundary after linear-HDR
  reconstruction/post work and before tone mapping. Native aliases the source;
  spatial dispatches into an output-resolution `RGBA16F` transient image;
  temporal writes ping-ponged output-resolution `RGBA16F` color and `R32F`
  depth history. Raster TAA is bypassed while temporal upscaling is active.
- Temporal reprojects with jitter-corrected canonical motion, clips history to
  the current 3x3 color neighborhood, and bounds reuse using depth variation,
  disocclusion, motion, and explicit resets.
- Raster and path graphs generate an output-resolution `R8_UNORM` reactive mask
  from current HDR/depth/motion and reprojected color/depth history before the
  temporal dispatch. The mask reduces history contribution around changed or
  newly visible content; `--debug-view reactive-mask` exposes it directly.
- Bloom reads the render-resolution HDR result before scaled upscaling and
  is composited during output-resolution tone mapping.
- `tools/validate_upscaling.sh` certifies exact implicit/explicit native output,
  deterministic spatial output, every named extent plus custom 0.75, capability
  and convention diagnostics, odd-size TAA/bloom resize recovery, animated path
  reset/TLAS behavior, timestamp budgets, and clean Vulkan validation.
- `tools/validate_temporal_upscaling.sh` certifies RMSE-0 repeats, first-frame and
  teleport rejection, stationary history acceptance, finite output, odd resize,
  bloom, animated path/TLAS updates, history memory, timestamp budgets, and
  warning-free Vulkan validation.
- Windows Release validation on an RTX 2060 SUPER measured the 1280x720 mask at
  0.878906 MiB transient and 0.089 ms average. Temporal reconstruction averaged
  0.144 ms for the raster moving-content gate and 0.267 ms for a 16-sample path
  capture. A 20-frame stress run recreated temporal history and masks across
  eight even and odd output sizes with clean core, synchronization, and
  best-practices validation.
- Swapchain recreation now preserves the live swapchain when Win32 temporarily
  reports a zero surface extent, and the event loop does not submit one extra
  frame after a close request is polled.

Not working yet:

- Spatial remains bilinear and spatial-only. The temporal reference establishes
  portable correctness and integration behavior, not production vendor-upscaler
  image quality.
- Backend-provided sharpening is not live. Raster TAA applies bounded
  presentation-only sharpening without feeding sharpened pixels back into
  history.
- FSR and XeSS remain future implementations behind `IUpscaler`; DLSS Super
  Resolution and DLSS Ray Reconstruction are live on Windows/NVIDIA.

## Latest Windows RTX / Ray Reconstruction Checkpoint

### 2026-08-31 atmosphere, clouds, and stars update

- Procedural planetary clouds, wind animation, cloud-aware sunlight, and the
  automatic night sky are live in both raster and path modes. Default cloud
  coverage/density are 0.42/0.8; CLI overrides are `--clouds`,
  `--cloud-coverage`, and `--cloud-density`.
- The dynamic environment image must retain `VK_IMAGE_USAGE_STORAGE_BIT` when
  atmosphere is enabled. Its first sampled-to-storage transition waits for the
  asynchronous startup upload; removing either rule causes invalid/no cloud
  cache writes or a transfer/graphics layout-transition hazard.
- Release CTest passes 20/20. A validated 640x360 raster atmosphere run is clean.
  The bounded 1280x720 Sponza/DLSS Quality/Ray Reconstruction run measured
  1.78 ms cloud cache, 16.29 ms path tracing, and 8.69 ms Ray Reconstruction on
  the RTX 2060 SUPER.
- The user's Fluid Ninja-style clone is a useful later reference for local
  smoke/fire fields, but it is a 2D stable-fluid field lofted for volume
  rendering and is intentionally not used for planet-scale clouds. NanoVDB is
  deferred until imported/simulated sparse local volumes are implemented.

This is the exact state at the end of the 2026-08-29 Windows session:

- GitHub `main` and local `main` both point to `b312d7a1e74a78fa32eae4ad7d99e0a0e1d29c9f`
  (`Add Windows RTX path tracing and DLSS Ray Reconstruction`).
- The Release executable built successfully at
  `build/windows-port/bin/Release/genesis_render.exe` using Visual Studio's
  bundled CMake. Release CTest passed **20/20** after extending the shader
  reflection test through Ray Reconstruction bindings 20-23.
- Hardware validation used an NVIDIA GeForce RTX 2060 SUPER with driver 595.79.
  Streamline reported both DLSS Super Resolution and DLSS Ray Reconstruction
  supported. The native Vulkan OFA warning and `NvLowLatencyVk.dll` warning are
  non-fatal for this graph; do not add an unknown DLL to silence them.
- The live Sponza test used the Khronos glTF scene at
  `reference/glTF-Sample-Assets/Models/Sponza/glTF/Sponza.gltf` and the local
  8K `reference/hdri/ulmer_muenster_8k.hdr`. The HDRI is intentionally ignored
  (97 MB test data), as are generated `diagnostic_*.png` captures and the local
  Vulkan capability dump.
- The accepted interactive configuration is 1280x720 output, DLSS Quality at
  853x480 internal resolution, four path bounces, one fresh sample per frame,
  DLSS Ray Reconstruction, +2 EV, and subtle fixed film grain. The graph is:
  `TLAS update -> noisy path tracing -> NVIDIA DLSS Ray Reconstruction -> tone
  mapping -> Present`.
- The last user-approved image has coherent reflected daylight on the Sponza
  floor, cloth, columns, and lower arcade while the enclosed rear passage stays
  darker. Movement performance and reconstruction were a large improvement over
  the previous local/NRD path.

Exact command used for that accepted run:

```powershell
.\build\windows-port\bin\Release\genesis_render.exe --render-mode path-traced --ray-tracing required --scene reference\glTF-Sample-Assets\Models\Sponza\glTF\Sponza.gltf --environment reference\hdri\ulmer_muenster_8k.hdr --camera-start interior --camera-input on --path-max-bounces 4 --path-interactive-samples 1 --path-denoiser ray-reconstruction --upscaler dlss --upscaler-quality quality --resolution 1280 720 --taa off --bloom off --film-grain 0.015 --validation off --exposure-ev 2
```

### Lighting diagnosis that must not be repeated

- Increasing exposure and changing two to four bounces did not initially fix
  the apparently black interior. Raw-radiance presentation looked equally dark,
  proving Ray Reconstruction was not removing the light.
- Fixed 128-frame direct-only and indirect-only captures at 640x360 proved that
  indirect transport exists and dominates the visible interior. The direct
  buffer contained almost only openings/sunlight; the indirect buffer contained
  the floor, arches, cloth, and walls. The real presentation problem was the
  compact ACES fit crushing valid low radiance when bright sun patches were in
  the same SDR image.
- Production presentation now uses the same logarithmic AgX display transform
  for raster and path output. It replaces the path-only shadow boost, retains
  color separation through bright skies and wet reflections, and leaves raw
  scene-linear buffers unchanged. The compact ACES fit remains only for isolated
  diagnostic views.
- The path hit shader no longer multiplies glTF baked ambient occlusion into
  environment or recursive throughput. Ray visibility already accounts for
  occlusion; repeated AO multiplication would bias physical transport darker.
- Do not compensate future sampling failures with an arbitrary ambient term.
  Use raw/direct/indirect captures to distinguish transport, reconstruction, and
  display-transform problems.

### Remaining limitations

- Ray Reconstruction denoises/reconstructs sampled light; it cannot create an
  important indirect path that the one-sample estimator never found. During
  motion, difficult multi-bounce regions can therefore still fluctuate or lag.
- The new shadow-latitude curve is static. There is no luminance histogram,
  automatic exposure, or temporally smoothed eye adaptation yet.
- There is no live performance HUD or target-frame-time dynamic resolution.
  Quality changes still require command-line options and restarts.
- Glass/transmission, broader punctual-light sampling, deforming geometry,
  production animation ingestion, and PBRT runtime scene import remain absent.
  `reference/pbrt-v4-scenes` is a pinned reference/stress source, not currently
  a supported runtime format.
- The signed Streamline SDK is deliberately not committed. A fresh Windows
  checkout must run `tools/setup_streamline.ps1`; the script verifies the
  pinned 2.12.0 archive hash before installation.

## Safest Next Steps

The user selected an NVIDIA/RTX-first direction. Do not spend the next slice on
FSR, XeSS, or another denoiser. The next image-quality bottleneck is getting the
important indirect light sampled consistently before Ray Reconstruction.

1. Implement bounded temporal/spatial reuse for indirect-light candidates—a
   renderer-owned screen-space radiance reservoir/cache in front of Ray
   Reconstruction. Reproject with the existing motion/depth/geometric-normal/
   material guides, reject disocclusions and camera cuts, clamp reuse, and reset
   exactly with the current path-history rules. Keep one fresh path sample per
   pixel so the result remains responsive rather than freezing cached light.
2. Add deterministic Sponza motion gates for that reuse: accepted-history ratio,
   raw versus reused indirect RMSE, disocclusion rejection, camera-teleport
   reset, finite/range checks, and RTX 2060 GPU timing. The success criterion is
   visibly steadier bounced light during movement without ghosting or inventing
   energy.
3. Add target-frame-time dynamic internal resolution around DLSS Quality/
   Balanced limits. Preserve 1280x720 output and use hysteresis so resolution
   does not oscillate. Start with selectable 30 and 45 FPS targets for the RTX
   2060 SUPER.
4. Add a minimal in-app performance HUD: FPS/frame time, path-tracing time, Ray
   Reconstruction time, internal/output resolution, bounce count, fresh spp,
   GPU memory, and current exposure. This is required before further performance
   claims.
5. Replace the static interior shadow lift with histogram-based automatic
   exposure and temporally smooth eye adaptation, retaining manual EV as
   compensation. Validate indoor/outdoor transitions for comfort and flicker.
6. Then expand materials and lighting: transmission/glass, additional punctual
   lights, stronger emissive coverage, and animated/deforming scene tests.

Keep the NRD/reference paths as diagnostic controls, but optimize and
validate the RTX/Ray Reconstruction graph first. The immediate continuation is
step 1, beginning with deterministic Sponza motion/quality gates before enabling
another reuse implementation.

## Clone, Build, and Test

Windows (from a Visual Studio Developer PowerShell with the Vulkan SDK installed):

```powershell
git submodule update --init --recursive thirdparty reference/glTF-Sample-Assets
.\tools\setup_streamline.ps1
cmake --preset windows
cmake --build --preset windows-debug
ctest --preset windows-debug

cmake --build --preset windows-release
ctest --preset windows-release
```

Linux:

```bash
git clone https://github.com/stevenlstarrjr/grevious_engine.git
cd grevious_engine
git submodule update --init --recursive thirdparty reference/glTF-Sample-Assets

cmake --preset debug
cmake --build --preset debug -j4
ctest --test-dir build/debug --output-on-failure

cmake --preset release
cmake --build --preset release -j4
ctest --test-dir build/release --output-on-failure
```

PBRT references are optional for normal builds:

```bash
git submodule update --init reference/pbr-book reference/pbrt-v4
git submodule update --init reference/pbrt-v4-scenes
```

Useful live validation:

```bash
tools/validate_post_processing.sh build/debug captures/current/post_processing_debug
tools/validate_directional_shadows.sh build/debug captures/current/directional_shadow
tools/validate_alpha_shadows.sh build/debug captures/current/alpha_shadows
tools/validate_occlusion_camera_motion.sh build/debug captures/current/occlusion_camera_motion
tools/validate_ray_tracing_fallback.sh build/debug captures/current/ray_tracing_fallback
tools/validate_tlas_updates.sh build/debug captures/current/tlas_updates
tools/validate_ray_query_shadows.sh build/debug captures/current/ray_query_shadows
tools/validate_path_tracing.sh build/release captures/current/path_tracing
tools/validate_aov_export.sh build/release captures/current/aov_export
tools/validate_path_animation.sh build/release captures/current/path_animation
tools/validate_upscaling.sh build/release captures/current/upscaling
```

These scripts require a working Wayland graphical session.

## Reference and Resource Submodules

- `reference/glTF-Sample-Assets`: Khronos integration-test resources.
- `reference/pbr-book`: rendering-theory reference.
- `reference/pbrt-v4`: implementation/provenance reference.
- `reference/pbrt-v4-scenes`: optional offline scenes and future conversion or
  stress-test resources; not a runtime scene format.

See `reference/README.md`. Keep every reference pinned. Updating a reference is
a deliberate change requiring corresponding test/provenance review.

## Architecture Boundaries

- Raster and RT share one GPU scene.
- Render Graph declarations own synchronization wherever practical.
- BLAS represents reusable mesh geometry; TLAS represents object instances.
- The current finest-LOD BLAS policy is a baseline, not permanent RT LOD.
- Bloom's current one-level implementation is a certified baseline; public
  `radius` and `intensity` remain future-pyramid compatible.
- A six-face cube sphere with crack-free Concurrent Binary Tree refinement is
  the planned planetary-terrain architecture. Mesh shaders are only an optional
  backend. Terrain remains planning-only while cloud work is active.
- NVIDIA RTX capability is the mandatory product baseline. Raster paths remain
  useful for material techniques and diagnostic comparison, not as support for
  non-RTX hardware.

## Primary Documents

- `README.md`: build, controls, evidence, and user-facing status.
- `ARCHITECTURE.md`: ownership, pass order, synchronization, post stack,
  shadows, RT foundation, and future terrain compatibility.
- `PBRT_PROVENANCE.md`: PBRT-derived concepts and reviewed revisions.
- `GPU_VENDOR_LIBRARIES.md`: NVIDIA RTX SDK candidates, non-product research
  references, and the renderer-owned ownership rules governing integrations.

Always distinguish:

```text
works now and was actually tested
implemented but awaiting hardware validation
planned for later
```

Report progress using game-development numbers: objects, polygons/triangles,
lights, texture capacity, material limitations, GPU memory, and frame time.
