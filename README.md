# Bloxel Renderer

Current developer handoff: [`HANDOFF.md`](HANDOFF.md)

A Linux/Wayland-only, real-time C++20 renderer built directly on Vulkan 1.3.
Milestones 1 through 7 are implemented at baseline. Milestone 8 has started
with capability-gated Vulkan ray-tracing device setup and a certified raster
fallback. Static BLAS/TLAS ownership and build recording are implemented but
remain RTX-unverified; ray-traced output is not live yet. The
demo owns the Vulkan
window/device/swapchain, suballocated GPU
resources, asynchronous uploads, SPIR-V shader reflection, descriptors, cached
dynamic-rendering/compute pipelines, and an initial render graph. The graph
orders GPU work-counter reset, projected-size LOD selection, 64-bit
visibility-layer filtering, frustum culling, previous-frame Hi-Z occlusion culling,
GPU-culled directional shadow mapping, environment-backed indirect PBR,
hierarchical-depth construction,
motion/normal output, temporal reconstruction, the certified single-level bloom
baseline, HDR tone
mapping, and presentation; tracks resource
lifetimes and cross-frame state; and emits synchronization2 barriers for
graph-owned and imported resources.
The asset milestone now includes an importer-neutral Scene IR, a bounded glTF
2.0/GLB adapter, and a validated Scene IR-to-GPU packing boundary. It imports
core metallic/roughness materials, encoded images, samplers, textures, static
mesh attributes, cameras, punctual lights, and a selected node hierarchy into
renderer-owned typed handles, then packs geometry, primitives, instances,
materials, visibility masks, and current/previous transforms into fixed-layout
GPU records. `KHR_materials_emissive_strength` is applied to linear emissive
factors before GPU packing.

PBRT-v4, the PBRT book, PBRT-v4 scenes, and Khronos glTF sample assets are
pinned under `reference/`. They are design and test references only and are not
linked into the renderer. See [`reference/README.md`](reference/README.md) for
their roles and selective initialization commands.

## Prerequisites

- CMake 3.25+
- Ninja (for the supplied presets)
- A C++20 compiler
- Vulkan 1.3 loader, headers, and a conforming driver
- A Wayland compositor and Wayland development/runtime libraries
- libdecor development files and a libdecor decoration plugin
- `glslc` for GLSL-to-SPIR-V compilation
- `VK_LAYER_KHRONOS_validation` for validation runs (strongly recommended)

SDL3, volk, Vulkan Memory Allocator, GLM, fastgltf, simdjson, stb, MikkTSpace,
ThorVG, and Yoga are pinned Git submodules. fastgltf is linked privately behind the
renderer-owned importer API, stb is private to bounded image decoding, and the
unmodified MikkTSpace implementation supplies the authoritative tangent-frame
callback algorithm through a renderer-owned adapter. The local simdjson target
prevents dependency downloads during configure. ThorVG
and Yoga are present but deliberately not linked until a renderer-facing vector
or layout use is approved.
After cloning, initialize the runtime dependencies and glTF test resources with:

```sh
git submodule update --init --recursive thirdparty reference/glTF-Sample-Assets
```

The PBRT references are optional for normal builds; initialize them selectively
using the commands in [`reference/README.md`](reference/README.md).

## Build and run

```sh
cmake --preset debug
cmake --build --preset debug
ctest --test-dir build/debug --output-on-failure
./build/debug/bin/renderer_demo
```

Useful development options:

```sh
./build/debug/bin/renderer_demo --frames 120
./build/debug/bin/renderer_demo --validation required
./build/debug/bin/renderer_demo --validation off
./build/debug/bin/renderer_demo --environment path/to/studio.hdr --exposure-ev -1
./build/debug/bin/renderer_demo --scene path/to/model.gltf --visibility-mask 0x3
./build/debug/bin/renderer_demo --frames 8 --capture capture.png --capture-frame 8
./build/debug/bin/renderer_demo --taa on --debug-view history-validity
./build/debug/bin/renderer_demo --bloom on --bloom-intensity 0.08 --bloom-radius 1
./build/debug/bin/renderer_demo --post-test impulse --post-value 16 --debug-view bloom
./build/debug/bin/renderer_demo --demo emissive-sphere --exposure-ev -3
./build/debug/bin/renderer_demo --demo directional-shadow --shadows on
./build/debug/bin/renderer_demo --demo directional-shadow --shadows off
./build/debug/bin/renderer_demo --demo alpha-shadow --shadows on
./build/debug/bin/renderer_demo --ray-tracing auto
./build/debug/bin/renderer_demo --ray-tracing off
./build/debug/bin/renderer_demo --occlusion-culling on --camera-script rotate
./build/debug/bin/renderer_demo --frames 12 --camera-script teleport --resize-script
./build/debug/bin/renderer_demo --frames 40 --resize-script stress
./build/debug/bin/renderer_demo --frames 40 --timing-output post_timings.txt
tools/validate_temporal.sh build/debug captures/current/temporal
tools/validate_post_processing.sh build/debug captures/current/post_processing_debug
tools/validate_directional_shadows.sh build/debug captures/current/directional_shadow
tools/validate_alpha_shadows.sh build/debug captures/current/alpha_shadows
tools/validate_occlusion_camera_motion.sh build/debug captures/current/occlusion_camera_motion
tools/validate_ray_tracing_fallback.sh build/debug captures/current/ray_tracing_fallback
```

`--frames` makes automated smoke tests deterministic. Validation defaults to
`available` in a debug build: the application enables the Khronos layer when it
is installed and emits a warning otherwise. CI validation jobs should use
`--validation required` so a missing layer is a hard failure.

SDL is built with Wayland as its only video backend; X11, dummy, offscreen, and
KMS/DRM video backends are disabled. Window startup also rejects any selected
driver other than Wayland. SDL dynamically loads libdecor for client-side
title-bar and resize decorations on compositors such as GNOME that do not
provide server-side decorations.

## Current validation evidence

- Debug and Release configure/build complete with the strict warning set.
- CLI, shader-reflection, render-graph, Scene IR, and glTF importer tests pass.
- PBRT-checked PBR math, glTF/PBR integration, bounded image decoding, IBL
  reference-baker, GPU scene packing, visibility-mask, temporal-reference, and
  post-processing, ray-tracing policy, acceleration-structure planning, and
  32,768-instance GPU-work scale tests bring the suite to sixteen tests in
  Debug and Release.
- A 600-frame, FIFO-paced optimized Intel Iris Xe bindless/indirect run passes
  Khronos core, best-practices, and synchronization validation with zero errors.
- A decorated 300-frame optimized run and a heavily resized 600-frame Debug run
  validate libdecor-backed GNOME Wayland presentation and swapchain recreation.
- A 300-frame optimized Hi-Z run and a 240-frame Debug transform/material ABI
  run pass Khronos core, best-practices, and synchronization validation with
  zero messages from the renderer path.
- The configured Hi-Z path matches an unculled reference at normalized RMSE
  `0` on rotating-camera frames 8, 16, and 24, close-approach frames 24 and 36,
  and the first post-teleport frame. The regression runs with all 36
  directional-shadow demo objects and core, best-practices, and synchronization
  validation enabled.
- A 240-frame Debug two-LOD procedural run and a 120-frame textured Damaged
  Helmet run pass core, best-practices, and synchronization validation through
  compacted meshlet work and `vkCmdDrawIndexedIndirectCount`. The established
  Damaged Helmet/Hangar regression remains within tolerance at normalized RMSE
  `4.59899e-05` (limit `0.025`).
- TAA-on certification includes a clean 120-frame translated-camera run, a
  clean 60-frame normal-mapped Damaged Helmet run, a 20-frame Khronos
  alpha-coverage run, and a deterministic 1280x720 -> 960x540 -> 1280x720
  history-recreation run. Static history accepts 99.7% of pixels after
  convergence; the scripted teleport rejects 100% on its reset frame.
- The directional-shadow demo renders 35 sphere casters and one receiver plane.
  Repeat captures are exact (`RMSE 0`), the shadow-on/off image delta is
  `0.0459746`, and the established non-shadow PBR regression remains at
  normalized RMSE `4.59899e-05` with shadows explicitly disabled.
- A separate two-quad alpha-shadow scene proves material-aware cutout shadows:
  opaque checker cells darken by at least `0.05` luminance, transparent holes
  remain within `0.01` of the shadows-off reference, shadow on/off differs by
  normalized RMSE `0.013191`, and repeated Debug and Release captures are exact
  (`RMSE 0`).
- On Intel Iris Xe, `--ray-tracing auto` reports no acceleration-structure or
  ray-query support and selects raster shadows. Its Debug and Release captures
  match explicit `--ray-tracing off` at `RMSE 0` under core, synchronization,
  and best-practices validation. `--ray-tracing required` stops with a clear
  unsupported-hardware error rather than creating an invalid device/path.

## Milestone 8 ray-tracing foundation — in progress

What works now is honest capability/device setup plus a tested CPU build plan.
The renderer distinguishes features advertised by the GPU from features
actually enabled on the Vulkan device. On a GPU exposing the complete
dependency set it enables
`VK_KHR_acceleration_structure`, deferred host operations, ray queries, and the
ray-tracing pipeline, then records useful limits such as maximum BLAS
primitives, TLAS instances, scratch alignment, recursion depth, and shader-group
alignment. Incomplete extension/feature sets are never presented as usable.

The scene now retains whole mesh/LOD triangle ranges alongside raster meshlets.
The static baseline plans one reusable triangle structure per mesh group and one
top-level entry per object, validates all triangle indices, handles shared
meshes, alpha-tested materials, double-sided/mirrored instances, and verifies
the Vulkan object-transform layout. On the current 36-object shadow demo that
means two reusable mesh structures, 2,306 unique triangles, and 80,642 triangles
after the meshes are instanced 36 times.

A capability-gated Vulkan implementation now owns BLAS/TLAS storage, queries
exact build sizes, uses correctly aligned non-overlapping scratch ranges, and
records the required BLAS-to-TLAS synchronization. The TLAS is allocated for
future updates and persistent RT resource bytes are logged. This code compiles
in Debug and Release, but its actual Vulkan commands remain **RTX-unverified**
until they run with validation on the RTX 2060 Super.

`--ray-tracing off|auto|required` expresses policy without changing the meaning
of `--shadows on|off`. `auto` is the default and safely selects the certified
raster shadow backend on this Intel GPU. `off` always uses raster. `required`
fails if the requested backend cannot truly run. This prevents a capable RTX
device from being reported as actively ray tracing before the renderer owns a
working acceleration structure and ray-query path.

What does not work yet: the BLAS/TLAS command path has not been run on RT
hardware, TLAS transforms are not updated each frame, no ray-query or
ray-pipeline shader is dispatched, and no ray-traced shadow is visible. There is
still no measured RTX object, polygon, memory, or frame-time limit. The next M8
slice is RTX validation of the static build followed by safe TLAS updates and a
basic ray-query shadow pass. Intel continues using the complete raster path and
allocates zero RT memory.

## Directional raster-shadow baseline

What works now is one real-time shadow-casting directional light: the sun.
`--shadows on|off` controls it explicitly. When enabled, the GPU builds a
separate shadow-visible indirect draw list for each of three camera-distance
ranges, renders three 2048x2048 depth maps, and the PBR pass selects the range
and applies a 3x3 filtered shadow test. Map centers move in whole shadow-texel
steps to reduce camera-motion shimmer. Each range also extends along the sun
direction across the loaded scene bounds, so an off-camera object can still
cast a long shadow into view. Shadow casters are not limited to objects visible
to the camera. Alpha-masked materials sample their base-color alpha and cutoff
in the shadow pass, so a leaf, fence, or cloth cutout casts its actual cutout
shape instead of its full rectangular polygon. Opaque materials take the same
path without sampling a texture. When disabled, shadow culling and drawing are skipped and
cleared 1x1 descriptor fallbacks prevent stale or undefined data from being
sampled.

The `--demo directional-shadow` scene contains 36 rendered objects: 35 spheres
casting shadows onto one plane. At the fine sphere LOD that is about 80,642
visible triangles. It is a correctness and visual demo, not the renderer's
maximum object or polygon claim. The separate GPU-work scale test currently
proves 32,768 instances expanding to 262,144 LOD/meshlet work candidates; a
game-ready maximum animated-object or polygon budget has not yet been measured.

Two current 40-frame Release runs on Intel Iris Xe at 1280x720 measured
`0.0468-0.0537 ms` for all three GPU culling passes and `0.7124-0.7949 ms`
for all three shadow-map draws, or `0.7592-0.8486 ms` combined. Three active
D32 2048x2048 maps use
48 MiB of graph-transient GPU memory; the 36-object demo's persistent per-range
work buffers use about 0.113 MiB. Debug and Release each pass all sixteen
tests. Static and scripted moving-camera repeat captures are exact (`RMSE 0`).
Live regression ran with
Vulkan core, synchronization, and best-practices validation enabled and
reported no renderer validation error. The host Vulkan loader still reports
and skips its unrelated unusable `dzn` driver.

This is a cascaded sun-shadow baseline, not production-complete shadows. It now
has the basic near/middle/far structure needed for long outdoor view distances,
but a large landscape has not yet been used to certify its practical shadow
distance. Planned later are configurable distance/split/resolution/quality,
smoother range transitions, point- and spot-light shadows, shadow LOD, and
transparent/blended-material shadowing. The current version supports one
shadow-casting sun, three fixed 2048 maps and fixed bias. Alpha masking is live;
partially transparent glass and smoke still cast a solid silhouette because
blended/transmission shadows need a different technique. Ray-traced shadows
remain optional future work and are not required for this path. Milestone 8
capability/device setup has begun, but this shadow output is still entirely
rasterized.

## Post-processing certification

The pre-Milestone-8 certification gate is complete on the current Intel Iris Xe
host and covers the current baseline:

```text
HDR/TAA source -> extraction + ceil-half downsample -> H blur -> V blur
               -> bloom composite -> exposure -> ACES fit -> sRGB output
```

The gate was completed before Milestone 8 began. `--bloom on|off` controls
ownership and scheduling;
off creates no bloom level images and records no bloom passes. The default
threshold is `1.2` with a `0.6` soft knee, so extraction support begins at
`0.6`. `--bloom-intensity` is the final composite contribution. The
dimensionless `--bloom-radius` requests perceptual/spatial extent (`1.0` is the
certified baseline); it is deliberately not specified in texels, filter taps,
pass count, resolution, or pyramid levels.

Deterministic CPU references and live GPU captures cover black/dark inputs,
threshold boundary values, single impulses, independent horizontal and
vertical filtering, controlled downsample patterns, edge/corner placement,
odd extents, monotonic HDR energy, finite extreme inputs, exposure EV steps,
ACES isolation, bloom off/on, static and moving TAA, history reset, camera
motion, resize reconstruction, and repeated resize stress. Current measured
results on Intel Iris Xe include exact black and dark-scene zero, exact
horizontal/vertical/axis impulse symmetry, exact repeated-run and static-TAA
agreement, no reset-frame ghost, and a measurable bloom-on delta. The
DamagedHelmet/Hangar PBR regression remains at normalized RMSE
`4.59899e-05` against the established golden (`0.025` limit).

The current 1280x720 single-level resources are two ceil-half `RGBA16F` graph
transients at 1.75781 MiB each. Accounted full-resolution post targets plus
bloom total 28.125 MiB with TAA off. Double-buffered temporal color/depth/normal
histories add 35.15625 MiB, totaling 63.28125 MiB with TAA on. The latest
Release baseline on this host measured 0.259499 ms for extraction/downsample,
0.351862 ms horizontal blur, 0.193483 ms vertical blur, and 0.360812 ms for
composite/exposure/tone mapping over 32 samples.

The current filter is a correctness baseline, not the final bloom architecture.
Bloom graph and memory ownership are represented as levels, so a future
downsample pyramid, per-level filtering, and progressive upsample/combine can
reuse the same Render Graph declarations and public intensity/radius controls.
The present small single-scale kernel cannot match the broad layered haze of
cinematic multi-scale bloom; stretching it far enough exposes separated sample
bands. Lens dirt, anamorphic/FFT/volumetric bloom, auto exposure, alternate tone
operators, and HDR-display output remain outside this gate.

The Milestone 5 path passes 600 paced frames with SDL3 Wayland windowing, volk
dispatch, VMA-backed resources, an `RGBA16F` target, a depth attachment,
directional and inverse-square point lighting, ACES-fit tone mapping, and
graph-generated synchronization under all three validation feature groups. It
adds linear HDR environment radiance, cosine-weighted diffuse irradiance,
GGX-prefiltered specular mips, a split-sum BRDF LUT, exposure compensation, and
a restrained half-resolution bloom chain.
VMA is contained behind the renderer-owned allocation API; it is not exposed to
passes or resource users. FIFO is the baseline presentation mode so visual tests
are inspectable; faster modes will be explicit runtime policy.

The loader reports and skips an unusable system D3D12/`dzn` ICD on this host;
that loader warning is not emitted by the renderer or the validation layer.
AMD and NVIDIA still require runtime certification on representative hardware.

The glTF ingestion gate uses the pinned
`reference/glTF-Sample-Assets` submodule directly rather than copying its data.
Current coverage
includes external-buffer glTF, image-bearing GLB, hierarchy evaluation, and the
98-material `MetalRoughSpheresNoTextures` sweep. Unsupported extensions,
animation, skins, morph targets, and unsupported vertex semantics fail with
actionable diagnostics instead of being ignored.

The PBR reference implementation uses GGX/Trowbridge-Reitz, PBRT's Smith joint
masking-shadowing form, Fresnel-Schlick, energy-conserving diffuse/specular
weighting, linear HDR lighting, and glTF's `alpha = roughness²` convention. The
local comparison assets now exercise importer/material compatibility and all 98
factor-only sphere materials exercise the CPU BRDF.
The exact PBRT-v4 revision, symbols, realtime adaptations, and validation links
are recorded in `PBRT_PROVENANCE.md`.

The image boundary decodes bounded PNG, JPEG, and Radiance HDR inputs into
renderer-owned data. The demo accepts `--environment`; without it, a
deterministic procedural HDR studio is used. CPU preprocessing is currently a
measurable reference path (153 ms in the optimized PBRT-environment run) that
will later move to the asset compiler or GPU preprocessing without changing the
runtime texture interface.

The pinned `reference/pbrt-v4-scenes` corpus supplies PBRT reference configurations,
textures, geometry, HDR environments, and reference renders. Near-term checks
will use the compact Ganesha environment, Killeroo area lighting, and Barcelona
Pavilion day/night setups; Bistro, San Miguel, and Landscape are later
GPU-scene stress cases. Most HDR assets are OpenEXR, which is the next bounded
image-decoder format gate. `.pbrt`, PLY, measured-BSDF, and NanoVDB files are not
accepted as runtime scene formats.

The completed baseline Milestone 6 path has compact GPU
instance/material/transform tables, 64-bit instance/view visibility masks,
bounded 64-vertex/64-triangle meshlets with bounds and normal-cone metadata,
projected-size LOD bands, race-free counter reset, compute layer/frustum
culling, previous-frame max-depth Hi-Z construction and occlusion culling,
atomic visible-work compaction, multi-draw indexed-indirect-count submission,
and a 64-slot variable-count bindless texture set. The CPU
residency gate packs complete Scene IR geometry, primitive draw metadata,
hierarchical world/normal/previous transforms, conservative world bounds, and
the full glTF material texture ABI. The live demo accepts static multi-primitive
`.gltf`/`.glb` scenes and their material textures; missing tangent frames for
normal-mapped primitives are generated through MikkTSpace. Material mip chains
use semantic-aware filtering for sRGB color, linear data, and tangent-space
normals. Source glTF primitives receive a single-LOD meshlet hierarchy; the
procedural certification scene supplies two real geometry LODs. Authored LOD
ingestion and offline LOD generation belong to the later asset compiler, while
the runtime ABI and GPU selector already accept multiple levels.

The Hi-Z pyramid represents the previous submitted frame, so occlusion bounds
are projected with that frame's unjittered camera and previous camera position.
Occlusion rejection is disabled while the camera moves, until a compatible
previous camera exists, and after a camera reset, teleport, or resize. Frustum
culling remains active during movement. `--occlusion-culling on|off` provides
an unculled correctness reference, while `--camera-script rotate|approach`
supplies deterministic rotation and close-approach regressions. Live scene objects are currently static; a
future animated-object path must provide previous-frame bounds or skip Hi-Z
rejection for bounds that changed.

The completed baseline Milestone 7 path uses a centered eight-sample
base-2/base-3 camera-jitter sequence, jittered raster projection, and
unjittered current/previous camera and rigid-object transforms for motion.
Geometry writes current-minus-previous UV motion, expected previous depth, and
world normals; the infinite environment uses direction-only reprojection so
camera translation produces no false sky parallax. The render graph resolves
into ping-ponged `RGBA16F` HDR color, `R32F` depth, and `RGBA16F` world-normal
histories, preserving a truthful cross-frame layout for both physical roles.
History is rejected for invalid/out-of-bounds reprojection, depth mismatch,
surface/sky changes, normal discontinuity, non-finite input, and explicit
camera/view/resource invalidation. Accepted history is clipped to a 3x3 current
neighborhood and weighted down with screen-space motion. Alpha-masked fragments
use derivative-preserving helper demotion, so rejected coverage writes no color,
motion, depth, or normals without corrupting neighboring implicit derivatives.

`--taa on|off`, `--debug-view final|motion|history-validity`,
`--camera-script none|translate|teleport|rotate|approach`, and `--resize-script`
expose the validation paths. Imported scenes remain static, but the GPU transform ABI and
CPU reference tests cover rigid previous/current object motion. Live animation,
skinning/morph motion, transparent/additive queues, reactive masks, temporal
upscaling, and denoising remain later milestones and must reuse this temporal
contract.
