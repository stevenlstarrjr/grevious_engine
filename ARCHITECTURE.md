# Renderer Architecture

## Scope and direction

This is a Linux/Wayland-only Vulkan rendering engine, not a game engine. It owns GPU
device state, rendering resources, scene representation, render scheduling, and
image reconstruction. It does not own physics, audio, scripting, gameplay,
networking, an ECS, or editor infrastructure.

The delivery sequence is raster PBR, hybrid raster/ray tracing, real-time GI,
then an experimental low-sample path-traced mode. Rasterization remains the
portable primary-visibility path; ray tracing is always optional.

PBRT-v4 and the fourth-edition PBRT book in the parent workspace are references
for radiometry, BSDFs, sampling, and light transport. The local PBRT-v4 scene
corpus supplies reference cameras, lights, environments, materials, geometry,
and rendered targets. Their parser and offline object model are not the runtime
architecture. The glTF sample repository supplies runtime-format test data;
individual asset licenses must be checked before redistribution.

## Ownership boundaries

```text
Renderer application
├── Core              logging, assertions, timing, CPU/GPU profiling interfaces
├── Platform          SDL3 lifetime, Wayland window/events, Vulkan surface setup
├── Vulkan            explicit Vulkan objects and capability records
│   ├── Instance      loader version, layers, debug messenger, surface
│   ├── Device        physical-device policy, feature chain, queues
│   ├── Swapchain     formats, image views, acquire/present lifecycle
│   ├── Resources     VMA-backed buffers, images/views, samplers, uploads
│   ├── Shaders       SPIR-V modules/reflection, descriptors, pipeline cache
│   └── Frame runtime command pools/buffers and synchronization
├── Render Graph      pass DAG, resource states/lifetimes, barriers, queue edges
├── Scene IR          importer-neutral source assets, hierarchy, typed handles
├── GPU Scene         packed instances, meshes/meshlets, lights, indirect work
├── Terrain (future)  CBT refinement, terrain residency, visibility, GPU work
├── Materials         GPU material records, texture bindings, BSDF evaluation
├── Lighting          clustered lists, IBL, direct and sampled-light strategies
├── Pipelines         raster, compute, and optional ray-tracing pipelines
├── Temporal          jitter, motion, histories, rejection, reconstruction
├── Denoising         effect-agnostic temporal/spatial filtering
├── Post Processing   exposure, tone map, bloom, grade, sharpen, optional effects
└── Upscaling         Native/FSR/XeSS/DLSS implementations behind IUpscaler
```

Dependencies point downward: high-level passes may use Vulkan resources, but
Vulkan wrappers never depend on render-graph, scene, material, or effect code.
The Vulkan layer mirrors Vulkan concepts instead of presenting a fake portable
graphics API. Every handle has one owner; borrowed handles are non-owning.

Third-party dependencies have narrow ownership boundaries. SDL3 is used only by
the platform layer for Wayland window/event/WSI integration. Its dynamically
loaded libdecor path supplies client-side decorations when the compositor does
not implement server-side decorations. volk loads Vulkan
entry points without changing the explicit Vulkan API exposed by the renderer.
VMA is private to `MemoryAllocator`; public headers contain no VMA types. GLM is
used for host-side vector/matrix math, while GPU records retain explicit layouts
and compile-time size checks. fastgltf and simdjson are private implementation
dependencies of the glTF adapter and do not appear in renderer public headers.
stb is private to the bounded PNG/JPEG/Radiance-HDR decoder; decoded pixels are
returned as renderer-owned plain data. ThorVG and Yoga are pinned but currently
unlinked, so vector graphics and UI layout do not leak into renderer ownership.
PBRT and glTF reference repositories remain data and theory inputs, never
runtime dependencies.

The render graph owns resources created through its transient API. Imported
resources (swapchain images, histories, persistent scene buffers) keep external
owners. Stable, strongly typed graph handles prevent raw Vulkan handles from
crossing pass declarations. Passes declare image/buffer reads and writes, queue
type, Vulkan stage/access/layout state, and an image subresource range.

Compilation freezes the graph, derives read/write hazard edges, merges explicit
dependencies, rejects cycles, produces a deterministic topological order, and
computes first/last resource use. Execution materializes synchronization2
barriers from the imported initial state and each declared use. The current
`GPU cascade culling -> material-aware directional shadow maps -> Environment + PBR raster ->
optional TAA -> optional Bloom -> Tone mapping -> Present transition` path
contains no handwritten render-target barrier.

Imported resources are rebound with an explicit initial state on each graph
execution. Graph-owned images and buffers retain their final tracked state
between executions. This distinction is required for cross-frame hazards: an
HDR target ends in shader-read state and a depth target ends with a late-fragment
write, so the next frame's writes must source those actual states even when the
new contents will fully replace the old contents.

The initial executor records one command buffer on one physical queue family.
Graphics, compute, and transfer pass types work when those roles alias that
family; a graph spanning distinct families is rejected before recording. A
later multi-queue executor must add submission batching, paired release/acquire
ownership barriers, and timeline edges. Transient resources are owned correctly
but do not alias memory yet; lifetime-based aliasing is an optimization after
the first measured transient-heavy pipeline.

## Scene IR boundary

The Scene IR is the single renderer-owned interchange boundary for source
importers. It represents images, samplers, textures, metallic/roughness
materials, mesh primitives, meshes, cameras, lights, nodes, and selected scene
roots without exposing glTF, PBRT, Vulkan, or scripting-runtime types.

IR objects use distinct 32-bit typed handles. Handles are deterministic,
append-only indices that remain stable for one `Scene`; invalid handles express
optional references and never alias index zero. They intentionally have no
generation counter because the source IR does not remove or recycle objects.
Generation-checked handles belong in the later mutable residency/runtime layer.

Validation checks resource references, attribute cardinality, index bounds,
finite transforms and material values, projection/light constraints, duplicate
parents/roots, and hierarchy cycles with collection/index paths in every error.
Evaluation runs only after validation and produces world transforms plus a flat
mesh-instance list. Renderable nodes carry a defaulted 64-bit visibility mask,
which evaluation copies to each derived mesh instance; imported glTF nodes
therefore remain visible without any format extension or metadata. Primitive
bounds are derived from positions. This keeps
hierarchy traversal and format decoding out of GPU-scene upload code.

The current IR is deliberately a source-stage structure: image payloads remain
encoded, all parentless nodes must be declared roots, and animation, skins,
morph targets, meshlets, LOD, compiled texture payloads, and residency metadata
are deferred. The glTF adapter populates this boundary with core glTF 2.0 and
GLB data. A later asset compiler will convert it into packed runtime/GPU records.

## Visibility layers, material queues, and graph passes

These are independent renderer dimensions:

- A visibility layer says which instances are eligible for a `RenderView`.
  Scene IR nodes and evaluated mesh instances hold a `RenderLayerMask`; GPU
  instances store the same 64 bits as low/high 32-bit words. A view owns a
  64-bit visibility mask. The split GPU representation preserves Vulkan 1.3
  portability without requiring the optional `shaderInt64` device feature.
- A material queue says how a surface is scheduled: opaque, alpha-masked,
  transparent, or additive. glTF alpha modes map into this separate Scene IR
  classification. A vegetation instance can therefore be alpha-masked without
  making “vegetation” a material or pass type.
- A render-graph pass says when and how work executes and continues to own only
  dependencies, resource states, barriers, queue transitions, and transient
  lifetimes. Layer bits are not pass IDs.

The default instance mask is `World`; the default main-view mask admits the
normal named renderer-test layers and excludes `Debug`. Renderer-authored code
may replace either mask directly or with constexpr bit helpers. The first GPU
culling operation intersects the instance's two mask words with the view's two
mask words, before frustum and previous-frame Hi-Z tests. A zero instance or
view mask rejects the instance. No strings, descriptor changes, allocations,
CPU draw filtering, or submissions per layer occur in the frame path.

The importer performs a metadata-only preflight before loading external files.
It permits local resources under the source directory by default, enforces
configurable source/external/aggregate byte and object-count budgets, validates
with fastgltf, and then validates the converted Scene IR. Only
`KHR_mesh_quantization` and `KHR_lights_punctual` are currently accepted.
Unsupported extensions and source features are rejected explicitly so no asset
silently renders with lost semantics. Scene selection imports only nodes
reachable from the chosen/default scene; assets without a scenes array import
all nodes and declare every parentless node as a root.

## Resource ownership and upload policy

The renderer-owned memory allocator returns move-only allocations while VMA
privately manages Vulkan memory blocks, buffer/image granularity, dedicated
allocations, and suballocation. The allocator is created with Vulkan 1.3 and
buffer-device-address support. Host-visible allocations are persistently mapped;
VMA performs non-coherent atom alignment and flush-range correction.

`Buffer` and `Image` own their Vulkan object plus one allocation; `ImageView`
and `Sampler` own only their matching Vulkan handles. The renderer owns
persistent scene resources, while the render graph owns graph-created transient
images and buffers.

Uploads use one staging allocation and one transfer submission per batch. Image
uploads address explicit mip levels and array layers, allowing a complete IBL
mip chain to share one transfer ticket. A
timeline-semaphore ticket connects transfer completion to the first graphics
use without a CPU wait. Resources shared by distinct graphics/transfer families
use concurrent sharing in this baseline. Milestone 2 deliberately supports one
outstanding batch; a measured staging ring is a later optimization.

GLSL is compiled to SPIR-V at build time. Runtime reflection covers shader
stage, vertex input locations, fixed-size descriptor bindings, and runtime
descriptor arrays; it validates the C++ pipeline interface before creation.
Runtime arrays become bounded variable-count, partially-bound,
update-after-bind Vulkan layouts. Descriptor layouts stay explicit Vulkan
objects, and pipeline cache data is persisted per build tree. Push-constant
reflection, controlled permutations, and hot reload are later shader-system
work.

## Vulkan capability policy

The foundation requires:

- Vulkan API 1.3; `VK_KHR_surface` plus `VK_KHR_wayland_surface`, requested
  through SDL3. The SDL X11 backend is disabled at configure time.
- `VK_KHR_swapchain` on the selected device.
- Core 1.3 dynamic rendering and synchronization2.
- Core 1.3 shader demotion to helper invocation. Alpha-mask `discard` requires
  derivative-preserving helper lanes so later implicit texture derivatives are
  valid; immediate invocation termination demonstrably corrupted normal-map
  filtering on Intel Iris Xe.
- Core 1.2 timeline semaphores, buffer device address, draw-indirect count, and
  descriptor indexing features needed by the planned bindless texture table.
- `multiDrawIndirect` and sampler anisotropy.
- A graphics queue and a presentation queue. Dedicated compute and transfer
  families are preferred, with safe aliasing to graphics when unavailable.

Acquire and present use binary semaphores as required by the WSI path. A timeline
semaphore tracks frame-slot retirement. Full-device idle waits are restricted to
shutdown and exceptional swapchain replacement, never steady-state rendering.
The baseline swapchain uses FIFO presentation so validation and visual tests are
paced and inspectable; low-latency present modes will be an explicit runtime
policy rather than an implicit platform-dependent default.

volk initializes before instance creation, loads instance and device dispatch
tables immediately after those objects are created, and finalizes after the
device and instance are destroyed. The Vulkan SDK remains a configure-time
source of headers and `glslc`; the executable does not link the loader directly.

Optional instance tooling is `VK_EXT_debug_utils`,
`VK_EXT_validation_features`, and `VK_LAYER_KHRONOS_validation`. Development and
CI should enable best-practices and synchronization validation when installed.

Optional device paths are detected independently. Milestone 8 now distinguishes
GPU-advertised support from features actually enabled on the logical device:

- `VK_KHR_acceleration_structure` (with deferred-host-operations and BDA)
- `VK_KHR_ray_tracing_pipeline` (with acceleration structures)
- `VK_KHR_ray_query` (with acceleration structures)
- `VK_EXT_mesh_shader`

Acceleration structures are enabled only with deferred host operations and the
required feature bit. Ray-query and ray-pipeline features/extensions are each
enabled only on top of that complete acceleration-structure set. The device
records AS geometry/primitive/instance limits, scratch alignment, ray recursion
depth, and shader-group alignment for later allocation validation. Mesh-shader
support is still detection-only because no mesh-shader owner/backend is live.

No vendor ray-tracing or upscaling extension is foundational. Optional features
retain raster fallbacks on AMD, Intel, NVIDIA, and conforming software Vulkan
devices.

## Milestone 8 ray-tracing foundation — in progress

The public `--ray-tracing off|auto|required` policy is independent from
`--shadows on|off` and from the internal Vulkan mechanism. The current planned
first shadow backend is ray query, but the policy can later select a ray
pipeline without changing the public control. Selection requires both enabled
acceleration structures and enabled ray queries plus a renderer-owned backend:

```text
Request + enabled device capabilities + implemented backend
    -> raster fallback OR ray-query shadows OR explicit required-mode failure
```

`auto` falls back to raster when hardware support is incomplete or the backend
is not implemented. `off` forces raster. `required` rejects either condition
instead of silently rendering a different path. This prevents capability
detection from being confused with a working ray-traced effect.

The scene residency boundary now retains whole primitive/LOD source ranges in
addition to raster meshlets. A deterministic CPU planner selects the finest
resident LOD for the current static baseline, produces one reusable BLAS
geometry plan per primitive group, and produces one TLAS instance per rendered
object. Shared meshes therefore do not duplicate triangle storage. The planner
validates index/vertex ranges, propagates alpha/double-sided/mirrored state,
checks Vulkan's 24-bit instance-index limit, and converts GLM transforms to the
row-major Vulkan 3x4 layout under unit test. The public/runtime LOD policy is
not permanently coupled to this initial finest-LOD choice.

The capability-gated Vulkan build path is implemented and compiles in Debug and
Release. It queries exact BLAS/TLAS sizes, owns each acceleration structure and
its backing buffer, supplies whole-mesh vertex/index device addresses, allocates
separate aligned scratch ranges for concurrent BLAS commands, inserts the
required BLAS-write to TLAS-read barrier, and waits for the exact startup build
fence rather than idling the device. TLAS storage is created update-capable and
the packed instance buffer remains persistent, but per-frame TLAS updates are
not implemented yet. Scratch is temporary; BLAS, TLAS, and instance input are
persistent and report their requested resource bytes on capable hardware.

The Intel Iris Xe host exposes none of the three RT feature sets, so no RT
buffer usage, acceleration structure, scratch buffer, or RT memory is created.
The 36-object directional-shadow demo still produces an inspectable CPU plan of
two reusable mesh structures, 2,306 unique triangles, 36 instances, and 80,642
triangles after instancing. Automatic and explicitly-off captures match exactly
(`RMSE 0`) in Debug and Release with core, synchronization, and best-practices
validation. Policy and build-planner tests simulate complete, partial, missing,
shared-mesh, alpha, transform, and invalid-geometry cases.

The Vulkan BLAS/TLAS command path has not executed on RT hardware yet and is
therefore implemented but **not certified** pending the RTX 2060 Super run.
Ray descriptors, per-frame TLAS updates, shader execution, ray-traced shadows,
RT GPU timing, and measured RTX memory cost remain unfinished. They must not be
described as working based only on compilation, CPU planning, and Intel fallback
tests. The current Vulkan 8-bit instance mask is deliberately left fully open;
the existing 64-bit raster render-layer policy must be preserved by the later
ray-query integration rather than silently truncated.

## Frame and data flow

The steady-state frame polls, acquires, updates, records, submits, and presents.
Command buffers and semaphores are allocated once per frame slot. Submission
waits only for that slot's timeline value. Swapchain images are transitioned with
synchronization2 and rendered inside a Vulkan 1.3 dynamic-rendering scope. The
current scene exercises race-free GPU work-counter reset, projected-size LOD
selection, compute visibility-layer and frustum culling, previous-frame Hi-Z
occlusion culling, atomic visible-work compaction, indirect-count multi-draw
indexed submission, per-frame mapped uniforms, device-local
instance/material/transform tables, and a variable-count bindless texture set.
World, inverse-transpose normal, and previous-world transforms have one fixed
CPU/GPU ABI; mirrored transforms carry an explicit tangent-handedness flag. The
graph constructs a full-resolution
`R32F` max-depth level followed by half-resolution levels down to 1x1; separate
graph images keep every dependency and layout transition explicit until
subresource-granular graph tracking is introduced. It uses an `RGBA16F` color
target, a sampled depth attachment, and a transfer-to-graphics timeline
dependency. Thirty-five spheres sweep metallic and perceptual-roughness values.
A directional light and finite-range point light are evaluated in linear HDR
alongside image-based lighting. The directional light samples three
GPU-generated 2048x2048 raster shadow maps. The IBL path
uses an unclamped lat-long environment, cosine-weighted diffuse irradiance,
eight GGX-prefiltered specular mips, and a two-channel split-sum BRDF LUT. Three
half-resolution graphics passes downsample and blur bright energy before the
bloom result is composited with exposure compensation and ACES-fit tone mapping.
Meshes use counter-clockwise front faces. The camera projection flips Y for
Vulkan framebuffer coordinates, so the raster pipeline retains CCW classification;
changing either side requires changing and visually validating the other.

The target render-graph order is:

```text
Visibility -> Shadow maps -> Depth -> Hi-Z -> GBuffer -> Lighting -> optional RT effects
      -> Temporal/Denoise -> Post -> Upscale -> Present
```

GPU-driven scene flow is `scene buffers -> flattened instance/LOD/meshlet
candidates -> counter reset -> projected-size LOD + frustum culling -> compacted
candidate count -> previous-frame Hi-Z culling -> compacted draw count ->
vkCmdDrawIndexedIndirectCount`. Meshlets bound both unique vertices and triangles
to 64 and retain an object-space sphere plus normal cone. Source glTF currently
produces one LOD per primitive; the procedural certification scene supplies two
real sphere LODs, and the packed runtime ABI accepts arbitrary decreasing
projected-diameter thresholds. CPU per-object submission is not an accepted
scaling path.

The max-depth Hi-Z pyramid belongs to the previous submitted frame. Occlusion
therefore projects static world-space bounds with the previous unjittered
view-projection and computes the bound's nearest depth from the previous camera
position. Using the current camera against previous depth is invalid and caused
whole-instance blinking during camera motion. Previous-frame occlusion rejection
is conservatively bypassed while the camera moves, until a compatible previous
camera exists, and after reset, teleport, or resize; current-frame visibility
layers, LOD selection, and frustum culling remain active. A deterministic
rotating-camera, close-approach, and teleport regression compares the configured
Hi-Z path against an unculled reference at RMSE `0` under Vulkan core,
synchronization, and best-practices validation. Current live scene transforms are static; future
animated transforms must supply previous-frame bounds or bypass previous-frame
occlusion for changed bounds.

## Directional raster-shadow baseline

The current production-raster foundation implements one shadow-casting
directional light independently of optional ray tracing. Its live order is:

```text
GPU scene + directional-light coverage
    -> reset independent shadow work counters
    -> compute per-cascade shadow visibility and compact indirect work
    -> three 2048x2048 shadow maps with material alpha testing
    -> camera-distance cascade selection + 3x3 PCF in HDR PBR
    -> temporal reconstruction and certified post processing
```

Each cascade owns separate visible-instance, indirect-draw, and draw-count
buffers, so camera visibility never incorrectly removes an off-camera caster.
The current near/middle/far splits cover 32%, 62%, and 100% of the camera depth
range. A conservative sphere encloses each perspective-frustum slice with 15%
padding, then its light-space X/Y center snaps to the 2048-map texel size to
reduce camera-motion shimmer. Light-axis depth is independent of spatial radius
and expands to the union of the receiving slice and loaded scene bounds. This
keeps off-camera casters that can project long shadows into the slice without
widening the map and losing near-shadow resolution. The shadow shader reuses
the normal GPU scene vertex, index, instance, transform, material, and bindless
texture records; every cascade remains GPU driven through
`vkCmdDrawIndexedIndirectCount`. Its fragment stage immediately keeps opaque
materials. For alpha-masked materials it evaluates the same base-color texture,
UV selector, factor alpha, and cutoff convention as the PBR pass, then discards
transparent fragments before depth is written. Alpha-blended materials are not
approximated as cutouts.

The three depth images are render-graph transients. The per-cascade shadow work
buffers are persistent renderer-owned imports because they are allocated with
the GPU scene capacity and reused each frame. Compute writes, indirect/depth
reads, material-buffer reads, and sampled alpha-texture reads are fully declared
to the graph, which derives synchronization2 barriers and image layouts. There
is no hand-authored shadow barrier. The comparison sampler uses
`CLAMP_TO_BORDER` with an opaque-white border, so samples outside light coverage
are lit rather than wrapping. Fixed depth bias and normal-dependent receiver
bias limit self-shadowing; the PBR pass averages a 3x3 texel PCF footprint.

`--shadows off` skips shadow compute and depth drawing. Three graph-cleared 1x1
depth fallbacks keep the fixed descriptor array valid and prevent undefined or
previous-frame data from being consumed. The public toggle is independent of
the demo selected with `--demo directional-shadow`.

This is a measured baseline, not the final outdoor-shadow architecture. The
collection of graph-owned depth images, per-cascade culling/work, split
metadata, and cascade selection in PBR are now live without replacing the
Render Graph, GPU Scene, or material contracts. Future public configuration may
change cascade count, resolution, split placement, shadow distance, filtering,
and transition blending without defining material behavior around three fixed
maps. Point and spot shadows require separate light-indexed work and suitable map representations.
Alpha-masked caster silhouettes are now correct. Fractional coverage from
alpha-blended, refractive, smoke, or transmission materials remains future
work; those materials currently cast a solid silhouette. Shadow distance,
resolution, bias, cascade count, and quality are not yet public controls.

The deterministic 36-object demo contains 35 sphere casters and one receiver
plane. Two current 40-frame Release Intel Iris Xe measurements at 1280x720 are
`0.0468-0.0537 ms` for all cascade culling plus `0.7124-0.7949 ms` for all map
draws. The three
active D32 maps account for 48 MiB of transient memory and the demo work buffers
for approximately 0.113 MiB of persistent memory. Static and moving-camera
repeated captures are exact (`RMSE 0`), shadow on/off differs by `RMSE
0.0459746`, the non-shadow PBR golden remains at normalized RMSE `4.59899e-05`,
and Debug/Release tests plus core, synchronization, and best-practices
validation are clean on the current host. The procedural alpha-cutout demo is
also exact across repeats; solid cells darken while transparent holes match the
unshadowed reference within `0.01` luminance. `dzn` remains an external loader
warning, not a renderer validation message. Large-landscape coverage and
cascade-transition quality still require a representative outdoor scene.

## Temporal reconstruction

Milestone 7 adds a reusable temporal boundary without coupling it to ray
tracing or an upscaler. `TemporalState` owns committed current/previous camera
matrices and an explicit history-valid bit. GPU transform records carry current
and previous world matrices, so the motion-vector ABI already covers rigid
object motion even though the current imported demo scenes are static. Raster
positions use an eight-sample, zero-mean base-2/base-3 camera-jitter cycle;
motion vectors use unjittered current and previous projections and follow one
fixed convention: current UV minus previous UV.

Opaque and alpha-masked geometry writes `RGBA16F` motion/expected-previous-depth
data plus a world-space normal and surface classification. The infinite sky
reprojects a homogeneous direction (`w = 0`), so camera translation never
creates false environment parallax. Color, `R32F` depth, and world-normal
histories are double-buffered. Both physical history roles end each submitted
frame in `GENERAL`, giving imported cross-frame render-graph state one truthful
layout regardless of which ping-pong image was most recently written.

The temporal resolve reconstructs the previous sample at
`currentUV - motion + (previousJitter - currentJitter) / extent`. It rejects
history after a global invalidation, invalid projection, non-finite value,
out-of-bounds reprojection, depth mismatch, surface/sky transition, or normal
dot below `0.85`. Accepted HDR history is clamped to the current 3x3
neighborhood and blended with a motion-adaptive weight. Large camera jumps,
large rotations, camera resets, environment changes, visibility-mask changes,
and swapchain recreation invalidate history; resize recreates all extent-bound
histories. `--debug-view motion|history-validity`, deterministic translate and
teleport camera scripts, and `--resize-script` provide inspectable validation
paths.

The current temporal certification covers opaque/alpha-masked rigid geometry,
camera motion, thin-edge disocclusion, the infinite environment, teleport, and
resize recovery. Transparent/additive scheduling, live animation/skinning and
morph transform updates, reactive masks, variance clipping, and temporal
upscaling remain later milestones; their data must feed this same motion and
history contract rather than create parallel temporal systems.

## Certified post-processing contract

The post-processing gate fixes the live ordering as:

```text
Linear HDR PBR
    -> TAA resolve when enabled
    -> bloom extraction + downsample when enabled
    -> horizontal blur
    -> vertical blur
    -> bloom composite
    -> exposure compensation
    -> ACES-fit tone mapping
    -> sRGB swapchain image
```

Bloom therefore consumes TAA-resolved HDR when TAA is on and the raster HDR
target when TAA is off. Bloom itself is not accumulated in a separate temporal
history. This preserves stable highlight sources without feeding blurred bloom
back into TAA. A camera teleport, explicit temporal reset, view reset, or resize
invalidates the HDR history before bloom reads it. Deterministic moving-source
captures verify that this order does not leave persistent bloom trails.

Bloom extraction and downsampling are one half-resolution graphics pass. The
five-sample filter weights the center by `0.5` and four diagonal samples by
`0.125` each. Extraction uses maximum RGB channel brightness, a nominal
threshold of `1.2`, and a `0.6` soft knee. Its support begins at `0.6`: values
at or below that lower knee produce zero, the interval from `0.6` through `1.8`
ramps quadratically, and the hard `brightness - 1.2` response dominates above
the knee. The scalar contribution multiplies the original RGB, preserving hue.
This is a soft-knee convention; “below threshold” does not mean zero throughout
the entire interval below `1.2`.

Horizontal and vertical passes implement the current separable, normalized
linear-sampling Gaussian approximation. The center weight is `0.227027`; paired
samples at offsets `1.384615` and `3.230769` use weights `0.316216` and
`0.070270`. Horizontal offsets divide by the actual half-resolution width and
vertical offsets by its height. The shared linear sampler clamps to edge, never
wraps. Bloom dimensions use ceil-half extents, `max(1, (extent + 1) / 2)`, so
odd source rows and columns remain covered. The final pass upsamples the blurred
image with the same linear clamp sampler and adds it to HDR. The certified
default composite intensity is `0.08`.

The public bloom `intensity` is the final bloom contribution at composite.
Public `radius` is a dimensionless request for perceptual/spatial bloom extent;
`1.0` preserves the certified baseline and larger values request broader bloom.
Neither control promises a texel count, kernel width, pass count, buffer
resolution, or pyramid depth. A private adapter maps perceptual radius to the
current filter scale. A future implementation may instead map the same value to
pyramid depth, per-level filtering, upsample weights, or level contributions
without changing the renderer-facing configuration.

Exposure is a single scalar `exp2(exposureEV)`, applied once after HDR and bloom
are composited. `EV +1` is therefore 2x and `EV -1` is 0.5x in linear space.
The existing ACES-fit curve then clamps display-referred RGB to `[0, 1]`; black
has no offset. `--debug-view hdr` visualizes the pre-bloom resolved input through
the fit, while `--debug-view bloom` visualizes the isolated blurred bloom image
through the fit. These are diagnostic encodings, not alternate production
orders.

The full-resolution HDR, motion, normal, and depth images and current bloom
level images are render-graph transients. Bloom memory and graph ownership are
represented as a collection of levels; certification instantiates one
ceil-half level with `RGBA16F` filtered and scratch images. No bloom image is
persistent. TAA color/depth/normal histories remain persistent renderer-owned
imports because they cross frames. With bloom disabled, the level collection is
empty, the graph records no extraction or blur passes, declares no bloom read,
and the tone shader does not sample its otherwise-valid fallback binding. At
1280x720 the accounted post resources are 7.03125 MiB each for HDR, motion, and
normals, 3.51562 MiB for depth, and 1.75781 MiB for each current bloom image:
28.125 MiB with TAA off. Double-buffered temporal histories add 35.15625 MiB,
for 63.28125 MiB with TAA on. Graph transients currently have correct ownership
but do not alias memory.

The current pass builder consumes the first level because the certified
baseline is:

```text
HDR extraction -> one half-resolution level -> H/V filtering -> composite
```

The level-based private resource and memory representation is deliberately not
a promise that bloom has one level. The Render Graph already accepts repeated
ordinary image reads/writes and pass declarations, so a future implementation
can add a downsample pyramid, per-level filtering, and progressive
upsample/combine without changing Render Graph interfaces or the public bloom
controls.

Every bloom read is declared as a fragment sampled read in
`SHADER_READ_ONLY_OPTIMAL`; every bloom write is a color-attachment write in
`COLOR_ATTACHMENT_OPTIMAL`. Tone mapping declares the selected HDR source,
bloom when enabled, motion debug input, frame uniform, and swapchain output.
The graph derives their synchronization2 barriers. No bloom or post pass adds a
manual image barrier. Normal frame-slot retirement uses timeline semaphores.
Resize is the existing exceptional full swapchain replacement path: it waits for
the device, rebuilds extent-bound graph transients/descriptors and temporal
histories, invalidates history, and recreates presentation semaphores. Steady
state rendering never uses `vkDeviceWaitIdle`; removing the exceptional rebuild
wait requires a generation-retired swapchain/resource owner and is future
lifetime work, not a post-correctness change.

`renderer_post` is the CPU reference for extraction, exact filter footprints,
linear clamp sampling, exposure, ACES fit, finite-value metrics, and resource
byte accounting. `--post-test` replaces only the raster HDR producer with
controlled black, constant, impulse, square, line, multiple-source,
checkerboard, or moving-source data; the live TAA, bloom, tone, graph, capture,
and presentation paths are unchanged. `tools/validate_post_processing.sh`
combines these references with deterministic GPU image metrics, edge/corner
runs, static/moving/reset temporal cases, glTF emissive integration, odd and
repeated resize, validation layers, and timestamp queries.

Correctness defects found by this gate were fixed: odd bloom dimensions now use
ceil-half coverage, bloom-off skips resources/work, and alpha-masked raster
fragments use derivative-preserving helper demotion rather than immediate
termination. The last fix restored the established normal-mapped PBR golden
from normalized RMSE `0.04097` to `4.59899e-05` while remaining validation-clean.

Known quality limits are separate from correctness: the current baseline is
single-scale, fixed-threshold, and uses a small separable kernel. Large public
radius values are mapped into that kernel and therefore cannot reproduce the
smooth, layered haze of a true bloom pyramid. Multi-scale downsample/filter/
upsample bloom remains a future quality implementation, not a permanent
architectural exclusion. There is no lens dirt, anamorphic/FFT/volumetric
bloom, auto exposure, alternate tone operator, or HDR-display output. Core glTF
emissive factors/textures and `KHR_materials_emissive_strength` are live; the
importer regression verifies that the standard strength multiplier reaches the
linear HDR material factor. Procedural inputs remain the authoritative isolated
post tests.

## Future CBT adaptive terrain architecture — PLANNED

Large-scale and planetary terrain is a future renderer milestone; no terrain
runtime is implemented by the current raster-shadow, ray-tracing, temporal, or
post-processing work. Its preferred refinement architecture is a Concurrent
Binary Tree (CBT), or a closely related binary refinement representation chosen
after evaluating the relevant research and reference implementations. CBT owns
compact topology/refinement state, deterministic parallel split and merge,
conforming adaptive triangulation, and bounded GPU traversal. Mesh shaders are
only an optional way to render CBT output; they are not the terrain algorithm.

The planned view path is:

```text
Coarse manifold base mesh (flat, cube sphere, icosahedron, or another topology)
    -> visibility layer and planetary horizon rejection
    -> frustum rejection
    -> projected geometric-error evaluation
    -> GPU CBT split / retain / merge with hysteresis
    -> shared Hi-Z rejection where useful
    -> visible conforming terrain work
    -> optional mesh-task/mesh-shader dispatch OR compute-generated indirect draws
    -> normal depth, PBR, shadow, temporal, and post-processing paths
```

Projected screen-space error is the primary geometric-detail control. Distance
may be an inexpensive input, but fixed distance bands are not the fundamental
LOD model. Planet surfaces are closed topological meshes rather than six
unrelated flat renderers, and horizon culling should reject hidden coarse
regions before detailed refinement. Split/merge thresholds require hysteresis
and deterministic state so geometry, motion vectors, TAA, and shadows do not
flicker as the camera moves from orbital to ground scale. Conforming refinement
is preferred over skirts; transition morphing remains an implementation-time,
measured quality option.

Terrain data keeps four boundaries explicit:

- CBT state is topology and refinement, not height storage.
- Height/displacement pages describe shape independently of CBT nodes.
- Geometry LOD is independent of texture/material mip and residency choices.
- Source formats are compiler inputs, never runtime terrain abstractions.

World Creator exports, procedural data, tiled heightmaps, satellite/elevation
datasets, and authored terrain will eventually enter a terrain importer and
asset compiler. The runtime dataset should contain validated tiles/pages,
geometric error, bounds and min/max height metadata, compression, normal or
gradient inputs, material masks, streaming page metadata, and optional CBT
bootstrap data. A coarse representation stays resident while finer geometry
and displacement pages load asynchronously, fall back safely, and can be
evicted. Terrain shading remains in the renderer's metallic/roughness PBR
system, with future biome, slope, height, and material layers. Virtual or sparse
texturing may supply planet-scale surface pages; the terrain contract must not
assume a few permanently resident giant textures or tie one texture page to one
CBT subdivision.

The terrain subsystem is planned to contain Terrain Asset, CBT State,
Refinement, Streaming, Visibility, GPU Data, Materials, and Rendering
components. Its persistent state remains terrain-owned, while passes and
rendered outputs integrate with the existing Render Graph, GPU profiling,
visibility masks, Hi-Z pyramid, bindless resources, PBR lighting, raster
shadows, temporal reconstruction, and optional ray tracing. Primary and shadow
views may request different refinement error thresholds. An RT terrain study
must evaluate patch/partitioned BLAS, streamed acceleration structures,
procedural intersection, or another bounded representation; rebuilding one
planet-wide highest-detail BLAS every frame is not an acceptable fixed design.

`VK_EXT_mesh_shader` systems may consume visible CBT bisectors through indirect
mesh-task dispatch. The mandatory non-mesh-shader path remains Vulkan compute
work generation followed by indexed indirect rendering, or another measured
portable GPU-driven backend. Neither path may require CPU traversal of every
terrain node each frame. Refinement, culling, work generation, raster cost,
terrain memory, and streaming bandwidth must be profiled independently against
16.67 ms (60 FPS) and 8.33 ms (120 FPS) targets.

Planet-scale coordinates require a later precision layer: high-precision CPU
world positions plus camera-relative/local GPU coordinates, floating-origin or
hierarchical transforms, and stable previous-frame coordinates. The current
32-bit world-space camera/instance records are suitable for present scenes but
are not the final planet-coordinate contract. Other future adjustment points
are persistent compute-generated terrain work/state, streamed stable resource
handles, descriptor capacity beyond the current small fully resident texture
table, and optional mesh-task indirect commands. Current Render Graph resource
and pass declarations can insert terrain compute passes, so these are planned
extensions rather than reasons to fork a terrain renderer now.

Modern CBT adaptive-geometry research, arbitrary-mesh CBT work, planetary CBT
examples, and the SIGGRAPH 2020 continuous-LOD/mesh-shader work are reference
concepts only. If a local `large_cbt` repository is added, it is treated like
PBRT: a reference implementation, not a runtime dependency or architecture to
copy blindly. CBT remains the preferred terrain strategy until implementation
measurements justify a documented change; generic meshlets, meshoptimizer, and
Nanite-style virtualized-geometry ideas may assist adjacent processing but do
not silently replace terrain-specific CBT refinement.

## PBRT-v4 technique disposition

Exact source-to-port traceability is maintained in `PBRT_PROVENANCE.md`; new
PBRT-derived work must update that ledger and its corresponding reference test.

| Technique | Decision | Real-time interpretation |
|---|---|---|
| BSDF evaluation | ADOPT | Preserve physical units, lobe behavior, and reciprocity in GPU functions. |
| BSDF sampling | ADAPT FOR REALTIME | Small fixed lobe sets and GPU-friendly PDFs for RT effects. |
| GGX sampling | ADOPT | Use visible-normal sampling where sampling is needed; raster uses direct evaluation. |
| Fresnel | ADOPT | Exact dielectric/conductor forms for reference; Schlick in the baseline PBR path. |
| Importance sampling | ADAPT FOR REALTIME | Spend very low ray budgets on high-contribution directions. |
| Multiple importance sampling | ADAPT FOR REALTIME | Use for path mode/direct lighting when its cost beats variance. |
| Next-event estimation | ADAPT FOR REALTIME | One or a few selected lights per bounce, with reuse. |
| Environment-map importance sampling | ADOPT | Precomputed luminance distribution with robust PDF conversion. |
| Area lights | ADAPT FOR REALTIME | Analytic or emissive-triangle sampling integrated with light selection. |
| Many-light sampling | ADAPT FOR REALTIME | GPU light lists first, reservoir/tree approaches later. |
| Light BVHs | DEFER | Evaluate after clustered lighting and measured emissive-scene pressure. |
| Sobol sequences | ADAPT FOR REALTIME | Scrambled low-dimensional samples across pixels/frames. |
| Blue-noise sampling | ADOPT | Drive low-spp spatial/temporal decorrelation. |
| Path throughput | ADOPT | Explicit linear-RGB throughput with guarded PDFs and finite-value checks. |
| Russian roulette | ADAPT FOR REALTIME | Only path mode; short fixed bounce caps usually terminate first. |
| Layered materials | DEFER | Start with one metallic/roughness model; retain composable lobe data. |
| Participating media | DEFER | Requires separate bounded-cost integration and temporal treatment. |
| Volumetric scattering | DEFER | Add after opaque lighting/GI; likely froxels plus optional ray queries. |
| Physically based cameras | ADAPT FOR REALTIME | Physical exposure and optional lens effects; raster pinhole remains baseline. |
| Ray differentials | ADAPT FOR REALTIME | Raster derivatives normally; approximate ray cones/differentials for RT LOD. |
| Texture filtering | ADAPT FOR REALTIME | Hardware anisotropic filtering plus ray-cone LOD for traced rays. |
| Acceleration structures | ADAPT FOR REALTIME | Vulkan BLAS/TLAS with update/compaction policy; raster fallback is mandatory. |

Full spectral rendering is REJECTED for the initial and near-term renderer: all
transport is linear RGB. The interfaces should avoid embedding display-referred
assumptions so later spectral experiments remain possible.

## Reference and PBR verification protocol

The local `pbrt-v4` source and fourth-edition `pbr-book` are correctness
references, never linked runtime dependencies. For GPU design, the wavefront
chapter and PBRT work queues inform structure-of-arrays layout, bounded work
queues, and explicit stage scheduling. For shading, PBRT's BxDF interfaces,
Trowbridge-Reitz distribution, Fresnel functions, sampling routines, and PDF
handling provide equations and edge-case references. The renderer will express
those ideas as compact GPU data and shader functions rather than reproduce
PBRT's polymorphic offline architecture.

The IBL reference baker adapts PBRT's visible-normal Trowbridge-Reitz sampling
and cosine-weighted integration to compact real-time lookup textures. It remains
linear RGB and precomputes bounded 2D lat-long textures rather than copying
PBRT's spectral infinite-light object model or per-path sampling machinery. The
current CPU bake is a correctness checkpoint and asset-interface reference;
offline compilation or Vulkan compute can replace it after profiling without
changing renderer-owned decoded image or GPU texture boundaries.

PBRT scene validation is staged by cost and supported formats. Ganesha's small
environment and the Killeroo/LTE Orb lighting setups are focused checks;
Barcelona Pavilion day/night checks environment and area-light balance; Bistro,
San Miguel, and Landscape become culling/residency stress cases only after a
deliberate asset-conversion path exists. OpenEXR decoding is required before the
corpus's HDR environments or reference images enter automated image tests.
Hair, BSSRDF, measured-BSDF, NanoVDB, and large offline path-depth cases remain
theory/reference inputs until their corresponding renderer milestones.

Before declaring the PBR milestone complete:

1. CPU/shader reference tests will check GGX distribution and masking,
   Fresnel-Schlick endpoints, PDF consistency, finite outputs, and energy bounds
   against equations verified from the local PBRT sources/book.
2. Analytic furnace tests and dielectric/metallic parameter sweeps will catch
   energy gain, roughness singularities, and color-space mistakes.
3. Visual tests will use assets from the local `glTF-Sample-Assets` repository,
   starting with `CompareMetallic`, `CompareRoughness`, `CompareNormal`,
   `MetalRoughSpheres`, `MetalRoughSpheresNoTextures`, and
   `NormalTangentTest`.

The baseline math portion is complete. CPU tests cover PBRT Trowbridge-Reitz and
Smith reference values, exact dielectric Fresnel endpoints, Fresnel-Schlick,
reciprocity, grazing/zero-roughness finite behavior, deterministic white-furnace
energy bounds, inverse-square attenuation, and preservation of HDR values. The
shader uses the same equations; SPIR-V reflection locks its resource interface.

Cross-module tests import the local `CompareMetallic`, `CompareRoughness`,
`CompareNormal`, `MetalRoughSpheres`, `MetalRoughSpheresNoTextures`, and
`NormalTangentTest` assets. Every one of the 98 factor-only sphere materials is
evaluated through the reference BRDF. `MetalRoughSpheresNoTextures` is also
packed through the full CPU residency gate, including unified geometry,
primitive draw ranges, instances, material records, and transform history.
PNG/JPEG/HDR decoding and IBL texture residency are implemented and tested, but
general Scene IR material-texture image residency is not; textured glTF assets
therefore still verify source semantics, bindings, geometry, and hierarchy
rather than image-space golden results.
`NormalTangentTest` intentionally omits tangent data; tangent reconstruction is
a required precondition for enabling normal maps in the live shader.
4. Each model's README/license is checked before use or redistribution. Tests
   consume the repository in place; sample assets are not copied into the
   engine by default.

This begins with linear-RGB metallic/roughness glTF semantics. Spectral
rendering, layered BSDFs, media, and offline sample counts are outside the PBR
gate.

## Host and future scripting boundary

The renderer is independent of scripting runtimes. A future host may bind
PocketPy, Lua, mruby, V8, or another VM through a narrow renderer façade made of
stable handles, plain value types, and batched frame-boundary commands. The host
owns VM lifetimes and converts script requests into renderer data; scripts never
own Vulkan handles, allocations, descriptor sets, synchronization primitives,
or render-graph callbacks.

This boundary permits several language adapters without making any one of them
foundational. No VM, language-specific type, reflection system, or scripting
dependency belongs in the renderer library itself.

## Milestone gates

- Milestone 1 is complete: configure/build, repeated present, resize handling,
  and validation are clean.
- Milestone 2 is complete: Debug and Release builds pass, SPIR-V reflection
  tests pass, and the textured direct/indexed demo survives an extended
  validation run without errors.
- Milestone 3 is complete: graph unit tests and Debug/Release builds pass, and
  the scene and present transitions generated by graph declarations survive an
  extended validation run without errors.
- The importer-neutral Scene IR gate is complete: typed-handle, geometry,
  hierarchy, bounds, and evaluation tests pass.
- The bounded glTF/GLB ingestion gate is complete and is exercised against the
  local Khronos sample assets.
- Milestone 4 is complete at baseline: the PBRT-checked CPU/shader math tests,
  depth-tested HDR sphere sweep, direct lights, tone mapping, and a 600-frame
  FIFO-paced Wayland validation run pass without errors.
- Milestone 5 is complete at baseline: bounded PNG/JPEG/Radiance-HDR decoding,
  external or procedural environments, diffuse irradiance, GGX-prefiltered
  specular radiance, BRDF LUT, exposure, and graph-synchronized bloom pass nine
  Debug/Release tests and a 600-frame optimized validation run.
- Milestone 6 is complete at baseline: packed GPU instances/materials/transforms,
  64-bit visibility masks, bounded meshlets, projected-size LOD bands, compute
  layer/frustum and previous-frame Hi-Z culling, atomic work compaction,
  `vkCmdDrawIndexedIndirectCount`, variable-count bindless textures, static
  multi-primitive glTF material/texture residency, and MikkTSpace tangent
  generation pass thirteen Debug/Release tests. A deterministic 32,768-instance
  test expands to 262,144 LOD/meshlet candidates; 240-frame procedural and
  120-frame textured glTF validation runs are clean, and the established PBR
  capture remains within its RMSE threshold. Authored/generated source LODs are
  an asset-compiler extension, not a runtime GPU-scene blocker.
- Milestone 7 is complete at baseline: centered camera jitter, unjittered
  camera/rigid-object motion vectors, infinite-sky reprojection, ping-ponged HDR
  color/depth/world-normal histories, depth/normal/disocclusion rejection,
  neighborhood clipping, motion-adaptive TAA, history resets, and temporal
  debug views pass thirteen Debug/Release tests. A 120-frame translated-camera
  run, 60-frame normal-mapped glTF run, 20-frame alpha-coverage run,
  deterministic teleport captures, and two-recreation resize run are clean
  under core, best-practices, and synchronization validation. Static history
  accepts 99.7% of pixels after convergence, while the scripted teleport
  rejects 100% on the reset frame.
- The pre-Milestone-8 post-processing gate is complete for the current
  single-level baseline. Fourteen Debug/Release tests pass, a 52-run Vulkan
  matrix covers isolated spatial/math/temporal/resize/emissive cases, core and
  synchronization validation report zero errors, best-practices validation has
  no renderer warning, and the established PBR golden remains within tolerance.
  The only host warning is the Vulkan loader skipping the unrelated unusable
  `dzn` ICD. Bloom resource/configuration contracts permit future pyramid
  expansion, but cinematic multi-scale bloom is not implemented by this gate.
  This gate completed before Milestone 8 began.
- The cascaded directional raster-shadow baseline is complete: one sun uses
  three GPU-culled 2048x2048 maps with texel-snapped camera-relative coverage,
  material-aware alpha-cutout indirect drawing, 3x3 filtered PBR sampling, explicit on/off
  behavior, deterministic static/moving-camera comparison, GPU timing and
  memory accounting, and Debug/Release plus Vulkan validation evidence.
  Large-landscape coverage, smoother transitions, local-light shadows,
  blended/transmission shadows, and configurable quality remain production-raster work.
  This raster prerequisite completed before Milestone 8 began.
- Milestone 8 is in progress at the capability/device-policy foundation. A
  complete supported AS/ray-query/ray-pipeline dependency set is enabled on
  capable devices, unsupported or unfinished automatic paths fall back to
  raster, and required mode fails explicitly. Fifteen Debug/Release tests pass;
  Intel automatic/off raster captures are exact under Vulkan validation. No
  BLAS, TLAS, ray shader, or ray-traced image exists yet, and the capable path
  remains unverified until it runs on the RTX 2060 Super.

Each milestone gets a measurable scene, automated smoke path, and validation
evidence before the next begins.
