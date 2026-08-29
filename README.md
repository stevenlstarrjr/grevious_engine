# Genesis Render

### A rendering engine for worlds worth getting lost in

Great graphics are not just about sharper reflections or a bigger polygon
count. They are about the moment a sunrise breaks through the trees, the way a
helmet catches the light, the shadow that warns you something is around the
corner, and a distant landscape that makes you want to find out what is over
the horizon.

Genesis Render is being built for those moments.

This is a from-scratch, real-time Vulkan renderer for Linux and Wayland. It is
not a game yet, and it is not a complete game-development suite. It is the
visual foundation a game can stand on: the part responsible for turning a
world, its materials, its lights, and its atmosphere into the image on your
screen.

## What players should feel

The technology should disappear once the game begins. Genesis Render's player-facing
goals are simple:

- **Believable worlds.** Metal should look like metal, rough stone should
  scatter light differently from polished glass, and bright objects should
  feel like they belong in the same scene as deep shadow.
- **A clear image in motion.** Camera movement should remain stable, edges
  should not crawl, and detail should survive without turning the picture into
  a blurry mess.
- **Long views and dense scenes.** The renderer should spend the GPU's time on
  what the player can actually see, leaving room for more objects, richer
  landscapes, and farther horizons.
- **Lighting with presence.** Sunlight, shadow, emissive surfaces, reflections,
  and eventually global illumination should help shape mood and gameplay—not
  exist merely as checkboxes.
- **A graceful path across hardware.** Rasterization remains the dependable
  foundation. Ray tracing will enhance supported machines without making older
  or integrated GPUs second-class citizens.

## What you can see today

The current build is a technology demo, but its image is already produced by a
real rendering pipeline rather than a collection of mockups. It includes:

- physically based materials, normal maps, emissive surfaces, HDR environment
  lighting, exposure, tone mapping, and bloom;
- temporal anti-aliasing with motion-aware history rejection, so the image can
  remain stable as the camera moves;
- a shadow-casting sun with live ray-query shadows for opaque and alpha-masked
  scenes on capable hardware, plus three filtered raster ranges as the portable
  and blended-material path;
- glTF/GLB scene loading for static meshes, materials, textures, cameras,
  lights, and scene hierarchies;
- GPU-selected detail levels, frustum and occlusion culling, meshlets, and
  indirect drawing to avoid spending work on invisible detail;
- the first ray-tracing effect, including hardware detection, validated
  acceleration-structure builds, synchronized per-frame TLAS updates, and
  visible directional shadows on an RTX GPU; and
- an experimental progressive path-traced mode with camera rays, direct sun
  visibility, one-to-four sampled diffuse/GGX metallic-roughness bounces
  (default two), throughput-based Russian roulette after the second bounce,
  explicit throughput/PDF accounting, bindless opaque material textures with
  ray-cone mip selection, luminance/solid-angle HDR environment sampling with
  power-heuristic MIS, deterministic sample accumulation, and the same
  exposure/tone-mapping output as raster mode. Motion-validated temporal
  radiance reconstruction now stabilizes the live one-sample moving-camera path
  while preserving raw progressive accumulation for inspection. Temporal
  luminance moments and one bounded depth/normal/material-guided 3x3 spatial
  pass further reduce one-sample error and fade as progressive convergence grows;
  and
- a renderer-owned upscaling boundary shared by raster and path modes. Its
  first `native` backend preserves exact 1:1 render/output resolution while
  carrying the color, depth, motion, jitter, exposure, and history-reset
  contract required by later FSR, XeSS, and DLSS implementations.

Opaque and alpha-masked scenes now select ray-query sun shadows automatically
when acceleration structures and ray queries are available. Unsupported devices
and scenes with blended or additive geometry retain the proven raster path;
explicit required mode reports either limitation instead of silently changing
backends.

## What Genesis Render is becoming

The destination is a hybrid renderer that can scale from a strong raster image
to richer ray-assisted lighting. The planned journey moves from the current PBR
and first progressive path-tracing baselines to denoised ray-traced effects and
real-time global illumination.

Future work also aims at large-scale and planetary terrain, more expressive
lighting and shadow types, better material coverage, denoising, temporal
upscaling, and the asset pipeline needed to turn detailed source art into
streamable game worlds.

The promise is not “maximum settings at any cost.” It is a renderer where each
new effect earns its frame time, where optional hardware features have solid
fallbacks, and where a beautiful still frame does not come at the expense of a
game that feels good to play.

## Try the current demo

The demo currently requires Linux, a Wayland session, Vulkan 1.3, a C++20
compiler, CMake 3.25+, Ninja, `glslc`, and the Wayland/libdecor development
libraries.

After cloning the repository, initialize its runtime dependencies and test
assets:

```sh
git submodule update --init --recursive thirdparty reference/glTF-Sample-Assets
```

Build, test, and launch:

```sh
cmake --preset debug
cmake --build --preset debug
ctest --test-dir build/debug --output-on-failure
./build/debug/bin/genesis_render
```

Some useful ways to explore it:

```sh
# A field of objects casting cascaded sun shadows
./build/debug/bin/genesis_render --demo directional-shadow --shadows on

# Require hardware ray-query shadows for that opaque scene
./build/debug/bin/genesis_render \
  --demo directional-shadow --shadows on --ray-tracing required

# Require alpha-tested ray-query traversal and shape-correct cutout shadows
./build/debug/bin/genesis_render \
  --demo alpha-shadow --shadows on --ray-tracing required

# Measure TLAS updates and the matched ray-query shadow cost in Release
./tools/validate_ray_query_timings.sh build/release

# Run the progressive path tracer on an opaque scene
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --demo directional-shadow --path-max-bounces 2 --taa off --bloom off

# Run an opaque textured glTF scene through the same path
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --scene reference/glTF-Sample-Assets/Models/DamagedHelmet/glTF/DamagedHelmet.gltf

# Reproduce convergence, environment/emissive variance, reset, and RTX timing gates
./tools/validate_path_tracing.sh build/release

# Export frame 32 as linear float AOVs while also writing a tone-mapped preview
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --demo directional-shadow --frames 32 \
  --aov-export captures/vfx_shot --aov-frame 32 \
  --capture captures/vfx_shot/preview.png --capture-frame 32

# Validate the binary AOV contract and its measured RTX readback pass
./tools/validate_aov_export.sh build/release

# Animate one rigid instance through TLAS updates and path-history reprojection
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --demo directional-shadow --rt-transform-script oscillate \
  --taa off --bloom off

# Validate object motion, accumulation resets, disocclusion, and exact repeats
./tools/validate_path_animation.sh build/release

# Validate the native 1:1 upscaler contract, graph boundary, and resize handling
./tools/validate_upscaling.sh build/release

# Select the current backend explicitly (native is also the default)
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --demo directional-shadow --upscaler native

# Inspect renderer-owned path outputs used by game reconstruction and VFX AOVs
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --demo directional-shadow --path-debug shading-normal

# Compare raw path radiance with the applied temporal reconstruction weight
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --demo directional-shadow --path-debug raw-radiance
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --demo directional-shadow --path-debug reconstruction-weight
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --demo directional-shadow --path-debug temporal-radiance
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --demo directional-shadow --path-debug variance

# Sample an HDR emissive sphere as an area light over a diffuse receiver
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --demo emissive-sphere --shadows off --path-emissive-sampling on

# Load a static glTF scene with an HDR environment
./build/debug/bin/genesis_render \
  --scene path/to/model.gltf \
  --environment path/to/studio.hdr
```

## Where the project stands

Genesis Render is under active development. Opaque and alpha-masked ray-query shadows
and the first opaque progressive path-tracing slice are validated in Debug and
Release on an RTX 2060 Super. The path tracer supports resident glTF material
textures with ray-cone mip selection, supports one to four sampled
metallic/roughness bounces with Russian roulette after the second, and
importance-samples both the HDR environment and resident emissive triangles with
MIS. The path dispatch exposes accumulated beauty/direct radiance with derived
indirect plus first-hit albedo, roughness, geometric/shading normals, motion,
current/expected-previous device depth, world hit distance, and motion-aware
history validity through `--path-debug`. One compute pass validates ping-ponged
geometry history and disocclusions during stationary and moving-camera frames;
a second reprojects accepted radiance, clamps it to the current 3x3 neighborhood,
applies bounded motion/convergence weighting, and tracks luminance moments. A
third applies one variance-, depth-, normal-, albedo-, roughness-, and
hit-distance-guided 3x3 spatial filter whose strength fades with convergence.
Raw, temporal, and final radiance plus weight and variance remain inspectable.
This is the first bounded spatial reconstruction baseline, not a production
multistage denoiser. `--aov-export` now exposes the shared renderer-owned output
contract through a Vulkan-independent `AovFrameSink`. The built-in v1 sink
writes 15 linear float32 PFM planes plus a JSON manifest: final, raw, and
temporal beauty; direct and derived indirect lighting; albedo; roughness;
geometric and shading normals; motion; current and expected-previous depth; hit
distance; history validity; and variance. PFM is the deterministic portable
baseline; production EXR, color management, render-job scheduling, and offline
denoiser integration remain future sink implementations rather than a second
renderer. A deterministic rigid-instance script now updates matching
frame-local GPU Scene and TLAS transforms, carries `previousWorld` into path
motion, resets the raw progressive accumulator each moving frame, and retains
validated temporal history across accepted moving surfaces. This proves the
renderer-side animation/history boundary, not a host-facing production
animation system.

Both live render modes now converge on `IUpscaler` before tone mapping. The
current native implementation is an identity boundary and therefore does not
yet improve performance or image resolution; reduced internal resolution and
FSR/XeSS/DLSS backends are the next upscaling work.

Production animation ingestion, alpha/transmission path
materials, more complete transport, RT scene scaling, skinning, morph targets,
transparent rendering, terrain, and a game-ready asset streaming pipeline are
still ahead.

This repository deliberately focuses on rendering. Physics, audio, gameplay,
networking, scripting, an entity system, and editor tools belong to the larger
game stack rather than this renderer.

For the engineering details, measured validation results, and contribution
context, see [the current handoff](HANDOFF.md),
[the renderer architecture](ARCHITECTURE.md), and
[the PBRT provenance record](PBRT_PROVENANCE.md).
