# Genesis Render

### A rendering engine for worlds worth getting lost in

Great graphics are not just about sharper reflections or a bigger polygon
count. They are about the moment a sunrise breaks through the trees, the way a
helmet catches the light, the shadow that warns you something is around the
corner, and a distant landscape that makes you want to find out what is over
the horizon.

Genesis Render is being built for those moments.

This is a from-scratch, real-time Vulkan renderer for Windows and Linux. It is
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
- **Built for RTX hardware.** Genesis targets NVIDIA RTX-capable graphics cards
  exclusively, allowing ray tracing, DLSS/Ray Reconstruction, and CUDA-backed
  techniques to be treated as first-class platform capabilities.

## What you can see today

The current build is a technology demo, but its image is already produced by a
real rendering pipeline rather than a collection of mockups. It includes:

- physically based materials, normal maps, emissive surfaces, HDR environment
  lighting, exposure, tone mapping, and bloom;
- temporal anti-aliasing with motion-aware history rejection, contrast-adaptive
  history weighting, and presentation-only sharpening, so the image remains
  stable without unnecessarily softening stationary detail;
- a shadow-casting sun with live ray-query shadows for opaque and alpha-masked
  scenes, plus three filtered raster ranges for blended materials and reference
  comparison;
- Eric Bruneton's precomputed Earth atmosphere with interactive Sun placement,
  astronomical time-of-day playback, a real-time procedural spherical cloud
  layer, cloud-aware sunlight attenuation, and an automatic nighttime star and
  Milky Way field. Clouds share one 512x256 dynamic HDR cache between raster
  and path modes instead of ray marching every camera pixel;
- a depth-aware camera-local rain field composited after temporal reconstruction,
  with physical fall speed, six irregular distance layers, stable analytic edge
  filtering, depth/normal-aware surface impacts, and rain-receiver materials
  that darken and reduce roughness for damp ground and shallow puddles. Shared
  directional wind advects clouds and controls rain slant, while the rain field
  retains a monotonic downward phase through gust and direction changes. A
  Sun-elevation response keeps rain and splash reflections faint at night
  instead of rendering them as self-lit white streaks;
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
  radiance reconstruction now stabilizes three fresh moving-camera samples per
  pixel by default while preserving raw progressive accumulation for inspection.
  Temporal luminance moments and three bounded depth/normal/material-guided
  atrous passes further reduce error and fade as progressive convergence grows;
  and
- a renderer-owned upscaling boundary shared by raster and path modes. The
  default `native` backend preserves exact 1:1 output, while the portable
  `spatial` backend renders at 0.5-1.0 scale and reconstructs a distinct
  output-resolution linear-HDR target before tone mapping. The renderer-owned
  `temporal` reference backend adds output-resolution color/depth history,
  jitter-corrected motion reprojection, neighborhood clipping, and an
  output-resolution reactive mask that bounds reuse around changed or newly
  visible content. Quality, balanced, and performance modes map to 2/3, 10/17,
  and 1/2 render scale; an explicit `--render-scale` retains custom testing. The
  backend contract names history and output ownership, reconstruction type,
  optional features, and exact
  color/depth/motion/jitter/exposure conventions without vendor types. Native
  Windows builds add NVIDIA Streamline DLSS Super Resolution and a path-traced
  DLSS Ray Reconstruction mode that consumes fresh noisy radiance plus depth,
  motion, diffuse/specular albedo, normal/roughness, and specular-hit-distance
  guides;

Opaque and alpha-masked scenes now select ray-query sun shadows automatically
when acceleration structures and ray queries are available. Scale-stable ray
origins and near-source candidate rejection prevent receiver self-shadowing.
Scenes with blended or additive geometry retain the proven raster shadow path;
explicit required mode reports that material limitation instead of silently
changing backends. GPUs without the required NVIDIA RTX feature set are outside
the supported hardware target.

## What Genesis Render is becoming

The destination is an RTX-focused hybrid renderer that combines rasterization
where it remains the best representation with ray-traced lighting and
reconstruction. The planned journey moves from the current PBR and first
progressive path-tracing baselines to denoised ray-traced effects and real-time
global illumination.

Future work also aims at large-scale and planetary terrain, more expressive
lighting and shadow types, better material coverage, denoising, temporal
upscaling, and the asset pipeline needed to turn detailed source art into
streamable game worlds.

The promise is not “maximum settings at any cost.” It is a renderer where each
new effect earns its frame time and fully uses the defined RTX platform without
letting a beautiful still frame come at the expense of a game that feels good
to play.

## Try the current demo

The demo supports native Windows and Linux. Both platforms require an NVIDIA
RTX-capable graphics card with Vulkan 1.3 acceleration-structure and ray-tracing
pipeline support, a compatible NVIDIA driver, a C++20 compiler, CMake 3.25+,
and `glslc`.

- Windows requires Visual Studio with the Desktop development with C++ workload
  and the Vulkan SDK. The standalone executable uses SDL, while embedders can
  link `Genesis::Renderer` and provide a native `RenderSurface` such as the
  supplied Win32 HWND surface without linking SDL.

### Editor embedding

`Genesis::Renderer` is the SDL-independent rendering library used by Genisis
Studio's Qt viewport. A host supplies an in-memory `scene::Scene`, camera, native
surface, and optional `EditorOverlayFrame`. Raster editor hosts can enable
`dynamicSceneUpdates` and call `updateSceneSnapshot` to stream geometry into
reserved GPU buffers without recreating the swapchain, pipelines, or IBL bake.
Clean rendered views leave editor overlays disabled. Empty editor documents are
supported through an internal zero-visibility residency anchor and do not need
host-authored placeholder geometry.
- Linux requires Ninja, a Wayland session, and the Wayland/libdecor development
  libraries.

After cloning the repository, initialize its runtime dependencies and test
assets:

```sh
git submodule update --init --recursive thirdparty reference/glTF-Sample-Assets
```

On Windows, open a Visual Studio Developer PowerShell and run:

```powershell
.\tools\setup_streamline.ps1
cmake --preset windows
cmake --build --preset windows-debug
ctest --preset windows-debug
.\build\windows\bin\Debug\genesis_render.exe
```

For an optimized Windows build, use `windows-release` and launch
`.\build\windows\bin\Release\genesis_render.exe`.

Smart App Control can block locally built executables that are not signed by a
publisher it trusts. Production binaries should be signed with a trusted
code-signing certificate; disabling Smart App Control is not a build step. See
[Microsoft's explanation](https://support.microsoft.com/en-US/Windows/Security/Threat-Malware-Protection/smart-app-control-has-blocked-part-of-this-app).

On Linux, build, test, and launch with:

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
  --demo directional-shadow --path-max-bounces 2 \
  --path-interactive-samples 3 --taa off --bloom off

# Run an opaque textured glTF scene through the same path
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --scene reference/glTF-Sample-Assets/Models/DamagedHelmet/glTF/DamagedHelmet.gltf

# Render Khronos Sponza under the interactive Bruneton atmosphere
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --demo atmosphere \
  --clouds on --cloud-coverage 0.42 --cloud-density 1.0 \
  --time-of-day on --time-hour 16 --day-cycle-seconds 120 \
  --scene reference/glTF-Sample-Assets/Models/Sponza/glTF/Sponza.gltf

# Atmosphere controls: Ctrl+left-drag moves the Sun, +/- changes exposure, and
# 1-9 selects the original Bruneton demo presets. Right-drag and WASD retain the normal camera controls
# for free-fly navigation through imported scenes such as Sponza.
# Time-of-day controls: T pauses/resumes, [ and ] slow down/speed up the cycle,
# and , and . scrub backward/forward by 15 minutes. Manual Sun dragging pauses
# playback; resuming continues smoothly from the manually placed Sun.
# Stars fade in automatically after sunset. Clouds use a 1.5-7.0 km planetary
# shell and move at 18 m/s by default; --clouds off retains the exact clear-sky
# atmosphere path.

# Add camera-local rain after temporal reconstruction. Intensity is normalized;
# tagged receiving materials become wet and upward-facing visible surfaces show
# short-lived contact ripples. Wind direction uses world-space horizontal
# degrees: 0 is +X and 90 is +Z. The same 18 m/s wind moves the clouds.
# Scene materials can opt into `rainReceiver` for capped puddles plus spreading
# dampness, or `wetCoat` for a patch-to-continuous glossy water film.
./build/release/bin/genesis_render \
  --demo atmosphere --rain on --rain-intensity 0.65 \
  --wind-speed 18 --wind-direction 14

# A zero speed provides a calm reference without changing the chosen direction.
./build/release/bin/genesis_render \
  --demo atmosphere --rain on --wind-speed 0

# Cycle through drizzle, downpour, storm, and a dry clearing interval. Over the
# default 90 seconds, intensity moves from 0 to 1 and back to 0, wind moves from
# 3 to 32 m/s, wetness accumulates and then evaporates, and low steam rises
# during the post-rain interval. Use --weather-cycle-seconds 12 for a fast cycle.
./build/release/bin/genesis_render \
  --demo weather --weather-cycle-seconds 90 \
  --render-mode path-traced --ray-tracing required \
  --path-denoiser ray-reconstruction --upscaler dlss \
  --upscaler-quality quality --taa off

# Capture a deterministic camera/altitude/Sun matrix and a labeled contact sheet.
# Add -CloudDebug transmittance to inspect cloud opacity, or shafts to inspect
# normalized additive shaft illumination without the beauty composite.
./tools/capture_atmosphere_debug_sweep.ps1

# Capture a Sun-facing cloud-edge shot as native beauty, production RR beauty,
# cloud opacity, and normalized shaft illumination in one labeled comparison sheet.
./tools/capture_light_shaft_debug.ps1

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

# Validate native and spatial raster/path upscaling, timing, and resize handling
./tools/validate_upscaling.sh build/release

# Validate temporal history, motion rejection, resize, timing, and repeatability
./tools/validate_temporal_upscaling.sh build/release

# Select the exact 1:1 backend explicitly (native is also the default)
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --demo directional-shadow --upscaler native

# Use the named half-resolution performance mode
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --demo directional-shadow --upscaler spatial \
  --upscaler-quality performance

# Use the portable temporal reference at half resolution
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --demo directional-shadow --upscaler temporal \
  --upscaler-quality performance

# Use native NVIDIA DLSS Super Resolution on Windows
.\build\windows\bin\Release\genesis_render.exe --render-mode path-traced --ray-tracing required --demo directional-shadow --upscaler dlss --upscaler-quality quality

# Feed one fresh path sample per frame into NVIDIA DLSS Ray Reconstruction
.\build\windows\bin\Release\genesis_render.exe --render-mode path-traced --ray-tracing required --demo directional-shadow --upscaler dlss --upscaler-quality quality --path-denoiser ray-reconstruction

# Add optional post-tone-map film grain (disabled by default; zero disables it)
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --demo directional-shadow --film-grain 0.015

# Light the sphere grid and matte backdrop exclusively from an HDRI
./build/release/bin/genesis_render \
  --render-mode path-traced --ray-tracing required \
  --demo hdri-backdrop --environment path/to/studio.hdr

# Explicit render scale remains available as a custom diagnostic mode
./build/release/bin/genesis_render \
  --demo pbr-grid --upscaler spatial --render-scale 0.75

# Keep a 1280x720 SDL window while path tracing internally at 1920x1080
./build/release/bin/genesis_render \
  --resolution 1280 720 --upscaler spatial --render-scale 1.5 \
  --render-mode path-traced --ray-tracing required \
  --demo atmosphere \
  --scene reference/glTF-Sample-Assets/Models/Sponza/glTF/Sponza.gltf

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

## Shot look controls

These controls are available on the Windows and Linux command line and through
`RendererConfig`. Both raster and path tracing accept the three look controls.

| Option | Effect |
| --- | --- |
| `--visible-sky-ev -6` | Darken the entire camera-visible environment by six stops. Environment lighting and reflections use the original radiance. Default is zero. |
| `--clay` | Preview the geometry with neutral 18% diffuse gray, zero metallic and 0.85 roughness. The original normals, alpha cutouts and lamp emission remain active. |
| `--white-balance-k 3200` | Set the camera white balance for a warm illuminant. The resulting image is cooler. Higher values warm the image. The default of 6500 K is neutral. Valid range is 2000–12000 K. |

White balance uses green-normalized linear RGB gains before the final AgX display
transform. The linear AOVs and diagnostic views retain their original values.

### Object and material ID passes

Add `--id-aovs` to a path-traced `--aov-export DIRECTORY` job. The selected frame
contains numeric `object_id.pfm` and `material_id.pfm` planes, lossless RGB
selection images `object_id.png` and `material_id.png`, and `id_legend.csv`
with names, IDs and swatches. In Fusion, load the PNG as data with no color
conversion and pick the desired RGB color. Use nearest-neighbor sampling if resizing.

Background has ID zero. glTF node and material IDs are their one-based scene
indices. An object's primitives share its object ID, and instances using the same
material share its material ID. The IDs are stable across captures of the same
scene layout. The mattes sample the visible opaque or alpha-tested surface with
a dedicated pixel-center ray and have hard edges. They use the internal render
resolution, like the other path AOVs. The default AOV export keeps its 15 original planes.
Use the native NRD path when exporting AOVs. DLSS Ray Reconstruction has a
separate output pipeline.

```powershell
.\build\windows\bin\Release\genesis_render.exe --render-mode path-traced --scene shot.glb --environment night.hdr --visible-sky-ev -6 --white-balance-k 5200 --id-aovs --aov-export build/shot-passes --frames 64 --aov-frame 64 --capture build/shot.png --capture-frame 64
```

To verify the controls with a real HDR environment, run
`python tools/validate_look_controls.py <renderer> <hdr-file> <test-output-directory>`
with NumPy and Pillow installed. It compares matched raster/path captures and
checks the named mattes, alpha cutouts, shared materials and preserved emission.

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
applies bounded motion/convergence weighting, and tracks luminance moments.
Moving frames reset the current-view accumulator and trace three fresh samples
per pixel by default, then feed back only depth/normal-validated, previously
denoised radiance with a stronger motion-aware weight. The
`--path-interactive-samples 1..8` option tunes that motion budget; stationary
frames return to one new progressive sample and progressively reduce feedback so
the fresh accumulator takes over. A
three-scale variance-guided atrous stage separates direct and indirect light,
then filters both at step widths 1, 2, and 4 using depth, normal, albedo,
roughness, hit distance, luminance, and temporal variance. Direct lighting keeps
tighter edge weights while indirect lighting receives the stronger filter, and
both strengths fade as samples converge. Background/environment pixels remain
unfiltered. Raw, temporal, and final radiance plus weight and variance remain
inspectable. Same-size swapchain recovery, including minimize/restore, preserves
the offscreen images and accumulated history. This is the vendor-neutral
interactive multi-scale reconstruction baseline,
not an adaptive or vendor-specific production denoiser. `--aov-export` now
exposes the shared renderer-owned output
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

Both live render modes now converge on `IUpscaler` before tone mapping. Native
remains the exact identity control. Spatial is the first real scaled path: it
allocates an output-resolution HDR target and performs a hardware-linear
fullscreen reconstruction from a configurable 0.5-1.0 render scale (default
quality mode at 2/3). Named quality, balanced, and performance modes are stable
at 2/3, 10/17, and 1/2 scale. Backend capabilities explicitly describe
identity/spatial/temporal reconstruction, history and output ownership,
reactive-mask, sharpening, dynamic-resolution support, and canonical temporal
input conventions. Spatial establishes portable resource and scheduling
semantics, not the image quality of a temporal upscaler. The renderer-owned
temporal reference and reactive-mask input are live. Windows additionally has
signed-runtime NVIDIA Streamline DLSS Super Resolution and DLSS Ray
Reconstruction adapters; FSR/XeSS and backend-provided sharpening remain future
work.

Production animation ingestion, alpha/transmission path materials, more
complete transport, RT scene scaling, skinning, morph targets, spatially simulated
wetness deposition/drainage/drying, local fog/smoke volumes, imported NanoVDB fields, transparent rendering,
terrain, and a game-ready asset streaming pipeline are still ahead.

This repository deliberately focuses on rendering. Physics, audio, gameplay,
networking, scripting, an entity system, and editor tools belong to the larger
game stack rather than this renderer.

For the engineering details, measured validation results, and contribution
context, see [the current handoff](HANDOFF.md),
[the renderer architecture](ARCHITECTURE.md), and
[the PBRT provenance record](PBRT_PROVENANCE.md).
