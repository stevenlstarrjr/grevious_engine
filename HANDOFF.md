# Renderer Handoff

Last updated: 2026-08-28

Repository: <https://github.com/stevenlstarrjr/grevious_engine>

Branch: `main`

## Scope

This is a Linux/Wayland-only, real-time C++20 Vulkan renderer, not a complete
game engine. PBRT-v4 and the PBRT book are rendering references, never runtime
dependencies. Do not add unrelated physics, audio, scripting, networking, ECS,
or editor systems here.

## What Works Now

- Vulkan 1.3 device/swapchain, synchronization2, dynamic rendering, timeline
  semaphores, asynchronous uploads, VMA resources, and validation.
- Render Graph-managed passes, barriers, layouts, and resource ownership.
- Metallic/roughness PBR, GGX, normal mapping, emissive materials, HDR IBL,
  exposure, ACES-fit tone mapping, and certified single-level bloom.
- GPU-driven projected-size LOD, meshlets, frustum culling, previous-frame Hi-Z
  occlusion, and indirect-count rendering.
- TAA, motion vectors, history rejection/reset, resize handling, and scripted
  temporal regressions.
- One shadow-casting sun with three GPU-culled 2048x2048 cascades, 3x3 PCF,
  stable centers, and alpha-cutout shadows.
- Honest Vulkan RT capability/device policy, a deterministic RT scene planner,
  and compiled static BLAS/TLAS ownership/build code. The GPU commands remain
  RTX-unverified.

## Concrete Progress Numbers

These are tested examples, not final engine maximums:

| Measurement | Current evidence |
|---|---:|
| Directional-shadow demo objects | 36 |
| Fine-LOD triangles after instancing | 80,642 |
| Unique triangles in its RT plan | 2,306 |
| Reusable RT mesh structures planned | 2 |
| GPU scene scale-test instances | 32,768 |
| GPU LOD/meshlet work candidates | 262,144 |
| Shadow-casting lights | 1 directional sun |
| Shadow maps | 3 x 2048x2048 D32 |
| Bindless texture descriptors | 64 current fixed capacity |
| Debug tests | 16/16 passing |
| Release tests | 16/16 passing |

No game-ready maximum animated-object count, polygon budget, clustered-light
count, streaming texture budget, or layered-material count has been measured.

## Hardware Validation

Validated on Intel Iris Xe:

- Debug and Release pass all 16 tests.
- Recorded Vulkan core, synchronization, and best-practices validation runs
  report no renderer errors.
- `--ray-tracing auto` and `--ray-tracing off` raster captures match at RMSE 0.
- Intel exposes no AS/ray-query/RT-pipeline support, allocates zero RT memory,
  and uses the certified raster path.

The host loader may report that it skipped its unrelated unusable `dzn` driver.

An NVIDIA RTX 2060 Super 8 GB is available on another PC for later validation.
Do not call the BLAS/TLAS path certified until it actually runs there with
Vulkan validation.

## Milestone 8 Status

Implemented and CPU-tested:

- `--ray-tracing off|auto|required` and complete optional-device policy.
- Whole primitive/LOD ranges retained alongside raster meshlets.
- One static BLAS plan per reusable primitive group and one TLAS instance per
  rendered object.
- Shared-mesh reuse, finest-resident-LOD selection, index validation,
  alpha/double-sided/mirrored classification, and Vulkan 3x4 transforms.
- Vulkan acceleration-structure ownership, exact size queries, aligned
  non-overlapping scratch, upload dependency, BLAS-to-TLAS barrier, exact fence
  wait, and requested persistent-byte reporting.

Not working yet:

- RT hardware validation of static BLAS/TLAS builds.
- Per-frame TLAS transform updates.
- Ray-query descriptors/shaders and visible ray-traced shadows.
- Alpha-texture evaluation during ray traversal.
- Preservation of the 64-bit renderer layer policy through Vulkan's 8-bit
  instance mask. Do not silently truncate layers.
- RTX memory, build/update timing, and practical object/triangle limits.

Raster shadows remain the selected and certified output backend.

## Safest Next Steps

1. Build on the RTX 2060 Super with Vulkan validation enabled.
2. Run `--ray-tracing auto` and confirm BLAS/TLAS counts, resource bytes,
   scratch bytes, synchronization, and clean destruction.
3. Compare automatic/off captures; they must remain identical before a ray
   shader consumes the TLAS.
4. Add deterministic, synchronized per-frame TLAS updates.
5. Add the first ray-query sun-shadow path behind the existing policy while
   retaining raster fallback.
6. Measure objects, instanced triangles, RT memory, build/update time, and frame
   cost before starting reflections.

Do not start RT reflections, GI, path tracing, terrain implementation,
cinematic bloom replacement, or vendor upscalers during this slice.

## Clone, Build, and Test

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
- CBT is the planned terrain refinement architecture. Mesh shaders are only an
  optional backend. Terrain remains planning-only.
- Raster operation on Intel, AMD, and NVIDIA must not depend on optional RT or
  mesh-shader support.

## Primary Documents

- `README.md`: build, controls, evidence, and user-facing status.
- `ARCHITECTURE.md`: ownership, pass order, synchronization, post stack,
  shadows, RT foundation, and future terrain compatibility.
- `PBRT_PROVENANCE.md`: PBRT-derived concepts and reviewed revisions.

Always distinguish:

```text
works now and was actually tested
implemented but awaiting hardware validation
planned for later
```

Report progress using game-development numbers: objects, polygons/triangles,
lights, texture capacity, material limitations, GPU memory, and frame time.
