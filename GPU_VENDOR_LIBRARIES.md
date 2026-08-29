# GPU Vendor Libraries and SDKs for Genesis Render

## Purpose

This document catalogs NVIDIA, AMD, and Intel graphics/rendering libraries and SDKs that may be useful to Genesis Render, a Linux/Wayland Vulkan renderer. Vendor technologies should remain optional components behind renderer-owned interfaces rather than becoming architectural foundations.

Genesis Render should continue to use Vulkan as the common low-level graphics API, retain a portable raster path, and provide fallbacks when vendor-specific features are unavailable.

---

## NVIDIA

| Library / SDK | Purpose | Potential Genesis Render Use | Priority |
|---|---|---|---|
| **DLSS Super Resolution / DLAA** | Temporal AI upscaling and anti-aliasing | `IUpscaler` backend | High |
| **DLSS Ray Reconstruction** | Neural reconstruction/denoising of ray-traced effects | Future RT reflections/GI/path-traced reconstruction | High, later |
| **DLSS Frame Generation** | Generates intermediate frames | Future high-end performance option | Later |
| **NVIDIA Real-Time Denoisers (NRD)** | Denoising for low-sample ray-traced lighting, shadows, AO, and reflections | RT shadow/reflection/GI denoising | High |
| **NVIDIA Reflex** | Reduces system/input latency | Future latency subsystem | Later |
| **Streamline** | Framework for integrating supported rendering features | Evaluate as an integration option where Linux/Vulkan support fits | Evaluate |
| **Nsight Graphics** | GPU debugging, frame inspection, shader/performance analysis | Development and profiling | High |
| **Vulkan RTX extensions/features** | Hardware ray tracing and related GPU capabilities | BLAS/TLAS, ray queries, ray pipelines | Already in use / core optional path |

### NVIDIA integration direction

```text
Genesis Render
  -> renderer-owned interfaces
      -> DLSS / DLAA
      -> NRD
      -> Ray Reconstruction
      -> Reflex
  -> Vulkan
      -> NVIDIA GPU
```

NRD is particularly relevant once Genesis Render begins using very low ray counts for reflections, GI, soft shadows, or other noisy effects.

---

## AMD / GPUOpen

| Library / SDK | Purpose | Potential Genesis Render Use | Priority |
|---|---|---|---|
| **FidelityFX Super Resolution (FSR)** | Temporal/spatial upscaling | `IUpscaler` backend | High |
| **FidelityFX Denoiser** | Denoising for ray-traced effects | RT shadow/reflection denoising | High |
| **FidelityFX SPD (Single Pass Downsampler)** | Efficient GPU mip/downsample generation | Hi-Z, bloom pyramids, environment maps, SSR structures | High |
| **FidelityFX CAS** | Contrast Adaptive Sharpening | Post/upscaling sharpening | Medium |
| **FidelityFX CACAO** | Ambient occlusion | Optional AO implementation/reference | Medium |
| **FidelityFX SSSR** | Stochastic screen-space reflections | Hybrid reflection path/reference | Medium |
| **FidelityFX Brixelizer / Brixelizer GI** | GPU spatial representation and real-time GI | Research/reference for future GI | Medium/Research |
| **FidelityFX Hybrid Shadows / Reflections** | Hybrid raster/ray-traced effects | Reference for Genesis Render hybrid effects | Medium |
| **FidelityFX LPM** | HDR/display and tone-mapping utilities | Reference/optional post-processing work | Low/Medium |
| **FidelityFX Variable Shading** | Variable-rate shading support | Future performance optimization | Later |
| **TressFX** | Hair/fur simulation and rendering | Character rendering | Later |
| **Radeon Rays** | Ray-intersection acceleration library | Reference/tooling; probably unnecessary as Genesis Render owns Vulkan BLAS/TLAS | Low |
| **Render Pipeline Shaders (RPS)** | Render-graph/render-pipeline framework | Study for render-graph ideas, aliasing and scheduling | Reference |
| **Vulkan Memory Allocator (VMA)** | Vulkan memory allocation/suballocation | Renderer memory management | Already in use |
| **GPUPerfAPI** | Hardware performance counters | GPU performance analysis | High |
| **Radeon GPU Profiler / Radeon Developer Tool Suite** | GPU profiling and debugging | Development and performance work | High |

### SPD is especially relevant

Potential uses include:

```text
Hi-Z pyramid
Bloom pyramid
Environment-map mip generation
SSR hierarchy
Future hierarchical GPU effects
```

Genesis Render already owns its render graph, so AMD RPS should primarily be treated as an implementation/reference source rather than replacing the existing graph.

Radeon Rays likewise should not replace Genesis Render's native Vulkan acceleration-structure and ray-query implementation unless a specific future requirement justifies another abstraction layer.

---

## Intel

| Library / SDK | Purpose | Potential Genesis Render Use | Priority |
|---|---|---|---|
| **XeSS Super Resolution (XeSS-SR)** | Temporal AI super resolution | `IUpscaler` backend | High |
| **XeSS Frame Generation (XeSS-FG)** | Generated intermediate frames | Future performance option | Later |
| **Xe Low Latency (XeLL)** | Latency reduction | Future latency subsystem | Later |
| **XeSS Inspector / developer tooling** | Validate/debug XeSS integration and inputs | Upscaler development | Medium/High |
| **Open Image Denoise (OIDN)** | ML-based image denoising | Progressive path tracer and offline/reference denoising | High for path tracing |
| **Embree** | High-performance ray-tracing kernels | CPU reference renderer, ray-tracing validation, tools | Medium/High |
| **Open PGL** | Path guiding | Future advanced path tracer | Research/Later |
| **Open VKL** | Volume computation/rendering | Future volumetric workloads | Specialized/Later |
| **oneAPI / SYCL** | Heterogeneous compute framework | Specialized compute/tools if needed | Low for core renderer |
| **Intel Graphics Performance Analyzers / VTune** | GPU/CPU profiling | Performance analysis | High |

### XeSS integration direction

XeSS belongs behind the same renderer interface as FSR and DLSS:

```text
IUpscaler
├── Native
├── Spatial
├── TemporalReference
├── FSR
├── XeSS
└── DLSS
```

Genesis Render's temporal infrastructure should provide the common inputs required by temporal upscalers, including low-resolution color, motion vectors, depth, jitter, exposure information where required, and reset/history state.

Current implementation status: the renderer-owned `IUpscaler`, strict 1:1
`Native`, portable `Spatial`, and portable `TemporalReference` backends are live
in both raster and path graphs. Spatial and temporal accept a 0.5-1.0 render
scale; temporal owns bounded output-resolution color/depth history and consumes
the common color, depth, motion, jitter, exposure, and reset contract. Quality,
balanced, and performance modes map to 2/3, 10/17, and 1/2 scale. The contract
fixes linear-HDR color, Vulkan device depth, jitter-free UV motion, render-pixel
jitter, exposure/reset data, reconstruction type, history/output ownership,
reactive-mask, sharpening, dynamic-resolution capability, and supported scale.
FSR, XeSS, DLSS, reactive-mask generation, and sharpening remain deferred
backends rather than implied capabilities.

### Embree as a reference backend

Embree does not need to replace Vulkan RT. A more useful role is independent CPU/reference validation:

```text
                 Scene IR
                    |
          +---------+---------+
          |                   |
     Vulkan renderer      Embree reference
          |                   |
          +---------+---------+
                    |
              image comparison
              RMSE / probes
```

This follows the same general philosophy as using PBRT as a mathematical/reference implementation without making PBRT the runtime architecture.

---

## Cross-vendor architecture

The preferred architecture keeps vendor implementations below Genesis Render-owned interfaces.

```text
                         GENESIS RENDER
                           |
                 Renderer-owned APIs
                           |
        +------------------+------------------+
        |                  |                  |
     Upscaling          Denoising          Latency
        |                  |                  |
   +----+----+        +----+------+      +----+----+
   |    |    |        |           |      |         |
  FSR XeSS DLSS   FidelityFX     NRD   Anti-Lag  Reflex/XeLL
                    Denoiser
        |
        +------------------+------------------+
                           |
                         Vulkan
                           |
              AMD / NVIDIA / Intel / Mesa
```

The exact latency integrations should be evaluated independently because their APIs and platform support differ.

---

## Recommended Genesis Render Interfaces

### Upscaling

```cpp
IUpscaler
  NativeUpscaler
  SpatialUpscaler
  FsrUpscaler
  XessUpscaler
  DlssUpscaler
```

The renderer owns the interface and resource contract. Vendor SDK objects remain private to each backend.

### Denoising

```cpp
IDenoiser
  GenesisRenderTemporalDenoiser
  FidelityFxDenoiser
  NrdDenoiser
  OidnDenoiser        // primarily progressive/offline path
```

Denoising should remain effect-aware where necessary rather than assuming one implementation works equally well for shadows, reflections, diffuse GI, and progressive path tracing.

### Latency

```cpp
ILatencyProvider
  NativeLatencyProvider
  NvidiaReflexProvider
  IntelXeLLProvider
  // AMD implementation where platform/API support fits
```

Do not require any vendor latency provider for normal rendering.

---

## Libraries to Integrate vs. Libraries to Study

### Strong candidates for actual integration

- NVIDIA DLSS / DLAA
- NVIDIA NRD
- AMD FidelityFX FSR
- AMD FidelityFX Denoiser
- AMD FidelityFX SPD
- Intel XeSS-SR
- Intel Open Image Denoise for the progressive path tracer
- Vendor profiling/debugging tools

### Primarily useful as references or specialized tools

- AMD RPS — Genesis Render already owns a render graph.
- AMD Radeon Rays — Genesis Render already owns native Vulkan BLAS/TLAS and ray-query infrastructure.
- Intel Embree — excellent independent CPU/reference ray tracer rather than a replacement for Vulkan RT.
- Intel Open PGL — useful when the path tracer becomes sophisticated enough to benefit from path guiding.
- Intel Open VKL — useful if Genesis Render develops serious volume rendering.
- AMD Brixelizer GI — worth studying while designing the future GI architecture.

---

## Suggested Evaluation Order

1. **AMD SPD** — directly relevant to existing hierarchical image operations such as Hi-Z and future bloom pyramids.
2. **FSR + XeSS + DLSS** — validate the `IUpscaler` abstraction against three genuinely different implementations.
3. **NVIDIA NRD + FidelityFX Denoiser** — evaluate when low-sample RT effects expand beyond hard ray-query shadows.
4. **Intel OIDN** — add alongside the experimental progressive path tracer.
5. **Nsight / Radeon GPU tools / Intel profiling tools** — establish comparable performance-analysis workflows across vendors.
6. **Ray Reconstruction / frame generation / latency technologies** — add only after the basic temporal-upscaling and RT pipelines are mature.
7. **Brixelizer GI / Open PGL / advanced vendor research libraries** — investigate as the GI and path-tracing architecture evolves.

---

## Core Rule

Vendor SDKs should accelerate or improve Genesis Render; they should not define Genesis Render.

```text
Genesis Render owns:
    Scene IR
    GPU Scene
    Materials
    Render Graph
    Temporal contracts
    Resource ownership
    Vulkan device/runtime
    Raster fallback
    RT policy
    Public renderer interfaces

Vendors provide optional implementations:
    Upscaling
    Denoising
    Frame generation
    Latency reduction
    Specialized algorithms
    Profiling/debugging tools
```

This keeps the renderer functional across AMD, NVIDIA, Intel, and conforming Vulkan/Mesa implementations while still allowing each GPU family to expose its strongest optional capabilities.
