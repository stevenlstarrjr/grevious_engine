#pragma once

#include "renderer/lighting/Atmosphere.hpp"
#include "renderer/output/AovExport.hpp"
#include "renderer/post/PostProcessing.hpp"
#include "renderer/ray_tracing/AccelerationStructurePlan.hpp"
#include "renderer/ray_tracing/RayTracingPolicy.hpp"
#include "renderer/render_graph/RenderGraph.hpp"
#include "renderer/renderer/EditorOverlay.hpp"
#include "renderer/temporal/TemporalState.hpp"
#include "renderer/upscaling/Upscaler.hpp"
#include "renderer/visibility/Visibility.hpp"
#include "renderer/vulkan/Buffer.hpp"
#include "renderer/vulkan/Descriptors.hpp"
#include "renderer/vulkan/Image.hpp"
#include "renderer/vulkan/MemoryAllocator.hpp"
#include "renderer/vulkan/Pipeline.hpp"
#include "renderer/vulkan/Sampler.hpp"
#include "renderer/vulkan/Shader.hpp"
#include "renderer/vulkan/Swapchain.hpp"
#include "renderer/vulkan/UploadContext.hpp"
#include "renderer/weather/Weather.hpp"

#include <volk.h>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace renderer::platform {
class RenderSurface;
}

namespace renderer::scene {
class Scene;
}

namespace renderer::nvidia {
class NrdVulkanDenoiser;
class StreamlineRuntime;
}

namespace renderer::vulkan {
class AccelerationStructure;
class Device;
class Instance;
} // namespace renderer::vulkan

namespace renderer {

enum class TemporalDebugView : std::uint32_t {
    none = 0,
    motion = 1,
    historyValidity = 2,
    bloom = 3,
    hdr = 4,
    reactiveMask = 20,
};
enum class TemporalCameraScript : std::uint32_t {
    none = 0,
    translate = 1,
    teleport = 2,
    rotate = 3,
    approach = 4,
};
enum class CameraStartView : std::uint32_t {
    front = 0,
    side = 1,
    interior = 2,
};
enum class RayTracingTransformScript : std::uint32_t {
    none = 0,
    oscillate = 1,
};
enum class RenderMode : std::uint32_t {
    rasterized = 0,
    pathTraced = 1,
};
enum class CloudDebugView : std::uint32_t {
    final = 1,
    transmittance = 2,
    shafts = 3,
};
enum class PathDenoiser : std::uint32_t {
    nrd = 0,
    rayReconstruction = 1,
};
enum class PathDebugView : std::uint32_t {
    final = 0,
    direct = 1,
    indirect = 2,
    albedo = 3,
    roughness = 4,
    geometricNormal = 5,
    shadingNormal = 6,
    motion = 7,
    depth = 8,
    hitDistance = 9,
    lightingResidual = 10,
    historyValidity = 11,
    rawRadiance = 12,
    reconstructionWeight = 13,
    temporalRadiance = 14,
    variance = 15,
};

// Perspective camera supplied by an editor or game framework. Coordinates use
// Genesis' right-handed, Y-up world convention. The renderer derives aspect
// ratio from the current drawable so docked Qt viewports resize naturally.
struct RendererCamera {
    enum class Projection : std::uint32_t {
        perspective,
        orthographic,
    };

    glm::vec3 position{0.0F, 0.0F, 1.0F};
    glm::vec3 forward{0.0F, 0.0F, -1.0F};
    Projection projection = Projection::perspective;
    float verticalFieldOfViewRadians = 0.872664626F;
    // Full vertical world-space span for an orthographic camera.
    float orthographicHeight = 10.0F;
    float nearPlane = 0.01F;
    float farPlane = 1000.0F;
};

inline constexpr std::uint32_t defaultPathBounceCount = 2U;
inline constexpr std::uint32_t maximumPathBounceCount = 4U;
inline constexpr std::uint32_t defaultPathInteractiveSampleCount = 3U;
inline constexpr std::uint32_t maximumPathInteractiveSampleCount = 8U;
inline constexpr std::uint32_t maximumBindlessTextureCount = 128U;
inline constexpr std::uint32_t pathRussianRouletteStartDepth = 2U;

struct RendererConfig {
    std::filesystem::path scenePath;
    // An editor can provide the same scene IR directly instead of serializing
    // to glTF first. Genesis packs this immutable snapshot into GPU residency.
    std::shared_ptr<const scene::Scene> sceneSnapshot;
    bool sceneAnimationStaticPose = false;
    std::filesystem::path environmentPath;
    // When present, replaces the fallback studio HDR environment with a
    // physically based precomputed atmosphere. Explicit environmentPath wins.
    std::optional<lighting::AtmosphereSettings> atmosphere;
    weather::RainSettings rain;
    // Demonstration-only continuous cycle from drizzle through storm and back.
    bool weatherDemoEnabled = false;
    float weatherDemoCycleSeconds = 90.0F;
    float exposureCompensationEv = 0.0F;
    float visibleSkyEv = 0.0F;
    float whiteBalanceKelvin = 6500.0F;
    bool clay = false;
    bool idAovs = false;
    float environmentRotationRadians = 0.0F;
    std::filesystem::path capturePath;
    std::filesystem::path timingOutputPath;
    std::uint64_t captureFrame = 8;
    std::shared_ptr<output::AovFrameSink> aovFrameSink;
    std::uint64_t aovExportFrame = 8;
    RenderView view;
    bool taaEnabled = true;
    bool bloomEnabled = true;
    // Final bloom contribution at composite; independent of filtering strategy.
    float bloomIntensity = post::bloomIntensity;
    // Dimensionless desired perceptual extent. 1.0 preserves the certified baseline.
    float bloomRadius = 1.0F;
    // Monochrome post-tone-map grain amplitude. Zero disables the effect.
    float filmGrainIntensity = 0.0F;
    bool emissiveSphereDemo = false;
    bool directionalShadowDemo = false;
    bool hdriBackdropDemo = false;
    bool alphaShadowDemo = false;
    bool atmosphereDemo = false;
    // Optional deterministic camera controls used by atmosphere/cloud capture
    // sweeps. Preset zero keeps the default view; 1-9 copy only the camera
    // position from the matching Bruneton preset, leaving Sun/exposure alone.
    std::uint32_t atmosphereCameraPreset = 0U;
    float atmosphereCameraPitchRadians = 0.0872664626F;
    float atmosphereCameraYawRadians = 0.0F;
    glm::vec2 cloudInitialOffsetKm{72.0F, -15.5F};
    // Replaces the visible atmosphere with a selected deterministic cloud
    // diagnostic. Cached secondary lighting remains physical.
    CloudDebugView cloudDebugView = CloudDebugView::final;
    bool timeOfDayEnabled = false;
    float timeOfDayHour = 16.0F;
    float timeOfDayCycleSeconds = 120.0F;
    float timeOfDayLatitudeDegrees = 35.0F;
    std::uint32_t timeOfDayDayOfYear = 172U;
    bool shadowsEnabled = true;
    ray_tracing::Request rayTracingRequest = ray_tracing::Request::automatic;
    RayTracingTransformScript rayTracingTransformScript = RayTracingTransformScript::none;
    RenderMode renderMode = RenderMode::rasterized;
    upscaling::Backend upscalingBackend = upscaling::Backend::native;
    upscaling::QualityMode upscalingQualityMode = upscaling::QualityMode::native;
    float upscalingRenderScale = 1.0F;
    nvidia::StreamlineRuntime* streamlineRuntime = nullptr;
    bool pathEnvironmentImportanceSampling = true;
    // Angular radius in radians used to convolve path-traced environment-light visibility.
    float pathShadowAngularRadius = 0.00872664626F;
    bool pathEmissiveTriangleSampling = true;
    std::uint32_t pathMaxBounces = defaultPathBounceCount;
    std::uint32_t pathInteractiveSamples = defaultPathInteractiveSampleCount;
    PathDenoiser pathDenoiser = PathDenoiser::nrd;
    PathDebugView pathDebugView = PathDebugView::final;
    bool occlusionCullingEnabled = true;
    post::TestPattern postTestPattern = post::TestPattern::none;
    float postTestSourceValue = 4.0F;
    glm::vec2 postTestPosition{0.5F};
    TemporalDebugView temporalDebugView = TemporalDebugView::none;
    TemporalCameraScript temporalCameraScript = TemporalCameraScript::none;
    CameraStartView cameraStartView = CameraStartView::front;
    bool interactiveCameraInput = true;
    // Reserves raster scene buffers for an editor to replace geometry without
    // recreating the renderer, swapchain, pipelines, or baked IBL resources.
    bool dynamicSceneUpdates = false;
    // Enables a host-supplied grid/gizmo/selection pass after tone mapping.
    // Rendered/final views leave this disabled for a clean presentation.
    bool editorOverlayEnabled = false;
};

class Renderer final {
  public:
    Renderer(platform::RenderSurface& surfaceHost, const vulkan::Instance& instance,
             const vulkan::Device& device, RendererConfig config = {});
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void drawFrame();
    void waitIdle();
    // Supplying a camera transfers navigation ownership to the host. Passing
    // nullopt returns to Genesis' built-in SDL/demo camera controller.
    void setCamera(std::optional<RendererCamera> camera);
    void setEditorOverlay(EditorOverlayFrame overlay);
    // Returns false when the replacement exceeds the reserved editor capacity
    // or changes resources that require a full renderer rebuild.
    bool updateSceneSnapshot(std::shared_ptr<const scene::Scene> sceneSnapshot);
    void setViewVisibilityMask(RenderLayerMask mask) {
        mainView_.visibilityMask = mask;
        temporalState_.invalidateHistory();
        hiZReady_ = false;
        pathTracingResetPending_ = true;
    }
    [[nodiscard]] RenderLayerMask viewVisibilityMask() const { return mainView_.visibilityMask; }
    void resetTemporalHistory() {
        temporalState_.reset();
        hiZReady_ = false;
        pathTracingResetPending_ = true;
    }

  private:
    struct Frame {
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        std::uint64_t completionValue = 0;
        bool timingPending = false;
    };

    enum class PathAovReadbackTarget : std::size_t {
        rawBeauty,
        directLighting,
        surface,
        normals,
        motionDepth,
        depth,
        historyValidity,
        temporalBeauty,
        beauty,
        count,
    };

    struct FrameUniform {
        glm::mat4 viewProjection{1.0F};
        glm::mat4 inverseViewProjection{1.0F};
        glm::mat4 unjitteredViewProjection{1.0F};
        glm::mat4 previousViewProjection{1.0F};
        glm::mat4 previousUnjitteredViewProjection{1.0F};
        glm::vec4 cameraPositionExposure{0.0F};
        glm::vec4 directionalDirectionIntensity{0.0F};
        glm::vec4 directionalColor{1.0F};
        glm::vec4 pointPositionIntensity{0.0F};
        glm::vec4 pointColorRange{1.0F};
        glm::vec4 environmentParameters{0.0F};
        glm::uvec4 cullingParameters{0U};
        // Low/high RenderLayerMask words, post-test ID, bloom-intensity float bits.
        glm::uvec4 visibilityMask{0U};
        // Current.xy and previous.zw camera jitter in pixel units.
        glm::vec4 temporalJitter{0.0F};
        // History valid, debug view, TAA enabled, reserved.
        glm::uvec4 temporalFlags{0U};
        // Cascade 0 XYZ center and W coverage radius. W=0 disables shadow sampling.
        glm::vec4 shadowSceneCenterRadius{0.0F};
        glm::vec4 shadowCascadeCenterRadius1{0.0F};
        glm::vec4 shadowCascadeCenterRadius2{0.0F};
        // XYZ are far view-distance splits; W is the active cascade count.
        glm::vec4 shadowCascadeSplits{0.0F};
        // XYZ are light-axis half-depths for cascades 0..2.
        glm::vec4 shadowCascadeDepthRadii{0.0F};
        // Previous camera position used with previous-frame Hi-Z occlusion data.
        glm::vec4 previousCameraPosition{0.0F};
        // Reference-demo camera XYZ and planet bottom radius W, in kilometers.
        glm::vec4 atmosphereCamera{0.0F};
        // Reference Z-up Sun direction; W enables the Bruneton lookup path.
        glm::vec4 atmosphereSunDirection{0.0F};
        // Inner/outer shell radii in km, coverage, and density.
        glm::vec4 cloudLayerParameters{0.0F};
        // Wind offset in km, horizontal scale in km, and enable flag.
        glm::vec4 cloudWeatherParameters{0.0F};
        // Rain intensity, simulation time, post-rain steam age, and accumulated wetness.
        glm::vec4 weatherParameters{0.0F};
        // RGB camera adaptation, W camera-visible environment multiplier.
        glm::vec4 lookColor{1.0F};
        // X clay override, ZW optional uint64 first-hit ID capture address.
        glm::uvec4 lookOptions{0U};
    };

    struct BloomControls {
        glm::vec4 intensityFilterScaleThresholdKnee{post::bloomIntensity, 1.0F,
                                                    post::bloomThreshold, post::bloomKnee};
    };

    struct TemporalUpscalingControls {
        glm::vec4 temporalJitter{0.0F};
        glm::uvec4 flags{0U};
    };

    struct BloomLevelGraphResources {
        VkExtent2D extent{};
        render_graph::ImageHandle filtered;
        render_graph::ImageHandle scratch;
    };

    static_assert(sizeof(FrameUniform) == 172 * sizeof(float));

    struct GraphFrameContext {
        std::uint32_t imageIndex = 0;
        std::size_t frameIndex = 0;
        std::size_t historyReadIndex = 0;
        std::size_t historyWriteIndex = 0;
        bool capture = false;
        bool exportAovs = false;
    };

    struct PathDenoisingControls {
        std::uint32_t sampleIndex = 0U;
        std::uint32_t stepWidth = 1U;
        std::uint32_t passIndex = 0U;
        std::uint32_t passCount = 1U;
        std::uint32_t samplesThisFrame = 1U;
    };

    static_assert(sizeof(PathDenoisingControls) == 5 * sizeof(std::uint32_t));

    struct PathTracingControls {
        std::uint32_t sampleIndex = 0U;
        std::uint32_t sampleInFrame = 0U;
        std::uint32_t samplesThisFrame = 1U;
        std::uint32_t fixedPrimarySample = 0U;
    };

    static_assert(sizeof(PathTracingControls) == 4 * sizeof(std::uint32_t));

    struct PathReconstructionControls {
        std::uint32_t sampleIndex = 0U;
        std::uint32_t samplesThisFrame = 1U;
    };

    static_assert(sizeof(PathReconstructionControls) == 2 * sizeof(std::uint32_t));

    struct RainControls {
        // intensity, simulation time, terminal speed, physical streak length
        glm::vec4 amountTimeSpeedLength{0.0F};
        // wind velocity XYZ and camera-local field distance
        glm::vec4 windMaximumDistance{0.0F};
    };

    static_assert(sizeof(RainControls) == 8 * sizeof(float));

    void createFrameResources();
    void createCaptureBuffer();
    void createTemporalResources();
    void createUpscalingHistoryResources();
    void createPathTracingResources();
    void destroyFrameResources();
    void createSceneResources();
    void createRayTracingResources(std::span<const gpu_scene::TransformRecord> transforms);
    void updateRayTracingInstanceBuffer(std::size_t frameIndex);
    void recordRayTracingTlasUpdate(VkCommandBuffer commandBuffer, std::size_t frameIndex);
    void createGraphicsPipelines();
    void createEditorOverlayResources();
    void createRenderGraph();
    void createPathTracingRenderGraph();
    void importUpscalingHistoryImages();
    void updateUpscalingDescriptor();
    void updateToneMappingDescriptor();
    void updateRainDescriptor();
    void updatePathTracingDescriptors();
    void updatePathHistoryDescriptors();
    void updatePathReconstructionDescriptors();
    void updatePathDenoisingDescriptors();
    void updateNrdDescriptors();
    void updateAtmosphereEnvironmentDescriptors();
    void updateShadowDescriptor();
    void updateHiZDescriptors();
    void resetCamera();
    void applyAtmosphereDemoCamera();
    void updateCamera();
    void updateFrameUniform(std::size_t frameIndex);
    void updateEditorOverlayBuffer(std::size_t frameIndex);
    void recreateSwapchain();
    void recordFrame(VkCommandBuffer commandBuffer, std::uint32_t imageIndex,
                     std::size_t frameIndex, bool capture, bool exportAovs);
    void recordPbrPass(VkCommandBuffer commandBuffer, const render_graph::RenderGraph& graph,
                       const GraphFrameContext& context);
    void recordPathTracingPass(VkCommandBuffer commandBuffer, const GraphFrameContext& context);
    void recordAtmosphereEnvironmentPass(VkCommandBuffer commandBuffer,
                                         const GraphFrameContext& context);
    void recordPathHistoryPass(VkCommandBuffer commandBuffer, const GraphFrameContext& context);
    void recordPathReconstructionPass(VkCommandBuffer commandBuffer,
                                       const GraphFrameContext& context);
    void recordPathDenoisingPass(VkCommandBuffer commandBuffer, const GraphFrameContext& context,
                                 std::size_t passIndex);
    void recordNrdPreparePass(VkCommandBuffer commandBuffer, const GraphFrameContext& context);
    void recordNrdDenoisingPass(VkCommandBuffer commandBuffer, const GraphFrameContext& context);
    void recordNrdResolvePass(VkCommandBuffer commandBuffer, const GraphFrameContext& context);
    void recordPathAovReadbackPass(VkCommandBuffer commandBuffer, const GraphFrameContext& context);
    void recordUpscalingPass(VkCommandBuffer commandBuffer, const render_graph::RenderGraph& graph,
                             const GraphFrameContext& context);
    void recordDlssPass(VkCommandBuffer commandBuffer, const render_graph::RenderGraph& graph,
                        const GraphFrameContext& context);
    void recordReactiveMaskPass(VkCommandBuffer commandBuffer,
                                const render_graph::RenderGraph& graph,
                                const GraphFrameContext& context);
    void recordCullingPass(VkCommandBuffer commandBuffer, const GraphFrameContext& context);
    void recordShadowCullingPass(VkCommandBuffer commandBuffer, const GraphFrameContext& context,
                                 std::size_t cascadeIndex);
    void recordShadowPass(VkCommandBuffer commandBuffer, const render_graph::RenderGraph& graph,
                          const GraphFrameContext& context, std::size_t cascadeIndex,
                          render_graph::ImageHandle shadowImage);
    void recordOcclusionCullingPass(VkCommandBuffer commandBuffer,
                                    const GraphFrameContext& context);
    void recordHiZPass(VkCommandBuffer commandBuffer, std::size_t level);
    void recordTaaPass(VkCommandBuffer commandBuffer, const GraphFrameContext& context);
    void recordToneMappingPass(VkCommandBuffer commandBuffer,
                               const render_graph::RenderGraph& graph,
                               const GraphFrameContext& context);
    void recordRainPass(VkCommandBuffer commandBuffer,
                        const GraphFrameContext& context);
    void updateWeatherDemo(float deltaSeconds);
    void recordEditorOverlayPass(VkCommandBuffer commandBuffer,
                                 const render_graph::RenderGraph& graph,
                                 const GraphFrameContext& context);
    void recordCapturePass(VkCommandBuffer commandBuffer, const render_graph::RenderGraph& graph);
    void writeCapture() const;
    void writePathAovExport() const;
    enum class GpuTimingPass : std::uint32_t {
        tlasUpdate,
        shadowCulling,
        shadowMap,
        pbrGeometry,
        atmosphereEnvironment,
        pathTracing,
        pathHistory,
        pathReconstruction,
        pathDenoising,
        nrdPrepare,
        nrdDenoising,
        nrdResolve,
        pathAovReadback,
        bloomDownsample,
        bloomHorizontal,
        bloomVertical,
        reactiveMask,
        upscaling,
        toneMapping,
        rain,
        count,
    };
    void recordBloomPass(VkCommandBuffer commandBuffer, const render_graph::RenderGraph& graph,
                         render_graph::ImageHandle target, const vulkan::GraphicsPipeline& pipeline,
                         std::size_t descriptorSetIndex, std::size_t frameIndex,
                         GpuTimingPass timingPass);
    void writeGpuTimestamp(VkCommandBuffer commandBuffer, std::size_t frameIndex,
                           GpuTimingPass pass, bool end) const;
    void collectGpuTimings(std::size_t frameIndex);
    void writeGpuTimingReport();
    [[nodiscard]] bool rasterShadowMapsEnabled() const {
        return config_.renderMode != RenderMode::pathTraced && config_.shadowsEnabled &&
               rayTracingShadowBackend_ == ray_tracing::ShadowBackend::raster;
    }
    [[nodiscard]] bool pathTracingEnabled() const {
        return config_.renderMode == RenderMode::pathTraced;
    }
    [[nodiscard]] bool rayReconstructionEnabled() const {
        return pathTracingEnabled() &&
               config_.pathDenoiser == PathDenoiser::rayReconstruction;
    }
    [[nodiscard]] bool upscalingDispatchEnabled() const {
        return upscaler_ != nullptr && upscaler_->backend() != upscaling::Backend::native;
    }
    [[nodiscard]] bool spatialUpscalingEnabled() const {
        return upscaler_ != nullptr && upscaler_->backend() == upscaling::Backend::spatial;
    }
    [[nodiscard]] bool temporalUpscalingEnabled() const {
        return upscaler_ != nullptr && upscaler_->backend() == upscaling::Backend::temporal;
    }
    [[nodiscard]] bool dlssUpscalingEnabled() const {
        return upscaler_ != nullptr && upscaler_->backend() == upscaling::Backend::dlss;
    }
    [[nodiscard]] VkExtent2D renderExtent() const;
    [[nodiscard]] render_graph::ImageHandle pathPresentationSourceImage() const;
    [[nodiscard]] const vulkan::ImageView& pathPresentationSourceView(std::size_t frameIndex) const;

    static constexpr std::size_t framesInFlight = 2;
    static constexpr std::size_t pathDenoisingPassCount = 3;
    static constexpr std::uint32_t maxBindlessTextures = maximumBindlessTextureCount;
    static constexpr std::uint32_t maxHiZLevels = 16;
    static constexpr std::uint32_t directionalShadowResolution = 2048;
    static constexpr std::size_t directionalShadowCascadeCount = 3;
    static constexpr std::size_t gpuTimingPassCount =
        static_cast<std::size_t>(GpuTimingPass::count);
    platform::RenderSurface& surfaceHost_;
    const vulkan::Instance& instance_;
    const vulkan::Device& device_;
    RendererConfig config_;
    std::unique_ptr<upscaling::IUpscaler> upscaler_;
    nvidia::StreamlineRuntime* streamlineRuntime_ = nullptr;
    ray_tracing::ShadowBackend rayTracingShadowBackend_ = ray_tracing::ShadowBackend::raster;
    ray_tracing::SelectionReason rayTracingSelectionReason_ =
        ray_tracing::SelectionReason::hardwareUnavailable;
    bool rayTracingFoundationEnabled_ = false;
    ray_tracing::SceneBuildPlan rayTracingBuildPlan_;
    RenderView mainView_;
    vulkan::Swapchain swapchain_;
    vulkan::MemoryAllocator memoryAllocator_;
    vulkan::PipelineCache pipelineCache_;
    std::unique_ptr<vulkan::Buffer> vertexBuffer_;
    std::unique_ptr<vulkan::Buffer> indexBuffer_;
    std::unique_ptr<vulkan::Buffer> rayTracingGeometryBuffer_;
    std::vector<std::unique_ptr<vulkan::AccelerationStructure>> bottomLevelStructures_;
    std::unique_ptr<vulkan::AccelerationStructure> topLevelStructure_;
    std::array<std::unique_ptr<vulkan::Buffer>, framesInFlight> rayTracingInstanceBuffers_{};
    std::array<std::unique_ptr<vulkan::Buffer>, framesInFlight> rayTracingUpdateScratchBuffers_{};
    std::vector<VkAccelerationStructureInstanceKHR> rayTracingPackedInstances_;
    std::vector<gpu_scene::TransformRecord> rayTracingBaseTransforms_;
    std::vector<gpu_scene::TransformRecord> rayTracingTransforms_;
    std::array<std::unique_ptr<vulkan::Buffer>, framesInFlight> pathAnimatedTransformBuffers_{};
    VkDeviceSize rayTracingUpdateScratchSize_ = 0;
    VkDeviceSize rayTracingScratchAlignment_ = 1;
    VkDeviceSize rayTracingPersistentBytes_ = 0;
    std::uint64_t rayTracingTlasUpdateCount_ = 0;
    std::uint64_t rayTracingScriptedUpdateCount_ = 0;
    bool rayTracingUpdateReportWritten_ = false;
    std::unique_ptr<vulkan::Buffer> instanceBuffer_;
    std::unique_ptr<vulkan::Buffer> materialBuffer_;
    std::unique_ptr<vulkan::Buffer> transformBuffer_;
    std::unique_ptr<vulkan::Buffer> meshletWorkBuffer_;
    std::unique_ptr<vulkan::Buffer> frustumVisibleInstanceBuffer_;
    std::unique_ptr<vulkan::Buffer> frustumIndirectDrawBuffer_;
    std::unique_ptr<vulkan::Buffer> frustumDrawCountBuffer_;
    std::unique_ptr<vulkan::Buffer> visibleInstanceBuffer_;
    std::unique_ptr<vulkan::Buffer> indirectDrawBuffer_;
    std::unique_ptr<vulkan::Buffer> drawCountBuffer_;
    std::array<std::unique_ptr<vulkan::Buffer>, directionalShadowCascadeCount>
        shadowVisibleInstanceBuffers_{};
    std::array<std::unique_ptr<vulkan::Buffer>, directionalShadowCascadeCount>
        shadowIndirectDrawBuffers_{};
    std::array<std::unique_ptr<vulkan::Buffer>, directionalShadowCascadeCount>
        shadowDrawCountBuffers_{};
    std::array<std::unique_ptr<vulkan::Buffer>, framesInFlight> uniformBuffers_{};
    std::unique_ptr<vulkan::Buffer> bloomControlsBuffer_;
    std::unique_ptr<vulkan::Buffer> captureReadbackBuffer_;
    static constexpr VkDeviceSize editorOverlayBufferSize = 4U * 1024U * 1024U;
    std::array<std::unique_ptr<vulkan::Buffer>, framesInFlight>
        editorOverlayVertexBuffers_{};
    std::unique_ptr<vulkan::Buffer> pathAovReadbackBuffer_;
    VkDeviceSize pathIdOffset_ = 0;
    glm::vec4 lookColor_{1.0F};
    std::vector<output::IdLabel> objectLabels_;
    std::vector<output::IdLabel> materialLabels_;
    static constexpr std::size_t pathAovReadbackTargetCount =
        static_cast<std::size_t>(PathAovReadbackTarget::count);
    std::array<VkDeviceSize, pathAovReadbackTargetCount> pathAovReadbackOffsets_{};
    std::unique_ptr<vulkan::Buffer> environmentSamplingBuffer_;
    std::unique_ptr<vulkan::Buffer> emissiveTriangleBuffer_;
#if RENDERER_ENABLE_NRD
    std::unique_ptr<nvidia::NrdVulkanDenoiser> nrdDenoiser_;
#endif
    std::vector<std::unique_ptr<vulkan::Image>> textureImages_;
    std::vector<std::unique_ptr<vulkan::ImageView>> textureViews_;
    std::vector<std::unique_ptr<vulkan::Sampler>> textureSamplers_;
    std::unique_ptr<vulkan::Image> environmentImage_;
    std::unique_ptr<vulkan::ImageView> environmentView_;
    std::unique_ptr<vulkan::Image> atmosphereTransmittanceImage_;
    std::unique_ptr<vulkan::ImageView> atmosphereTransmittanceView_;
    std::unique_ptr<vulkan::Image> atmosphereScatteringImage_;
    std::unique_ptr<vulkan::ImageView> atmosphereScatteringView_;
    std::unique_ptr<vulkan::Image> atmosphereIrradianceImage_;
    std::unique_ptr<vulkan::ImageView> atmosphereIrradianceView_;
    std::unique_ptr<vulkan::Image> cloudBaseNoiseImage_;
    std::unique_ptr<vulkan::ImageView> cloudBaseNoiseView_;
    std::unique_ptr<vulkan::Image> cloudDetailNoiseImage_;
    std::unique_ptr<vulkan::ImageView> cloudDetailNoiseView_;
    std::unique_ptr<vulkan::Image> diffuseIrradianceImage_;
    std::unique_ptr<vulkan::ImageView> diffuseIrradianceView_;
    std::unique_ptr<vulkan::Image> prefilteredSpecularImage_;
    std::unique_ptr<vulkan::ImageView> prefilteredSpecularView_;
    std::unique_ptr<vulkan::Image> brdfLutImage_;
    std::unique_ptr<vulkan::ImageView> brdfLutView_;
    std::unique_ptr<vulkan::Image> pathTracingAccumulationImage_;
    std::unique_ptr<vulkan::ImageView> pathTracingAccumulationView_;
    std::unique_ptr<vulkan::Image> pathTracingDirectImage_;
    std::unique_ptr<vulkan::ImageView> pathTracingDirectView_;
    std::unique_ptr<vulkan::Image> pathTracingSurfaceImage_;
    std::unique_ptr<vulkan::ImageView> pathTracingSurfaceView_;
    std::unique_ptr<vulkan::Image> pathTracingNormalImage_;
    std::unique_ptr<vulkan::ImageView> pathTracingNormalView_;
    std::unique_ptr<vulkan::Image> pathTracingMotionDepthImage_;
    std::unique_ptr<vulkan::ImageView> pathTracingMotionDepthView_;
    std::unique_ptr<vulkan::Image> pathTracingDepthImage_;
    std::unique_ptr<vulkan::ImageView> pathTracingDepthView_;
    std::unique_ptr<vulkan::Image> rayReconstructionDiffuseAlbedoImage_;
    std::unique_ptr<vulkan::ImageView> rayReconstructionDiffuseAlbedoView_;
    std::unique_ptr<vulkan::Image> rayReconstructionSpecularAlbedoImage_;
    std::unique_ptr<vulkan::ImageView> rayReconstructionSpecularAlbedoView_;
    std::unique_ptr<vulkan::Image> rayReconstructionNormalRoughnessImage_;
    std::unique_ptr<vulkan::ImageView> rayReconstructionNormalRoughnessView_;
    std::unique_ptr<vulkan::Image> rayReconstructionSpecularHitDistanceImage_;
    std::unique_ptr<vulkan::ImageView> rayReconstructionSpecularHitDistanceView_;
    std::array<std::unique_ptr<vulkan::Image>, 2> pathHistoryNormalImages_{};
    std::array<std::unique_ptr<vulkan::ImageView>, 2> pathHistoryNormalViews_{};
    std::array<std::unique_ptr<vulkan::Image>, 2> pathHistoryDepthImages_{};
    std::array<std::unique_ptr<vulkan::ImageView>, 2> pathHistoryDepthViews_{};
    std::unique_ptr<vulkan::Image> pathHistoryValidityImage_;
    std::unique_ptr<vulkan::ImageView> pathHistoryValidityView_;
    std::array<std::unique_ptr<vulkan::Image>, 2> pathReconstructionImages_{};
    std::array<std::unique_ptr<vulkan::ImageView>, 2> pathReconstructionViews_{};
    std::array<std::unique_ptr<vulkan::Image>, 2> pathMomentImages_{};
    std::array<std::unique_ptr<vulkan::ImageView>, 2> pathMomentViews_{};
    std::array<std::unique_ptr<vulkan::Image>, pathDenoisingPassCount - 1U>
        pathDenoisingDirectImages_{};
    std::array<std::unique_ptr<vulkan::ImageView>, pathDenoisingPassCount - 1U>
        pathDenoisingDirectViews_{};
    std::array<std::unique_ptr<vulkan::Image>, pathDenoisingPassCount - 1U>
        pathDenoisingIndirectImages_{};
    std::array<std::unique_ptr<vulkan::ImageView>, pathDenoisingPassCount - 1U>
        pathDenoisingIndirectViews_{};
    std::unique_ptr<vulkan::Image> pathDenoisedImage_;
    std::unique_ptr<vulkan::ImageView> pathDenoisedView_;
    std::unique_ptr<vulkan::Image> nrdMotionImage_;
    std::unique_ptr<vulkan::ImageView> nrdMotionView_;
    std::unique_ptr<vulkan::Image> nrdNormalRoughnessImage_;
    std::unique_ptr<vulkan::ImageView> nrdNormalRoughnessView_;
    std::unique_ptr<vulkan::Image> nrdViewZImage_;
    std::unique_ptr<vulkan::ImageView> nrdViewZView_;
    std::unique_ptr<vulkan::Image> nrdDiffuseInputImage_;
    std::unique_ptr<vulkan::ImageView> nrdDiffuseInputView_;
    std::unique_ptr<vulkan::Image> nrdSpecularInputImage_;
    std::unique_ptr<vulkan::ImageView> nrdSpecularInputView_;
    std::unique_ptr<vulkan::Image> nrdMaterialImage_;
    std::unique_ptr<vulkan::ImageView> nrdMaterialView_;
    std::unique_ptr<vulkan::Image> nrdDiffuseOutputImage_;
    std::unique_ptr<vulkan::ImageView> nrdDiffuseOutputView_;
    std::unique_ptr<vulkan::Image> nrdSpecularOutputImage_;
    std::unique_ptr<vulkan::ImageView> nrdSpecularOutputView_;
    std::array<std::unique_ptr<vulkan::Image>, 2> historyColorImages_{};
    std::array<std::unique_ptr<vulkan::ImageView>, 2> historyColorViews_{};
    std::array<std::unique_ptr<vulkan::Image>, 2> historyDepthImages_{};
    std::array<std::unique_ptr<vulkan::ImageView>, 2> historyDepthViews_{};
    std::array<std::unique_ptr<vulkan::Image>, 2> historyNormalImages_{};
    std::array<std::unique_ptr<vulkan::ImageView>, 2> historyNormalViews_{};
    std::array<std::unique_ptr<vulkan::Image>, framesInFlight> upscalingHistoryColorImages_{};
    std::array<std::unique_ptr<vulkan::ImageView>, framesInFlight> upscalingHistoryColorViews_{};
    std::array<std::unique_ptr<vulkan::Image>, framesInFlight> upscalingHistoryDepthImages_{};
    std::array<std::unique_ptr<vulkan::ImageView>, framesInFlight> upscalingHistoryDepthViews_{};
    std::unique_ptr<vulkan::Image> dlssOutputImage_;
    std::unique_ptr<vulkan::ImageView> dlssOutputView_;
    std::unique_ptr<vulkan::Sampler> iblSampler_;
    std::unique_ptr<vulkan::Sampler> brdfLutSampler_;
    std::unique_ptr<vulkan::Sampler> proceduralNoiseSampler_;
    std::unique_ptr<vulkan::Sampler> hiZSampler_;
    std::unique_ptr<vulkan::Sampler> shadowSampler_;
    vulkan::UploadContext uploadContext_;
    std::unique_ptr<vulkan::ShaderModule> vertexShader_;
    std::unique_ptr<vulkan::ShaderModule> fragmentShader_;
    std::unique_ptr<vulkan::ShaderModule> shadowVertexShader_;
    std::unique_ptr<vulkan::ShaderModule> shadowFragmentShader_;
    std::unique_ptr<vulkan::ShaderModule> shadowCullShader_;
    std::unique_ptr<vulkan::ShaderModule> frustumCullShader_;
    std::unique_ptr<vulkan::ShaderModule> occlusionCullShader_;
    std::unique_ptr<vulkan::ShaderModule> hiZCopyShader_;
    std::unique_ptr<vulkan::ShaderModule> hiZReduceShader_;
    std::unique_ptr<vulkan::ShaderModule> skyFragmentShader_;
    std::unique_ptr<vulkan::ShaderModule> postTestFragmentShader_;
    std::unique_ptr<vulkan::ShaderModule> bloomDownsampleShader_;
    std::unique_ptr<vulkan::ShaderModule> bloomBlurHorizontalShader_;
    std::unique_ptr<vulkan::ShaderModule> bloomBlurVerticalShader_;
    std::unique_ptr<vulkan::ShaderModule> spatialUpscalingFragmentShader_;
    std::unique_ptr<vulkan::ShaderModule> reactiveMaskFragmentShader_;
    std::unique_ptr<vulkan::ShaderModule> temporalUpscalingFragmentShader_;
    std::unique_ptr<vulkan::ShaderModule> toneMappingVertexShader_;
    std::unique_ptr<vulkan::ShaderModule> toneMappingFragmentShader_;
    std::unique_ptr<vulkan::ShaderModule> rainFragmentShader_;
    std::unique_ptr<vulkan::ShaderModule> editorOverlayVertexShader_;
    std::unique_ptr<vulkan::ShaderModule> editorOverlayFragmentShader_;
    std::unique_ptr<vulkan::ShaderModule> taaFragmentShader_;
    std::unique_ptr<vulkan::ShaderModule> atmosphereEnvironmentShader_;
    std::unique_ptr<vulkan::ShaderModule> pathTracingRayGenerationShader_;
    std::unique_ptr<vulkan::ShaderModule> pathTracingRadianceMissShader_;
    std::unique_ptr<vulkan::ShaderModule> pathTracingShadowMissShader_;
    std::unique_ptr<vulkan::ShaderModule> pathTracingClosestHitShader_;
    std::unique_ptr<vulkan::ShaderModule> pathTracingAnyHitShader_;
    std::unique_ptr<vulkan::ShaderModule> pathHistoryShader_;
    std::unique_ptr<vulkan::ShaderModule> pathReconstructionShader_;
    std::unique_ptr<vulkan::ShaderModule> pathDenoisingShader_;
    std::unique_ptr<vulkan::ShaderModule> pathNrdPrepareShader_;
    std::unique_ptr<vulkan::ShaderModule> pathNrdResolveShader_;
    std::unique_ptr<vulkan::DescriptorSetLayout> pathTracingDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> pathTracingDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> atmosphereEnvironmentDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> atmosphereEnvironmentDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> pathHistoryDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> pathHistoryDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> pathReconstructionDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> pathReconstructionDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> pathDenoisingDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> pathDenoisingDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> pathNrdPrepareDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> pathNrdPrepareDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> pathNrdResolveDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> pathNrdResolveDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> descriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> descriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> bindlessDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> bindlessDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> cullingDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> cullingDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> shadowDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> shadowDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> occlusionDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> occlusionDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> hiZDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> hiZDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> bloomDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> bloomDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> taaDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> taaDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> upscalingDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> upscalingDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> reactiveMaskDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> reactiveMaskDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> temporalUpscalingDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> temporalUpscalingDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> toneMappingDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> toneMappingDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorSetLayout> rainDescriptorSetLayout_;
    std::unique_ptr<vulkan::DescriptorPool> rainDescriptorPool_;
    std::unique_ptr<vulkan::GraphicsPipeline> graphicsPipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> shadowPipeline_;
    std::unique_ptr<vulkan::ComputePipeline> shadowCullPipeline_;
    std::unique_ptr<vulkan::ComputePipeline> frustumCullPipeline_;
    std::unique_ptr<vulkan::ComputePipeline> occlusionCullPipeline_;
    std::unique_ptr<vulkan::ComputePipeline> hiZCopyPipeline_;
    std::unique_ptr<vulkan::ComputePipeline> hiZReducePipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> skyPipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> postTestPipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> bloomDownsamplePipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> bloomBlurHorizontalPipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> bloomBlurVerticalPipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> taaPipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> spatialUpscalingPipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> reactiveMaskPipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> temporalUpscalingPipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> toneMappingPipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> rainPipeline_;
    std::unique_ptr<vulkan::ComputePipeline> atmosphereEnvironmentPipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> editorSceneTrianglePipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> editorSelectionTrianglePipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> editorDepthLinePipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> editorDepthTrianglePipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> editorOverlayTrianglePipeline_;
    std::unique_ptr<vulkan::GraphicsPipeline> editorOverlayLinePipeline_;
    std::unique_ptr<vulkan::RayTracingPipeline> pathTracingPipeline_;
    std::unique_ptr<vulkan::ComputePipeline> pathHistoryPipeline_;
    std::unique_ptr<vulkan::ComputePipeline> pathReconstructionPipeline_;
    std::unique_ptr<vulkan::ComputePipeline> pathDenoisingPipeline_;
    std::unique_ptr<vulkan::ComputePipeline> pathNrdPreparePipeline_;
    std::unique_ptr<vulkan::ComputePipeline> pathNrdResolvePipeline_;
    std::unique_ptr<render_graph::RenderGraph> renderGraph_;
    render_graph::ImageHandle graphSwapchainImage_;
    std::vector<render_graph::ImageHandle> graphTextureImages_;
    render_graph::ImageHandle graphEnvironmentImage_;
    render_graph::ImageHandle graphAtmosphereTransmittanceImage_;
    render_graph::ImageHandle graphAtmosphereScatteringImage_;
    render_graph::ImageHandle graphAtmosphereIrradianceImage_;
    render_graph::ImageHandle graphDiffuseIrradianceImage_;
    render_graph::ImageHandle graphPrefilteredSpecularImage_;
    render_graph::ImageHandle graphBrdfLutImage_;
    render_graph::ImageHandle graphPathTracingAccumulationImage_;
    render_graph::ImageHandle graphPathTracingDirectImage_;
    render_graph::ImageHandle graphPathTracingSurfaceImage_;
    render_graph::ImageHandle graphPathTracingNormalImage_;
    render_graph::ImageHandle graphPathTracingMotionDepthImage_;
    render_graph::ImageHandle graphPathTracingDepthImage_;
    render_graph::ImageHandle graphRayReconstructionDiffuseAlbedoImage_;
    render_graph::ImageHandle graphRayReconstructionSpecularAlbedoImage_;
    render_graph::ImageHandle graphRayReconstructionNormalRoughnessImage_;
    render_graph::ImageHandle graphRayReconstructionSpecularHitDistanceImage_;
    render_graph::ImageHandle graphPathHistoryNormalReadImage_;
    render_graph::ImageHandle graphPathHistoryNormalWriteImage_;
    render_graph::ImageHandle graphPathHistoryDepthReadImage_;
    render_graph::ImageHandle graphPathHistoryDepthWriteImage_;
    render_graph::ImageHandle graphPathHistoryValidityImage_;
    render_graph::ImageHandle graphPathReconstructionReadImage_;
    render_graph::ImageHandle graphPathReconstructionWriteImage_;
    render_graph::ImageHandle graphPathMomentReadImage_;
    render_graph::ImageHandle graphPathMomentWriteImage_;
    std::array<render_graph::ImageHandle, pathDenoisingPassCount - 1U>
        graphPathDenoisingDirectImages_{};
    std::array<render_graph::ImageHandle, pathDenoisingPassCount - 1U>
        graphPathDenoisingIndirectImages_{};
    render_graph::ImageHandle graphPathDenoisedImage_;
    render_graph::ImageHandle graphNrdMotionImage_;
    render_graph::ImageHandle graphNrdNormalRoughnessImage_;
    render_graph::ImageHandle graphNrdViewZImage_;
    render_graph::ImageHandle graphNrdDiffuseInputImage_;
    render_graph::ImageHandle graphNrdSpecularInputImage_;
    render_graph::ImageHandle graphNrdMaterialImage_;
    render_graph::ImageHandle graphNrdDiffuseOutputImage_;
    render_graph::ImageHandle graphNrdSpecularOutputImage_;
    std::array<render_graph::ImageHandle, directionalShadowCascadeCount>
        graphDirectionalShadowImages_{};
    render_graph::ImageHandle graphHdrImage_;
    render_graph::ImageHandle graphMotionImage_;
    render_graph::ImageHandle graphNormalImage_;
    render_graph::ImageHandle graphHistoryColorReadImage_;
    render_graph::ImageHandle graphHistoryColorWriteImage_;
    render_graph::ImageHandle graphHistoryDepthReadImage_;
    render_graph::ImageHandle graphHistoryDepthWriteImage_;
    render_graph::ImageHandle graphHistoryNormalReadImage_;
    render_graph::ImageHandle graphHistoryNormalWriteImage_;
    std::vector<BloomLevelGraphResources> graphBloomLevels_;
    render_graph::ImageHandle graphDepthImage_;
    render_graph::ImageHandle graphUpscalingSourceImage_;
    render_graph::ImageHandle graphReactiveMaskImage_;
    render_graph::ImageHandle graphUpscaledImage_;
    render_graph::ImageHandle graphUpscaledDepthImage_;
    render_graph::ImageHandle graphUpscalingHistoryColorReadImage_;
    render_graph::ImageHandle graphUpscalingHistoryColorWriteImage_;
    render_graph::ImageHandle graphUpscalingHistoryDepthReadImage_;
    render_graph::ImageHandle graphUpscalingHistoryDepthWriteImage_;
    render_graph::ImageHandle graphDlssOutputImage_;
    std::vector<render_graph::ImageHandle> graphHiZImages_;
    std::vector<VkExtent2D> hiZExtents_;
    render_graph::BufferHandle graphVertexBuffer_;
    render_graph::BufferHandle graphIndexBuffer_;
    render_graph::BufferHandle graphInstanceBuffer_;
    render_graph::BufferHandle graphMaterialBuffer_;
    render_graph::BufferHandle graphTransformBuffer_;
    render_graph::BufferHandle graphRayTracingGeometryBuffer_;
    render_graph::BufferHandle graphEnvironmentSamplingBuffer_;
    render_graph::BufferHandle graphEmissiveTriangleBuffer_;
    render_graph::BufferHandle graphMeshletWorkBuffer_;
    render_graph::BufferHandle graphFrustumVisibleInstanceBuffer_;
    render_graph::BufferHandle graphFrustumIndirectDrawBuffer_;
    render_graph::BufferHandle graphFrustumDrawCountBuffer_;
    render_graph::BufferHandle graphVisibleInstanceBuffer_;
    render_graph::BufferHandle graphIndirectDrawBuffer_;
    render_graph::BufferHandle graphDrawCountBuffer_;
    std::array<render_graph::BufferHandle, directionalShadowCascadeCount>
        graphShadowVisibleInstanceBuffers_{};
    std::array<render_graph::BufferHandle, directionalShadowCascadeCount>
        graphShadowIndirectDrawBuffers_{};
    std::array<render_graph::BufferHandle, directionalShadowCascadeCount>
        graphShadowDrawCountBuffers_{};
    render_graph::BufferHandle graphUniformBuffer_;
    render_graph::BufferHandle graphEditorOverlayVertexBuffer_;
    render_graph::BufferHandle graphCaptureBuffer_;
    render_graph::BufferHandle graphPathAovReadbackBuffer_;
    std::array<Frame, framesInFlight> frames_{};
    VkSemaphore frameTimeline_ = VK_NULL_HANDLE;
    VkQueryPool timestampQueryPool_ = VK_NULL_HANDLE;
    float timestampPeriodNanoseconds_ = 0.0F;
    std::uint32_t timestampValidBits_ = 0;
    std::array<double, gpuTimingPassCount> gpuTimingTotalMilliseconds_{};
    std::array<std::uint64_t, gpuTimingPassCount> gpuTimingSampleCounts_{};
    bool timingReportWritten_ = false;
    // Present-wait semaphores belong to acquired images. Reusing one by frame
    // slot can race a presentation engine that has not consumed its wait yet.
    std::vector<VkSemaphore> renderFinishedSemaphores_;
    std::vector<bool> imageInitialized_;
    std::size_t frameIndex_ = 0;
    std::uint64_t nextTimelineValue_ = 1;
    std::uint64_t renderedFrameCount_ = 0;
    std::uint32_t sceneWorkItemCount_ = 0;
    glm::vec3 sceneCenter_{0.0F};
    glm::vec3 sceneMinimum_{0.0F};
    glm::vec3 sceneMaximum_{0.0F};
    float sceneRadius_ = 1.0F;
    glm::vec3 cameraPosition_{0.0F, 0.0F, 1.0F};
    float cameraYaw_ = -1.57079632679F;
    float cameraPitch_ = 0.0F;
    float environmentRotationRadians_ = 0.0F;
    lighting::AtmosphereDemoView atmosphereDemoView_{};
    float atmosphereBaseExposureEv_ = 0.0F;
    float timeOfDayHour_ = 16.0F;
    float timeOfDayCycleSeconds_ = 120.0F;
    bool timeOfDayPaused_ = false;
    // Start on a representative broken-cloud region instead of the unusually
    // large clear cell at the procedural field's origin.
    glm::vec2 cloudWindOffsetKm_{72.0F, -15.5F};
    float rainSimulationTimeSeconds_ = 0.0F;
    float rainWetnessAccumulation_ = 0.0F;
    float rainSteamAgeSeconds_ = 0.0F;
    float weatherDemoTimeSeconds_ = 0.0F;
    float weatherDemoBaseWindDirectionRadians_ = 0.0F;
    float cameraNear_ = 0.01F;
    float cameraFar_ = 1000.0F;
    float cameraFovRadians_ = 0.7330382858F;
    std::optional<RendererCamera> externalCamera_;
    struct EditorOverlayRange {
        std::uint32_t first = 0U;
        std::uint32_t count = 0U;
    };
    struct EditorOverlayRanges {
        EditorOverlayRange sceneTriangles;
        EditorOverlayRange selectedFaceTriangles;
        EditorOverlayRange gridLines;
        EditorOverlayRange xRayMeshLines;
        EditorOverlayRange xRayDepthScreenTriangles;
        EditorOverlayRange xRayTriangles;
        EditorOverlayRange meshLines;
        EditorOverlayRange overlayWorldLines;
        EditorOverlayRange depthScreenTriangles;
        EditorOverlayRange overlayScreenTriangles;
        EditorOverlayRange overlayScreenLines;
    };
    EditorOverlayFrame editorOverlay_;
    EditorOverlayRanges editorOverlayRanges_;
    float temporalScriptOriginYaw_ = -1.57079632679F;
    glm::vec3 temporalScriptOrigin_{0.0F};
    std::chrono::steady_clock::time_point previousCameraUpdate_{};
    VkFormat hdrFormat_ = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat motionFormat_ = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat normalFormat_ = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat temporalDepthFormat_ = VK_FORMAT_R32_SFLOAT;
    VkFormat pathTracingFormat_ = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkFormat pathTracingAovFormat_ = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat pathMomentFormat_ = VK_FORMAT_R16G16_SFLOAT;
    VkFormat reactiveMaskFormat_ = VK_FORMAT_R8_UNORM;
    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;
    temporal::TemporalState temporalState_;
    upscaling::FramePlan upscalingFramePlan_;
    vulkan::UploadTicket sceneUpload_{};
    bool sceneUploadPending_ = false;
    bool hiZReady_ = false;
    bool cameraMovedThisFrame_ = true;
    bool temporalImageLayoutsInitialized_ = false;
    bool upscalingHistoryLayoutsInitialized_ = false;
    bool dlssOutputLayoutInitialized_ = false;
    bool captureComplete_ = false;
    bool pathAovExportComplete_ = false;
    std::uint32_t pathTracingSampleIndex_ = 0;
    std::uint32_t pathSamplesThisFrame_ = 1U;
    std::uint64_t pathTracingResetCount_ = 0;
    bool pathTracingImageInitialized_ = false;
    bool pathHistoryLayoutsInitialized_ = false;
    bool nrdExternalLayoutsInitialized_ = false;
    bool pathTracingResetPending_ = false;
    bool atmosphereEnvironmentUpdatePending_ = true;
    std::uint32_t atmosphereHistoryResetFramesRemaining_ = 0U;
    bool pathTracingReportWritten_ = false;
};

} // namespace renderer
