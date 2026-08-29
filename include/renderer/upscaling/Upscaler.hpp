#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace renderer::upscaling {

enum class Backend { native, spatial, temporal, dlss };
enum class QualityMode { native, quality, balanced, performance, custom };
enum class ReconstructionType { identity, spatial, temporal };
enum class HistoryOwnership { none, renderer, backend };
enum class OutputOwnership { renderer, backend };
enum class ColorConvention { linearHdrBeforeExposure };
enum class DepthConvention { vulkanDeviceZeroToOne };
enum class MotionVectorConvention { currentMinusPreviousUv };
enum class JitterConvention { renderPixelOffsets };
enum class ExposureConvention { linearMultiplier };

inline constexpr float minimumSpatialRenderScale = 0.5F;
inline constexpr float maximumSpatialRenderScale = 1.0F;
inline constexpr float qualitySpatialRenderScale = 2.0F / 3.0F;
inline constexpr float balancedSpatialRenderScale = 10.0F / 17.0F;
inline constexpr float performanceSpatialRenderScale = 0.5F;
inline constexpr float defaultSpatialRenderScale = qualitySpatialRenderScale;

struct BackendConfiguration {
    Backend backend = Backend::native;
    QualityMode qualityMode = QualityMode::native;
    float customRenderScale = 1.0F;
};

struct BackendCapabilities {
    ReconstructionType reconstruction = ReconstructionType::identity;
    HistoryOwnership historyOwnership = HistoryOwnership::none;
    OutputOwnership outputOwnership = OutputOwnership::renderer;
    bool reactiveMask = false;
    bool sharpening = false;
    bool dynamicResolution = false;
    float minimumRenderScale = 1.0F;
    float maximumRenderScale = 1.0F;

    friend bool operator==(const BackendCapabilities&, const BackendCapabilities&) = default;
};

struct InputConventions {
    ColorConvention color = ColorConvention::linearHdrBeforeExposure;
    DepthConvention depth = DepthConvention::vulkanDeviceZeroToOne;
    MotionVectorConvention motionVectors = MotionVectorConvention::currentMinusPreviousUv;
    JitterConvention jitter = JitterConvention::renderPixelOffsets;
    ExposureConvention exposure = ExposureConvention::linearMultiplier;
    bool motionVectorsExcludeJitter = true;

    friend bool operator==(const InputConventions&, const InputConventions&) = default;
};

struct Extent {
    std::uint32_t width = 0U;
    std::uint32_t height = 0U;

    friend bool operator==(Extent, Extent) = default;
};

struct Jitter {
    float x = 0.0F;
    float y = 0.0F;
};

struct InputRequirements {
    bool color = true;
    bool depth = true;
    bool motionVectors = true;
    bool jitter = true;
    bool exposure = true;
    bool reactiveMask = false;
};

struct FrameInput {
    Extent renderExtent;
    Extent outputExtent;
    bool colorAvailable = false;
    bool depthAvailable = false;
    bool motionVectorsAvailable = false;
    bool jitterAvailable = false;
    bool exposureAvailable = false;
    bool reactiveMaskAvailable = false;
    Jitter currentJitterPixels;
    Jitter previousJitterPixels;
    float exposureScale = 1.0F;
    bool resetHistory = true;
};

struct FramePlan {
    Backend backend = Backend::native;
    QualityMode qualityMode = QualityMode::native;
    Extent renderExtent;
    Extent outputExtent;
    InputRequirements inputs;
    Jitter currentJitterPixels;
    Jitter previousJitterPixels;
    float exposureScale = 1.0F;
    bool resetHistory = true;
    bool dispatchRequired = false;
};

class IUpscaler {
  public:
    virtual ~IUpscaler() = default;

    [[nodiscard]] virtual Backend backend() const = 0;
    [[nodiscard]] virtual std::string_view name() const = 0;
    [[nodiscard]] virtual QualityMode qualityMode() const = 0;
    [[nodiscard]] virtual float renderScale() const = 0;
    [[nodiscard]] virtual BackendCapabilities capabilities() const = 0;
    [[nodiscard]] virtual InputConventions inputConventions() const = 0;
    [[nodiscard]] virtual InputRequirements inputRequirements() const = 0;
    [[nodiscard]] virtual Extent renderExtent(Extent outputExtent) const = 0;
    [[nodiscard]] virtual FramePlan prepareFrame(const FrameInput& input) const = 0;
};

[[nodiscard]] std::string_view backendName(Backend backend);
[[nodiscard]] std::optional<Backend> parseBackend(std::string_view name);
[[nodiscard]] std::string_view qualityModeName(QualityMode qualityMode);
[[nodiscard]] std::optional<QualityMode> parseQualityMode(std::string_view name);
[[nodiscard]] float qualityModeRenderScale(QualityMode qualityMode);
[[nodiscard]] std::string_view reconstructionTypeName(ReconstructionType reconstruction);
[[nodiscard]] std::string_view historyOwnershipName(HistoryOwnership ownership);
[[nodiscard]] std::string_view outputOwnershipName(OutputOwnership ownership);
[[nodiscard]] std::string_view colorConventionName(ColorConvention convention);
[[nodiscard]] std::string_view depthConventionName(DepthConvention convention);
[[nodiscard]] std::string_view motionVectorConventionName(MotionVectorConvention convention);
[[nodiscard]] std::string_view jitterConventionName(JitterConvention convention);
[[nodiscard]] std::string_view exposureConventionName(ExposureConvention convention);
[[nodiscard]] std::unique_ptr<IUpscaler> createUpscaler(BackendConfiguration configuration);
[[nodiscard]] std::unique_ptr<IUpscaler> createUpscaler(Backend backend,
                                                        float renderScale = 1.0F);

} // namespace renderer::upscaling
