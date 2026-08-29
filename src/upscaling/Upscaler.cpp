#include "renderer/upscaling/Upscaler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace renderer::upscaling {
namespace {

void requireExtent(Extent extent, std::string_view label) {
    if (extent.width == 0U || extent.height == 0U) {
        throw std::invalid_argument(std::string(label) + " must be nonzero");
    }
}

void requireInput(bool available, bool required, std::string_view label) {
    if (required && !available) {
        throw std::invalid_argument("Upscaler requires " + std::string(label));
    }
}

void validateFrameInput(const FrameInput& input, InputRequirements requirements) {
    requireExtent(input.renderExtent, "Upscaler render extent");
    requireExtent(input.outputExtent, "Upscaler output extent");
    requireInput(input.colorAvailable, requirements.color, "linear color");
    requireInput(input.depthAvailable, requirements.depth, "depth");
    requireInput(input.motionVectorsAvailable, requirements.motionVectors, "motion vectors");
    requireInput(input.jitterAvailable, requirements.jitter, "camera jitter");
    requireInput(input.exposureAvailable, requirements.exposure, "exposure");
    requireInput(input.reactiveMaskAvailable, requirements.reactiveMask, "a reactive mask");
    if (input.jitterAvailable && (!std::isfinite(input.currentJitterPixels.x) ||
                                  !std::isfinite(input.currentJitterPixels.y) ||
                                  !std::isfinite(input.previousJitterPixels.x) ||
                                  !std::isfinite(input.previousJitterPixels.y))) {
        throw std::invalid_argument("Upscaler camera jitter must be finite");
    }
    if (input.exposureAvailable &&
        (!std::isfinite(input.exposureScale) || input.exposureScale <= 0.0F)) {
        throw std::invalid_argument("Upscaler exposure scale must be finite and positive");
    }
}

class NativeUpscaler final : public IUpscaler {
  public:
    [[nodiscard]] Backend backend() const override { return Backend::native; }
    [[nodiscard]] std::string_view name() const override { return "native"; }
    [[nodiscard]] QualityMode qualityMode() const override { return QualityMode::native; }
    [[nodiscard]] float renderScale() const override { return 1.0F; }
    [[nodiscard]] BackendCapabilities capabilities() const override { return {}; }
    [[nodiscard]] InputConventions inputConventions() const override { return {}; }
    [[nodiscard]] InputRequirements inputRequirements() const override { return {}; }

    [[nodiscard]] Extent renderExtent(Extent outputExtent) const override {
        requireExtent(outputExtent, "Native upscaler output extent");
        return outputExtent;
    }

    [[nodiscard]] FramePlan prepareFrame(const FrameInput& input) const override {
        const InputRequirements requirements = inputRequirements();
        validateFrameInput(input, requirements);
        if (input.renderExtent != input.outputExtent) {
            throw std::invalid_argument(
                "Native upscaling requires identical render and output extents");
        }
        return {
            .backend = Backend::native,
            .qualityMode = QualityMode::native,
            .renderExtent = input.renderExtent,
            .outputExtent = input.outputExtent,
            .inputs = requirements,
            .currentJitterPixels = input.currentJitterPixels,
            .previousJitterPixels = input.previousJitterPixels,
            .exposureScale = input.exposureScale,
            .resetHistory = input.resetHistory,
            .dispatchRequired = false,
        };
    }
};

class SpatialUpscaler final : public IUpscaler {
  public:
    SpatialUpscaler(QualityMode qualityMode, float renderScale)
        : qualityMode_(qualityMode), renderScale_(renderScale) {
        if (!std::isfinite(renderScale_) || renderScale_ < minimumSpatialRenderScale ||
            renderScale_ > maximumSpatialRenderScale) {
            throw std::invalid_argument("Spatial render scale must be finite and in [0.5, 1.0]");
        }
        if (qualityMode_ == QualityMode::native) {
            throw std::invalid_argument("Spatial upscaling requires a scaled quality mode");
        }
    }

    [[nodiscard]] Backend backend() const override { return Backend::spatial; }
    [[nodiscard]] std::string_view name() const override { return "spatial"; }
    [[nodiscard]] QualityMode qualityMode() const override { return qualityMode_; }
    [[nodiscard]] float renderScale() const override { return renderScale_; }
    [[nodiscard]] BackendCapabilities capabilities() const override {
        return {
            .reconstruction = ReconstructionType::spatial,
            .historyOwnership = HistoryOwnership::none,
            .outputOwnership = OutputOwnership::renderer,
            .reactiveMask = false,
            .sharpening = false,
            .dynamicResolution = false,
            .minimumRenderScale = minimumSpatialRenderScale,
            .maximumRenderScale = maximumSpatialRenderScale,
        };
    }
    [[nodiscard]] InputConventions inputConventions() const override { return {}; }
    [[nodiscard]] InputRequirements inputRequirements() const override {
        return {
            .color = true,
            .depth = false,
            .motionVectors = false,
            .jitter = false,
            .exposure = false,
            .reactiveMask = false,
        };
    }

    [[nodiscard]] Extent renderExtent(Extent outputExtent) const override {
        requireExtent(outputExtent, "Spatial upscaler output extent");
        const auto scaledDimension = [this](std::uint32_t dimension) {
            const float scaled = static_cast<float>(dimension) * renderScale_;
            return std::max(1U, static_cast<std::uint32_t>(std::lround(scaled)));
        };
        return {
            .width = scaledDimension(outputExtent.width),
            .height = scaledDimension(outputExtent.height),
        };
    }

    [[nodiscard]] FramePlan prepareFrame(const FrameInput& input) const override {
        const InputRequirements requirements = inputRequirements();
        validateFrameInput(input, requirements);
        if (input.renderExtent != renderExtent(input.outputExtent)) {
            throw std::invalid_argument(
                "Spatial upscaling render extent does not match its configured scale");
        }
        return {
            .backend = Backend::spatial,
            .qualityMode = qualityMode_,
            .renderExtent = input.renderExtent,
            .outputExtent = input.outputExtent,
            .inputs = requirements,
            .currentJitterPixels = input.currentJitterPixels,
            .previousJitterPixels = input.previousJitterPixels,
            .exposureScale = input.exposureScale,
            .resetHistory = input.resetHistory,
            .dispatchRequired = true,
        };
    }

  private:
    QualityMode qualityMode_ = QualityMode::quality;
    float renderScale_ = defaultSpatialRenderScale;
};

class TemporalUpscaler final : public IUpscaler {
  public:
    TemporalUpscaler(QualityMode qualityMode, float renderScale)
        : qualityMode_(qualityMode), renderScale_(renderScale) {
        if (!std::isfinite(renderScale_) || renderScale_ < minimumSpatialRenderScale ||
            renderScale_ > maximumSpatialRenderScale) {
            throw std::invalid_argument("Temporal render scale must be finite and in [0.5, 1.0]");
        }
        if (qualityMode_ == QualityMode::native) {
            throw std::invalid_argument("Temporal upscaling requires a scaled quality mode");
        }
    }

    [[nodiscard]] Backend backend() const override { return Backend::temporal; }
    [[nodiscard]] std::string_view name() const override { return "temporal"; }
    [[nodiscard]] QualityMode qualityMode() const override { return qualityMode_; }
    [[nodiscard]] float renderScale() const override { return renderScale_; }
    [[nodiscard]] BackendCapabilities capabilities() const override {
        return {
            .reconstruction = ReconstructionType::temporal,
            .historyOwnership = HistoryOwnership::backend,
            .outputOwnership = OutputOwnership::renderer,
            .reactiveMask = true,
            .sharpening = false,
            .dynamicResolution = false,
            .minimumRenderScale = minimumSpatialRenderScale,
            .maximumRenderScale = maximumSpatialRenderScale,
        };
    }
    [[nodiscard]] InputConventions inputConventions() const override { return {}; }
    [[nodiscard]] InputRequirements inputRequirements() const override {
        return {
            .color = true,
            .depth = true,
            .motionVectors = true,
            .jitter = true,
            .exposure = true,
            .reactiveMask = true,
        };
    }

    [[nodiscard]] Extent renderExtent(Extent outputExtent) const override {
        requireExtent(outputExtent, "Temporal upscaler output extent");
        const auto scaledDimension = [this](std::uint32_t dimension) {
            const float scaled = static_cast<float>(dimension) * renderScale_;
            return std::max(1U, static_cast<std::uint32_t>(std::lround(scaled)));
        };
        return {
            .width = scaledDimension(outputExtent.width),
            .height = scaledDimension(outputExtent.height),
        };
    }

    [[nodiscard]] FramePlan prepareFrame(const FrameInput& input) const override {
        const InputRequirements requirements = inputRequirements();
        validateFrameInput(input, requirements);
        if (input.renderExtent != renderExtent(input.outputExtent)) {
            throw std::invalid_argument(
                "Temporal upscaling render extent does not match its configured scale");
        }
        return {
            .backend = Backend::temporal,
            .qualityMode = qualityMode_,
            .renderExtent = input.renderExtent,
            .outputExtent = input.outputExtent,
            .inputs = requirements,
            .currentJitterPixels = input.currentJitterPixels,
            .previousJitterPixels = input.previousJitterPixels,
            .exposureScale = input.exposureScale,
            .resetHistory = input.resetHistory,
            .dispatchRequired = true,
        };
    }

  private:
    QualityMode qualityMode_ = QualityMode::quality;
    float renderScale_ = defaultSpatialRenderScale;
};

class DlssUpscaler final : public IUpscaler {
  public:
    DlssUpscaler(QualityMode qualityMode, float renderScale)
        : qualityMode_(qualityMode), renderScale_(renderScale) {
        if (!std::isfinite(renderScale_) || renderScale_ < minimumSpatialRenderScale ||
            renderScale_ > maximumSpatialRenderScale) {
            throw std::invalid_argument("DLSS render scale must be finite and in [0.5, 1.0]");
        }
        if (qualityMode_ == QualityMode::native) {
            throw std::invalid_argument("DLSS requires a scaled quality mode");
        }
    }

    [[nodiscard]] Backend backend() const override { return Backend::dlss; }
    [[nodiscard]] std::string_view name() const override { return "dlss"; }
    [[nodiscard]] QualityMode qualityMode() const override { return qualityMode_; }
    [[nodiscard]] float renderScale() const override { return renderScale_; }
    [[nodiscard]] BackendCapabilities capabilities() const override {
        return {
            .reconstruction = ReconstructionType::temporal,
            .historyOwnership = HistoryOwnership::backend,
            .outputOwnership = OutputOwnership::backend,
            .reactiveMask = false,
            .sharpening = false,
            .dynamicResolution = false,
            .minimumRenderScale = minimumSpatialRenderScale,
            .maximumRenderScale = maximumSpatialRenderScale,
        };
    }
    [[nodiscard]] InputConventions inputConventions() const override { return {}; }
    [[nodiscard]] InputRequirements inputRequirements() const override {
        return {
            .color = true,
            .depth = true,
            .motionVectors = true,
            .jitter = true,
            .exposure = true,
            .reactiveMask = false,
        };
    }

    [[nodiscard]] Extent renderExtent(Extent outputExtent) const override {
        requireExtent(outputExtent, "DLSS output extent");
        const auto scaledDimension = [this](std::uint32_t dimension) {
            return std::max(
                1U, static_cast<std::uint32_t>(
                        std::lround(static_cast<float>(dimension) * renderScale_)));
        };
        return {.width = scaledDimension(outputExtent.width),
                .height = scaledDimension(outputExtent.height)};
    }

    [[nodiscard]] FramePlan prepareFrame(const FrameInput& input) const override {
        const InputRequirements requirements = inputRequirements();
        validateFrameInput(input, requirements);
        if (input.renderExtent != renderExtent(input.outputExtent)) {
            throw std::invalid_argument(
                "DLSS render extent does not match its configured quality scale");
        }
        return {
            .backend = Backend::dlss,
            .qualityMode = qualityMode_,
            .renderExtent = input.renderExtent,
            .outputExtent = input.outputExtent,
            .inputs = requirements,
            .currentJitterPixels = input.currentJitterPixels,
            .previousJitterPixels = input.previousJitterPixels,
            .exposureScale = input.exposureScale,
            .resetHistory = input.resetHistory,
            .dispatchRequired = true,
        };
    }

  private:
    QualityMode qualityMode_ = QualityMode::quality;
    float renderScale_ = defaultSpatialRenderScale;
};

} // namespace

std::string_view backendName(Backend backend) {
    switch (backend) {
    case Backend::native:
        return "native";
    case Backend::spatial:
        return "spatial";
    case Backend::temporal:
        return "temporal";
    case Backend::dlss:
        return "dlss";
    }
    return "unknown";
}

std::optional<Backend> parseBackend(std::string_view name) {
    if (name == "native") {
        return Backend::native;
    }
    if (name == "spatial") {
        return Backend::spatial;
    }
    if (name == "temporal") {
        return Backend::temporal;
    }
    if (name == "dlss") {
        return Backend::dlss;
    }
    return std::nullopt;
}

std::string_view qualityModeName(QualityMode qualityMode) {
    switch (qualityMode) {
    case QualityMode::native:
        return "native";
    case QualityMode::quality:
        return "quality";
    case QualityMode::balanced:
        return "balanced";
    case QualityMode::performance:
        return "performance";
    case QualityMode::custom:
        return "custom";
    }
    return "unknown";
}

std::optional<QualityMode> parseQualityMode(std::string_view name) {
    if (name == "native") {
        return QualityMode::native;
    }
    if (name == "quality") {
        return QualityMode::quality;
    }
    if (name == "balanced") {
        return QualityMode::balanced;
    }
    if (name == "performance") {
        return QualityMode::performance;
    }
    if (name == "custom") {
        return QualityMode::custom;
    }
    return std::nullopt;
}

float qualityModeRenderScale(QualityMode qualityMode) {
    switch (qualityMode) {
    case QualityMode::native:
        return 1.0F;
    case QualityMode::quality:
        return qualitySpatialRenderScale;
    case QualityMode::balanced:
        return balancedSpatialRenderScale;
    case QualityMode::performance:
        return performanceSpatialRenderScale;
    case QualityMode::custom:
        throw std::invalid_argument("Custom upscaling quality requires an explicit render scale");
    }
    throw std::invalid_argument("Unknown upscaling quality mode");
}

std::string_view reconstructionTypeName(ReconstructionType reconstruction) {
    switch (reconstruction) {
    case ReconstructionType::identity:
        return "identity";
    case ReconstructionType::spatial:
        return "spatial";
    case ReconstructionType::temporal:
        return "temporal";
    }
    return "unknown";
}

std::string_view historyOwnershipName(HistoryOwnership ownership) {
    switch (ownership) {
    case HistoryOwnership::none:
        return "none";
    case HistoryOwnership::renderer:
        return "renderer";
    case HistoryOwnership::backend:
        return "backend";
    }
    return "unknown";
}

std::string_view outputOwnershipName(OutputOwnership ownership) {
    switch (ownership) {
    case OutputOwnership::renderer:
        return "renderer";
    case OutputOwnership::backend:
        return "backend";
    }
    return "unknown";
}

std::string_view colorConventionName(ColorConvention convention) {
    switch (convention) {
    case ColorConvention::linearHdrBeforeExposure:
        return "linear-hdr-before-exposure";
    }
    return "unknown";
}

std::string_view depthConventionName(DepthConvention convention) {
    switch (convention) {
    case DepthConvention::vulkanDeviceZeroToOne:
        return "vulkan-device-0-to-1";
    }
    return "unknown";
}

std::string_view motionVectorConventionName(MotionVectorConvention convention) {
    switch (convention) {
    case MotionVectorConvention::currentMinusPreviousUv:
        return "current-minus-previous-uv";
    }
    return "unknown";
}

std::string_view jitterConventionName(JitterConvention convention) {
    switch (convention) {
    case JitterConvention::renderPixelOffsets:
        return "render-pixel-offsets";
    }
    return "unknown";
}

std::string_view exposureConventionName(ExposureConvention convention) {
    switch (convention) {
    case ExposureConvention::linearMultiplier:
        return "linear-multiplier";
    }
    return "unknown";
}

std::unique_ptr<IUpscaler> createUpscaler(BackendConfiguration configuration) {
    switch (configuration.backend) {
    case Backend::native:
        if (configuration.qualityMode != QualityMode::native ||
            !std::isfinite(configuration.customRenderScale) ||
            std::abs(configuration.customRenderScale - 1.0F) > 1.0e-6F) {
            throw std::invalid_argument("Native upscaling requires native quality at scale 1.0");
        }
        return std::make_unique<NativeUpscaler>();
    case Backend::spatial: {
        if (configuration.qualityMode == QualityMode::native) {
            throw std::invalid_argument("Spatial upscaling requires a scaled quality mode");
        }
        const float renderScale = configuration.qualityMode == QualityMode::custom
                                      ? configuration.customRenderScale
                                      : qualityModeRenderScale(configuration.qualityMode);
        return std::make_unique<SpatialUpscaler>(configuration.qualityMode, renderScale);
    }
    case Backend::temporal: {
        if (configuration.qualityMode == QualityMode::native) {
            throw std::invalid_argument("Temporal upscaling requires a scaled quality mode");
        }
        const float renderScale = configuration.qualityMode == QualityMode::custom
                                      ? configuration.customRenderScale
                                      : qualityModeRenderScale(configuration.qualityMode);
        return std::make_unique<TemporalUpscaler>(configuration.qualityMode, renderScale);
    }
    case Backend::dlss: {
        if (configuration.qualityMode == QualityMode::native ||
            configuration.qualityMode == QualityMode::custom) {
            throw std::invalid_argument("DLSS requires quality, balanced, or performance mode");
        }
        const float renderScale = qualityModeRenderScale(configuration.qualityMode);
        return std::make_unique<DlssUpscaler>(configuration.qualityMode, renderScale);
    }
    }
    throw std::invalid_argument("Unknown upscaling backend");
}

std::unique_ptr<IUpscaler> createUpscaler(Backend backend, float renderScale) {
    return createUpscaler({
        .backend = backend,
        .qualityMode = backend == Backend::native ? QualityMode::native : QualityMode::custom,
        .customRenderScale = renderScale,
    });
}

} // namespace renderer::upscaling
