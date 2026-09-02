#include "renderer/upscaling/Upscaler.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <typename Function> void requireInvalid(Function&& function, std::string_view message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

renderer::upscaling::FrameInput completeInput() {
    return {
        .renderExtent = {1920U, 1080U},
        .outputExtent = {1920U, 1080U},
        .colorAvailable = true,
        .depthAvailable = true,
        .motionVectorsAvailable = true,
        .jitterAvailable = true,
        .exposureAvailable = true,
        .reactiveMaskAvailable = false,
        .currentJitterPixels = {0.25F, -0.125F},
        .previousJitterPixels = {-0.25F, 0.125F},
        .exposureScale = 2.0F,
        .resetHistory = true,
    };
}

void testBackendIdentity() {
    using namespace renderer::upscaling;
    require(parseBackend("native") == Backend::native &&
                parseBackend("spatial") == Backend::spatial &&
                parseBackend("dlss") == Backend::dlss &&
                parseBackend("temporal") == Backend::temporal &&
                backendName(Backend::native) == "native" &&
                backendName(Backend::spatial) == "spatial" &&
                backendName(Backend::temporal) == "temporal" &&
                backendName(Backend::dlss) == "dlss",
            "Upscaling backend naming or parsing is incorrect");
    const std::unique_ptr<IUpscaler> upscaler = createUpscaler(Backend::native);
    require(upscaler && upscaler->backend() == Backend::native && upscaler->name() == "native",
            "Native upscaler factory returned the wrong backend");
}

void testQualityModes() {
    using namespace renderer::upscaling;
    require(parseQualityMode("quality") == QualityMode::quality &&
                parseQualityMode("balanced") == QualityMode::balanced &&
                parseQualityMode("performance") == QualityMode::performance &&
                parseQualityMode("custom") == QualityMode::custom && !parseQualityMode("ultra") &&
                qualityModeName(QualityMode::native) == "native" &&
                qualityModeName(QualityMode::quality) == "quality" &&
                qualityModeName(QualityMode::balanced) == "balanced" &&
                qualityModeName(QualityMode::performance) == "performance" &&
                qualityModeName(QualityMode::custom) == "custom",
            "Upscaling quality naming or parsing is incorrect");
    require(std::abs(qualityModeRenderScale(QualityMode::native) - 1.0F) < 1.0e-6F &&
                std::abs(qualityModeRenderScale(QualityMode::quality) - 2.0F / 3.0F) < 1.0e-6F &&
                std::abs(qualityModeRenderScale(QualityMode::balanced) - 10.0F / 17.0F) < 1.0e-6F &&
                std::abs(qualityModeRenderScale(QualityMode::performance) - 0.5F) < 1.0e-6F,
            "Named upscaling quality modes use incorrect scales");
    requireInvalid([&] { (void)qualityModeRenderScale(QualityMode::custom); },
                   "Custom quality produced a scale without explicit configuration");
}

void testCapabilitiesAndConventions() {
    using namespace renderer::upscaling;
    const std::unique_ptr<IUpscaler> native = createUpscaler(Backend::native);
    const std::unique_ptr<IUpscaler> spatial = createUpscaler({
        .backend = Backend::spatial,
        .qualityMode = QualityMode::quality,
    });
    const std::unique_ptr<IUpscaler> temporal = createUpscaler({
        .backend = Backend::temporal,
        .qualityMode = QualityMode::balanced,
    });
    const std::unique_ptr<IUpscaler> dlss = createUpscaler({
        .backend = Backend::dlss,
        .qualityMode = QualityMode::quality,
    });
    const BackendCapabilities nativeCapabilities = native->capabilities();
    const BackendCapabilities spatialCapabilities = spatial->capabilities();
    require(nativeCapabilities.reconstruction == ReconstructionType::identity &&
                nativeCapabilities.historyOwnership == HistoryOwnership::none &&
                nativeCapabilities.outputOwnership == OutputOwnership::renderer &&
                !nativeCapabilities.reactiveMask && !nativeCapabilities.sharpening &&
                !nativeCapabilities.dynamicResolution &&
                nativeCapabilities.minimumRenderScale == 1.0F &&
                nativeCapabilities.maximumRenderScale == 1.0F,
            "Native backend capabilities are incorrect");
    require(spatialCapabilities.reconstruction == ReconstructionType::spatial &&
                spatialCapabilities.historyOwnership == HistoryOwnership::none &&
                spatialCapabilities.outputOwnership == OutputOwnership::renderer &&
                !spatialCapabilities.reactiveMask && !spatialCapabilities.sharpening &&
                !spatialCapabilities.dynamicResolution &&
                spatialCapabilities.minimumRenderScale == minimumSpatialRenderScale &&
                spatialCapabilities.maximumRenderScale == maximumSpatialRenderScale,
            "Spatial backend capabilities are incorrect");
    const BackendCapabilities temporalCapabilities = temporal->capabilities();
    require(temporalCapabilities.reconstruction == ReconstructionType::temporal &&
                temporalCapabilities.historyOwnership == HistoryOwnership::backend &&
                temporalCapabilities.outputOwnership == OutputOwnership::renderer &&
                temporalCapabilities.reactiveMask && !temporalCapabilities.sharpening &&
                !temporalCapabilities.dynamicResolution &&
                temporalCapabilities.minimumRenderScale == minimumSpatialRenderScale &&
                temporalCapabilities.maximumRenderScale == maximumUpscalingRenderScale,
            "Temporal backend capabilities are incorrect");
    const BackendCapabilities dlssCapabilities = dlss->capabilities();
    require(dlssCapabilities.reconstruction == ReconstructionType::temporal &&
                dlssCapabilities.historyOwnership == HistoryOwnership::backend &&
                dlssCapabilities.outputOwnership == OutputOwnership::backend &&
                !dlssCapabilities.reactiveMask && !dlssCapabilities.sharpening &&
                !dlssCapabilities.dynamicResolution,
            "DLSS backend capabilities are incorrect");
    const InputConventions conventions = spatial->inputConventions();
    require(conventions == native->inputConventions() &&
                conventions.color == ColorConvention::linearHdrBeforeExposure &&
                conventions.depth == DepthConvention::vulkanDeviceZeroToOne &&
                conventions.motionVectors == MotionVectorConvention::currentMinusPreviousUv &&
                conventions.jitter == JitterConvention::renderPixelOffsets &&
                conventions.exposure == ExposureConvention::linearMultiplier &&
                conventions.motionVectorsExcludeJitter,
            "Canonical temporal input conventions are incorrect");
    require(reconstructionTypeName(spatialCapabilities.reconstruction) == "spatial" &&
                historyOwnershipName(spatialCapabilities.historyOwnership) == "none" &&
                outputOwnershipName(spatialCapabilities.outputOwnership) == "renderer" &&
                colorConventionName(conventions.color) == "linear-hdr-before-exposure" &&
                depthConventionName(conventions.depth) == "vulkan-device-0-to-1" &&
                motionVectorConventionName(conventions.motionVectors) ==
                    "current-minus-previous-uv" &&
                jitterConventionName(conventions.jitter) == "render-pixel-offsets" &&
                exposureConventionName(conventions.exposure) == "linear-multiplier",
            "Capability or convention diagnostic names are incorrect");
}

void testSpatialExtentPolicy() {
    using namespace renderer::upscaling;
    const std::unique_ptr<IUpscaler> upscaler =
        createUpscaler({.backend = Backend::spatial, .qualityMode = QualityMode::quality});
    require(upscaler->backend() == Backend::spatial && upscaler->name() == "spatial" &&
                upscaler->qualityMode() == QualityMode::quality &&
                std::abs(upscaler->renderScale() - defaultSpatialRenderScale) < 1.0e-6F,
            "Spatial upscaler factory returned the wrong configuration");
    require(upscaler->renderExtent({1920U, 1080U}) == Extent{1280U, 720U} &&
                upscaler->renderExtent({1279U, 719U}) == Extent{853U, 479U},
            "Spatial upscaling produced an incorrect rounded render extent");
    const std::unique_ptr<IUpscaler> supersampler = createUpscaler(Backend::spatial, 1.5F);
    require(supersampler->renderExtent({1280U, 720U}) == Extent{1920U, 1080U},
            "Spatial supersampling did not produce a 1080p render extent for a 720p window");
    requireInvalid([&] { (void)createUpscaler(Backend::spatial, 0.49F); },
                   "Spatial upscaling accepted a scale below its supported range");
    requireInvalid([&] { (void)createUpscaler(Backend::spatial, 2.01F); },
                   "Spatial upscaling accepted a scale above its supported range");
    requireInvalid([&] { (void)createUpscaler(Backend::temporal, 1.01F); },
                   "Temporal upscaling accepted a supersampling scale");
    requireInvalid([&] { (void)createUpscaler(Backend::native, 0.75F); },
                   "Native upscaling accepted a reduced render scale");
    requireInvalid(
        [&] {
            (void)createUpscaler({.backend = Backend::native, .qualityMode = QualityMode::quality});
        },
        "Native upscaling accepted a scaled quality mode");
    requireInvalid(
        [&] {
            (void)createUpscaler({.backend = Backend::spatial, .qualityMode = QualityMode::native});
        },
        "Spatial upscaling accepted native quality");
    const std::unique_ptr<IUpscaler> balanced =
        createUpscaler({.backend = Backend::spatial, .qualityMode = QualityMode::balanced});
    const std::unique_ptr<IUpscaler> performance =
        createUpscaler({.backend = Backend::spatial, .qualityMode = QualityMode::performance});
    require(balanced->renderExtent({1700U, 850U}) == Extent{1000U, 500U} &&
                performance->renderExtent({1920U, 1080U}) == Extent{960U, 540U},
            "Balanced or performance quality produced an incorrect extent");
}

void testNativeExtentPolicy() {
    using namespace renderer::upscaling;
    const std::unique_ptr<IUpscaler> upscaler = createUpscaler(Backend::native);
    require(upscaler->renderExtent({1279U, 719U}) == Extent{1279U, 719U},
            "Native upscaling changed the output extent");
    requireInvalid([&] { (void)upscaler->renderExtent({0U, 720U}); },
                   "Native upscaling accepted a zero output extent");
}

void testCanonicalInputsAndReset() {
    using namespace renderer::upscaling;
    const std::unique_ptr<IUpscaler> upscaler = createUpscaler(Backend::native);
    const InputRequirements requirements = upscaler->inputRequirements();
    require(requirements.color && requirements.depth && requirements.motionVectors &&
                requirements.jitter && requirements.exposure && !requirements.reactiveMask,
            "Native upscaling does not expose the canonical reconstruction inputs");
    const FramePlan resetPlan = upscaler->prepareFrame(completeInput());
    require(resetPlan.backend == Backend::native && resetPlan.resetHistory &&
                resetPlan.qualityMode == QualityMode::native && !resetPlan.dispatchRequired &&
                resetPlan.renderExtent == resetPlan.outputExtent &&
                resetPlan.currentJitterPixels.x == 0.25F &&
                resetPlan.previousJitterPixels.y == 0.125F && resetPlan.exposureScale == 2.0F,
            "Native frame plan changed identity or reset semantics");
    FrameInput continued = completeInput();
    continued.resetHistory = false;
    require(!upscaler->prepareFrame(continued).resetHistory,
            "Native frame plan failed to propagate continued history");
}

void testInvalidInputs() {
    using namespace renderer::upscaling;
    const std::unique_ptr<IUpscaler> upscaler = createUpscaler(Backend::native);
    FrameInput missingDepth = completeInput();
    missingDepth.depthAvailable = false;
    requireInvalid([&] { (void)upscaler->prepareFrame(missingDepth); },
                   "Native upscaling accepted missing depth");
    FrameInput mismatchedExtent = completeInput();
    mismatchedExtent.renderExtent = {1280U, 720U};
    requireInvalid([&] { (void)upscaler->prepareFrame(mismatchedExtent); },
                   "Native upscaling accepted a scaled input");
    FrameInput invalidJitter = completeInput();
    invalidJitter.currentJitterPixels.x = std::numeric_limits<float>::quiet_NaN();
    requireInvalid([&] { (void)upscaler->prepareFrame(invalidJitter); },
                   "Native upscaling accepted non-finite jitter");
    FrameInput invalidExposure = completeInput();
    invalidExposure.exposureScale = 0.0F;
    requireInvalid([&] { (void)upscaler->prepareFrame(invalidExposure); },
                   "Native upscaling accepted invalid exposure");
}

void testSpatialFramePlan() {
    using namespace renderer::upscaling;
    const std::unique_ptr<IUpscaler> upscaler = createUpscaler(Backend::spatial, 0.5F);
    FrameInput input = completeInput();
    input.renderExtent = {960U, 540U};
    input.depthAvailable = false;
    input.motionVectorsAvailable = false;
    input.jitterAvailable = false;
    input.exposureAvailable = false;
    const FramePlan plan = upscaler->prepareFrame(input);
    require(
        plan.backend == Backend::spatial && plan.qualityMode == QualityMode::custom &&
            plan.dispatchRequired && plan.resetHistory && plan.renderExtent == Extent{960U, 540U} &&
            plan.outputExtent == Extent{1920U, 1080U} && plan.inputs.color && !plan.inputs.depth &&
            !plan.inputs.motionVectors && !plan.inputs.jitter && !plan.inputs.exposure,
        "Spatial frame plan does not describe the color-only scaling dispatch");
    input.renderExtent.width = 959U;
    requireInvalid([&] { (void)upscaler->prepareFrame(input); },
                   "Spatial upscaling accepted a render extent that disagrees with its scale");
}

void testTemporalFramePlan() {
    using namespace renderer::upscaling;
    const std::unique_ptr<IUpscaler> upscaler = createUpscaler({
        .backend = Backend::temporal,
        .qualityMode = QualityMode::performance,
    });
    FrameInput input = completeInput();
    input.renderExtent = {960U, 540U};
    input.reactiveMaskAvailable = true;
    input.resetHistory = false;
    const FramePlan plan = upscaler->prepareFrame(input);
    require(plan.backend == Backend::temporal && plan.qualityMode == QualityMode::performance &&
                plan.dispatchRequired && !plan.resetHistory &&
                plan.renderExtent == Extent{960U, 540U} &&
                plan.outputExtent == Extent{1920U, 1080U} && plan.inputs.color &&
                plan.inputs.depth && plan.inputs.motionVectors && plan.inputs.jitter &&
                plan.inputs.exposure && plan.inputs.reactiveMask &&
                std::abs(plan.exposureScale - 2.0F) < 1.0e-6F,
            "Temporal frame plan does not expose its full temporal contract");
    input.motionVectorsAvailable = false;
    requireInvalid([&] { (void)upscaler->prepareFrame(input); },
                   "Temporal upscaling accepted missing motion vectors");
    input.motionVectorsAvailable = true;
    input.reactiveMaskAvailable = false;
    requireInvalid([&] { (void)upscaler->prepareFrame(input); },
                   "Temporal upscaling accepted a missing reactive mask");
    requireInvalid(
        [&] {
            (void)createUpscaler(
                {.backend = Backend::temporal, .qualityMode = QualityMode::native});
        },
        "Temporal upscaling accepted native quality");
    requireInvalid(
        [&] {
            (void)createUpscaler(
                {.backend = Backend::dlss, .qualityMode = QualityMode::custom,
                 .customRenderScale = 0.75F});
        },
        "DLSS accepted a custom scale instead of an NVIDIA quality mode");
}

void testDlssFramePlan() {
    using namespace renderer::upscaling;
    const std::unique_ptr<IUpscaler> upscaler = createUpscaler({
        .backend = Backend::dlss,
        .qualityMode = QualityMode::quality,
    });
    FrameInput input = completeInput();
    input.renderExtent = {1280U, 720U};
    input.resetHistory = false;
    const FramePlan plan = upscaler->prepareFrame(input);
    require(plan.backend == Backend::dlss && plan.qualityMode == QualityMode::quality &&
                plan.dispatchRequired && !plan.resetHistory &&
                plan.renderExtent == Extent{1280U, 720U} &&
                plan.outputExtent == Extent{1920U, 1080U} && plan.inputs.color &&
                plan.inputs.depth && plan.inputs.motionVectors && plan.inputs.jitter &&
                plan.inputs.exposure && !plan.inputs.reactiveMask &&
                upscaler->capabilities().outputOwnership == OutputOwnership::backend,
            "DLSS frame plan does not expose its Streamline contract");
    input.depthAvailable = false;
    requireInvalid([&] { (void)upscaler->prepareFrame(input); },
                   "DLSS accepted missing depth");
}

} // namespace

int main() {
    try {
        testBackendIdentity();
        testQualityModes();
        testCapabilitiesAndConventions();
        testNativeExtentPolicy();
        testSpatialExtentPolicy();
        testCanonicalInputsAndReset();
        testInvalidInputs();
        testSpatialFramePlan();
        testTemporalFramePlan();
        testDlssFramePlan();
        std::cout << "Upscaling tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Upscaling test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
