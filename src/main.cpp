#include "renderer/core/Log.hpp"
#if RENDERER_ENABLE_STREAMLINE
#include "renderer/nvidia/StreamlineRuntime.hpp"
#endif
#include "renderer/output/AovExport.hpp"
#include "renderer/platform/Window.hpp"
#include "renderer/renderer/Renderer.hpp"
#include "renderer/vulkan/Device.hpp"
#include "renderer/vulkan/Instance.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

enum class ResizeScript { none, standard, odd, stress };

#if RENDERER_ENABLE_STREAMLINE
class StreamlineShutdownGuard final {
  public:
    explicit StreamlineShutdownGuard(renderer::nvidia::StreamlineRuntime* runtime)
        : runtime_(runtime) {}

    ~StreamlineShutdownGuard() {
        if (runtime_ != nullptr) {
            runtime_->shutdown();
        }
    }

    StreamlineShutdownGuard(const StreamlineShutdownGuard&) = delete;
    StreamlineShutdownGuard& operator=(const StreamlineShutdownGuard&) = delete;

  private:
    renderer::nvidia::StreamlineRuntime* runtime_ = nullptr;
};
#endif

struct Options {
    std::optional<std::uint64_t> frameLimit;
    std::filesystem::path scenePath;
    bool sceneAnimationStaticPose = false;
    std::filesystem::path environmentPath;
    float exposureCompensationEv = 0.0F;
    float environmentRotationDegrees = 0.0F;
    std::filesystem::path capturePath;
    std::filesystem::path timingOutputPath;
    std::uint64_t captureFrame = 8;
    std::filesystem::path aovExportDirectory;
    std::uint64_t aovExportFrame = 8;
    renderer::RenderLayerMask visibilityMask = renderer::defaultViewVisibilityMask;
    bool taaEnabled = true;
    bool bloomEnabled = true;
    float bloomIntensity = renderer::post::bloomIntensity;
    float bloomRadius = 1.0F;
    float filmGrainIntensity = 0.015F;
    bool emissiveSphereDemo = false;
    bool directionalShadowDemo = false;
    bool hdriBackdropDemo = false;
    bool alphaShadowDemo = false;
    bool shadowsEnabled = true;
    renderer::ray_tracing::Request rayTracingRequest = renderer::ray_tracing::Request::automatic;
    renderer::vulkan::HardwarePolicy hardwarePolicy =
        renderer::vulkan::HardwarePolicy::nvidiaRtx;
    renderer::RayTracingTransformScript rayTracingTransformScript =
        renderer::RayTracingTransformScript::none;
    renderer::RenderMode renderMode = renderer::RenderMode::rasterized;
    renderer::upscaling::Backend upscalingBackend = renderer::upscaling::Backend::native;
    renderer::upscaling::QualityMode upscalingQualityMode =
        renderer::upscaling::QualityMode::native;
    bool upscalingQualityModeSpecified = false;
    float upscalingRenderScale = 1.0F;
    bool upscalingRenderScaleSpecified = false;
    bool pathEnvironmentImportanceSampling = true;
    bool pathEmissiveTriangleSampling = true;
    std::uint32_t pathMaxBounces = renderer::defaultPathBounceCount;
    std::uint32_t pathInteractiveSamples = renderer::defaultPathInteractiveSampleCount;
    renderer::PathDenoiser pathDenoiser = renderer::PathDenoiser::nrd;
    renderer::PathDebugView pathDebugView = renderer::PathDebugView::final;
    bool occlusionCullingEnabled = true;
    renderer::post::TestPattern postTestPattern = renderer::post::TestPattern::none;
    float postTestSourceValue = 4.0F;
    glm::vec2 postTestPosition{0.5F};
    std::uint32_t width = 1280;
    std::uint32_t height = 720;
    renderer::TemporalDebugView temporalDebugView = renderer::TemporalDebugView::none;
    renderer::TemporalCameraScript temporalCameraScript = renderer::TemporalCameraScript::none;
    renderer::CameraStartView cameraStartView = renderer::CameraStartView::front;
    bool interactiveCameraInput = true;
    ResizeScript resizeScript = ResizeScript::none;
    bool captureFrameSpecified = false;
    bool aovExportFrameSpecified = false;
    renderer::vulkan::ValidationMode validation =
#if RENDERER_VALIDATION_DEFAULT
        renderer::vulkan::ValidationMode::available;
#else
        renderer::vulkan::ValidationMode::off;
#endif
};

float parseExposure(std::string_view text) {
    float value = 0.0F;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value) ||
        value < -16.0F || value > 16.0F) {
        throw std::runtime_error("--exposure-ev requires a finite value in [-16, 16]");
    }
    return value;
}

float parsePostSourceValue(std::string_view text) {
    float value = 0.0F;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value) ||
        value < 0.0F || value > 65504.0F) {
        throw std::runtime_error("--post-value requires a finite value in [0, 65504]");
    }
    return value;
}

float parseBloomIntensity(std::string_view text) {
    float value = 0.0F;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value) ||
        value < 0.0F || value > 16.0F) {
        throw std::runtime_error("--bloom-intensity requires a finite value in [0, 16]");
    }
    return value;
}

float parseBloomRadius(std::string_view text) {
    float value = 0.0F;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value) ||
        value < 0.25F || value > 16.0F) {
        throw std::runtime_error("--bloom-radius requires a finite value in [0.25, 16]");
    }
    return value;
}

float parseFilmGrainIntensity(std::string_view text) {
    float value = 0.0F;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value) ||
        value < 0.0F || value > 0.1F) {
        throw std::runtime_error("--film-grain requires a finite value in [0, 0.1]");
    }
    return value;
}

float parseRenderScale(std::string_view text) {
    float value = 0.0F;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value) ||
        value < renderer::upscaling::minimumSpatialRenderScale ||
        value > renderer::upscaling::maximumSpatialRenderScale) {
        throw std::runtime_error("--render-scale requires a finite value in [0.5, 1.0]");
    }
    return value;
}

float parseUnitCoordinate(std::string_view text) {
    float value = 0.0F;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value) ||
        value < 0.0F || value > 1.0F) {
        throw std::runtime_error("--post-position requires two finite coordinates in [0, 1]");
    }
    return value;
}

std::uint32_t parseExtent(std::string_view text) {
    std::uint32_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0U || value > 16384U) {
        throw std::runtime_error("--resolution requires dimensions in [1, 16384]");
    }
    return value;
}

std::uint32_t parsePathBounceCount(std::string_view text) {
    std::uint32_t value = 0U;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0U ||
        value > renderer::maximumPathBounceCount) {
        throw std::runtime_error("--path-max-bounces requires an integer in [1, 4]");
    }
    return value;
}

std::uint32_t parsePathInteractiveSampleCount(std::string_view text) {
    std::uint32_t value = 0U;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0U ||
        value > renderer::maximumPathInteractiveSampleCount) {
        throw std::runtime_error("--path-interactive-samples requires an integer in [1, 8]");
    }
    return value;
}

renderer::post::TestPattern parsePostTestPattern(std::string_view value) {
    using renderer::post::TestPattern;
    if (value == "off") {
        return TestPattern::none;
    }
    if (value == "black") {
        return TestPattern::black;
    }
    if (value == "constant") {
        return TestPattern::constant;
    }
    if (value == "impulse") {
        return TestPattern::impulse;
    }
    if (value == "small-square") {
        return TestPattern::square;
    }
    if (value == "horizontal-line") {
        return TestPattern::horizontalLine;
    }
    if (value == "vertical-line") {
        return TestPattern::verticalLine;
    }
    if (value == "multiple") {
        return TestPattern::multipleSources;
    }
    if (value == "checkerboard") {
        return TestPattern::checkerboard;
    }
    if (value == "moving-square") {
        return TestPattern::movingSquare;
    }
    throw std::runtime_error(
        "--post-test requires off, black, constant, impulse, small-square, horizontal-line, "
        "vertical-line, multiple, checkerboard, or moving-square");
}

float parseEnvironmentRotation(std::string_view text) {
    float value = 0.0F;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || !std::isfinite(value)) {
        throw std::runtime_error("--environment-rotation requires a finite angle in degrees");
    }
    return std::remainder(value, 360.0F);
}

std::uint64_t parseFrameCount(std::string_view text) {
    std::uint64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0) {
        throw std::runtime_error("--frames requires a positive integer");
    }
    return value;
}

renderer::RenderLayerMask parseVisibilityMask(std::string_view text) {
    int base = 10;
    if (text.starts_with("0x") || text.starts_with("0X")) {
        text.remove_prefix(2);
        base = 16;
    }
    renderer::RenderLayerMask value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value, base);
    if (text.empty() || error != std::errc{} || end != text.data() + text.size()) {
        throw std::runtime_error(
            "--visibility-mask requires an unsigned decimal or 0x-prefixed 64-bit mask");
    }
    return value;
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help") {
            std::cout << "Usage: genesis_render [--frames N] "
                         "[--validation off|available|required] "
                         "[--scene PATH] [--scene-animation reject|static-pose] "
                         "[--environment PATH] [--environment-rotation DEGREES] "
                         "[--exposure-ev STOPS] [--capture PATH] "
                         "[--timing-output PATH] "
                         "[--capture-frame N] [--visibility-mask MASK] "
                         "[--aov-export DIRECTORY] [--aov-frame N] "
                         "[--taa on|off] [--bloom on|off] [--bloom-intensity VALUE] "
                         "[--bloom-radius VALUE] "
                         "[--film-grain 0.0..0.1] "
                         "[--gpu-policy nvidia-rtx|portable] "
                         "[--shadows on|off] [--ray-tracing off|auto|required] "
                         "[--render-mode rasterized|path-traced] "
                         "[--upscaler native|spatial|temporal|dlss] "
                         "[--upscaler-quality quality|balanced|performance] "
                         "[--render-scale 0.5..1.0] "
                         "[--path-environment-sampling importance|uniform] "
                         "[--path-emissive-sampling on|off] "
                         "[--path-max-bounces 1|2|3|4] "
                         "[--path-interactive-samples 1..8] "
                         "[--path-denoiser nrd|ray-reconstruction] "
                         "[--path-debug final|direct|indirect|albedo|roughness|"
                         "geometric-normal|shading-normal|motion|depth|hit-distance|"
                         "lighting-residual|history-validity|raw-radiance|"
                         "reconstruction-weight|temporal-radiance|variance] "
                         "[--rt-transform-script none|oscillate] "
                         "[--occlusion-culling on|off] "
                         "[--demo pbr-grid|emissive-sphere|directional-shadow|hdri-backdrop|"
                         "alpha-shadow] "
                         "[--debug-view final|motion|history-validity|bloom|hdr|reactive-mask] "
                         "[--post-test PATTERN] [--post-value LINEAR_HDR] "
                         "[--post-position X Y] "
                         "[--resolution WIDTH HEIGHT] "
                         "[--camera-script none|translate|teleport|rotate|approach] "
                         "[--camera-start front|side|interior] "
                         "[--camera-input on|off] "
                         "[--resize-script [standard|odd|stress]]\n";
            std::exit(0);
        }
        if (argument == "--frames" && index + 1 < argc) {
            options.frameLimit = parseFrameCount(argv[++index]);
            continue;
        }
        if (argument == "--validation" && index + 1 < argc) {
            const std::string_view mode(argv[++index]);
            if (mode == "off") {
                options.validation = renderer::vulkan::ValidationMode::off;
            } else if (mode == "available") {
                options.validation = renderer::vulkan::ValidationMode::available;
            } else if (mode == "required") {
                options.validation = renderer::vulkan::ValidationMode::required;
            } else {
                throw std::runtime_error("Unknown validation mode: " + std::string(mode));
            }
            continue;
        }
        if (argument == "--gpu-policy" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "nvidia-rtx") {
                options.hardwarePolicy = renderer::vulkan::HardwarePolicy::nvidiaRtx;
            } else if (value == "portable") {
                options.hardwarePolicy = renderer::vulkan::HardwarePolicy::portable;
            } else {
                throw std::runtime_error("--gpu-policy requires nvidia-rtx or portable");
            }
            continue;
        }
        if (argument == "--environment" && index + 1 < argc) {
            options.environmentPath = argv[++index];
            if (options.environmentPath.empty()) {
                throw std::runtime_error("--environment requires a non-empty path");
            }
            continue;
        }
        if (argument == "--scene" && index + 1 < argc) {
            options.scenePath = argv[++index];
            if (options.scenePath.empty()) {
                throw std::runtime_error("--scene requires a non-empty path");
            }
            continue;
        }
        if (argument == "--scene-animation" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "reject") {
                options.sceneAnimationStaticPose = false;
            } else if (value == "static-pose") {
                options.sceneAnimationStaticPose = true;
            } else {
                throw std::runtime_error("--scene-animation requires reject or static-pose");
            }
            continue;
        }
        if (argument == "--exposure-ev" && index + 1 < argc) {
            options.exposureCompensationEv = parseExposure(argv[++index]);
            continue;
        }
        if (argument == "--environment-rotation" && index + 1 < argc) {
            options.environmentRotationDegrees = parseEnvironmentRotation(argv[++index]);
            continue;
        }
        if (argument == "--capture" && index + 1 < argc) {
            options.capturePath = argv[++index];
            if (options.capturePath.empty()) {
                throw std::runtime_error("--capture requires a non-empty PNG path");
            }
            continue;
        }
        if (argument == "--timing-output" && index + 1 < argc) {
            options.timingOutputPath = argv[++index];
            if (options.timingOutputPath.empty()) {
                throw std::runtime_error("--timing-output requires a non-empty path");
            }
            continue;
        }
        if (argument == "--capture-frame" && index + 1 < argc) {
            options.captureFrame = parseFrameCount(argv[++index]);
            options.captureFrameSpecified = true;
            continue;
        }
        if (argument == "--aov-export" && index + 1 < argc) {
            options.aovExportDirectory = argv[++index];
            if (options.aovExportDirectory.empty()) {
                throw std::runtime_error("--aov-export requires a non-empty directory");
            }
            continue;
        }
        if (argument == "--aov-frame" && index + 1 < argc) {
            options.aovExportFrame = parseFrameCount(argv[++index]);
            options.aovExportFrameSpecified = true;
            continue;
        }
        if (argument == "--visibility-mask" && index + 1 < argc) {
            options.visibilityMask = parseVisibilityMask(argv[++index]);
            continue;
        }
        if (argument == "--taa" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "on") {
                options.taaEnabled = true;
            } else if (value == "off") {
                options.taaEnabled = false;
            } else {
                throw std::runtime_error("--taa requires on or off");
            }
            continue;
        }
        if (argument == "--bloom" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "on") {
                options.bloomEnabled = true;
            } else if (value == "off") {
                options.bloomEnabled = false;
            } else {
                throw std::runtime_error("--bloom requires on or off");
            }
            continue;
        }
        if (argument == "--bloom-intensity" && index + 1 < argc) {
            options.bloomIntensity = parseBloomIntensity(argv[++index]);
            continue;
        }
        if (argument == "--bloom-radius" && index + 1 < argc) {
            options.bloomRadius = parseBloomRadius(argv[++index]);
            continue;
        }
        if (argument == "--film-grain" && index + 1 < argc) {
            options.filmGrainIntensity = parseFilmGrainIntensity(argv[++index]);
            continue;
        }
        if (argument == "--shadows" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "on") {
                options.shadowsEnabled = true;
            } else if (value == "off") {
                options.shadowsEnabled = false;
            } else {
                throw std::runtime_error("--shadows requires on or off");
            }
            continue;
        }
        if (argument == "--ray-tracing" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "off") {
                options.rayTracingRequest = renderer::ray_tracing::Request::disabled;
            } else if (value == "auto") {
                options.rayTracingRequest = renderer::ray_tracing::Request::automatic;
            } else if (value == "required") {
                options.rayTracingRequest = renderer::ray_tracing::Request::required;
            } else {
                throw std::runtime_error("--ray-tracing requires off, auto, or required");
            }
            continue;
        }
        if (argument == "--render-mode" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "rasterized") {
                options.renderMode = renderer::RenderMode::rasterized;
            } else if (value == "path-traced") {
                options.renderMode = renderer::RenderMode::pathTraced;
            } else {
                throw std::runtime_error("--render-mode requires rasterized or path-traced");
            }
            continue;
        }
        if (argument == "--upscaler" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            const std::optional<renderer::upscaling::Backend> backend =
                renderer::upscaling::parseBackend(value);
            if (!backend) {
                throw std::runtime_error(
                    "--upscaler requires native, spatial, temporal, or dlss");
            }
            options.upscalingBackend = *backend;
            continue;
        }
        if (argument == "--render-scale" && index + 1 < argc) {
            options.upscalingRenderScale = parseRenderScale(argv[++index]);
            options.upscalingRenderScaleSpecified = true;
            continue;
        }
        if (argument == "--upscaler-quality" && index + 1 < argc) {
            const std::optional<renderer::upscaling::QualityMode> qualityMode =
                renderer::upscaling::parseQualityMode(argv[++index]);
            if (!qualityMode || *qualityMode == renderer::upscaling::QualityMode::native ||
                *qualityMode == renderer::upscaling::QualityMode::custom) {
                throw std::runtime_error(
                    "--upscaler-quality requires quality, balanced, or performance");
            }
            options.upscalingQualityMode = *qualityMode;
            options.upscalingQualityModeSpecified = true;
            continue;
        }
        if (argument == "--path-debug" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "final") {
                options.pathDebugView = renderer::PathDebugView::final;
            } else if (value == "direct") {
                options.pathDebugView = renderer::PathDebugView::direct;
            } else if (value == "indirect") {
                options.pathDebugView = renderer::PathDebugView::indirect;
            } else if (value == "albedo") {
                options.pathDebugView = renderer::PathDebugView::albedo;
            } else if (value == "roughness") {
                options.pathDebugView = renderer::PathDebugView::roughness;
            } else if (value == "geometric-normal") {
                options.pathDebugView = renderer::PathDebugView::geometricNormal;
            } else if (value == "shading-normal") {
                options.pathDebugView = renderer::PathDebugView::shadingNormal;
            } else if (value == "motion") {
                options.pathDebugView = renderer::PathDebugView::motion;
            } else if (value == "depth") {
                options.pathDebugView = renderer::PathDebugView::depth;
            } else if (value == "hit-distance") {
                options.pathDebugView = renderer::PathDebugView::hitDistance;
            } else if (value == "lighting-residual") {
                options.pathDebugView = renderer::PathDebugView::lightingResidual;
            } else if (value == "history-validity") {
                options.pathDebugView = renderer::PathDebugView::historyValidity;
            } else if (value == "raw-radiance") {
                options.pathDebugView = renderer::PathDebugView::rawRadiance;
            } else if (value == "reconstruction-weight") {
                options.pathDebugView = renderer::PathDebugView::reconstructionWeight;
            } else if (value == "temporal-radiance") {
                options.pathDebugView = renderer::PathDebugView::temporalRadiance;
            } else if (value == "variance") {
                options.pathDebugView = renderer::PathDebugView::variance;
            } else {
                throw std::runtime_error(
                    "--path-debug requires final, direct, indirect, albedo, roughness, "
                    "geometric-normal, shading-normal, motion, depth, hit-distance, or "
                    "lighting-residual, history-validity, raw-radiance, or "
                    "reconstruction-weight, temporal-radiance, or variance");
            }
            continue;
        }
        if (argument == "--path-environment-sampling" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "importance") {
                options.pathEnvironmentImportanceSampling = true;
            } else if (value == "uniform") {
                options.pathEnvironmentImportanceSampling = false;
            } else {
                throw std::runtime_error(
                    "--path-environment-sampling requires importance or uniform");
            }
            continue;
        }
        if (argument == "--path-emissive-sampling" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "on") {
                options.pathEmissiveTriangleSampling = true;
            } else if (value == "off") {
                options.pathEmissiveTriangleSampling = false;
            } else {
                throw std::runtime_error("--path-emissive-sampling requires on or off");
            }
            continue;
        }
        if (argument == "--path-max-bounces" && index + 1 < argc) {
            options.pathMaxBounces = parsePathBounceCount(argv[++index]);
            continue;
        }
        if (argument == "--path-interactive-samples" && index + 1 < argc) {
            options.pathInteractiveSamples = parsePathInteractiveSampleCount(argv[++index]);
            continue;
        }
        if (argument == "--path-denoiser" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "nrd") {
                options.pathDenoiser = renderer::PathDenoiser::nrd;
            } else if (value == "ray-reconstruction") {
                options.pathDenoiser = renderer::PathDenoiser::rayReconstruction;
            } else {
                throw std::runtime_error(
                    "--path-denoiser requires nrd or ray-reconstruction");
            }
            continue;
        }
        if (argument == "--rt-transform-script" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "none") {
                options.rayTracingTransformScript = renderer::RayTracingTransformScript::none;
            } else if (value == "oscillate") {
                options.rayTracingTransformScript = renderer::RayTracingTransformScript::oscillate;
            } else {
                throw std::runtime_error("--rt-transform-script requires none or oscillate");
            }
            continue;
        }
        if (argument == "--occlusion-culling" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "on") {
                options.occlusionCullingEnabled = true;
            } else if (value == "off") {
                options.occlusionCullingEnabled = false;
            } else {
                throw std::runtime_error("--occlusion-culling requires on or off");
            }
            continue;
        }
        if (argument == "--camera-input" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "on") {
                options.interactiveCameraInput = true;
            } else if (value == "off") {
                options.interactiveCameraInput = false;
            } else {
                throw std::runtime_error("--camera-input requires on or off");
            }
            continue;
        }
        if (argument == "--camera-start" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "front") {
                options.cameraStartView = renderer::CameraStartView::front;
            } else if (value == "side") {
                options.cameraStartView = renderer::CameraStartView::side;
            } else if (value == "interior") {
                options.cameraStartView = renderer::CameraStartView::interior;
            } else {
                throw std::runtime_error("--camera-start requires front, side, or interior");
            }
            continue;
        }
        if (argument == "--demo" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "pbr-grid") {
                options.emissiveSphereDemo = false;
                options.directionalShadowDemo = false;
                options.hdriBackdropDemo = false;
                options.alphaShadowDemo = false;
            } else if (value == "emissive-sphere") {
                options.emissiveSphereDemo = true;
                options.directionalShadowDemo = false;
                options.hdriBackdropDemo = false;
                options.alphaShadowDemo = false;
            } else if (value == "directional-shadow") {
                options.emissiveSphereDemo = false;
                options.directionalShadowDemo = true;
                options.hdriBackdropDemo = false;
                options.alphaShadowDemo = false;
            } else if (value == "hdri-backdrop") {
                options.emissiveSphereDemo = false;
                options.directionalShadowDemo = false;
                options.hdriBackdropDemo = true;
                options.alphaShadowDemo = false;
            } else if (value == "alpha-shadow") {
                options.emissiveSphereDemo = false;
                options.directionalShadowDemo = false;
                options.hdriBackdropDemo = false;
                options.alphaShadowDemo = true;
            } else {
                throw std::runtime_error(
                    "--demo requires pbr-grid, emissive-sphere, directional-shadow, "
                    "hdri-backdrop, or alpha-shadow");
            }
            continue;
        }
        if (argument == "--post-test" && index + 1 < argc) {
            options.postTestPattern = parsePostTestPattern(argv[++index]);
            continue;
        }
        if (argument == "--post-value" && index + 1 < argc) {
            options.postTestSourceValue = parsePostSourceValue(argv[++index]);
            continue;
        }
        if (argument == "--post-position" && index + 2 < argc) {
            options.postTestPosition.x = parseUnitCoordinate(argv[++index]);
            options.postTestPosition.y = parseUnitCoordinate(argv[++index]);
            continue;
        }
        if (argument == "--resolution" && index + 2 < argc) {
            options.width = parseExtent(argv[++index]);
            options.height = parseExtent(argv[++index]);
            continue;
        }
        if (argument == "--debug-view" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "final") {
                options.temporalDebugView = renderer::TemporalDebugView::none;
            } else if (value == "motion") {
                options.temporalDebugView = renderer::TemporalDebugView::motion;
            } else if (value == "history-validity") {
                options.temporalDebugView = renderer::TemporalDebugView::historyValidity;
            } else if (value == "bloom") {
                options.temporalDebugView = renderer::TemporalDebugView::bloom;
            } else if (value == "hdr") {
                options.temporalDebugView = renderer::TemporalDebugView::hdr;
            } else if (value == "reactive-mask") {
                options.temporalDebugView = renderer::TemporalDebugView::reactiveMask;
            } else {
                throw std::runtime_error(
                    "--debug-view requires final, motion, history-validity, bloom, hdr, or "
                    "reactive-mask");
            }
            continue;
        }
        if (argument == "--camera-script" && index + 1 < argc) {
            const std::string_view value(argv[++index]);
            if (value == "none") {
                options.temporalCameraScript = renderer::TemporalCameraScript::none;
            } else if (value == "translate") {
                options.temporalCameraScript = renderer::TemporalCameraScript::translate;
            } else if (value == "teleport") {
                options.temporalCameraScript = renderer::TemporalCameraScript::teleport;
            } else if (value == "rotate") {
                options.temporalCameraScript = renderer::TemporalCameraScript::rotate;
            } else if (value == "approach") {
                options.temporalCameraScript = renderer::TemporalCameraScript::approach;
            } else {
                throw std::runtime_error(
                    "--camera-script requires none, translate, teleport, rotate, or approach");
            }
            continue;
        }
        if (argument == "--resize-script") {
            options.resizeScript = ResizeScript::standard;
            if (index + 1 < argc && std::string_view(argv[index + 1]) == "standard") {
                ++index;
            } else if (index + 1 < argc && std::string_view(argv[index + 1]) == "odd") {
                ++index;
                options.resizeScript = ResizeScript::odd;
            } else if (index + 1 < argc && std::string_view(argv[index + 1]) == "stress") {
                ++index;
                options.resizeScript = ResizeScript::stress;
            }
            continue;
        }
        throw std::runtime_error("Unknown or incomplete argument: " + std::string(argument));
    }
    if (options.captureFrameSpecified && options.capturePath.empty()) {
        throw std::runtime_error("--capture-frame requires --capture");
    }
    if (options.aovExportFrameSpecified && options.aovExportDirectory.empty()) {
        throw std::runtime_error("--aov-frame requires --aov-export");
    }
    if (!options.capturePath.empty() && options.frameLimit.has_value() &&
        *options.frameLimit < options.captureFrame) {
        throw std::runtime_error("--frames must reach the requested capture frame");
    }
    if (!options.aovExportDirectory.empty() && options.frameLimit.has_value() &&
        *options.frameLimit < options.aovExportFrame) {
        throw std::runtime_error("--frames must reach the requested AOV export frame");
    }
    if (!options.taaEnabled &&
        options.temporalDebugView == renderer::TemporalDebugView::historyValidity) {
        throw std::runtime_error("history-validity debug view requires --taa on");
    }
    if (!options.bloomEnabled && options.temporalDebugView == renderer::TemporalDebugView::bloom) {
        throw std::runtime_error("bloom debug view requires --bloom on");
    }
    if (options.temporalDebugView == renderer::TemporalDebugView::reactiveMask &&
        options.upscalingBackend != renderer::upscaling::Backend::temporal) {
        throw std::runtime_error("reactive-mask debug view requires --upscaler temporal");
    }
    if (options.renderMode == renderer::RenderMode::pathTraced) {
        if (options.rayTracingRequest == renderer::ray_tracing::Request::disabled) {
            throw std::runtime_error("--render-mode path-traced cannot use --ray-tracing off");
        }
        if (options.postTestPattern != renderer::post::TestPattern::none ||
            options.temporalDebugView != renderer::TemporalDebugView::none) {
            throw std::runtime_error(
                "--render-mode path-traced requires --post-test off and --debug-view final");
        }
    } else if (options.pathDebugView != renderer::PathDebugView::final) {
        throw std::runtime_error("--path-debug requires --render-mode path-traced");
    }
    if (options.pathDenoiser == renderer::PathDenoiser::rayReconstruction) {
        if (options.renderMode != renderer::RenderMode::pathTraced) {
            throw std::runtime_error(
                "--path-denoiser ray-reconstruction requires --render-mode path-traced");
        }
        if (options.upscalingBackend != renderer::upscaling::Backend::dlss) {
            throw std::runtime_error(
                "--path-denoiser ray-reconstruction requires --upscaler dlss");
        }
        if (options.pathDebugView != renderer::PathDebugView::final) {
            throw std::runtime_error(
                "--path-denoiser ray-reconstruction requires --path-debug final");
        }
        if (!options.aovExportDirectory.empty()) {
            throw std::runtime_error(
                "--path-denoiser ray-reconstruction cannot currently export NRD AOVs");
        }
    }
    if (!options.aovExportDirectory.empty() &&
        options.renderMode != renderer::RenderMode::pathTraced) {
        throw std::runtime_error("--aov-export requires --render-mode path-traced");
    }
    if (options.upscalingQualityModeSpecified && options.upscalingRenderScaleSpecified) {
        throw std::runtime_error("--upscaler-quality and --render-scale are mutually exclusive");
    }
    if (options.upscalingBackend == renderer::upscaling::Backend::dlss &&
        options.upscalingRenderScaleSpecified) {
        throw std::runtime_error(
            "--upscaler dlss requires a named --upscaler-quality mode");
    }
    if (options.upscalingBackend == renderer::upscaling::Backend::spatial ||
        options.upscalingBackend == renderer::upscaling::Backend::temporal ||
        options.upscalingBackend == renderer::upscaling::Backend::dlss) {
        if (options.upscalingRenderScaleSpecified) {
            options.upscalingQualityMode = renderer::upscaling::QualityMode::custom;
        } else {
            if (!options.upscalingQualityModeSpecified) {
                options.upscalingQualityMode = renderer::upscaling::QualityMode::quality;
            }
            options.upscalingRenderScale =
                renderer::upscaling::qualityModeRenderScale(options.upscalingQualityMode);
        }
    } else {
        if (options.upscalingQualityModeSpecified) {
            throw std::runtime_error(
                "--upscaler-quality requires --upscaler spatial, temporal, or dlss");
        }
        if (options.upscalingRenderScaleSpecified &&
            std::abs(options.upscalingRenderScale - 1.0F) > 1.0e-6F) {
            throw std::runtime_error("--upscaler native requires --render-scale 1.0");
        }
        options.upscalingQualityMode = renderer::upscaling::QualityMode::native;
    }
    if ((options.emissiveSphereDemo || options.directionalShadowDemo || options.hdriBackdropDemo ||
         options.alphaShadowDemo) &&
        !options.scenePath.empty()) {
        throw std::runtime_error("Procedural --demo modes cannot be combined with --scene");
    }
    if (options.sceneAnimationStaticPose && options.scenePath.empty()) {
        throw std::runtime_error("--scene-animation static-pose requires --scene");
    }
    return options;
}

int run(const Options& options) {
#if RENDERER_ENABLE_STREAMLINE
    std::unique_ptr<renderer::nvidia::StreamlineRuntime> streamline;
    if (options.upscalingBackend == renderer::upscaling::Backend::dlss) {
        streamline = std::make_unique<renderer::nvidia::StreamlineRuntime>(
            std::filesystem::path{});
    }
#else
    if (options.upscalingBackend == renderer::upscaling::Backend::dlss) {
        throw std::runtime_error("This Genesis build does not include NVIDIA Streamline");
    }
#endif
    renderer::platform::Window window(options.width, options.height, "Genesis Render — Vulkan 1.3");
    renderer::vulkan::Instance instance(
        window, options.validation,
#if RENDERER_ENABLE_STREAMLINE
        streamline ? streamline->vulkanGetInstanceProcAddr() : nullptr
#else
        nullptr
#endif
    );
    std::uint64_t frameCount = 0;
    {
        renderer::vulkan::Device device(instance, options.hardwarePolicy);
#if RENDERER_ENABLE_STREAMLINE
        StreamlineShutdownGuard streamlineShutdown(streamline.get());
        if (streamline &&
            !streamline->featureSupported(
                renderer::nvidia::StreamlineFeature::dlssSuperResolution,
                device.physicalHandle())) {
            throw std::runtime_error(
                "DLSS Super Resolution is unavailable on the selected GPU or driver");
        }
        if (streamline &&
            options.pathDenoiser == renderer::PathDenoiser::rayReconstruction &&
            !streamline->featureSupported(
                renderer::nvidia::StreamlineFeature::dlssRayReconstruction,
                device.physicalHandle())) {
            throw std::runtime_error(
                "DLSS Ray Reconstruction is unavailable on the selected GPU or driver");
        }
#endif
        {
            std::shared_ptr<renderer::output::AovFrameSink> aovFrameSink;
            if (!options.aovExportDirectory.empty()) {
                aovFrameSink = std::make_shared<renderer::output::PfmAovDirectorySink>(
                    options.aovExportDirectory);
            }
            renderer::Renderer renderer(
                window, instance, device,
                {.scenePath = options.scenePath,
                 .sceneAnimationStaticPose = options.sceneAnimationStaticPose,
                 .environmentPath = options.environmentPath,
                 .exposureCompensationEv = options.exposureCompensationEv,
                 .environmentRotationRadians =
                     options.environmentRotationDegrees * std::numbers::pi_v<float> / 180.0F,
                 .capturePath = options.capturePath,
                 .timingOutputPath = options.timingOutputPath,
                 .captureFrame = options.captureFrame,
                 .aovFrameSink = std::move(aovFrameSink),
                 .aovExportFrame = options.aovExportFrame,
                 .view = {.visibilityMask = options.visibilityMask},
                 .taaEnabled = options.taaEnabled,
                 .bloomEnabled = options.bloomEnabled,
                 .bloomIntensity = options.bloomIntensity,
                 .bloomRadius = options.bloomRadius,
                 .filmGrainIntensity = options.filmGrainIntensity,
                 .emissiveSphereDemo = options.emissiveSphereDemo,
                 .directionalShadowDemo = options.directionalShadowDemo,
                 .hdriBackdropDemo = options.hdriBackdropDemo,
                 .alphaShadowDemo = options.alphaShadowDemo,
                 .shadowsEnabled = options.shadowsEnabled,
                 .rayTracingRequest = options.rayTracingRequest,
                 .rayTracingTransformScript = options.rayTracingTransformScript,
                 .renderMode = options.renderMode,
                 .upscalingBackend = options.upscalingBackend,
                 .upscalingQualityMode = options.upscalingQualityMode,
                 .upscalingRenderScale = options.upscalingRenderScale,
                 .streamlineRuntime =
#if RENDERER_ENABLE_STREAMLINE
                     streamline.get(),
#else
                     nullptr,
#endif
                 .pathEnvironmentImportanceSampling = options.pathEnvironmentImportanceSampling,
                 .pathEmissiveTriangleSampling = options.pathEmissiveTriangleSampling,
                 .pathMaxBounces = options.pathMaxBounces,
                 .pathInteractiveSamples = options.pathInteractiveSamples,
                 .pathDenoiser = options.pathDenoiser,
                 .pathDebugView = options.pathDebugView,
                 .occlusionCullingEnabled = options.occlusionCullingEnabled,
                 .postTestPattern = options.postTestPattern,
                 .postTestSourceValue = options.postTestSourceValue,
                 .postTestPosition = options.postTestPosition,
                 .temporalDebugView = options.temporalDebugView,
                 .temporalCameraScript = options.temporalCameraScript,
                 .cameraStartView = options.cameraStartView,
                 .interactiveCameraInput = options.interactiveCameraInput});
            while (!window.shouldClose() &&
                   (!options.frameLimit.has_value() || frameCount < *options.frameLimit)) {
                window.pollEvents();
                if (window.shouldClose()) {
                    break;
                }
                if (options.resizeScript == ResizeScript::standard && frameCount == 2U) {
                    window.resize(960, 540);
                } else if (options.resizeScript == ResizeScript::standard && frameCount == 6U) {
                    window.resize(options.width, options.height);
                } else if (options.resizeScript == ResizeScript::odd && frameCount == 2U) {
                    window.resize(1279, 719);
                } else if (options.resizeScript == ResizeScript::odd && frameCount == 6U) {
                    window.resize(options.width, options.height);
                } else if (options.resizeScript == ResizeScript::stress && frameCount >= 2U &&
                           frameCount <= 16U && (frameCount & 1U) == 0U) {
                    constexpr std::array sizes{std::pair{1280U, 720U},  std::pair{1279U, 719U},
                                               std::pair{2560U, 1440U}, std::pair{1920U, 1080U},
                                               std::pair{960U, 541U},   std::pair{1601U, 901U},
                                               std::pair{1279U, 719U},  std::pair{1920U, 1080U}};
                    const auto [width, height] = sizes[((frameCount - 2U) / 2U) % sizes.size()];
                    window.resize(width, height);
                }
                renderer.drawFrame();
                ++frameCount;
            }
            renderer.waitIdle();
        }
    }
    if (instance.validationErrorCount() != 0) {
        renderer::core::log(renderer::core::LogLevel::error, "Validation reported ",
                            instance.validationErrorCount(), " error message(s)");
        return 2;
    }
    renderer::core::log(renderer::core::LogLevel::info, "Clean shutdown after ", frameCount,
                        " frames");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parseOptions(argc, argv));
    } catch (const std::exception& error) {
        renderer::core::log(renderer::core::LogLevel::error, error.what());
        return 1;
    }
}
