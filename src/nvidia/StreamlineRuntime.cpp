#include "renderer/nvidia/StreamlineRuntime.hpp"

#include "renderer/core/Log.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <sl.h>
#include <sl_dlss.h>
#include <sl_dlss_d.h>
#include <sl_security.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>

namespace renderer::nvidia {
namespace {

constexpr std::array requestedFeatures{sl::kFeatureDLSS, sl::kFeatureDLSS_RR};
constexpr char genesisProjectId[] = "20c89835-a2b3-4388-a1e8-5446e03ff2f7";

std::string resultName(sl::Result result) {
    switch (result) {
    case sl::Result::eOk:
        return "ok";
    case sl::Result::eErrorDriverOutOfDate:
        return "driver out of date";
    case sl::Result::eErrorOSOutOfDate:
        return "OS out of date";
    case sl::Result::eErrorNoSupportedAdapterFound:
        return "no supported adapter";
    case sl::Result::eErrorAdapterNotSupported:
        return "adapter not supported";
    case sl::Result::eErrorNoPlugins:
        return "no plugins";
    case sl::Result::eErrorNGXFailed:
        return "NGX initialization failed";
    case sl::Result::eErrorMissingProxy:
        return "missing API proxy";
    case sl::Result::eErrorInvalidIntegration:
        return "invalid integration";
    case sl::Result::eErrorFeatureMissing:
        return "feature missing";
    case sl::Result::eErrorFeatureNotSupported:
        return "feature not supported";
    case sl::Result::eErrorFeatureFailedToLoad:
        return "feature failed to load";
    case sl::Result::eErrorFeatureMissingDependency:
        return "feature dependency missing";
    default:
        return "result " + std::to_string(static_cast<int>(result));
    }
}

std::string featureName(StreamlineFeature feature) {
    switch (feature) {
    case StreamlineFeature::dlssSuperResolution:
        return "DLSS Super Resolution";
    case StreamlineFeature::dlssRayReconstruction:
        return "DLSS Ray Reconstruction";
    }
    return "unknown Streamline feature";
}

sl::Feature nativeFeature(StreamlineFeature feature) {
    switch (feature) {
    case StreamlineFeature::dlssSuperResolution:
        return sl::kFeatureDLSS;
    case StreamlineFeature::dlssRayReconstruction:
        return sl::kFeatureDLSS_RR;
    }
    throw std::invalid_argument("Unknown Streamline feature");
}

sl::DLSSMode nativeQuality(DlssQuality quality) {
    switch (quality) {
    case DlssQuality::quality:
        return sl::DLSSMode::eMaxQuality;
    case DlssQuality::balanced:
        return sl::DLSSMode::eBalanced;
    case DlssQuality::performance:
        return sl::DLSSMode::eMaxPerformance;
    }
    throw std::invalid_argument("Unknown DLSS quality mode");
}

template <typename Handle> void* nativeHandle(Handle handle) {
    if constexpr (std::is_pointer_v<Handle>) {
        return reinterpret_cast<void*>(handle);
    } else {
        return reinterpret_cast<void*>(static_cast<std::uintptr_t>(handle));
    }
}

sl::Resource resource(const StreamlineImage& image) {
    if (image.image == VK_NULL_HANDLE || image.view == VK_NULL_HANDLE ||
        image.layout == VK_IMAGE_LAYOUT_UNDEFINED || image.extent.width == 0U ||
        image.extent.height == 0U || image.format == VK_FORMAT_UNDEFINED || image.usage == 0U) {
        throw std::invalid_argument("Streamline image metadata is incomplete");
    }
    sl::Resource result(sl::ResourceType::eTex2d, nativeHandle(image.image),
                        nativeHandle(image.memory), nativeHandle(image.view),
                        static_cast<std::uint32_t>(image.layout));
    result.width = image.extent.width;
    result.height = image.extent.height;
    result.nativeFormat = static_cast<std::uint32_t>(image.format);
    result.mipLevels = image.mipLevels;
    result.arrayLayers = image.arrayLayers;
    result.flags = image.flags;
    result.usage = image.usage;
    return result;
}

sl::float4x4 matrix(const std::array<float, 16>& values) {
    sl::float4x4 result{};
    for (std::uint32_t row = 0U; row < 4U; ++row) {
        result.setRow(row, {values[row * 4U], values[row * 4U + 1U],
                            values[row * 4U + 2U], values[row * 4U + 3U]});
    }
    return result;
}

template <typename FunctionPointer>
FunctionPointer loadFunction(HMODULE module, const char* name) {
    const FARPROC address = GetProcAddress(module, name);
    if (address == nullptr) {
        throw std::runtime_error(std::string("Streamline interposer is missing export ") + name);
    }
    return reinterpret_cast<FunctionPointer>(address);
}

void streamlineLog(sl::LogType type, const char* message) {
    const core::LogLevel level = type == sl::LogType::eError   ? core::LogLevel::error
                                 : type == sl::LogType::eWarn ? core::LogLevel::warning
                                                              : core::LogLevel::info;
    core::log(level, "Streamline: ", message != nullptr ? message : "<no message>");
}

std::filesystem::path executableDirectory() {
    std::vector<wchar_t> path(32768U, L'\0');
    const DWORD length =
        GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0U || length >= path.size()) {
        throw std::runtime_error("Could not resolve the Genesis executable directory");
    }
    return std::filesystem::path(std::wstring(path.data(), length)).parent_path();
}

} // namespace

struct StreamlineRuntime::Functions {
    PFun_slInit* init = nullptr;
    PFun_slShutdown* shutdown = nullptr;
    PFun_slIsFeatureSupported* isFeatureSupported = nullptr;
    PFun_slGetFeatureFunction* getFeatureFunction = nullptr;
    PFun_slGetNewFrameToken* getNewFrameToken = nullptr;
    PFun_slSetConstants* setConstants = nullptr;
    PFun_slSetTagForFrame* setTagForFrame = nullptr;
    PFun_slEvaluateFeature* evaluateFeature = nullptr;
    PFun_slFreeResources* freeResources = nullptr;
    PFun_slDLSSSetOptions* dlssSetOptions = nullptr;
    PFun_slDLSSDSetOptions* dlssdSetOptions = nullptr;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    bool dlssConfigured = false;
    bool rayReconstructionConfigured = false;
    sl::DLSSMode dlssMode = sl::DLSSMode::eOff;
    std::uint32_t outputWidth = 0U;
    std::uint32_t outputHeight = 0U;
};

StreamlineRuntime::StreamlineRuntime(std::filesystem::path runtimeDirectory)
    : runtimeDirectory_(runtimeDirectory.empty()
                            ? executableDirectory()
                            : std::filesystem::absolute(std::move(runtimeDirectory))
                                  .lexically_normal()) {
    const std::filesystem::path interposerPath = runtimeDirectory_ / "sl.interposer.dll";
    if (!std::filesystem::is_regular_file(interposerPath)) {
        throw std::runtime_error("Streamline runtime is missing: " + interposerPath.string());
    }
    if (!sl::security::verifyEmbeddedSignature(interposerPath.c_str())) {
        throw std::runtime_error("Refusing to load an unsigned or invalid Streamline interposer: " +
                                 interposerPath.string());
    }

    const HMODULE module = LoadLibraryExW(
        interposerPath.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (module == nullptr) {
        throw std::runtime_error("Could not securely load Streamline interposer (Windows error " +
                                 std::to_string(GetLastError()) + "): " +
                                 interposerPath.string());
    }
    module_ = module;
    try {
        functions_ = new Functions{
            .init = loadFunction<PFun_slInit*>(module, "slInit"),
            .shutdown = loadFunction<PFun_slShutdown*>(module, "slShutdown"),
            .isFeatureSupported =
                loadFunction<PFun_slIsFeatureSupported*>(module, "slIsFeatureSupported"),
            .getFeatureFunction =
                loadFunction<PFun_slGetFeatureFunction*>(module, "slGetFeatureFunction"),
            .getNewFrameToken =
                loadFunction<PFun_slGetNewFrameToken*>(module, "slGetNewFrameToken"),
            .setConstants = loadFunction<PFun_slSetConstants*>(module, "slSetConstants"),
            .setTagForFrame =
                loadFunction<PFun_slSetTagForFrame*>(module, "slSetTagForFrame"),
            .evaluateFeature =
                loadFunction<PFun_slEvaluateFeature*>(module, "slEvaluateFeature"),
            .freeResources = loadFunction<PFun_slFreeResources*>(module, "slFreeResources"),
            .vkGetInstanceProcAddr =
                loadFunction<PFN_vkGetInstanceProcAddr>(module, "vkGetInstanceProcAddr"),
        };

        const wchar_t* pluginPath = runtimeDirectory_.c_str();
        sl::Preferences preferences{};
        preferences.logLevel = sl::LogLevel::eDefault;
        preferences.pathsToPlugins = &pluginPath;
        preferences.numPathsToPlugins = 1U;
        preferences.logMessageCallback = streamlineLog;
        preferences.flags = sl::PreferenceFlags::eDisableCLStateTracking |
                            sl::PreferenceFlags::eDisableDebugText |
                            sl::PreferenceFlags::eUseFrameBasedResourceTagging;
        preferences.featuresToLoad = requestedFeatures.data();
        preferences.numFeaturesToLoad =
            static_cast<std::uint32_t>(requestedFeatures.size());
        preferences.engine = sl::EngineType::eCustom;
        preferences.engineVersion = "0.1.0";
        preferences.projectId = genesisProjectId;
        preferences.renderAPI = sl::RenderAPI::eVulkan;

        const sl::Result result = functions_->init(preferences, sl::kSDKVersion);
        if (result != sl::Result::eOk) {
            throw std::runtime_error("Streamline initialization failed: " + resultName(result));
        }
        initialized_ = true;
        core::log(core::LogLevel::info, "NVIDIA Streamline 2.12.0 initialized from ",
                  runtimeDirectory_.string(), " (signed production runtime)");
    } catch (...) {
        delete functions_;
        functions_ = nullptr;
        FreeLibrary(module);
        module_ = nullptr;
        throw;
    }
}

StreamlineRuntime::~StreamlineRuntime() {
    shutdown();
    delete functions_;
    functions_ = nullptr;
    if (module_ != nullptr) {
        FreeLibrary(static_cast<HMODULE>(module_));
        module_ = nullptr;
    }
}

void StreamlineRuntime::shutdown() {
    if (!initialized_ || functions_ == nullptr || functions_->shutdown == nullptr) {
        return;
    }
    const sl::Result result = functions_->shutdown();
    if (result != sl::Result::eOk) {
        core::log(core::LogLevel::warning, "Streamline shutdown returned ", resultName(result));
    }
    initialized_ = false;
}

PFN_vkGetInstanceProcAddr StreamlineRuntime::vulkanGetInstanceProcAddr() const {
    if (!initialized_ || functions_ == nullptr || functions_->vkGetInstanceProcAddr == nullptr) {
        throw std::logic_error("Streamline Vulkan proxy requested before initialization");
    }
    return functions_->vkGetInstanceProcAddr;
}

bool StreamlineRuntime::featureSupported(StreamlineFeature feature,
                                         VkPhysicalDevice physicalDevice) const {
    if (!initialized_ || functions_ == nullptr || functions_->isFeatureSupported == nullptr) {
        throw std::logic_error("Streamline feature query requested before initialization");
    }
    sl::AdapterInfo adapter{};
    adapter.vkPhysicalDevice = physicalDevice;
    const sl::Result result =
        functions_->isFeatureSupported(nativeFeature(feature), adapter);
    if (result == sl::Result::eOk) {
        core::log(core::LogLevel::info, featureName(feature), " is supported on the selected GPU");
        return true;
    }
    if (result == sl::Result::eErrorFeatureNotSupported ||
        result == sl::Result::eErrorAdapterNotSupported ||
        result == sl::Result::eErrorNoSupportedAdapterFound ||
        result == sl::Result::eErrorFeatureMissing ||
        result == sl::Result::eErrorFeatureFailedToLoad) {
        core::log(core::LogLevel::warning, featureName(feature), " unavailable: ",
                  resultName(result));
        return false;
    }
    throw std::runtime_error(featureName(feature) + " support query failed: " +
                             resultName(result));
}

void StreamlineRuntime::evaluateDlss(const DlssFrame& frame) {
    if (!initialized_ || functions_ == nullptr) {
        throw std::logic_error("DLSS evaluation requested before Streamline initialization");
    }
    if (frame.commandBuffer == VK_NULL_HANDLE || frame.cameraNear <= 0.0F ||
        frame.cameraFar <= frame.cameraNear || frame.cameraFovRadians <= 0.0F ||
        frame.cameraAspectRatio <= 0.0F || frame.exposureScale <= 0.0F) {
        throw std::invalid_argument("DLSS frame constants are incomplete");
    }

    if (functions_->dlssSetOptions == nullptr) {
        void* function = nullptr;
        const sl::Result importResult = functions_->getFeatureFunction(
            sl::kFeatureDLSS, "slDLSSSetOptions", function);
        if (importResult != sl::Result::eOk || function == nullptr) {
            throw std::runtime_error("Could not import slDLSSSetOptions: " +
                                     resultName(importResult));
        }
        functions_->dlssSetOptions = reinterpret_cast<PFun_slDLSSSetOptions*>(function);
    }

    const sl::DLSSMode mode = nativeQuality(frame.quality);
    if (!functions_->dlssConfigured || functions_->dlssMode != mode ||
        functions_->outputWidth != frame.outputColor.extent.width ||
        functions_->outputHeight != frame.outputColor.extent.height) {
        sl::DLSSOptions options{};
        options.mode = mode;
        options.outputWidth = frame.outputColor.extent.width;
        options.outputHeight = frame.outputColor.extent.height;
        options.preExposure = 1.0F;
        options.exposureScale = 1.0F;
        options.colorBuffersHDR = sl::Boolean::eTrue;
        options.useAutoExposure = sl::Boolean::eFalse;
        options.alphaUpscalingEnabled = sl::Boolean::eFalse;
        const sl::Result result = functions_->dlssSetOptions(sl::ViewportHandle{0U}, options);
        if (result != sl::Result::eOk) {
            throw std::runtime_error("slDLSSSetOptions failed: " + resultName(result));
        }
        functions_->dlssConfigured = true;
        functions_->dlssMode = mode;
        functions_->outputWidth = frame.outputColor.extent.width;
        functions_->outputHeight = frame.outputColor.extent.height;
        core::log(core::LogLevel::info, "DLSS configured at ", frame.inputColor.extent.width, "x",
                  frame.inputColor.extent.height, " -> ", frame.outputColor.extent.width, "x",
                  frame.outputColor.extent.height);
    }

    sl::FrameToken* token = nullptr;
    const sl::Result tokenResult = functions_->getNewFrameToken(token, &frame.frameIndex);
    if (tokenResult != sl::Result::eOk || token == nullptr) {
        throw std::runtime_error("slGetNewFrameToken failed: " + resultName(tokenResult));
    }
    const sl::ViewportHandle viewport{0U};

    sl::Constants constants{};
    constants.cameraViewToClip = matrix(frame.cameraViewToClip);
    constants.clipToCameraView = matrix(frame.clipToCameraView);
    constants.clipToPrevClip = matrix(frame.clipToPrevClip);
    constants.prevClipToClip = matrix(frame.prevClipToClip);
    constants.jitterOffset = {frame.jitterX, frame.jitterY};
    // Genesis stores current-minus-previous UV. DLSS consumes the opposite
    // direction, so the normalization scale also performs the sign flip.
    constants.mvecScale = {-1.0F, -1.0F};
    constants.cameraPinholeOffset = {0.0F, 0.0F};
    constants.cameraPos = {frame.cameraPosition[0], frame.cameraPosition[1],
                           frame.cameraPosition[2]};
    constants.cameraUp = {frame.cameraUp[0], frame.cameraUp[1], frame.cameraUp[2]};
    constants.cameraRight = {frame.cameraRight[0], frame.cameraRight[1],
                              frame.cameraRight[2]};
    constants.cameraFwd = {frame.cameraForward[0], frame.cameraForward[1],
                            frame.cameraForward[2]};
    constants.cameraNear = frame.cameraNear;
    constants.cameraFar = frame.cameraFar;
    constants.cameraFOV = frame.cameraFovRadians;
    constants.cameraAspectRatio = frame.cameraAspectRatio;
    constants.depthInverted = sl::Boolean::eFalse;
    constants.cameraMotionIncluded = sl::Boolean::eTrue;
    constants.motionVectors3D = sl::Boolean::eFalse;
    constants.reset = frame.resetHistory ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.orthographicProjection = sl::Boolean::eFalse;
    constants.motionVectorsDilated = sl::Boolean::eFalse;
    constants.motionVectorsJittered = sl::Boolean::eFalse;
    const sl::Result constantsResult = functions_->setConstants(constants, *token, viewport);
    if (constantsResult != sl::Result::eOk) {
        throw std::runtime_error("slSetConstants(DLSS) failed: " +
                                 resultName(constantsResult));
    }

    sl::Resource inputColor = resource(frame.inputColor);
    sl::Resource outputColor = resource(frame.outputColor);
    sl::Resource depth = resource(frame.depth);
    sl::Resource motionVectors = resource(frame.motionVectors);
    const sl::Extent inputExtent{.top = 0U,
                                 .left = 0U,
                                 .width = frame.inputColor.extent.width,
                                 .height = frame.inputColor.extent.height};
    const sl::Extent outputExtent{.top = 0U,
                                  .left = 0U,
                                  .width = frame.outputColor.extent.width,
                                  .height = frame.outputColor.extent.height};
    const std::array tags{
        sl::ResourceTag{&inputColor, sl::kBufferTypeScalingInputColor,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &inputExtent},
        sl::ResourceTag{&outputColor, sl::kBufferTypeScalingOutputColor,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &outputExtent},
        sl::ResourceTag{&depth, sl::kBufferTypeDepth,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &inputExtent},
        sl::ResourceTag{&motionVectors, sl::kBufferTypeMotionVectors,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &inputExtent},
    };
    auto* commandBuffer = reinterpret_cast<sl::CommandBuffer*>(frame.commandBuffer);
    const sl::Result tagResult = functions_->setTagForFrame(
        *token, viewport, tags.data(), static_cast<std::uint32_t>(tags.size()), commandBuffer);
    if (tagResult != sl::Result::eOk) {
        throw std::runtime_error("slSetTagForFrame(DLSS) failed: " + resultName(tagResult));
    }

    const sl::BaseStructure* inputs[]{&viewport};
    const sl::Result evaluateResult = functions_->evaluateFeature(
        sl::kFeatureDLSS, *token, inputs, 1U, commandBuffer);
    if (evaluateResult != sl::Result::eOk) {
        throw std::runtime_error("slEvaluateFeature(DLSS) failed: " +
                                 resultName(evaluateResult));
    }
}

void StreamlineRuntime::evaluateRayReconstruction(
    const RayReconstructionFrame& reconstructionFrame) {
    const DlssFrame& frame = reconstructionFrame.dlss;
    if (!initialized_ || functions_ == nullptr) {
        throw std::logic_error(
            "DLSS Ray Reconstruction evaluation requested before Streamline initialization");
    }
    if (frame.commandBuffer == VK_NULL_HANDLE || frame.cameraNear <= 0.0F ||
        frame.cameraFar <= frame.cameraNear || frame.cameraFovRadians <= 0.0F ||
        frame.cameraAspectRatio <= 0.0F || frame.exposureScale <= 0.0F) {
        throw std::invalid_argument("DLSS Ray Reconstruction frame constants are incomplete");
    }

    if (functions_->dlssSetOptions == nullptr) {
        void* function = nullptr;
        const sl::Result importResult = functions_->getFeatureFunction(
            sl::kFeatureDLSS, "slDLSSSetOptions", function);
        if (importResult != sl::Result::eOk || function == nullptr) {
            throw std::runtime_error("Could not import slDLSSSetOptions: " +
                                     resultName(importResult));
        }
        functions_->dlssSetOptions = reinterpret_cast<PFun_slDLSSSetOptions*>(function);
    }
    if (functions_->dlssdSetOptions == nullptr) {
        void* function = nullptr;
        const sl::Result importResult = functions_->getFeatureFunction(
            sl::kFeatureDLSS_RR, "slDLSSDSetOptions", function);
        if (importResult != sl::Result::eOk || function == nullptr) {
            throw std::runtime_error("Could not import slDLSSDSetOptions: " +
                                     resultName(importResult));
        }
        functions_->dlssdSetOptions = reinterpret_cast<PFun_slDLSSDSetOptions*>(function);
    }

    const sl::DLSSMode mode = nativeQuality(frame.quality);
    if (!functions_->dlssConfigured || functions_->dlssMode != mode ||
        functions_->outputWidth != frame.outputColor.extent.width ||
        functions_->outputHeight != frame.outputColor.extent.height) {
        sl::DLSSOptions dlssOptions{};
        dlssOptions.mode = mode;
        dlssOptions.outputWidth = frame.outputColor.extent.width;
        dlssOptions.outputHeight = frame.outputColor.extent.height;
        dlssOptions.preExposure = 1.0F;
        dlssOptions.exposureScale = 1.0F;
        dlssOptions.colorBuffersHDR = sl::Boolean::eTrue;
        dlssOptions.useAutoExposure = sl::Boolean::eFalse;
        dlssOptions.alphaUpscalingEnabled = sl::Boolean::eFalse;
        const sl::Result result =
            functions_->dlssSetOptions(sl::ViewportHandle{0U}, dlssOptions);
        if (result != sl::Result::eOk) {
            throw std::runtime_error("slDLSSSetOptions for Ray Reconstruction failed: " +
                                     resultName(result));
        }
        functions_->dlssConfigured = true;
        functions_->dlssMode = mode;
        functions_->outputWidth = frame.outputColor.extent.width;
        functions_->outputHeight = frame.outputColor.extent.height;
    }

    sl::DLSSDOptions reconstructionOptions{};
    reconstructionOptions.mode = mode;
    reconstructionOptions.outputWidth = frame.outputColor.extent.width;
    reconstructionOptions.outputHeight = frame.outputColor.extent.height;
    reconstructionOptions.preExposure = 1.0F;
    reconstructionOptions.exposureScale = 1.0F;
    reconstructionOptions.colorBuffersHDR = sl::Boolean::eTrue;
    reconstructionOptions.normalRoughnessMode = sl::DLSSDNormalRoughnessMode::ePacked;
    reconstructionOptions.worldToCameraView = matrix(reconstructionFrame.worldToCameraView);
    reconstructionOptions.cameraViewToWorld = matrix(reconstructionFrame.cameraViewToWorld);
    reconstructionOptions.alphaUpscalingEnabled = sl::Boolean::eFalse;
    const sl::Result optionsResult =
        functions_->dlssdSetOptions(sl::ViewportHandle{0U}, reconstructionOptions);
    if (optionsResult != sl::Result::eOk) {
        throw std::runtime_error("slDLSSDSetOptions failed: " + resultName(optionsResult));
    }
    if (!functions_->rayReconstructionConfigured) {
        core::log(core::LogLevel::info, "DLSS Ray Reconstruction configured at ",
                  frame.inputColor.extent.width, "x", frame.inputColor.extent.height, " -> ",
                  frame.outputColor.extent.width, "x", frame.outputColor.extent.height);
    }
    functions_->rayReconstructionConfigured = true;

    sl::FrameToken* token = nullptr;
    const sl::Result tokenResult = functions_->getNewFrameToken(token, &frame.frameIndex);
    if (tokenResult != sl::Result::eOk || token == nullptr) {
        throw std::runtime_error("slGetNewFrameToken failed: " + resultName(tokenResult));
    }
    const sl::ViewportHandle viewport{0U};

    sl::Constants constants{};
    constants.cameraViewToClip = matrix(frame.cameraViewToClip);
    constants.clipToCameraView = matrix(frame.clipToCameraView);
    constants.clipToPrevClip = matrix(frame.clipToPrevClip);
    constants.prevClipToClip = matrix(frame.prevClipToClip);
    constants.jitterOffset = {frame.jitterX, frame.jitterY};
    constants.mvecScale = {-1.0F, -1.0F};
    constants.cameraPinholeOffset = {0.0F, 0.0F};
    constants.cameraPos = {frame.cameraPosition[0], frame.cameraPosition[1],
                           frame.cameraPosition[2]};
    constants.cameraUp = {frame.cameraUp[0], frame.cameraUp[1], frame.cameraUp[2]};
    constants.cameraRight = {frame.cameraRight[0], frame.cameraRight[1],
                             frame.cameraRight[2]};
    constants.cameraFwd = {frame.cameraForward[0], frame.cameraForward[1],
                           frame.cameraForward[2]};
    constants.cameraNear = frame.cameraNear;
    constants.cameraFar = frame.cameraFar;
    constants.cameraFOV = frame.cameraFovRadians;
    constants.cameraAspectRatio = frame.cameraAspectRatio;
    constants.depthInverted = sl::Boolean::eFalse;
    constants.cameraMotionIncluded = sl::Boolean::eTrue;
    constants.motionVectors3D = sl::Boolean::eFalse;
    constants.reset = frame.resetHistory ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.orthographicProjection = sl::Boolean::eFalse;
    constants.motionVectorsDilated = sl::Boolean::eFalse;
    constants.motionVectorsJittered = sl::Boolean::eFalse;
    const sl::Result constantsResult = functions_->setConstants(constants, *token, viewport);
    if (constantsResult != sl::Result::eOk) {
        throw std::runtime_error("slSetConstants(DLSS Ray Reconstruction) failed: " +
                                 resultName(constantsResult));
    }

    sl::Resource inputColor = resource(frame.inputColor);
    sl::Resource outputColor = resource(frame.outputColor);
    sl::Resource depth = resource(frame.depth);
    sl::Resource motionVectors = resource(frame.motionVectors);
    sl::Resource diffuseAlbedo = resource(reconstructionFrame.diffuseAlbedo);
    sl::Resource specularAlbedo = resource(reconstructionFrame.specularAlbedo);
    sl::Resource normalRoughness = resource(reconstructionFrame.normalRoughness);
    sl::Resource specularHitDistance = resource(reconstructionFrame.specularHitDistance);
    const sl::Extent inputExtent{.top = 0U,
                                 .left = 0U,
                                 .width = frame.inputColor.extent.width,
                                 .height = frame.inputColor.extent.height};
    const sl::Extent outputExtent{.top = 0U,
                                  .left = 0U,
                                  .width = frame.outputColor.extent.width,
                                  .height = frame.outputColor.extent.height};
    const std::array tags{
        sl::ResourceTag{&inputColor, sl::kBufferTypeScalingInputColor,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &inputExtent},
        sl::ResourceTag{&outputColor, sl::kBufferTypeScalingOutputColor,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &outputExtent},
        sl::ResourceTag{&depth, sl::kBufferTypeDepth,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &inputExtent},
        sl::ResourceTag{&motionVectors, sl::kBufferTypeMotionVectors,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &inputExtent},
        sl::ResourceTag{&diffuseAlbedo, sl::kBufferTypeAlbedo,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &inputExtent},
        sl::ResourceTag{&specularAlbedo, sl::kBufferTypeSpecularAlbedo,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &inputExtent},
        sl::ResourceTag{&normalRoughness, sl::kBufferTypeNormalRoughness,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &inputExtent},
        sl::ResourceTag{&specularHitDistance, sl::kBufferTypeSpecularHitDistance,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &inputExtent},
    };
    auto* commandBuffer = reinterpret_cast<sl::CommandBuffer*>(frame.commandBuffer);
    const sl::Result tagResult = functions_->setTagForFrame(
        *token, viewport, tags.data(), static_cast<std::uint32_t>(tags.size()), commandBuffer);
    if (tagResult != sl::Result::eOk) {
        throw std::runtime_error("slSetTagForFrame(DLSS Ray Reconstruction) failed: " +
                                 resultName(tagResult));
    }

    const sl::BaseStructure* inputs[]{&viewport};
    const sl::Result evaluateResult = functions_->evaluateFeature(
        sl::kFeatureDLSS_RR, *token, inputs, 1U, commandBuffer);
    if (evaluateResult != sl::Result::eOk) {
        throw std::runtime_error("slEvaluateFeature(DLSS Ray Reconstruction) failed: " +
                                 resultName(evaluateResult));
    }
}

void StreamlineRuntime::releaseDlssResources() {
    if (!initialized_ || functions_ == nullptr) {
        return;
    }
    const bool rayReconstructionWasConfigured = functions_->rayReconstructionConfigured;
    if (rayReconstructionWasConfigured) {
        const sl::Result result =
            functions_->freeResources(sl::kFeatureDLSS_RR, sl::ViewportHandle{0U});
        if (result != sl::Result::eOk) {
            core::log(core::LogLevel::warning,
                      "slFreeResources(DLSS Ray Reconstruction) returned ",
                      resultName(result));
        }
        functions_->rayReconstructionConfigured = false;
    }
    // RR requires DLSS options, but it does not allocate a separate SR context.
    if (functions_->dlssConfigured && !rayReconstructionWasConfigured) {
        const sl::Result result =
            functions_->freeResources(sl::kFeatureDLSS, sl::ViewportHandle{0U});
        if (result != sl::Result::eOk) {
            core::log(core::LogLevel::warning, "slFreeResources(DLSS) returned ",
                      resultName(result));
        }
    }
    functions_->dlssConfigured = false;
    functions_->dlssMode = sl::DLSSMode::eOff;
    functions_->outputWidth = 0U;
    functions_->outputHeight = 0U;
}

} // namespace renderer::nvidia
