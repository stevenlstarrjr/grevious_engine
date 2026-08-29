#pragma once

#include <string_view>

namespace renderer::ray_tracing {

// Public policy describes intent, not a Vulkan implementation detail.
enum class Request {
    disabled,
    automatic,
    required,
};

enum class ShadowBackend {
    raster,
    rayQuery,
};

enum class SelectionReason {
    explicitlyDisabled,
    hardwareUnavailable,
    backendUnavailable,
    sceneUnsupported,
    rayQuerySelected,
};

struct Capabilities {
    bool accelerationStructure = false;
    bool rayQuery = false;
    bool rayTracingPipeline = false;
};

struct ShadowBackendSelection {
    bool accepted = true;
    ShadowBackend backend = ShadowBackend::raster;
    SelectionReason reason = SelectionReason::hardwareUnavailable;
};

[[nodiscard]] ShadowBackendSelection selectShadowBackend(Request request,
                                                         const Capabilities& capabilities,
                                                         bool rayQueryBackendImplemented);
[[nodiscard]] ShadowBackendSelection requireSceneCompatibility(
    Request request, ShadowBackendSelection selection, bool sceneCompatible);
[[nodiscard]] std::string_view backendName(ShadowBackend backend);
[[nodiscard]] std::string_view selectionReason(SelectionReason reason);

} // namespace renderer::ray_tracing
