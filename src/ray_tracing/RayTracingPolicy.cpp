#include "renderer/ray_tracing/RayTracingPolicy.hpp"

namespace renderer::ray_tracing {

ShadowBackendSelection selectShadowBackend(Request request, const Capabilities& capabilities,
                                           bool rayQueryBackendImplemented) {
    if (request == Request::disabled) {
        return {.reason = SelectionReason::explicitlyDisabled};
    }

    const bool hardwareReady = capabilities.accelerationStructure && capabilities.rayQuery;
    if (!hardwareReady) {
        return {
            .accepted = request != Request::required,
            .reason = SelectionReason::hardwareUnavailable,
        };
    }
    if (!rayQueryBackendImplemented) {
        return {
            .accepted = request != Request::required,
            .reason = SelectionReason::backendUnavailable,
        };
    }
    return {
        .backend = ShadowBackend::rayQuery,
        .reason = SelectionReason::rayQuerySelected,
    };
}

std::string_view backendName(ShadowBackend backend) {
    switch (backend) {
    case ShadowBackend::raster:
        return "raster";
    case ShadowBackend::rayQuery:
        return "ray query";
    }
    return "unknown";
}

std::string_view selectionReason(SelectionReason reason) {
    switch (reason) {
    case SelectionReason::explicitlyDisabled:
        return "ray tracing was explicitly disabled";
    case SelectionReason::hardwareUnavailable:
        return "acceleration structures and ray queries are unavailable";
    case SelectionReason::backendUnavailable:
        return "the GPU is capable but the ray-query shadow backend is not implemented yet";
    case SelectionReason::rayQuerySelected:
        return "the ray-query backend is available";
    }
    return "unknown selection reason";
}

} // namespace renderer::ray_tracing
