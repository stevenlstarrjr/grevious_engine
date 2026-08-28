#include "renderer/ray_tracing/RayTracingPolicy.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void testExplicitRasterSelection() {
    using namespace renderer::ray_tracing;
    const ShadowBackendSelection selection = selectShadowBackend(
        Request::disabled,
        {.accelerationStructure = true, .rayQuery = true, .rayTracingPipeline = true}, true);
    require(selection.accepted && selection.backend == ShadowBackend::raster &&
                selection.reason == SelectionReason::explicitlyDisabled,
            "Explicitly disabled ray tracing did not force the raster backend");
}

void testAutomaticFallbacks() {
    using namespace renderer::ray_tracing;
    const ShadowBackendSelection unsupported = selectShadowBackend(Request::automatic, {}, false);
    require(unsupported.accepted && unsupported.backend == ShadowBackend::raster &&
                unsupported.reason == SelectionReason::hardwareUnavailable,
            "Automatic mode did not accept the unsupported-hardware raster fallback");

    const ShadowBackendSelection incompleteHardware = selectShadowBackend(
        Request::automatic, {.accelerationStructure = true, .rayQuery = false}, true);
    require(incompleteHardware.accepted &&
                incompleteHardware.reason == SelectionReason::hardwareUnavailable,
            "Acceleration structures without ray queries were treated as a complete backend");

    const ShadowBackendSelection unfinishedBackend = selectShadowBackend(
        Request::automatic, {.accelerationStructure = true, .rayQuery = true}, false);
    require(unfinishedBackend.accepted && unfinishedBackend.backend == ShadowBackend::raster &&
                unfinishedBackend.reason == SelectionReason::backendUnavailable,
            "Automatic mode selected an unimplemented ray-query backend");
}

void testRequiredModeFailures() {
    using namespace renderer::ray_tracing;
    const ShadowBackendSelection unsupported = selectShadowBackend(Request::required, {}, false);
    require(!unsupported.accepted && unsupported.reason == SelectionReason::hardwareUnavailable,
            "Required mode accepted hardware without the required RT features");

    const ShadowBackendSelection unfinishedBackend = selectShadowBackend(
        Request::required, {.accelerationStructure = true, .rayQuery = true}, false);
    require(!unfinishedBackend.accepted &&
                unfinishedBackend.reason == SelectionReason::backendUnavailable,
            "Required mode accepted an unimplemented ray-query backend");
}

void testRayQuerySelection() {
    using namespace renderer::ray_tracing;
    const ShadowBackendSelection selection = selectShadowBackend(
        Request::automatic,
        {.accelerationStructure = true, .rayQuery = true, .rayTracingPipeline = false}, true);
    require(selection.accepted && selection.backend == ShadowBackend::rayQuery &&
                selection.reason == SelectionReason::rayQuerySelected,
            "A complete ray-query path was not selected");
    require(backendName(selection.backend) == "ray query" &&
                !selectionReason(selection.reason).empty(),
            "Ray-tracing policy descriptions are incomplete");
}

} // namespace

int main() {
    try {
        testExplicitRasterSelection();
        testAutomaticFallbacks();
        testRequiredModeFailures();
        testRayQuerySelection();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
