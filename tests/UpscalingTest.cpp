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

template <typename Function>
void requireInvalid(Function&& function, std::string_view message) {
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
    require(parseBackend("native") == Backend::native && !parseBackend("dlss") &&
                backendName(Backend::native) == "native",
            "Upscaling backend naming or parsing is incorrect");
    const std::unique_ptr<IUpscaler> upscaler = createUpscaler(Backend::native);
    require(upscaler && upscaler->backend() == Backend::native &&
                upscaler->name() == "native",
            "Native upscaler factory returned the wrong backend");
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
                !resetPlan.dispatchRequired &&
                resetPlan.renderExtent == resetPlan.outputExtent,
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

} // namespace

int main() {
    try {
        testBackendIdentity();
        testNativeExtentPolicy();
        testCanonicalInputsAndReset();
        testInvalidInputs();
        std::cout << "Upscaling tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Upscaling test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
