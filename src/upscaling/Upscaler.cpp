#include "renderer/upscaling/Upscaler.hpp"

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
    if (input.jitterAvailable &&
        (!std::isfinite(input.currentJitterPixels.x) ||
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
            .renderExtent = input.renderExtent,
            .outputExtent = input.outputExtent,
            .inputs = requirements,
            .resetHistory = input.resetHistory,
            .dispatchRequired = false,
        };
    }
};

} // namespace

std::string_view backendName(Backend backend) {
    switch (backend) {
    case Backend::native:
        return "native";
    }
    return "unknown";
}

std::optional<Backend> parseBackend(std::string_view name) {
    if (name == "native") {
        return Backend::native;
    }
    return std::nullopt;
}

std::unique_ptr<IUpscaler> createUpscaler(Backend backend) {
    switch (backend) {
    case Backend::native:
        return std::make_unique<NativeUpscaler>();
    }
    throw std::invalid_argument("Unknown upscaling backend");
}

} // namespace renderer::upscaling
