#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

namespace renderer::upscaling {

enum class Backend { native };

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
    Extent renderExtent;
    Extent outputExtent;
    InputRequirements inputs;
    bool resetHistory = true;
    bool dispatchRequired = false;
};

class IUpscaler {
  public:
    virtual ~IUpscaler() = default;

    [[nodiscard]] virtual Backend backend() const = 0;
    [[nodiscard]] virtual std::string_view name() const = 0;
    [[nodiscard]] virtual InputRequirements inputRequirements() const = 0;
    [[nodiscard]] virtual Extent renderExtent(Extent outputExtent) const = 0;
    [[nodiscard]] virtual FramePlan prepareFrame(const FrameInput& input) const = 0;
};

[[nodiscard]] std::string_view backendName(Backend backend);
[[nodiscard]] std::optional<Backend> parseBackend(std::string_view name);
[[nodiscard]] std::unique_ptr<IUpscaler> createUpscaler(Backend backend);

} // namespace renderer::upscaling
