#pragma once

#include <volk.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace renderer::platform {

// Renderer-demo controls only. This plain snapshot keeps window-toolkit types
// out of the renderer and does not establish a game-input subsystem.
struct RendererInput {
    bool moveForward = false;
    bool moveBackward = false;
    bool moveLeft = false;
    bool moveRight = false;
    bool moveUp = false;
    bool moveDown = false;
    bool speedBoost = false;
    bool resetCamera = false;
    bool rotateEnvironmentLeft = false;
    bool rotateEnvironmentRight = false;
    bool mouseLookActive = false;
    float mouseDeltaX = 0.0F;
    float mouseDeltaY = 0.0F;
    bool atmosphereDragActive = false;
    bool atmosphereSunDrag = false;
    float atmosphereDeltaX = 0.0F;
    float atmosphereDeltaY = 0.0F;
    float atmosphereWheelDelta = 0.0F;
    int atmosphereExposureSteps = 0;
    std::uint32_t atmospherePreset = 0U;
    bool timeOfDayToggle = false;
    int timeOfDaySpeedSteps = 0;
    int timeOfDayScrubSteps = 0;
};

// Toolkit-neutral boundary between Genesis and the native window that owns the
// Vulkan presentation surface. SDL and Qt/Win32 hosts implement the same small
// contract while Genesis retains ownership of Vulkan and the swapchain.
class RenderSurface {
  public:
    virtual ~RenderSurface() = default;

    RenderSurface(const RenderSurface&) = delete;
    RenderSurface& operator=(const RenderSurface&) = delete;

    [[nodiscard]] virtual bool shouldClose() const = 0;
    [[nodiscard]] virtual std::pair<std::uint32_t, std::uint32_t> waitForDrawableSize() = 0;
    [[nodiscard]] virtual bool consumeFramebufferResize() = 0;
    [[nodiscard]] virtual RendererInput consumeRendererInput() = 0;
    [[nodiscard]] virtual std::vector<const char*> requiredVulkanExtensions() const = 0;
    [[nodiscard]] virtual VkSurfaceKHR createSurface(VkInstance instance) const = 0;
    virtual void destroySurface(VkInstance instance, VkSurfaceKHR surface) const = 0;

  protected:
    RenderSurface() = default;
};

} // namespace renderer::platform
