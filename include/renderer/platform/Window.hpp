#pragma once

#include "renderer/platform/RenderSurface.hpp"

#include <cstdint>
#include <string_view>

struct SDL_Window;
union SDL_Event;

namespace renderer::platform {

class Window final : public RenderSurface {
  public:
    Window(std::uint32_t width, std::uint32_t height, std::string_view title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    [[nodiscard]] bool shouldClose() const override;
    void pollEvents();
    void resize(std::uint32_t width, std::uint32_t height);
    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> waitForDrawableSize() override;
    [[nodiscard]] bool consumeFramebufferResize() override;
    [[nodiscard]] RendererInput consumeRendererInput() override;
    [[nodiscard]] std::vector<const char*> requiredVulkanExtensions() const override;
    [[nodiscard]] VkSurfaceKHR createSurface(VkInstance instance) const override;
    void destroySurface(VkInstance instance, VkSurfaceKHR surface) const override;

  private:
    void processEvent(const SDL_Event& event);
    void setMouseLook(bool enabled);

    SDL_Window* handle_ = nullptr;
    bool closeRequested_ = false;
    bool framebufferResized_ = false;
    bool mouseLookActive_ = false;
    bool atmosphereSunDragActive_ = false;
    bool resetCameraRequested_ = false;
    float mouseDeltaX_ = 0.0F;
    float mouseDeltaY_ = 0.0F;
    float atmosphereSunDeltaX_ = 0.0F;
    float atmosphereSunDeltaY_ = 0.0F;
    int atmosphereExposureSteps_ = 0;
    std::uint32_t atmospherePreset_ = 0U;
    bool timeOfDayToggle_ = false;
    int timeOfDaySpeedSteps_ = 0;
    int timeOfDayScrubSteps_ = 0;
};

} // namespace renderer::platform
