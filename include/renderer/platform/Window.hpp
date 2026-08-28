#pragma once

#include <volk.h>

#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

struct SDL_Window;
union SDL_Event;

namespace renderer::platform {

// Renderer-demo controls only. This plain snapshot keeps SDL types out of the
// renderer and does not establish a game-input subsystem.
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
};

class Window final {
  public:
    Window(std::uint32_t width, std::uint32_t height, std::string_view title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    [[nodiscard]] bool shouldClose() const;
    void pollEvents();
    void resize(std::uint32_t width, std::uint32_t height);
    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> waitForDrawableSize();
    [[nodiscard]] bool consumeFramebufferResize();
    [[nodiscard]] RendererInput consumeRendererInput();
    [[nodiscard]] std::vector<const char*> requiredVulkanExtensions() const;
    [[nodiscard]] VkSurfaceKHR createSurface(VkInstance instance) const;
    void destroySurface(VkInstance instance, VkSurfaceKHR surface) const;

  private:
    void processEvent(const SDL_Event& event);
    void setMouseLook(bool enabled);

    SDL_Window* handle_ = nullptr;
    bool closeRequested_ = false;
    bool framebufferResized_ = false;
    bool mouseLookActive_ = false;
    bool resetCameraRequested_ = false;
    float mouseDeltaX_ = 0.0F;
    float mouseDeltaY_ = 0.0F;
};

} // namespace renderer::platform
