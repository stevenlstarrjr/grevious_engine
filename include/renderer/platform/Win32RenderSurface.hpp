#pragma once

#include "renderer/platform/RenderSurface.hpp"

#include <cstdint>

namespace renderer::platform {

// Adapts an existing Win32 child or top-level window to Genesis without
// depending on SDL or Qt. A Qt QWindow host passes reinterpret_cast<void*>(winId()).
// Genesis owns the VkSurfaceKHR and swapchain; the host continues to own HWND
// lifetime, sizing, input, and its event loop.
class Win32RenderSurface final : public RenderSurface {
  public:
    explicit Win32RenderSurface(void* nativeWindowHandle);
    ~Win32RenderSurface() override = default;

    Win32RenderSurface(const Win32RenderSurface&) = delete;
    Win32RenderSurface& operator=(const Win32RenderSurface&) = delete;

    [[nodiscard]] bool shouldClose() const override;
    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t> waitForDrawableSize() override;
    [[nodiscard]] bool consumeFramebufferResize() override;
    [[nodiscard]] RendererInput consumeRendererInput() override;
    [[nodiscard]] std::vector<const char*> requiredVulkanExtensions() const override;
    [[nodiscard]] VkSurfaceKHR createSurface(VkInstance instance) const override;
    void destroySurface(VkInstance instance, VkSurfaceKHR surface) const override;

    // Call from the Qt resize event after the native child has its new size.
    void notifyFramebufferResize();
    // Held controls persist; resetCamera and mouse deltas are consumed once.
    void setRendererInput(RendererInput input);
    void requestClose();
    [[nodiscard]] void* nativeWindowHandle() const { return nativeWindowHandle_; }

  private:
    void* nativeWindowHandle_ = nullptr;
    bool closeRequested_ = false;
    bool framebufferResized_ = false;
    RendererInput rendererInput_{};
};

} // namespace renderer::platform
