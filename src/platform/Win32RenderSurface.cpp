#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "renderer/platform/Win32RenderSurface.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace renderer::platform {

Win32RenderSurface::Win32RenderSurface(void* nativeWindowHandle)
    : nativeWindowHandle_(nativeWindowHandle) {
    if (nativeWindowHandle_ == nullptr ||
        !IsWindow(static_cast<HWND>(nativeWindowHandle_))) {
        throw std::invalid_argument("Win32RenderSurface requires a valid HWND");
    }
}

bool Win32RenderSurface::shouldClose() const { return closeRequested_; }

std::pair<std::uint32_t, std::uint32_t> Win32RenderSurface::waitForDrawableSize() {
    RECT clientRect{};
    if (!GetClientRect(static_cast<HWND>(nativeWindowHandle_), &clientRect)) {
        throw std::runtime_error("GetClientRect failed for the Genesis render surface");
    }
    const LONG width = clientRect.right - clientRect.left;
    const LONG height = clientRect.bottom - clientRect.top;
    if (width <= 0 || height <= 0) {
        return {0U, 0U};
    }
    if (static_cast<unsigned long>(width) > std::numeric_limits<std::uint32_t>::max() ||
        static_cast<unsigned long>(height) > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Win32 drawable size exceeds the Vulkan extent range");
    }
    return {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};
}

bool Win32RenderSurface::consumeFramebufferResize() {
    const bool resized = framebufferResized_;
    framebufferResized_ = false;
    return resized;
}

RendererInput Win32RenderSurface::consumeRendererInput() {
    const RendererInput input = rendererInput_;
    rendererInput_.resetCamera = false;
    rendererInput_.mouseDeltaX = 0.0F;
    rendererInput_.mouseDeltaY = 0.0F;
    return input;
}

std::vector<const char*> Win32RenderSurface::requiredVulkanExtensions() const {
    return {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
}

VkSurfaceKHR Win32RenderSurface::createSurface(VkInstance instance) const {
    const HWND window = static_cast<HWND>(nativeWindowHandle_);
    HINSTANCE applicationInstance = reinterpret_cast<HINSTANCE>(
        GetWindowLongPtrW(window, GWLP_HINSTANCE));
    if (applicationInstance == nullptr) {
        applicationInstance = GetModuleHandleW(nullptr);
    }
    const VkWin32SurfaceCreateInfoKHR createInfo{
        .sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
        .hinstance = applicationInstance,
        .hwnd = window,
    };
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    const VkResult result = vkCreateWin32SurfaceKHR(instance, &createInfo, nullptr, &surface);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("vkCreateWin32SurfaceKHR failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
    }
    return surface;
}

void Win32RenderSurface::destroySurface(VkInstance instance, VkSurfaceKHR surface) const {
    vkDestroySurfaceKHR(instance, surface, nullptr);
}

void Win32RenderSurface::notifyFramebufferResize() { framebufferResized_ = true; }

void Win32RenderSurface::setRendererInput(RendererInput input) {
    rendererInput_ = input;
}

void Win32RenderSurface::requestClose() { closeRequested_ = true; }

} // namespace renderer::platform
