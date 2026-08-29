#include "renderer/platform/Window.hpp"

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace renderer::platform {
namespace {

[[nodiscard]] std::runtime_error sdlError(std::string_view operation) {
    return std::runtime_error(std::string(operation) + ": " + SDL_GetError());
}

} // namespace

Window::Window(std::uint32_t width, std::uint32_t height, std::string_view title) {
    if (width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        height > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("SDL window dimensions exceed the platform range");
    }
#if defined(__linux__)
    if (!SDL_SetHintWithPriority(SDL_HINT_VIDEO_WAYLAND_ALLOW_LIBDECOR, "1", SDL_HINT_OVERRIDE) ||
        !SDL_SetHintWithPriority(SDL_HINT_VIDEO_WAYLAND_PREFER_LIBDECOR, "1", SDL_HINT_OVERRIDE)) {
        throw sdlError("SDL Wayland decoration policy");
    }
#endif
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        throw sdlError("SDL_Init(video)");
    }
    const char* videoDriver = SDL_GetCurrentVideoDriver();
#if defined(_WIN32)
    constexpr std::string_view requiredVideoDriver = "windows";
#else
    constexpr std::string_view requiredVideoDriver = "wayland";
#endif
    if (videoDriver == nullptr || std::string_view(videoDriver) != requiredVideoDriver) {
        const std::string selected = videoDriver == nullptr ? "none" : videoDriver;
        SDL_Quit();
        throw std::runtime_error("Renderer requires SDL's " + std::string(requiredVideoDriver) +
                                 " video driver; selected: " + selected);
    }

    const std::string ownedTitle(title);
    constexpr SDL_WindowFlags flags =
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    handle_ = SDL_CreateWindow(ownedTitle.c_str(), static_cast<int>(width),
                               static_cast<int>(height), flags);
    if (handle_ == nullptr) {
        const std::runtime_error error = sdlError("SDL_CreateWindow(Vulkan)");
        SDL_Quit();
        throw error;
    }

    // SDL guarantees an initial pixel-size event. Consume it here so the first
    // rendered frame does not immediately rebuild a freshly-created swapchain.
    pollEvents();
    framebufferResized_ = false;
}

Window::~Window() {
    if (handle_ != nullptr) {
        if (mouseLookActive_) {
            (void)SDL_SetWindowRelativeMouseMode(handle_, false);
        }
        SDL_DestroyWindow(handle_);
    }
    SDL_Quit();
}

bool Window::shouldClose() const { return closeRequested_; }

void Window::pollEvents() {
    SDL_Event event{};
    while (SDL_PollEvent(&event)) {
        processEvent(event);
    }
}

void Window::resize(std::uint32_t width, std::uint32_t height) {
    if (width == 0U || height == 0U ||
        width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        height > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("SDL window resize requires nonzero platform dimensions");
    }
    if (!SDL_SetWindowSize(handle_, static_cast<int>(width), static_cast<int>(height)) ||
        !SDL_SyncWindow(handle_)) {
        throw sdlError("SDL_SetWindowSize");
    }
    pollEvents();
    framebufferResized_ = true;
}

std::pair<std::uint32_t, std::uint32_t> Window::waitForDrawableSize() {
    int width = 0;
    int height = 0;
    if (!SDL_GetWindowSizeInPixels(handle_, &width, &height)) {
        throw sdlError("SDL_GetWindowSizeInPixels");
    }
    while ((width == 0 || height == 0) && !closeRequested_) {
        SDL_Event event{};
        if (!SDL_WaitEvent(&event)) {
            throw sdlError("SDL_WaitEvent");
        }
        processEvent(event);
        if (!SDL_GetWindowSizeInPixels(handle_, &width, &height)) {
            throw sdlError("SDL_GetWindowSizeInPixels");
        }
    }
    return {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};
}

bool Window::consumeFramebufferResize() {
    const bool resized = framebufferResized_;
    framebufferResized_ = false;
    return resized;
}

RendererInput Window::consumeRendererInput() {
    int keyCount = 0;
    const bool* keys = SDL_GetKeyboardState(&keyCount);
    const auto pressed = [&](SDL_Scancode scancode) {
        const int index = static_cast<int>(scancode);
        return keys != nullptr && index >= 0 && index < keyCount && keys[index];
    };
    RendererInput input{
        .moveForward = pressed(SDL_SCANCODE_W),
        .moveBackward = pressed(SDL_SCANCODE_S),
        .moveLeft = pressed(SDL_SCANCODE_A),
        .moveRight = pressed(SDL_SCANCODE_D),
        .moveUp = pressed(SDL_SCANCODE_E),
        .moveDown = pressed(SDL_SCANCODE_Q),
        .speedBoost = pressed(SDL_SCANCODE_LSHIFT) || pressed(SDL_SCANCODE_RSHIFT),
        .resetCamera = resetCameraRequested_,
        .rotateEnvironmentLeft = pressed(SDL_SCANCODE_LEFT),
        .rotateEnvironmentRight = pressed(SDL_SCANCODE_RIGHT),
        .mouseLookActive = mouseLookActive_,
        .mouseDeltaX = mouseDeltaX_,
        .mouseDeltaY = mouseDeltaY_,
    };
    resetCameraRequested_ = false;
    mouseDeltaX_ = 0.0F;
    mouseDeltaY_ = 0.0F;
    return input;
}

std::vector<const char*> Window::requiredVulkanExtensions() const {
    Uint32 count = 0;
    const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&count);
    if (extensions == nullptr || count == 0) {
        throw sdlError("SDL_Vulkan_GetInstanceExtensions");
    }
    return {extensions, extensions + count};
}

VkSurfaceKHR Window::createSurface(VkInstance instance) const {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(handle_, instance, nullptr, &surface)) {
        throw sdlError("SDL_Vulkan_CreateSurface");
    }
    return surface;
}

void Window::destroySurface(VkInstance instance, VkSurfaceKHR surface) const {
    SDL_Vulkan_DestroySurface(instance, surface, nullptr);
}

void Window::setMouseLook(bool enabled) {
    if (mouseLookActive_ == enabled) {
        return;
    }
    if (!SDL_SetWindowRelativeMouseMode(handle_, enabled)) {
        throw sdlError(enabled ? "SDL_SetWindowRelativeMouseMode(enable)"
                               : "SDL_SetWindowRelativeMouseMode(disable)");
    }
    mouseLookActive_ = enabled;
    mouseDeltaX_ = 0.0F;
    mouseDeltaY_ = 0.0F;
}

void Window::processEvent(const SDL_Event& event) {
    if (event.type == SDL_EVENT_QUIT) {
        closeRequested_ = true;
        return;
    }
    const SDL_WindowID windowId = SDL_GetWindowID(handle_);
    if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == windowId) {
        closeRequested_ = true;
    } else if (event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED &&
               event.window.windowID == windowId) {
        framebufferResized_ = true;
    } else if (event.type == SDL_EVENT_WINDOW_FOCUS_LOST && event.window.windowID == windowId) {
        setMouseLook(false);
    } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.windowID == windowId &&
               event.button.button == SDL_BUTTON_RIGHT) {
        setMouseLook(true);
    } else if (event.type == SDL_EVENT_MOUSE_BUTTON_UP && event.button.windowID == windowId &&
               event.button.button == SDL_BUTTON_RIGHT) {
        setMouseLook(false);
    } else if (event.type == SDL_EVENT_MOUSE_MOTION && event.motion.windowID == windowId &&
               mouseLookActive_) {
        mouseDeltaX_ += event.motion.xrel;
        mouseDeltaY_ += event.motion.yrel;
    } else if (event.type == SDL_EVENT_KEY_DOWN && event.key.windowID == windowId &&
               !event.key.repeat) {
        if (event.key.scancode == SDL_SCANCODE_ESCAPE) {
            setMouseLook(false);
        } else if (event.key.scancode == SDL_SCANCODE_R) {
            resetCameraRequested_ = true;
        }
    }
}

} // namespace renderer::platform
