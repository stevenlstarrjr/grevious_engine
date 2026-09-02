#pragma once

#include <volk.h>

#include <atomic>
#include <cstdint>

namespace renderer::platform {
class RenderSurface;
}

namespace renderer::vulkan {

enum class ValidationMode { off, available, required };

class Instance final {
  public:
    Instance(const platform::RenderSurface& surfaceHost, ValidationMode validationMode,
             PFN_vkGetInstanceProcAddr customGetInstanceProcAddr = nullptr);
    ~Instance();

    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;

    [[nodiscard]] VkInstance handle() const { return instance_; }
    [[nodiscard]] VkSurfaceKHR surface() const { return surface_; }
    [[nodiscard]] bool validationEnabled() const { return validationEnabled_; }
    [[nodiscard]] std::uint32_t validationErrorCount() const {
        return validationErrorCount_.load(std::memory_order_relaxed);
    }

  private:
    const platform::RenderSurface& surfaceHost_;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
    bool validationEnabled_ = false;
    std::atomic_uint32_t validationErrorCount_{0};
};

} // namespace renderer::vulkan
