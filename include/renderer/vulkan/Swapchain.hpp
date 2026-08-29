#pragma once

#include <volk.h>

#include <cstdint>
#include <vector>

namespace renderer::vulkan {

class Device;
class Instance;

class Swapchain final {
  public:
    Swapchain(const Instance& instance, const Device& device, std::uint32_t width,
              std::uint32_t height, bool transferSource = false);
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    [[nodiscard]] bool recreate(std::uint32_t width, std::uint32_t height);
    [[nodiscard]] VkResult acquire(VkSemaphore semaphore, std::uint32_t& imageIndex) const;

    [[nodiscard]] VkSwapchainKHR handle() const { return swapchain_; }
    [[nodiscard]] VkFormat format() const { return format_; }
    [[nodiscard]] VkExtent2D extent() const { return extent_; }
    [[nodiscard]] std::uint32_t imageCount() const {
        return static_cast<std::uint32_t>(images_.size());
    }
    [[nodiscard]] VkImage image(std::uint32_t index) const { return images_.at(index); }
    [[nodiscard]] VkImageView imageView(std::uint32_t index) const { return views_.at(index); }

  private:
    [[nodiscard]] bool create(std::uint32_t width, std::uint32_t height,
                              VkSwapchainKHR oldSwapchain);
    void destroyViews();

    const Instance& instance_;
    const Device& device_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR colorSpace_ = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    bool transferSource_ = false;
    VkExtent2D extent_{};
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
};

} // namespace renderer::vulkan
