#include "renderer/vulkan/Swapchain.hpp"

#include "renderer/core/Log.hpp"
#include "renderer/vulkan/Device.hpp"
#include "renderer/vulkan/Instance.hpp"
#include "renderer/vulkan/VulkanError.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace renderer::vulkan {
namespace {

VkSurfaceFormatKHR selectFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    constexpr std::array preferred{
        VK_FORMAT_B8G8R8A8_SRGB,
        VK_FORMAT_R8G8B8A8_SRGB,
        VK_FORMAT_B8G8R8A8_UNORM,
        VK_FORMAT_R8G8B8A8_UNORM,
    };
    for (const VkFormat desired : preferred) {
        const auto found = std::ranges::find_if(formats, [desired](const auto& format) {
            return format.format == desired &&
                   format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
        });
        if (found != formats.end()) {
            return *found;
        }
    }
    return formats.front();
}

VkPresentModeKHR selectPresentMode(const std::vector<VkPresentModeKHR>& modes) {
    // FIFO is the portable, paced baseline. An explicit low-latency present-mode
    // policy can opt into mailbox later; the validation/demo path must remain
    // visually inspectable and must not render hundreds of discarded frames.
    if (std::ranges::find(modes, VK_PRESENT_MODE_FIFO_KHR) == modes.end()) {
        throw std::runtime_error("Surface does not expose required FIFO presentation");
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D selectExtent(const VkSurfaceCapabilitiesKHR& capabilities, std::uint32_t width,
                        std::uint32_t height) {
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
        return capabilities.currentExtent;
    }
    return {
        .width =
            std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        .height = std::clamp(height, capabilities.minImageExtent.height,
                             capabilities.maxImageExtent.height),
    };
}

VkCompositeAlphaFlagBitsKHR selectCompositeAlpha(const VkSurfaceCapabilitiesKHR& capabilities) {
    constexpr std::array preferred{
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
    };
    for (const VkCompositeAlphaFlagBitsKHR mode : preferred) {
        if ((capabilities.supportedCompositeAlpha & mode) != 0) {
            return mode;
        }
    }
    throw std::runtime_error("Surface reports no supported composite-alpha mode");
}

} // namespace

Swapchain::Swapchain(const Instance& instance, const Device& device, std::uint32_t width,
                     std::uint32_t height, bool transferSource)
    : instance_(instance), device_(device), transferSource_(transferSource) {
    if (!create(width, height, VK_NULL_HANDLE)) {
        throw std::runtime_error("Surface has no drawable extent for initial swapchain creation");
    }
}

Swapchain::~Swapchain() {
    destroyViews();
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_.handle(), swapchain_, nullptr);
    }
}

bool Swapchain::recreate(std::uint32_t width, std::uint32_t height) {
    const VkSwapchainKHR oldSwapchain = swapchain_;
    const std::vector<VkImageView> oldViews = views_;
    if (!create(width, height, oldSwapchain)) {
        return false;
    }
    for (VkImageView view : oldViews) {
        vkDestroyImageView(device_.handle(), view, nullptr);
    }
    vkDestroySwapchainKHR(device_.handle(), oldSwapchain, nullptr);
    return true;
}

VkResult Swapchain::acquire(VkSemaphore semaphore, std::uint32_t& imageIndex) const {
    return vkAcquireNextImageKHR(device_.handle(), swapchain_,
                                 std::numeric_limits<std::uint64_t>::max(), semaphore,
                                 VK_NULL_HANDLE, &imageIndex);
}

bool Swapchain::create(std::uint32_t width, std::uint32_t height, VkSwapchainKHR oldSwapchain) {
    VkSurfaceCapabilitiesKHR capabilities{};
    check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device_.physicalHandle(), instance_.surface(),
                                                    &capabilities),
          "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

    std::uint32_t formatCount = 0;
    check(vkGetPhysicalDeviceSurfaceFormatsKHR(device_.physicalHandle(), instance_.surface(),
                                               &formatCount, nullptr),
          "vkGetPhysicalDeviceSurfaceFormatsKHR(count)");
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    check(vkGetPhysicalDeviceSurfaceFormatsKHR(device_.physicalHandle(), instance_.surface(),
                                               &formatCount, formats.data()),
          "vkGetPhysicalDeviceSurfaceFormatsKHR");

    std::uint32_t modeCount = 0;
    check(vkGetPhysicalDeviceSurfacePresentModesKHR(device_.physicalHandle(), instance_.surface(),
                                                    &modeCount, nullptr),
          "vkGetPhysicalDeviceSurfacePresentModesKHR(count)");
    std::vector<VkPresentModeKHR> modes(modeCount);
    check(vkGetPhysicalDeviceSurfacePresentModesKHR(device_.physicalHandle(), instance_.surface(),
                                                    &modeCount, modes.data()),
          "vkGetPhysicalDeviceSurfacePresentModesKHR");
    if (formats.empty() || modes.empty()) {
        throw std::runtime_error("Surface no longer has swapchain formats or present modes");
    }
    if ((capabilities.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0) {
        throw std::runtime_error("Surface swapchain images cannot be color attachments");
    }
    if (transferSource_ &&
        (capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0) {
        throw std::runtime_error("Surface swapchain images do not support requested capture");
    }

    const VkSurfaceFormatKHR selectedFormat = selectFormat(formats);
    const VkPresentModeKHR presentMode = selectPresentMode(modes);
    const VkExtent2D selectedExtent = selectExtent(capabilities, width, height);
    if (selectedExtent.width == 0U || selectedExtent.height == 0U) {
        return false;
    }
    std::uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount != 0) {
        imageCount = std::min(imageCount, capabilities.maxImageCount);
    }

    const auto& families = device_.queueFamilies();
    const std::array sharingFamilies{families.graphics, families.present};
    const bool separatePresentQueue = families.graphics != families.present;
    const VkSwapchainCreateInfoKHR createInfo{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = instance_.surface(),
        .minImageCount = imageCount,
        .imageFormat = selectedFormat.format,
        .imageColorSpace = selectedFormat.colorSpace,
        .imageExtent = selectedExtent,
        .imageArrayLayers = 1,
        .imageUsage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            (transferSource_ ? static_cast<VkImageUsageFlags>(VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                             : VkImageUsageFlags{0}),
        .imageSharingMode =
            separatePresentQueue ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = separatePresentQueue ? 2U : 0U,
        .pQueueFamilyIndices = separatePresentQueue ? sharingFamilies.data() : nullptr,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = selectCompositeAlpha(capabilities),
        .presentMode = presentMode,
        .clipped = VK_TRUE,
        .oldSwapchain = oldSwapchain,
    };
    VkSwapchainKHR newSwapchain = VK_NULL_HANDLE;
    check(vkCreateSwapchainKHR(device_.handle(), &createInfo, nullptr, &newSwapchain),
          "vkCreateSwapchainKHR");

    std::vector<VkImage> newImages;
    std::vector<VkImageView> newViews;
    try {
        check(vkGetSwapchainImagesKHR(device_.handle(), newSwapchain, &imageCount, nullptr),
              "vkGetSwapchainImagesKHR(count)");
        newImages.resize(imageCount);
        check(
            vkGetSwapchainImagesKHR(device_.handle(), newSwapchain, &imageCount, newImages.data()),
            "vkGetSwapchainImagesKHR");

        newViews.reserve(newImages.size());
        for (VkImage image : newImages) {
            const VkImageViewCreateInfo viewInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = selectedFormat.format,
                .components =
                    {
                        .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                    },
                .subresourceRange =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
            };
            VkImageView view = VK_NULL_HANDLE;
            check(vkCreateImageView(device_.handle(), &viewInfo, nullptr, &view),
                  "vkCreateImageView(swapchain)");
            newViews.push_back(view);
        }
    } catch (...) {
        for (VkImageView view : newViews) {
            vkDestroyImageView(device_.handle(), view, nullptr);
        }
        vkDestroySwapchainKHR(device_.handle(), newSwapchain, nullptr);
        throw;
    }

    swapchain_ = newSwapchain;
    format_ = selectedFormat.format;
    colorSpace_ = selectedFormat.colorSpace;
    extent_ = selectedExtent;
    images_ = std::move(newImages);
    views_ = std::move(newViews);

    core::log(core::LogLevel::info, "Swapchain: ", extent_.width, 'x', extent_.height,
              ", images=", images_.size(), ", format=", static_cast<int>(format_),
              ", present mode=", static_cast<int>(presentMode));
    return true;
}

void Swapchain::destroyViews() {
    for (VkImageView view : views_) {
        vkDestroyImageView(device_.handle(), view, nullptr);
    }
    views_.clear();
}

} // namespace renderer::vulkan
