#pragma once

#include "renderer/vulkan/MemoryAllocator.hpp"

#include <volk.h>

#include <cstdint>
#include <span>

namespace renderer::vulkan {

class Device;

struct ImageCreateInfo {
    VkExtent3D extent{1, 1, 1};
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    VkImageType type = VK_IMAGE_TYPE_2D;
    VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    std::uint32_t mipLevels = 1;
    std::uint32_t arrayLayers = 1;
    VkImageCreateFlags flags = 0;
    VkMemoryPropertyFlags requiredMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkMemoryPropertyFlags preferredMemory = 0;
    std::span<const std::uint32_t> queueFamilies{};
};

class Image final {
  public:
    Image(const Device& device, MemoryAllocator& allocator, const ImageCreateInfo& createInfo);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    [[nodiscard]] VkImage handle() const { return image_; }
    [[nodiscard]] VkFormat format() const { return format_; }
    [[nodiscard]] VkExtent3D extent() const { return extent_; }
    [[nodiscard]] std::uint32_t mipLevels() const { return mipLevels_; }
    [[nodiscard]] std::uint32_t arrayLayers() const { return arrayLayers_; }

  private:
    const Device& device_;
    VkImage image_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent3D extent_{};
    std::uint32_t mipLevels_ = 0;
    std::uint32_t arrayLayers_ = 0;
    Allocation allocation_;
};

struct ImageViewCreateInfo {
    VkImageViewType type = VK_IMAGE_VIEW_TYPE_2D;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    std::uint32_t baseMipLevel = 0;
    std::uint32_t levelCount = VK_REMAINING_MIP_LEVELS;
    std::uint32_t baseArrayLayer = 0;
    std::uint32_t layerCount = VK_REMAINING_ARRAY_LAYERS;
};

class ImageView final {
  public:
    ImageView(const Device& device, const Image& image, const ImageViewCreateInfo& createInfo = {});
    ~ImageView();

    ImageView(const ImageView&) = delete;
    ImageView& operator=(const ImageView&) = delete;

    [[nodiscard]] VkImageView handle() const { return view_; }

  private:
    const Device& device_;
    VkImageView view_ = VK_NULL_HANDLE;
};

} // namespace renderer::vulkan
