#include "renderer/vulkan/Image.hpp"

#include "renderer/vulkan/Device.hpp"
#include "renderer/vulkan/VulkanError.hpp"

#include <stdexcept>

namespace renderer::vulkan {

Image::Image(const Device& device, MemoryAllocator& allocator, const ImageCreateInfo& createInfo)
    : device_(device), format_(createInfo.format), extent_(createInfo.extent),
      mipLevels_(createInfo.mipLevels), arrayLayers_(createInfo.arrayLayers) {
    if (format_ == VK_FORMAT_UNDEFINED || createInfo.usage == 0 || extent_.width == 0 ||
        extent_.height == 0 || extent_.depth == 0 || mipLevels_ == 0 || arrayLayers_ == 0) {
        throw std::runtime_error("Image create info is incomplete");
    }
    const bool concurrent = createInfo.queueFamilies.size() > 1;
    const VkImageCreateInfo imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = createInfo.flags,
        .imageType = createInfo.type,
        .format = format_,
        .extent = extent_,
        .mipLevels = mipLevels_,
        .arrayLayers = arrayLayers_,
        .samples = createInfo.samples,
        .tiling = createInfo.tiling,
        .usage = createInfo.usage,
        .sharingMode = concurrent ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount =
            concurrent ? static_cast<std::uint32_t>(createInfo.queueFamilies.size()) : 0U,
        .pQueueFamilyIndices = concurrent ? createInfo.queueFamilies.data() : nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    check(vkCreateImage(device_.handle(), &imageInfo, nullptr, &image_), "vkCreateImage");

    try {
        allocation_ = allocator.allocateForImage(image_, createInfo.requiredMemory,
                                                 createInfo.preferredMemory);
        allocator.bindImage(image_, allocation_);
    } catch (...) {
        vkDestroyImage(device_.handle(), image_, nullptr);
        image_ = VK_NULL_HANDLE;
        throw;
    }
}

Image::~Image() {
    if (image_ != VK_NULL_HANDLE) {
        vkDestroyImage(device_.handle(), image_, nullptr);
    }
    allocation_.reset();
}

ImageView::ImageView(const Device& device, const Image& image,
                     const ImageViewCreateInfo& createInfo)
    : device_(device) {
    const VkImageViewCreateInfo viewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image.handle(),
        .viewType = createInfo.type,
        .format = image.format(),
        .components =
            {
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
            },
        .subresourceRange =
            {
                .aspectMask = createInfo.aspect,
                .baseMipLevel = createInfo.baseMipLevel,
                .levelCount = createInfo.levelCount,
                .baseArrayLayer = createInfo.baseArrayLayer,
                .layerCount = createInfo.layerCount,
            },
    };
    check(vkCreateImageView(device_.handle(), &viewInfo, nullptr, &view_), "vkCreateImageView");
}

ImageView::~ImageView() {
    if (view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_.handle(), view_, nullptr);
    }
}

} // namespace renderer::vulkan
