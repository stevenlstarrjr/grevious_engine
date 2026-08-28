#include "renderer/vulkan/Buffer.hpp"

#include "renderer/vulkan/Device.hpp"
#include "renderer/vulkan/VulkanError.hpp"

#include <cstring>
#include <stdexcept>

namespace renderer::vulkan {

Buffer::Buffer(const Device& device, MemoryAllocator& allocator, const BufferCreateInfo& createInfo)
    : device_(device), size_(createInfo.size), usage_(createInfo.usage) {
    if (size_ == 0 || usage_ == 0) {
        throw std::runtime_error("Buffer requires non-zero size and usage");
    }
    const bool concurrent = createInfo.queueFamilies.size() > 1;
    const VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size_,
        .usage = usage_,
        .sharingMode = concurrent ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount =
            concurrent ? static_cast<std::uint32_t>(createInfo.queueFamilies.size()) : 0U,
        .pQueueFamilyIndices = concurrent ? createInfo.queueFamilies.data() : nullptr,
    };
    check(vkCreateBuffer(device_.handle(), &bufferInfo, nullptr, &buffer_), "vkCreateBuffer");

    try {
        allocation_ = allocator.allocateForBuffer(buffer_, createInfo.requiredMemory,
                                                  createInfo.preferredMemory,
                                                  createInfo.randomHostAccess);
        allocator.bindBuffer(buffer_, allocation_);
    } catch (...) {
        vkDestroyBuffer(device_.handle(), buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        throw;
    }
}

Buffer::~Buffer() {
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_.handle(), buffer_, nullptr);
    }
    allocation_.reset();
}

VkDeviceAddress Buffer::deviceAddress() const {
    if ((usage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) == 0) {
        throw std::runtime_error("Buffer was not created for device-address queries");
    }
    const VkBufferDeviceAddressInfo addressInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buffer_,
    };
    return vkGetBufferDeviceAddress(device_.handle(), &addressInfo);
}

void Buffer::write(std::span<const std::byte> data, VkDeviceSize offset) {
    if (allocation_.mappedData() == nullptr) {
        throw std::runtime_error("Cannot write directly to a non-host-visible buffer");
    }
    if (offset > size_ || data.size() > size_ - offset) {
        throw std::runtime_error("Buffer write exceeds allocation bounds");
    }
    auto* destination = static_cast<std::byte*>(allocation_.mappedData()) + offset;
    std::memcpy(destination, data.data(), data.size());
    allocation_.flush();
}

} // namespace renderer::vulkan
