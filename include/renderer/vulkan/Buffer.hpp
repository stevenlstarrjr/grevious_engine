#pragma once

#include "renderer/vulkan/MemoryAllocator.hpp"

#include <volk.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace renderer::vulkan {

class Device;

struct BufferCreateInfo {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    VkMemoryPropertyFlags requiredMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkMemoryPropertyFlags preferredMemory = 0;
    bool randomHostAccess = false;
    std::span<const std::uint32_t> queueFamilies{};
};

class Buffer final {
  public:
    Buffer(const Device& device, MemoryAllocator& allocator, const BufferCreateInfo& createInfo);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) = delete;
    Buffer& operator=(Buffer&&) = delete;

    [[nodiscard]] VkBuffer handle() const { return buffer_; }
    [[nodiscard]] VkDeviceSize size() const { return size_; }
    [[nodiscard]] void* mappedData() const { return allocation_.mappedData(); }
    [[nodiscard]] VkDeviceAddress deviceAddress() const;

    void write(std::span<const std::byte> data, VkDeviceSize offset = 0);
    void invalidate() const { allocation_.invalidate(); }

  private:
    const Device& device_;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    VkBufferUsageFlags usage_ = 0;
    Allocation allocation_;
};

} // namespace renderer::vulkan
