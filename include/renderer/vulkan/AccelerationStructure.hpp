#pragma once

#include "renderer/vulkan/Buffer.hpp"

#include <volk.h>

#include <cstdint>

namespace renderer::vulkan {

class Device;
class MemoryAllocator;

struct AccelerationStructureCreateInfo {
    VkAccelerationStructureTypeKHR type = VK_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_KHR;
    VkDeviceSize size = 0;
};

// Owns both the Vulkan acceleration-structure object and its backing buffer.
// Build scratch is deliberately external and short-lived.
class AccelerationStructure final {
  public:
    AccelerationStructure(const Device& device, MemoryAllocator& allocator,
                          const AccelerationStructureCreateInfo& createInfo);
    ~AccelerationStructure();

    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;
    AccelerationStructure(AccelerationStructure&&) = delete;
    AccelerationStructure& operator=(AccelerationStructure&&) = delete;

    [[nodiscard]] VkAccelerationStructureKHR handle() const { return handle_; }
    [[nodiscard]] VkAccelerationStructureTypeKHR type() const { return type_; }
    [[nodiscard]] VkDeviceSize size() const { return storage_.size(); }
    [[nodiscard]] VkDeviceAddress deviceAddress() const;

  private:
    const Device& device_;
    Buffer storage_;
    VkAccelerationStructureKHR handle_ = VK_NULL_HANDLE;
    VkAccelerationStructureTypeKHR type_ = VK_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_KHR;
};

} // namespace renderer::vulkan
