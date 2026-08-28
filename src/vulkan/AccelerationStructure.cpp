#include "renderer/vulkan/AccelerationStructure.hpp"

#include "renderer/vulkan/Device.hpp"
#include "renderer/vulkan/VulkanError.hpp"

#include <stdexcept>

namespace renderer::vulkan {
namespace {

BufferCreateInfo storageBufferInfo(const Device& device,
                                   const AccelerationStructureCreateInfo& createInfo) {
    if (!device.optionalCapabilities().accelerationStructure ||
        vkCreateAccelerationStructureKHR == nullptr || createInfo.size == 0U ||
        createInfo.type == VK_ACCELERATION_STRUCTURE_TYPE_MAX_ENUM_KHR) {
        throw std::runtime_error("Acceleration-structure creation is unavailable or invalid");
    }
    return {
        .size = createInfo.size,
        .usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .requiredMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };
}

} // namespace

AccelerationStructure::AccelerationStructure(const Device& device, MemoryAllocator& allocator,
                                             const AccelerationStructureCreateInfo& createInfo)
    : device_(device), storage_(device, allocator, storageBufferInfo(device, createInfo)),
      type_(createInfo.type) {
    const VkAccelerationStructureCreateInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .buffer = storage_.handle(),
        .offset = 0,
        .size = storage_.size(),
        .type = type_,
    };
    check(vkCreateAccelerationStructureKHR(device_.handle(), &info, nullptr, &handle_),
          "vkCreateAccelerationStructureKHR");
}

AccelerationStructure::~AccelerationStructure() {
    if (handle_ != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructureKHR(device_.handle(), handle_, nullptr);
    }
}

VkDeviceAddress AccelerationStructure::deviceAddress() const {
    if (handle_ == VK_NULL_HANDLE || vkGetAccelerationStructureDeviceAddressKHR == nullptr) {
        throw std::runtime_error("Acceleration-structure device address is unavailable");
    }
    const VkAccelerationStructureDeviceAddressInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
        .accelerationStructure = handle_,
    };
    const VkDeviceAddress address =
        vkGetAccelerationStructureDeviceAddressKHR(device_.handle(), &info);
    if (address == 0U) {
        throw std::runtime_error("Vulkan returned a null acceleration-structure device address");
    }
    return address;
}

} // namespace renderer::vulkan
