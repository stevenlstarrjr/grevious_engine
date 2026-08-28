#pragma once

#include <volk.h>

#include <cstdint>

namespace renderer::vulkan {

class Instance;

struct QueueFamilies {
    std::uint32_t graphics = VK_QUEUE_FAMILY_IGNORED;
    std::uint32_t present = VK_QUEUE_FAMILY_IGNORED;
    std::uint32_t compute = VK_QUEUE_FAMILY_IGNORED;
    std::uint32_t transfer = VK_QUEUE_FAMILY_IGNORED;
};

struct OptionalCapabilities {
    bool accelerationStructure = false;
    bool rayTracingPipeline = false;
    bool rayQuery = false;
    bool meshShader = false;
    std::uint64_t maxAccelerationStructureGeometryCount = 0;
    std::uint64_t maxAccelerationStructurePrimitiveCount = 0;
    std::uint64_t maxAccelerationStructureInstanceCount = 0;
    std::uint32_t minAccelerationStructureScratchAlignment = 0;
    std::uint32_t maxRayRecursionDepth = 0;
    std::uint32_t shaderGroupHandleSize = 0;
    std::uint32_t shaderGroupHandleAlignment = 0;
    std::uint32_t shaderGroupBaseAlignment = 0;
};

class Device final {
  public:
    explicit Device(const Instance& instance);
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    [[nodiscard]] VkPhysicalDevice physicalHandle() const { return physicalDevice_; }
    [[nodiscard]] VkInstance instanceHandle() const { return instanceHandle_; }
    [[nodiscard]] VkDevice handle() const { return device_; }
    [[nodiscard]] VkQueue graphicsQueue() const { return graphicsQueue_; }
    [[nodiscard]] VkQueue presentQueue() const { return presentQueue_; }
    [[nodiscard]] VkQueue computeQueue() const { return computeQueue_; }
    [[nodiscard]] VkQueue transferQueue() const { return transferQueue_; }
    [[nodiscard]] const QueueFamilies& queueFamilies() const { return queueFamilies_; }
    [[nodiscard]] const OptionalCapabilities& supportedOptionalCapabilities() const {
        return supportedOptionalCapabilities_;
    }
    [[nodiscard]] const OptionalCapabilities& optionalCapabilities() const {
        return enabledOptionalCapabilities_;
    }

  private:
    VkInstance instanceHandle_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    QueueFamilies queueFamilies_{};
    OptionalCapabilities supportedOptionalCapabilities_{};
    OptionalCapabilities enabledOptionalCapabilities_{};
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VkQueue computeQueue_ = VK_NULL_HANDLE;
    VkQueue transferQueue_ = VK_NULL_HANDLE;
};

} // namespace renderer::vulkan
