#include "renderer/vulkan/MemoryAllocator.hpp"

#include "renderer/core/Log.hpp"
#include "renderer/vulkan/Device.hpp"
#include "renderer/vulkan/VulkanError.hpp"

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

#include <atomic>
#include <stdexcept>
#include <utility>

namespace renderer::vulkan {

struct MemoryAllocator::Impl {
    VmaAllocator allocator = VK_NULL_HANDLE;
    std::atomic_size_t liveAllocations = 0;
};

namespace {

[[nodiscard]] VmaAllocation asVmaAllocation(void* allocation) {
    return static_cast<VmaAllocation>(allocation);
}

} // namespace

Allocation::~Allocation() { reset(); }

Allocation::Allocation(Allocation&& other) noexcept { *this = std::move(other); }

Allocation& Allocation::operator=(Allocation&& other) noexcept {
    if (this != &other) {
        reset();
        allocator_ = std::exchange(other.allocator_, nullptr);
        allocation_ = std::exchange(other.allocation_, nullptr);
        memory_ = std::exchange(other.memory_, VK_NULL_HANDLE);
        offset_ = std::exchange(other.offset_, 0);
        size_ = std::exchange(other.size_, 0);
        mappedData_ = std::exchange(other.mappedData_, nullptr);
        properties_ = std::exchange(other.properties_, 0);
    }
    return *this;
}

void Allocation::flush() const {
    if (valid() && mappedData_ != nullptr &&
        (properties_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
        allocator_->flush(*this);
    }
}

void Allocation::invalidate() const {
    if (valid() && mappedData_ != nullptr &&
        (properties_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
        allocator_->invalidate(*this);
    }
}

void Allocation::reset() {
    if (allocator_ != nullptr) {
        allocator_->release(*this);
    }
}

MemoryAllocator::MemoryAllocator(const Device& device)
    : device_(device), impl_(std::make_unique<Impl>()) {
    VmaAllocatorCreateInfo allocatorInfo{
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = device_.physicalHandle(),
        .device = device_.handle(),
        .instance = device_.instanceHandle(),
        .vulkanApiVersion = VK_API_VERSION_1_3,
    };
    VmaVulkanFunctions functions{};
    check(vmaImportVulkanFunctionsFromVolk(&allocatorInfo, &functions),
          "vmaImportVulkanFunctionsFromVolk");
    allocatorInfo.pVulkanFunctions = &functions;
    check(vmaCreateAllocator(&allocatorInfo, &impl_->allocator), "vmaCreateAllocator");
}

MemoryAllocator::~MemoryAllocator() {
    const std::size_t liveAllocations = impl_->liveAllocations.load();
    if (liveAllocations != 0) {
        core::log(core::LogLevel::error, "Destroying VMA allocator with ", liveAllocations,
                  " live allocation(s)");
    }
    if (impl_->allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(impl_->allocator);
    }
}

Allocation MemoryAllocator::allocateForBuffer(VkBuffer buffer,
                                              VkMemoryPropertyFlags requiredProperties,
                                              VkMemoryPropertyFlags preferredProperties,
                                              bool randomHostAccess) {
    if (buffer == VK_NULL_HANDLE) {
        throw std::runtime_error("Cannot allocate memory for a null Vulkan buffer");
    }
    return allocateResource(buffer, VK_NULL_HANDLE, requiredProperties, preferredProperties,
                            randomHostAccess);
}

Allocation MemoryAllocator::allocateForImage(VkImage image,
                                             VkMemoryPropertyFlags requiredProperties,
                                             VkMemoryPropertyFlags preferredProperties) {
    if (image == VK_NULL_HANDLE) {
        throw std::runtime_error("Cannot allocate memory for a null Vulkan image");
    }
    return allocateResource(VK_NULL_HANDLE, image, requiredProperties, preferredProperties);
}

Allocation MemoryAllocator::allocateResource(VkBuffer buffer, VkImage image,
                                             VkMemoryPropertyFlags requiredProperties,
                                             VkMemoryPropertyFlags preferredProperties,
                                             bool randomHostAccess) {
    VmaAllocationCreateInfo createInfo{
        .usage = VMA_MEMORY_USAGE_UNKNOWN,
        .requiredFlags = requiredProperties,
        .preferredFlags = preferredProperties,
    };
    if ((requiredProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        createInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                           (randomHostAccess ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                                             : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    }

    VmaAllocation vmaAllocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocationInfo{};
    if (buffer != VK_NULL_HANDLE) {
        check(vmaAllocateMemoryForBuffer(impl_->allocator, buffer, &createInfo, &vmaAllocation,
                                         &allocationInfo),
              "vmaAllocateMemoryForBuffer");
    } else {
        check(vmaAllocateMemoryForImage(impl_->allocator, image, &createInfo, &vmaAllocation,
                                        &allocationInfo),
              "vmaAllocateMemoryForImage");
    }

    VkMemoryPropertyFlags properties = 0;
    vmaGetMemoryTypeProperties(impl_->allocator, allocationInfo.memoryType, &properties);

    Allocation allocation;
    allocation.allocator_ = this;
    allocation.allocation_ = vmaAllocation;
    allocation.memory_ = allocationInfo.deviceMemory;
    allocation.offset_ = allocationInfo.offset;
    allocation.size_ = allocationInfo.size;
    allocation.mappedData_ = allocationInfo.pMappedData;
    allocation.properties_ = properties;
    impl_->liveAllocations.fetch_add(1, std::memory_order_relaxed);
    return allocation;
}

void MemoryAllocator::bindBuffer(VkBuffer buffer, const Allocation& allocation) const {
    if (buffer == VK_NULL_HANDLE || !allocation.valid() || allocation.allocator_ != this) {
        throw std::runtime_error("Invalid buffer allocation binding");
    }
    check(vmaBindBufferMemory(impl_->allocator, asVmaAllocation(allocation.allocation_), buffer),
          "vmaBindBufferMemory");
}

void MemoryAllocator::bindImage(VkImage image, const Allocation& allocation) const {
    if (image == VK_NULL_HANDLE || !allocation.valid() || allocation.allocator_ != this) {
        throw std::runtime_error("Invalid image allocation binding");
    }
    check(vmaBindImageMemory(impl_->allocator, asVmaAllocation(allocation.allocation_), image),
          "vmaBindImageMemory");
}

void MemoryAllocator::release(Allocation& allocation) noexcept {
    vmaFreeMemory(impl_->allocator, asVmaAllocation(allocation.allocation_));
    impl_->liveAllocations.fetch_sub(1, std::memory_order_relaxed);

    allocation.allocator_ = nullptr;
    allocation.allocation_ = nullptr;
    allocation.memory_ = VK_NULL_HANDLE;
    allocation.offset_ = 0;
    allocation.size_ = 0;
    allocation.mappedData_ = nullptr;
    allocation.properties_ = 0;
}

void MemoryAllocator::flush(const Allocation& allocation) const {
    check(vmaFlushAllocation(impl_->allocator, asVmaAllocation(allocation.allocation_), 0,
                             VK_WHOLE_SIZE),
          "vmaFlushAllocation");
}

void MemoryAllocator::invalidate(const Allocation& allocation) const {
    check(vmaInvalidateAllocation(impl_->allocator, asVmaAllocation(allocation.allocation_), 0,
                                  VK_WHOLE_SIZE),
          "vmaInvalidateAllocation");
}

} // namespace renderer::vulkan
