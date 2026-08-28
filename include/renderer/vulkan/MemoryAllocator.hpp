#pragma once

#include <volk.h>

#include <memory>

namespace renderer::vulkan {

class Device;
class MemoryAllocator;

// Renderer-owned allocation handle. VMA remains an implementation detail so
// resource users and future host-language bindings never depend on its types.
class Allocation final {
  public:
    Allocation() = default;
    ~Allocation();

    Allocation(const Allocation&) = delete;
    Allocation& operator=(const Allocation&) = delete;
    Allocation(Allocation&& other) noexcept;
    Allocation& operator=(Allocation&& other) noexcept;

    [[nodiscard]] bool valid() const { return allocation_ != nullptr; }
    [[nodiscard]] VkDeviceMemory memory() const { return memory_; }
    [[nodiscard]] VkDeviceSize offset() const { return offset_; }
    [[nodiscard]] VkDeviceSize size() const { return size_; }
    [[nodiscard]] void* mappedData() const { return mappedData_; }
    [[nodiscard]] VkMemoryPropertyFlags memoryProperties() const { return properties_; }

    void flush() const;
    void invalidate() const;
    void reset();

  private:
    friend class MemoryAllocator;

    MemoryAllocator* allocator_ = nullptr;
    void* allocation_ = nullptr;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize offset_ = 0;
    VkDeviceSize size_ = 0;
    void* mappedData_ = nullptr;
    VkMemoryPropertyFlags properties_ = 0;
};

class MemoryAllocator final {
  public:
    explicit MemoryAllocator(const Device& device);
    ~MemoryAllocator();

    MemoryAllocator(const MemoryAllocator&) = delete;
    MemoryAllocator& operator=(const MemoryAllocator&) = delete;

    [[nodiscard]] Allocation allocateForBuffer(VkBuffer buffer,
                                               VkMemoryPropertyFlags requiredProperties,
                                               VkMemoryPropertyFlags preferredProperties = 0,
                                               bool randomHostAccess = false);
    [[nodiscard]] Allocation allocateForImage(VkImage image,
                                              VkMemoryPropertyFlags requiredProperties,
                                              VkMemoryPropertyFlags preferredProperties = 0);

    void bindBuffer(VkBuffer buffer, const Allocation& allocation) const;
    void bindImage(VkImage image, const Allocation& allocation) const;

  private:
    struct Impl;

    [[nodiscard]] Allocation allocateResource(VkBuffer buffer, VkImage image,
                                              VkMemoryPropertyFlags requiredProperties,
                                              VkMemoryPropertyFlags preferredProperties,
                                              bool randomHostAccess = false);
    void release(Allocation& allocation) noexcept;
    void flush(const Allocation& allocation) const;
    void invalidate(const Allocation& allocation) const;

    friend class Allocation;

    const Device& device_;
    std::unique_ptr<Impl> impl_;
};

} // namespace renderer::vulkan
