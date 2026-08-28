#pragma once

#include <volk.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace renderer::vulkan {

class Buffer;
class Device;
class Image;
class MemoryAllocator;

struct BufferUpload {
    Buffer* destination = nullptr;
    VkDeviceSize destinationOffset = 0;
    std::span<const std::byte> data{};
};

struct ImageUpload {
    Image* destination = nullptr;
    std::span<const std::byte> data{};
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    std::uint32_t mipLevel = 0;
    std::uint32_t arrayLayer = 0;
    // Zero selects the full extent of the requested mip level.
    VkExtent3D extent{};
    VkImageLayout finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
};

struct UploadTicket {
    VkSemaphore semaphore = VK_NULL_HANDLE;
    std::uint64_t value = 0;
};

class UploadContext final {
  public:
    UploadContext(const Device& device, MemoryAllocator& allocator);
    ~UploadContext();

    UploadContext(const UploadContext&) = delete;
    UploadContext& operator=(const UploadContext&) = delete;

    [[nodiscard]] UploadTicket upload(std::span<const BufferUpload> bufferUploads,
                                      std::span<const ImageUpload> imageUploads = {});
    void wait(std::uint64_t value) const;

  private:
    const Device& device_;
    MemoryAllocator& allocator_;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    std::uint64_t nextValue_ = 1;
    std::uint64_t lastSubmittedValue_ = 0;
    std::vector<std::unique_ptr<Buffer>> retainedStagingBuffers_;
};

} // namespace renderer::vulkan
