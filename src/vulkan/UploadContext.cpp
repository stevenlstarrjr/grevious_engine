#include "renderer/vulkan/UploadContext.hpp"

#include "renderer/vulkan/Buffer.hpp"
#include "renderer/vulkan/Device.hpp"
#include "renderer/vulkan/Image.hpp"
#include "renderer/vulkan/MemoryAllocator.hpp"
#include "renderer/vulkan/VulkanError.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

namespace renderer::vulkan {
namespace {

VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

VkDeviceSize texelBlockByteSize(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
        return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32_SFLOAT:
        return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return 16;
    default:
        throw std::runtime_error("UploadContext does not know the image format's texel block size");
    }
}

VkDeviceSize imageOffsetAlignment(const ImageUpload& upload) {
    return std::max<VkDeviceSize>(4, texelBlockByteSize(upload.destination->format()));
}

} // namespace

UploadContext::UploadContext(const Device& device, MemoryAllocator& allocator)
    : device_(device), allocator_(allocator) {
    const VkCommandPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = device_.queueFamilies().transfer,
    };
    check(vkCreateCommandPool(device_.handle(), &poolInfo, nullptr, &commandPool_),
          "vkCreateCommandPool(upload)");

    const VkSemaphoreTypeCreateInfo timelineType{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0,
    };
    const VkSemaphoreCreateInfo semaphoreInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &timelineType,
    };
    try {
        check(vkCreateSemaphore(device_.handle(), &semaphoreInfo, nullptr, &timeline_),
              "vkCreateSemaphore(upload timeline)");
    } catch (...) {
        vkDestroyCommandPool(device_.handle(), commandPool_, nullptr);
        commandPool_ = VK_NULL_HANDLE;
        throw;
    }
}

UploadContext::~UploadContext() {
    if (lastSubmittedValue_ != 0) {
        const VkSemaphoreWaitInfo waitInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .semaphoreCount = 1,
            .pSemaphores = &timeline_,
            .pValues = &lastSubmittedValue_,
        };
        (void)vkWaitSemaphores(device_.handle(), &waitInfo,
                               std::numeric_limits<std::uint64_t>::max());
    }
    retainedStagingBuffers_.clear();
    if (timeline_ != VK_NULL_HANDLE) {
        vkDestroySemaphore(device_.handle(), timeline_, nullptr);
    }
    if (commandPool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_.handle(), commandPool_, nullptr);
    }
}

UploadTicket UploadContext::upload(std::span<const BufferUpload> bufferUploads,
                                   std::span<const ImageUpload> imageUploads) {
    if (bufferUploads.empty() && imageUploads.empty()) {
        throw std::runtime_error("Upload batch cannot be empty");
    }
    // One outstanding batch keeps command-pool reuse explicit. The API still
    // submits asynchronously; later versions can grow this into a staging ring.
    if (lastSubmittedValue_ != 0) {
        wait(lastSubmittedValue_);
        retainedStagingBuffers_.clear();
        check(vkResetCommandPool(device_.handle(), commandPool_, 0), "vkResetCommandPool(upload)");
    }

    VkDeviceSize totalSize = 0;
    for (const BufferUpload& upload : bufferUploads) {
        if (upload.destination == nullptr || upload.data.empty() ||
            upload.destinationOffset > upload.destination->size() ||
            upload.data.size() > upload.destination->size() - upload.destinationOffset) {
            throw std::runtime_error("Invalid buffer upload range");
        }
        totalSize = alignUp(totalSize, 4);
        totalSize += upload.data.size();
    }
    for (const ImageUpload& upload : imageUploads) {
        if (upload.destination == nullptr || upload.data.empty() ||
            upload.mipLevel >= upload.destination->mipLevels() ||
            upload.arrayLayer >= upload.destination->arrayLayers()) {
            throw std::runtime_error("Invalid image upload range");
        }
        totalSize = alignUp(totalSize, imageOffsetAlignment(upload));
        totalSize += upload.data.size();
    }

    auto staging =
        std::make_unique<Buffer>(device_, allocator_,
                                 BufferCreateInfo{
                                     .size = totalSize,
                                     .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                     .requiredMemory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                     .preferredMemory = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                 });

    std::vector<VkBufferCopy> bufferRegions;
    bufferRegions.reserve(bufferUploads.size());
    std::vector<VkDeviceSize> imageOffsets;
    imageOffsets.reserve(imageUploads.size());
    VkDeviceSize sourceOffset = 0;
    for (const BufferUpload& upload : bufferUploads) {
        sourceOffset = alignUp(sourceOffset, 4);
        staging->write(upload.data, sourceOffset);
        bufferRegions.push_back({
            .srcOffset = sourceOffset,
            .dstOffset = upload.destinationOffset,
            .size = upload.data.size(),
        });
        sourceOffset += upload.data.size();
    }
    for (const ImageUpload& upload : imageUploads) {
        sourceOffset = alignUp(sourceOffset, imageOffsetAlignment(upload));
        staging->write(upload.data, sourceOffset);
        imageOffsets.push_back(sourceOffset);
        sourceOffset += upload.data.size();
    }

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    const VkCommandBufferAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = commandPool_,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    check(vkAllocateCommandBuffers(device_.handle(), &allocateInfo, &commandBuffer),
          "vkAllocateCommandBuffers(upload)");
    const VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    check(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer(upload)");
    for (std::size_t index = 0; index < bufferUploads.size(); ++index) {
        vkCmdCopyBuffer(commandBuffer, staging->handle(),
                        bufferUploads[index].destination->handle(), 1, &bufferRegions[index]);
    }
    for (std::size_t index = 0; index < imageUploads.size(); ++index) {
        const ImageUpload& upload = imageUploads[index];
        const VkExtent3D baseExtent = upload.destination->extent();
        const VkExtent3D mipExtent{
            .width = std::max(1U, baseExtent.width >> upload.mipLevel),
            .height = std::max(1U, baseExtent.height >> upload.mipLevel),
            .depth = std::max(1U, baseExtent.depth >> upload.mipLevel),
        };
        const VkExtent3D copyExtent =
            upload.extent.width == 0 && upload.extent.height == 0 && upload.extent.depth == 0
                ? mipExtent
                : upload.extent;
        if (copyExtent.width == 0 || copyExtent.height == 0 || copyExtent.depth == 0 ||
            copyExtent.width > mipExtent.width || copyExtent.height > mipExtent.height ||
            copyExtent.depth > mipExtent.depth) {
            throw std::runtime_error("Image upload extent exceeds its mip level");
        }
        const VkImageMemoryBarrier2 toTransfer{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
            .srcAccessMask = VK_ACCESS_2_NONE,
            .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = upload.destination->handle(),
            .subresourceRange =
                {
                    .aspectMask = upload.aspect,
                    .baseMipLevel = upload.mipLevel,
                    .levelCount = 1,
                    .baseArrayLayer = upload.arrayLayer,
                    .layerCount = 1,
                },
        };
        const VkDependencyInfo toTransferDependency{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &toTransfer,
        };
        vkCmdPipelineBarrier2(commandBuffer, &toTransferDependency);

        const VkBufferImageCopy copy{
            .bufferOffset = imageOffsets[index],
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource =
                {
                    .aspectMask = upload.aspect,
                    .mipLevel = upload.mipLevel,
                    .baseArrayLayer = upload.arrayLayer,
                    .layerCount = 1,
                },
            .imageOffset = {0, 0, 0},
            .imageExtent = copyExtent,
        };
        vkCmdCopyBufferToImage(commandBuffer, staging->handle(), upload.destination->handle(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        const VkImageMemoryBarrier2 toFinal{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_NONE,
            .dstAccessMask = VK_ACCESS_2_NONE,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = upload.finalLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = upload.destination->handle(),
            .subresourceRange = toTransfer.subresourceRange,
        };
        const VkDependencyInfo toFinalDependency{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &toFinal,
        };
        vkCmdPipelineBarrier2(commandBuffer, &toFinalDependency);
    }
    check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer(upload)");

    const std::uint64_t signalValue = nextValue_++;
    const VkCommandBufferSubmitInfo commandInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = commandBuffer,
    };
    const VkSemaphoreSubmitInfo signalInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = timeline_,
        .value = signalValue,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
    };
    const VkSubmitInfo2 submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signalInfo,
    };
    check(vkQueueSubmit2(device_.transferQueue(), 1, &submitInfo, VK_NULL_HANDLE),
          "vkQueueSubmit2(upload)");

    lastSubmittedValue_ = signalValue;
    retainedStagingBuffers_.push_back(std::move(staging));
    return {.semaphore = timeline_, .value = signalValue};
}

void UploadContext::wait(std::uint64_t value) const {
    const VkSemaphoreWaitInfo waitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &timeline_,
        .pValues = &value,
    };
    check(vkWaitSemaphores(device_.handle(), &waitInfo, std::numeric_limits<std::uint64_t>::max()),
          "vkWaitSemaphores(upload)");
}

} // namespace renderer::vulkan
