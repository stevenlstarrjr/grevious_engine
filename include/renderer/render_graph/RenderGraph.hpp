#pragma once

#include <volk.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace renderer::vulkan {
class Device;
class MemoryAllocator;
struct BufferCreateInfo;
struct ImageCreateInfo;
struct ImageViewCreateInfo;
} // namespace renderer::vulkan

namespace renderer::render_graph {

enum class QueueType { graphics, compute, transfer };

struct QueueFamilyMapping {
    std::uint32_t graphics = VK_QUEUE_FAMILY_IGNORED;
    std::uint32_t compute = VK_QUEUE_FAMILY_IGNORED;
    std::uint32_t transfer = VK_QUEUE_FAMILY_IGNORED;
};

struct ImageHandle {
    static constexpr std::uint32_t invalid = ~std::uint32_t{0};
    std::uint32_t index = invalid;

    [[nodiscard]] explicit operator bool() const { return index != invalid; }
    friend bool operator==(ImageHandle, ImageHandle) = default;
};

struct BufferHandle {
    static constexpr std::uint32_t invalid = ~std::uint32_t{0};
    std::uint32_t index = invalid;

    [[nodiscard]] explicit operator bool() const { return index != invalid; }
    friend bool operator==(BufferHandle, BufferHandle) = default;
};

struct PassHandle {
    static constexpr std::uint32_t invalid = ~std::uint32_t{0};
    std::uint32_t index = invalid;

    [[nodiscard]] explicit operator bool() const { return index != invalid; }
    friend bool operator==(PassHandle, PassHandle) = default;
};

struct ImageState {
    VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 access = VK_ACCESS_2_NONE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

struct BufferState {
    VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 access = VK_ACCESS_2_NONE;
};

struct ResourceLifetime {
    bool used = false;
    std::size_t firstPass = 0;
    std::size_t lastPass = 0;
};

class RenderGraph final {
  public:
    using PassCallback =
        std::function<void(VkCommandBuffer, const RenderGraph&, const void* userData)>;

    explicit RenderGraph(QueueFamilyMapping queueFamilies);
    RenderGraph(const vulkan::Device& device, vulkan::MemoryAllocator& allocator);
    ~RenderGraph();

    RenderGraph(const RenderGraph&) = delete;
    RenderGraph& operator=(const RenderGraph&) = delete;
    RenderGraph(RenderGraph&&) = delete;
    RenderGraph& operator=(RenderGraph&&) = delete;

    [[nodiscard]] ImageHandle importImage(std::string name, VkImage image,
                                          VkImageSubresourceRange subresource,
                                          ImageState initialState, QueueType initialQueue,
                                          bool concurrentSharing = false);
    [[nodiscard]] BufferHandle importBuffer(std::string name, VkBuffer buffer, VkDeviceSize offset,
                                            VkDeviceSize size, BufferState initialState,
                                            QueueType initialQueue, bool concurrentSharing = false);
    [[nodiscard]] ImageHandle createTransientImage(std::string name,
                                                   const vulkan::ImageCreateInfo& createInfo,
                                                   const vulkan::ImageViewCreateInfo& viewInfo);
    [[nodiscard]] BufferHandle createTransientBuffer(std::string name,
                                                     const vulkan::BufferCreateInfo& createInfo);

    void bindImportedImage(ImageHandle handle, VkImage image, ImageState initialState,
                           QueueType initialQueue);
    void bindImportedBuffer(BufferHandle handle, VkBuffer buffer, VkDeviceSize offset,
                            VkDeviceSize size, BufferState initialState, QueueType initialQueue);

    [[nodiscard]] PassHandle addPass(std::string name, QueueType queue, PassCallback callback = {});
    void addDependency(PassHandle before, PassHandle after);
    void readImage(PassHandle pass, ImageHandle image, ImageState state);
    void writeImage(PassHandle pass, ImageHandle image, ImageState state);
    void readWriteImage(PassHandle pass, ImageHandle image, ImageState state);
    void readBuffer(PassHandle pass, BufferHandle buffer, BufferState state);
    void writeBuffer(PassHandle pass, BufferHandle buffer, BufferState state);
    void readWriteBuffer(PassHandle pass, BufferHandle buffer, BufferState state);

    void compile();
    void execute(VkCommandBuffer commandBuffer, QueueType commandQueue,
                 const void* userData = nullptr) const;

    [[nodiscard]] VkImage image(ImageHandle handle) const;
    [[nodiscard]] VkImageView imageView(ImageHandle handle) const;
    [[nodiscard]] VkBuffer buffer(BufferHandle handle) const;
    [[nodiscard]] ResourceLifetime lifetime(ImageHandle handle) const;
    [[nodiscard]] ResourceLifetime lifetime(BufferHandle handle) const;
    [[nodiscard]] std::span<const std::string> orderedPassNames() const;
    [[nodiscard]] bool compiled() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace renderer::render_graph
