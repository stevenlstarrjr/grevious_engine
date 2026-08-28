#include "renderer/render_graph/RenderGraph.hpp"

#include "renderer/vulkan/Buffer.hpp"
#include "renderer/vulkan/Device.hpp"
#include "renderer/vulkan/Image.hpp"
#include "renderer/vulkan/MemoryAllocator.hpp"

#include <algorithm>
#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace renderer::render_graph {
namespace {

enum class AccessMode { read, write, readWrite };

[[nodiscard]] bool writes(AccessMode mode) { return mode != AccessMode::read; }

[[nodiscard]] bool accessContainsWrite(VkAccessFlags2 access) {
    constexpr VkAccessFlags2 writeAccess =
        VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT |
        VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT |
        VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    return (access & writeAccess) != 0;
}

[[nodiscard]] std::uint32_t queueFamily(QueueFamilyMapping mapping, QueueType queue) {
    switch (queue) {
    case QueueType::graphics:
        return mapping.graphics;
    case QueueType::compute:
        return mapping.compute;
    case QueueType::transfer:
        return mapping.transfer;
    }
    throw std::runtime_error("Unknown render-graph queue type");
}

[[nodiscard]] QueueFamilyMapping mappingForDevice(const vulkan::Device& device) {
    const auto& families = device.queueFamilies();
    return {
        .graphics = families.graphics,
        .compute = families.compute,
        .transfer = families.transfer,
    };
}

} // namespace

struct RenderGraph::Impl {
    struct ImageResource {
        std::string name;
        VkImage image = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkImageSubresourceRange subresource{};
        ImageState initialState{};
        QueueType initialQueue = QueueType::graphics;
        bool hasInitialQueue = true;
        bool imported = true;
        bool concurrentSharing = false;
        bool hasExecutionState = false;
        ResourceLifetime lifetime{};
        std::unique_ptr<vulkan::Image> ownedImage;
        std::unique_ptr<vulkan::ImageView> ownedView;
    };

    struct BufferResource {
        std::string name;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
        BufferState initialState{};
        QueueType initialQueue = QueueType::graphics;
        bool hasInitialQueue = true;
        bool imported = true;
        bool concurrentSharing = false;
        bool hasExecutionState = false;
        ResourceLifetime lifetime{};
        std::unique_ptr<vulkan::Buffer> ownedBuffer;
    };

    struct ImageUse {
        ImageHandle resource;
        ImageState state;
        AccessMode mode = AccessMode::read;
    };

    struct BufferUse {
        BufferHandle resource;
        BufferState state;
        AccessMode mode = AccessMode::read;
    };

    struct Pass {
        std::string name;
        QueueType queue = QueueType::graphics;
        PassCallback callback;
        std::vector<PassHandle> dependencies;
        std::vector<ImageUse> images;
        std::vector<BufferUse> buffers;
    };

    struct TrackedImage {
        ImageState state{};
        QueueType queue = QueueType::graphics;
        bool hasQueue = false;
        bool hasAccess = false;
        bool previousWrite = false;
    };

    struct TrackedBuffer {
        BufferState state{};
        QueueType queue = QueueType::graphics;
        bool hasQueue = false;
        bool hasAccess = false;
        bool previousWrite = false;
    };

    QueueFamilyMapping queueFamilies{};
    const vulkan::Device* device = nullptr;
    vulkan::MemoryAllocator* allocator = nullptr;
    std::vector<ImageResource> images;
    std::vector<BufferResource> buffers;
    std::vector<Pass> passes;
    std::vector<std::uint32_t> order;
    std::vector<std::string> orderedNames;
    mutable std::vector<TrackedImage> trackedImages;
    mutable std::vector<TrackedBuffer> trackedBuffers;
    mutable std::vector<VkImageMemoryBarrier2> imageBarrierScratch;
    mutable std::vector<VkBufferMemoryBarrier2> bufferBarrierScratch;
    bool isCompiled = false;

    void requireMutable() const {
        if (isCompiled) {
            throw std::runtime_error("Cannot mutate a compiled render graph");
        }
    }

    [[nodiscard]] ImageResource& checked(ImageHandle handle) {
        if (!handle || handle.index >= images.size()) {
            throw std::runtime_error("Invalid render-graph image handle");
        }
        return images[handle.index];
    }

    [[nodiscard]] const ImageResource& checked(ImageHandle handle) const {
        if (!handle || handle.index >= images.size()) {
            throw std::runtime_error("Invalid render-graph image handle");
        }
        return images[handle.index];
    }

    [[nodiscard]] BufferResource& checked(BufferHandle handle) {
        if (!handle || handle.index >= buffers.size()) {
            throw std::runtime_error("Invalid render-graph buffer handle");
        }
        return buffers[handle.index];
    }

    [[nodiscard]] const BufferResource& checked(BufferHandle handle) const {
        if (!handle || handle.index >= buffers.size()) {
            throw std::runtime_error("Invalid render-graph buffer handle");
        }
        return buffers[handle.index];
    }

    [[nodiscard]] Pass& checked(PassHandle handle) {
        if (!handle || handle.index >= passes.size()) {
            throw std::runtime_error("Invalid render-graph pass handle");
        }
        return passes[handle.index];
    }

    [[nodiscard]] const Pass& checked(PassHandle handle) const {
        if (!handle || handle.index >= passes.size()) {
            throw std::runtime_error("Invalid render-graph pass handle");
        }
        return passes[handle.index];
    }

    void addImageUse(PassHandle passHandle, ImageHandle imageHandle, ImageState state,
                     AccessMode mode) {
        requireMutable();
        (void)checked(imageHandle);
        Pass& pass = checked(passHandle);
        if (std::ranges::any_of(pass.images, [imageHandle](const ImageUse& use) {
                return use.resource == imageHandle;
            })) {
            throw std::runtime_error("A pass may declare each image only once");
        }
        if (state.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
            throw std::runtime_error("A render-graph image use requires a defined layout");
        }
        if (writes(mode) != accessContainsWrite(state.access)) {
            throw std::runtime_error(
                "Render-graph image access flags disagree with the declared access mode");
        }
        pass.images.push_back({.resource = imageHandle, .state = state, .mode = mode});
    }

    void addBufferUse(PassHandle passHandle, BufferHandle bufferHandle, BufferState state,
                      AccessMode mode) {
        requireMutable();
        (void)checked(bufferHandle);
        Pass& pass = checked(passHandle);
        if (std::ranges::any_of(pass.buffers, [bufferHandle](const BufferUse& use) {
                return use.resource == bufferHandle;
            })) {
            throw std::runtime_error("A pass may declare each buffer only once");
        }
        if (writes(mode) != accessContainsWrite(state.access)) {
            throw std::runtime_error(
                "Render-graph buffer access flags disagree with the declared access mode");
        }
        pass.buffers.push_back({.resource = bufferHandle, .state = state, .mode = mode});
    }
};

RenderGraph::RenderGraph(QueueFamilyMapping queueFamilies) : impl_(std::make_unique<Impl>()) {
    impl_->queueFamilies = queueFamilies;
}

RenderGraph::RenderGraph(const vulkan::Device& device, vulkan::MemoryAllocator& allocator)
    : RenderGraph(mappingForDevice(device)) {
    impl_->device = &device;
    impl_->allocator = &allocator;
}

RenderGraph::~RenderGraph() = default;

ImageHandle RenderGraph::importImage(std::string name, VkImage image,
                                     VkImageSubresourceRange subresource, ImageState initialState,
                                     QueueType initialQueue, bool concurrentSharing) {
    impl_->requireMutable();
    if (name.empty() || subresource.aspectMask == 0 || subresource.levelCount == 0 ||
        subresource.layerCount == 0) {
        throw std::runtime_error("Imported render-graph image metadata is incomplete");
    }
    if (std::ranges::any_of(impl_->images, [&name](const Impl::ImageResource& value) {
            return value.name == name;
        })) {
        throw std::runtime_error("Duplicate render-graph image name: " + name);
    }
    const ImageHandle handle{.index = static_cast<std::uint32_t>(impl_->images.size())};
    impl_->images.push_back({
        .name = std::move(name),
        .image = image,
        .subresource = subresource,
        .initialState = initialState,
        .initialQueue = initialQueue,
        .concurrentSharing = concurrentSharing,
    });
    return handle;
}

BufferHandle RenderGraph::importBuffer(std::string name, VkBuffer buffer, VkDeviceSize offset,
                                       VkDeviceSize size, BufferState initialState,
                                       QueueType initialQueue, bool concurrentSharing) {
    impl_->requireMutable();
    if (name.empty() || size == 0) {
        throw std::runtime_error("Imported render-graph buffer metadata is incomplete");
    }
    if (std::ranges::any_of(impl_->buffers, [&name](const Impl::BufferResource& value) {
            return value.name == name;
        })) {
        throw std::runtime_error("Duplicate render-graph buffer name: " + name);
    }
    const BufferHandle handle{.index = static_cast<std::uint32_t>(impl_->buffers.size())};
    impl_->buffers.push_back({
        .name = std::move(name),
        .buffer = buffer,
        .offset = offset,
        .size = size,
        .initialState = initialState,
        .initialQueue = initialQueue,
        .concurrentSharing = concurrentSharing,
    });
    return handle;
}

ImageHandle RenderGraph::createTransientImage(std::string name,
                                              const vulkan::ImageCreateInfo& createInfo,
                                              const vulkan::ImageViewCreateInfo& viewInfo) {
    impl_->requireMutable();
    if (impl_->device == nullptr || impl_->allocator == nullptr) {
        throw std::runtime_error("Transient images require a device-backed render graph");
    }
    auto ownedImage =
        std::make_unique<vulkan::Image>(*impl_->device, *impl_->allocator, createInfo);
    auto ownedView = std::make_unique<vulkan::ImageView>(*impl_->device, *ownedImage, viewInfo);
    const VkImageSubresourceRange subresource{
        .aspectMask = viewInfo.aspect,
        .baseMipLevel = viewInfo.baseMipLevel,
        .levelCount = viewInfo.levelCount == VK_REMAINING_MIP_LEVELS
                          ? ownedImage->mipLevels() - viewInfo.baseMipLevel
                          : viewInfo.levelCount,
        .baseArrayLayer = viewInfo.baseArrayLayer,
        .layerCount = viewInfo.layerCount == VK_REMAINING_ARRAY_LAYERS
                          ? ownedImage->arrayLayers() - viewInfo.baseArrayLayer
                          : viewInfo.layerCount,
    };
    const ImageHandle handle =
        importImage(std::move(name), ownedImage->handle(), subresource, {}, QueueType::graphics,
                    createInfo.queueFamilies.size() > 1);
    Impl::ImageResource& resource = impl_->checked(handle);
    resource.view = ownedView->handle();
    resource.imported = false;
    resource.hasInitialQueue = false;
    resource.ownedImage = std::move(ownedImage);
    resource.ownedView = std::move(ownedView);
    return handle;
}

BufferHandle RenderGraph::createTransientBuffer(std::string name,
                                                const vulkan::BufferCreateInfo& createInfo) {
    impl_->requireMutable();
    if (impl_->device == nullptr || impl_->allocator == nullptr) {
        throw std::runtime_error("Transient buffers require a device-backed render graph");
    }
    auto ownedBuffer =
        std::make_unique<vulkan::Buffer>(*impl_->device, *impl_->allocator, createInfo);
    const BufferHandle handle =
        importBuffer(std::move(name), ownedBuffer->handle(), 0, ownedBuffer->size(), {},
                     QueueType::graphics, createInfo.queueFamilies.size() > 1);
    Impl::BufferResource& resource = impl_->checked(handle);
    resource.imported = false;
    resource.hasInitialQueue = false;
    resource.ownedBuffer = std::move(ownedBuffer);
    return handle;
}

void RenderGraph::bindImportedImage(ImageHandle handle, VkImage image, ImageState initialState,
                                    QueueType initialQueue) {
    Impl::ImageResource& resource = impl_->checked(handle);
    if (!resource.imported) {
        throw std::runtime_error("Cannot bind a graph-owned transient image");
    }
    resource.image = image;
    resource.initialState = initialState;
    resource.initialQueue = initialQueue;
    resource.hasInitialQueue = true;
}

void RenderGraph::bindImportedBuffer(BufferHandle handle, VkBuffer buffer, VkDeviceSize offset,
                                     VkDeviceSize size, BufferState initialState,
                                     QueueType initialQueue) {
    Impl::BufferResource& resource = impl_->checked(handle);
    if (!resource.imported || size == 0) {
        throw std::runtime_error("Cannot bind the render-graph buffer");
    }
    resource.buffer = buffer;
    resource.offset = offset;
    resource.size = size;
    resource.initialState = initialState;
    resource.initialQueue = initialQueue;
    resource.hasInitialQueue = true;
}

PassHandle RenderGraph::addPass(std::string name, QueueType queue, PassCallback callback) {
    impl_->requireMutable();
    if (name.empty()) {
        throw std::runtime_error("Render-graph pass name cannot be empty");
    }
    if (std::ranges::any_of(impl_->passes,
                            [&name](const Impl::Pass& value) { return value.name == name; })) {
        throw std::runtime_error("Duplicate render-graph pass name: " + name);
    }
    const PassHandle handle{.index = static_cast<std::uint32_t>(impl_->passes.size())};
    impl_->passes.push_back(
        {.name = std::move(name), .queue = queue, .callback = std::move(callback)});
    return handle;
}

void RenderGraph::addDependency(PassHandle before, PassHandle after) {
    impl_->requireMutable();
    (void)impl_->checked(before);
    Impl::Pass& destination = impl_->checked(after);
    if (before == after) {
        throw std::runtime_error("A render-graph pass cannot depend on itself");
    }
    if (std::ranges::find(destination.dependencies, before) == destination.dependencies.end()) {
        destination.dependencies.push_back(before);
    }
}

void RenderGraph::readImage(PassHandle pass, ImageHandle image, ImageState state) {
    impl_->addImageUse(pass, image, state, AccessMode::read);
}

void RenderGraph::writeImage(PassHandle pass, ImageHandle image, ImageState state) {
    impl_->addImageUse(pass, image, state, AccessMode::write);
}

void RenderGraph::readWriteImage(PassHandle pass, ImageHandle image, ImageState state) {
    impl_->addImageUse(pass, image, state, AccessMode::readWrite);
}

void RenderGraph::readBuffer(PassHandle pass, BufferHandle buffer, BufferState state) {
    impl_->addBufferUse(pass, buffer, state, AccessMode::read);
}

void RenderGraph::writeBuffer(PassHandle pass, BufferHandle buffer, BufferState state) {
    impl_->addBufferUse(pass, buffer, state, AccessMode::write);
}

void RenderGraph::readWriteBuffer(PassHandle pass, BufferHandle buffer, BufferState state) {
    impl_->addBufferUse(pass, buffer, state, AccessMode::readWrite);
}

void RenderGraph::compile() {
    impl_->requireMutable();
    if (impl_->passes.empty()) {
        throw std::runtime_error("Cannot compile an empty render graph");
    }

    const std::size_t passCount = impl_->passes.size();
    std::vector<std::unordered_set<std::uint32_t>> edges(passCount);
    std::vector<std::uint32_t> indegrees(passCount, 0);
    const auto addEdge = [&](std::uint32_t before, std::uint32_t after) {
        if (before != after && edges[before].insert(after).second) {
            ++indegrees[after];
        }
    };

    for (std::uint32_t passIndex = 0; passIndex < passCount; ++passIndex) {
        for (PassHandle dependency : impl_->passes[passIndex].dependencies) {
            addEdge(dependency.index, passIndex);
        }
    }

    const auto addHazards = [&](std::size_t resourceCount, auto usesForPass) {
        for (std::uint32_t resourceIndex = 0; resourceIndex < resourceCount; ++resourceIndex) {
            std::optional<std::uint32_t> lastWriter;
            std::vector<std::uint32_t> readers;
            for (std::uint32_t passIndex = 0; passIndex < passCount; ++passIndex) {
                const std::optional<AccessMode> mode = usesForPass(passIndex, resourceIndex);
                if (!mode.has_value()) {
                    continue;
                }
                if (writes(*mode)) {
                    if (lastWriter.has_value()) {
                        addEdge(*lastWriter, passIndex);
                    }
                    for (std::uint32_t reader : readers) {
                        addEdge(reader, passIndex);
                    }
                    readers.clear();
                    lastWriter = passIndex;
                } else {
                    if (lastWriter.has_value()) {
                        addEdge(*lastWriter, passIndex);
                    }
                    readers.push_back(passIndex);
                }
            }
        }
    };

    addHazards(
        impl_->images.size(),
        [&](std::uint32_t passIndex, std::uint32_t resourceIndex) -> std::optional<AccessMode> {
            const auto& uses = impl_->passes[passIndex].images;
            const auto found =
                std::ranges::find_if(uses, [resourceIndex](const Impl::ImageUse& use) {
                    return use.resource.index == resourceIndex;
                });
            return found == uses.end() ? std::nullopt : std::optional(found->mode);
        });
    addHazards(
        impl_->buffers.size(),
        [&](std::uint32_t passIndex, std::uint32_t resourceIndex) -> std::optional<AccessMode> {
            const auto& uses = impl_->passes[passIndex].buffers;
            const auto found =
                std::ranges::find_if(uses, [resourceIndex](const Impl::BufferUse& use) {
                    return use.resource.index == resourceIndex;
                });
            return found == uses.end() ? std::nullopt : std::optional(found->mode);
        });

    std::priority_queue<std::uint32_t, std::vector<std::uint32_t>, std::greater<>> ready;
    for (std::uint32_t passIndex = 0; passIndex < passCount; ++passIndex) {
        if (indegrees[passIndex] == 0) {
            ready.push(passIndex);
        }
    }
    impl_->order.clear();
    while (!ready.empty()) {
        const std::uint32_t passIndex = ready.top();
        ready.pop();
        impl_->order.push_back(passIndex);
        for (std::uint32_t successor : edges[passIndex]) {
            if (--indegrees[successor] == 0) {
                ready.push(successor);
            }
        }
    }
    if (impl_->order.size() != passCount) {
        throw std::runtime_error("Render graph contains a dependency cycle");
    }

    const auto updateLifetime = [](ResourceLifetime& lifetime, std::size_t passIndex) {
        if (!lifetime.used) {
            lifetime = {.used = true, .firstPass = passIndex, .lastPass = passIndex};
        } else {
            lifetime.lastPass = passIndex;
        }
    };
    for (auto& resource : impl_->images) {
        resource.lifetime = {};
    }
    for (auto& resource : impl_->buffers) {
        resource.lifetime = {};
    }
    impl_->orderedNames.clear();
    impl_->orderedNames.reserve(passCount);
    std::size_t maximumImageUses = 0;
    std::size_t maximumBufferUses = 0;
    for (std::size_t orderedIndex = 0; orderedIndex < impl_->order.size(); ++orderedIndex) {
        const Impl::Pass& pass = impl_->passes[impl_->order[orderedIndex]];
        impl_->orderedNames.push_back(pass.name);
        maximumImageUses = std::max(maximumImageUses, pass.images.size());
        maximumBufferUses = std::max(maximumBufferUses, pass.buffers.size());
        for (const Impl::ImageUse& use : pass.images) {
            updateLifetime(impl_->images[use.resource.index].lifetime, orderedIndex);
        }
        for (const Impl::BufferUse& use : pass.buffers) {
            updateLifetime(impl_->buffers[use.resource.index].lifetime, orderedIndex);
        }
    }
    impl_->trackedImages.resize(impl_->images.size());
    impl_->trackedBuffers.resize(impl_->buffers.size());
    for (auto& resource : impl_->images) {
        resource.hasExecutionState = false;
    }
    for (auto& resource : impl_->buffers) {
        resource.hasExecutionState = false;
    }
    impl_->imageBarrierScratch.reserve(maximumImageUses);
    impl_->bufferBarrierScratch.reserve(maximumBufferUses);
    impl_->isCompiled = true;
}

void RenderGraph::execute(VkCommandBuffer commandBuffer, QueueType commandQueue,
                          const void* userData) const {
    if (!impl_->isCompiled || commandBuffer == VK_NULL_HANDLE) {
        throw std::runtime_error("Render graph must be compiled before execution");
    }
    const std::uint32_t commandFamily = queueFamily(impl_->queueFamilies, commandQueue);
    for (std::uint32_t passIndex : impl_->order) {
        const Impl::Pass& pass = impl_->passes[passIndex];
        if (queueFamily(impl_->queueFamilies, pass.queue) != commandFamily) {
            throw std::runtime_error(
                "The initial render-graph executor cannot span physical queue families");
        }
    }

    auto& trackedImages = impl_->trackedImages;
    for (std::size_t index = 0; index < impl_->images.size(); ++index) {
        const Impl::ImageResource& resource = impl_->images[index];
        if (resource.lifetime.used && resource.image == VK_NULL_HANDLE) {
            throw std::runtime_error("Render-graph image is not bound: " + resource.name);
        }
        if (resource.lifetime.used && resource.hasInitialQueue && !resource.concurrentSharing &&
            queueFamily(impl_->queueFamilies, resource.initialQueue) != commandFamily) {
            throw std::runtime_error(
                "The initial render-graph executor cannot acquire an exclusive image from "
                "another queue family");
        }
        // Imported resources are rebound by their owner and therefore begin at
        // the state supplied for this execution. Graph-owned resources keep
        // their actual final state across executions; resetting those to an
        // empty state would lose cross-frame write dependencies.
        if (resource.imported || !resource.hasExecutionState) {
            trackedImages[index] = {
                .state = resource.initialState,
                .queue = resource.initialQueue,
                .hasQueue = resource.hasInitialQueue,
                .hasAccess = resource.initialState.access != VK_ACCESS_2_NONE,
                .previousWrite = accessContainsWrite(resource.initialState.access),
            };
        }
    }
    auto& trackedBuffers = impl_->trackedBuffers;
    for (std::size_t index = 0; index < impl_->buffers.size(); ++index) {
        const Impl::BufferResource& resource = impl_->buffers[index];
        if (resource.lifetime.used && resource.buffer == VK_NULL_HANDLE) {
            throw std::runtime_error("Render-graph buffer is not bound: " + resource.name);
        }
        if (resource.lifetime.used && resource.hasInitialQueue && !resource.concurrentSharing &&
            queueFamily(impl_->queueFamilies, resource.initialQueue) != commandFamily) {
            throw std::runtime_error(
                "The initial render-graph executor cannot acquire an exclusive buffer from "
                "another queue family");
        }
        if (resource.imported || !resource.hasExecutionState) {
            trackedBuffers[index] = {
                .state = resource.initialState,
                .queue = resource.initialQueue,
                .hasQueue = resource.hasInitialQueue,
                .hasAccess = resource.initialState.access != VK_ACCESS_2_NONE,
                .previousWrite = accessContainsWrite(resource.initialState.access),
            };
        }
    }

    for (std::uint32_t passIndex : impl_->order) {
        const Impl::Pass& pass = impl_->passes[passIndex];
        auto& imageBarriers = impl_->imageBarrierScratch;
        auto& bufferBarriers = impl_->bufferBarrierScratch;
        imageBarriers.clear();
        bufferBarriers.clear();

        for (const Impl::ImageUse& use : pass.images) {
            const Impl::ImageResource& resource = impl_->images[use.resource.index];
            Impl::TrackedImage& tracked = trackedImages[use.resource.index];
            const bool targetWrite = writes(use.mode);
            const bool layoutChange = tracked.state.layout != use.state.layout;
            const bool memoryHazard = tracked.hasAccess && (tracked.previousWrite || targetWrite);
            if (layoutChange || memoryHazard) {
                imageBarriers.push_back({
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .srcStageMask = tracked.state.stages,
                    .srcAccessMask = tracked.state.access,
                    .dstStageMask = use.state.stages,
                    .dstAccessMask = use.state.access,
                    .oldLayout = tracked.state.layout,
                    .newLayout = use.state.layout,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = resource.image,
                    .subresourceRange = resource.subresource,
                });
                tracked.state = use.state;
            } else if (tracked.hasAccess && !targetWrite) {
                tracked.state.stages |= use.state.stages;
                tracked.state.access |= use.state.access;
                tracked.state.layout = use.state.layout;
            } else {
                tracked.state = use.state;
            }
            tracked.queue = pass.queue;
            tracked.hasQueue = true;
            tracked.hasAccess = true;
            tracked.previousWrite = targetWrite;
        }

        for (const Impl::BufferUse& use : pass.buffers) {
            const Impl::BufferResource& resource = impl_->buffers[use.resource.index];
            Impl::TrackedBuffer& tracked = trackedBuffers[use.resource.index];
            const bool targetWrite = writes(use.mode);
            const bool memoryHazard = tracked.hasAccess && (tracked.previousWrite || targetWrite);
            if (memoryHazard) {
                bufferBarriers.push_back({
                    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                    .srcStageMask = tracked.state.stages,
                    .srcAccessMask = tracked.state.access,
                    .dstStageMask = use.state.stages,
                    .dstAccessMask = use.state.access,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .buffer = resource.buffer,
                    .offset = resource.offset,
                    .size = resource.size,
                });
                tracked.state = use.state;
            } else if (tracked.hasAccess && !targetWrite) {
                tracked.state.stages |= use.state.stages;
                tracked.state.access |= use.state.access;
            } else {
                tracked.state = use.state;
            }
            tracked.queue = pass.queue;
            tracked.hasQueue = true;
            tracked.hasAccess = true;
            tracked.previousWrite = targetWrite;
        }

        if (!imageBarriers.empty() || !bufferBarriers.empty()) {
            const VkDependencyInfo dependencyInfo{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .bufferMemoryBarrierCount = static_cast<std::uint32_t>(bufferBarriers.size()),
                .pBufferMemoryBarriers = bufferBarriers.data(),
                .imageMemoryBarrierCount = static_cast<std::uint32_t>(imageBarriers.size()),
                .pImageMemoryBarriers = imageBarriers.data(),
            };
            vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
        }
        if (pass.callback) {
            pass.callback(commandBuffer, *this, userData);
        }
    }

    for (auto& resource : impl_->images) {
        if (!resource.imported && resource.lifetime.used) {
            resource.hasExecutionState = true;
        }
    }
    for (auto& resource : impl_->buffers) {
        if (!resource.imported && resource.lifetime.used) {
            resource.hasExecutionState = true;
        }
    }
}

VkImage RenderGraph::image(ImageHandle handle) const { return impl_->checked(handle).image; }

VkImageView RenderGraph::imageView(ImageHandle handle) const { return impl_->checked(handle).view; }

VkBuffer RenderGraph::buffer(BufferHandle handle) const { return impl_->checked(handle).buffer; }

ResourceLifetime RenderGraph::lifetime(ImageHandle handle) const {
    if (!impl_->isCompiled) {
        throw std::runtime_error("Render-graph lifetimes require compilation");
    }
    return impl_->checked(handle).lifetime;
}

ResourceLifetime RenderGraph::lifetime(BufferHandle handle) const {
    if (!impl_->isCompiled) {
        throw std::runtime_error("Render-graph lifetimes require compilation");
    }
    return impl_->checked(handle).lifetime;
}

std::span<const std::string> RenderGraph::orderedPassNames() const { return impl_->orderedNames; }

bool RenderGraph::compiled() const { return impl_->isCompiled; }

} // namespace renderer::render_graph
