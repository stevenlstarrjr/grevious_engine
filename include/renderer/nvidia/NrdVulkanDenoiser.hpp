#pragma once

#include "renderer/nvidia/NrdContext.hpp"

#include <volk.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace renderer::vulkan {
class Buffer;
class ComputePipeline;
class DescriptorPool;
class DescriptorSetLayout;
class Device;
class Image;
class ImageView;
class MemoryAllocator;
class Sampler;
class ShaderModule;
} // namespace renderer::vulkan

namespace renderer::nvidia {

struct NrdExternalResource {
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
};

using NrdExternalResourceTable =
    std::array<NrdExternalResource,
               static_cast<std::size_t>(nrd::ResourceType::MAX_NUM)>;

// Native Vulkan execution backend for NRD. It owns NRD's private texture
// pools, embedded-SPIR-V compute pipelines, per-frame descriptors, samplers,
// and streamed constant data. External guide/signal images remain renderer
// owned and are supplied to record().
class NrdVulkanDenoiser final {
  public:
    NrdVulkanDenoiser(const vulkan::Device& device,
                      vulkan::MemoryAllocator& allocator, VkExtent2D extent,
                      VkPipelineCache pipelineCache,
                      std::uint32_t queuedFrameCount,
                      nrd::Denoiser denoiser =
                          nrd::Denoiser::RELAX_DIFFUSE_SPECULAR);
    ~NrdVulkanDenoiser();

    NrdVulkanDenoiser(const NrdVulkanDenoiser&) = delete;
    NrdVulkanDenoiser& operator=(const NrdVulkanDenoiser&) = delete;

    [[nodiscard]] VkExtent2D extent() const { return extent_; }
    [[nodiscard]] std::uint64_t permanentPoolBytes() const {
        return permanentPoolBytes_;
    }
    [[nodiscard]] std::uint64_t transientPoolBytes() const {
        return transientPoolBytes_;
    }
    [[nodiscard]] std::uint32_t pipelineCount() const;

    // All external images must be in VK_IMAGE_LAYOUT_GENERAL. The caller must
    // establish ray-tracing/compute dependencies before this command sequence.
    void record(VkCommandBuffer commandBuffer, std::uint32_t frameSlot,
                const nrd::CommonSettings& settings,
                const NrdExternalResourceTable& externalResources);

  private:
    struct PoolTexture;

    [[nodiscard]] const NrdExternalResource&
    resolveResource(const nrd::ResourceDesc& resource,
                    const NrdExternalResourceTable& externalResources) const;
    void createTexturePool(vulkan::MemoryAllocator& allocator);
    void createDescriptors(vulkan::MemoryAllocator& allocator);
    void createPipelines(VkPipelineCache pipelineCache);
    void transitionTexturePool(VkCommandBuffer commandBuffer);

    const vulkan::Device& device_;
    VkExtent2D extent_{};
    std::uint32_t queuedFrameCount_ = 0;
    NrdContext context_;
    std::vector<std::unique_ptr<PoolTexture>> permanentPool_;
    std::vector<std::unique_ptr<PoolTexture>> transientPool_;
    std::vector<std::unique_ptr<vulkan::ShaderModule>> shaders_;
    std::vector<std::unique_ptr<vulkan::ComputePipeline>> pipelines_;
    std::unique_ptr<vulkan::DescriptorSetLayout> resourceLayout_;
    std::unique_ptr<vulkan::DescriptorSetLayout> constantsLayout_;
    std::unique_ptr<vulkan::DescriptorPool> resourceDescriptorPool_;
    std::unique_ptr<vulkan::DescriptorPool> constantsDescriptorPool_;
    std::unique_ptr<vulkan::Sampler> nearestSampler_;
    std::unique_ptr<vulkan::Sampler> linearSampler_;
    std::unique_ptr<vulkan::Buffer> constantsBuffer_;
    VkDeviceSize constantStride_ = 0;
    std::uint32_t setsPerFrame_ = 0;
    std::uint64_t permanentPoolBytes_ = 0;
    std::uint64_t transientPoolBytes_ = 0;
    bool texturePoolInitialized_ = false;
};

} // namespace renderer::nvidia
