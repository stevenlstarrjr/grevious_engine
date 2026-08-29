#include "renderer/nvidia/NrdVulkanDenoiser.hpp"

#include "renderer/nvidia/NrdVulkan.hpp"
#include "renderer/vulkan/Buffer.hpp"
#include "renderer/vulkan/Descriptors.hpp"
#include "renderer/vulkan/Device.hpp"
#include "renderer/vulkan/Image.hpp"
#include "renderer/vulkan/Pipeline.hpp"
#include "renderer/vulkan/Sampler.hpp"
#include "renderer/vulkan/Shader.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace renderer::nvidia {
namespace {

constexpr VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    return alignment <= 1 ? value : (value + alignment - 1) & ~(alignment - 1);
}

std::uint32_t divideUp(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1U) / divisor;
}

VkDescriptorType descriptorType(nrd::DescriptorType type) {
    switch (type) {
    case nrd::DescriptorType::TEXTURE:
        return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case nrd::DescriptorType::STORAGE_TEXTURE:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case nrd::DescriptorType::MAX_NUM:
        break;
    }
    throw std::invalid_argument("Unsupported NRD descriptor type");
}

VkSamplerCreateInfo samplerDescription(VkFilter filter) {
    return {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = filter,
        .minFilter = filter,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxLod = 0.0F,
    };
}

} // namespace

struct NrdVulkanDenoiser::PoolTexture {
    std::unique_ptr<vulkan::Image> image;
    std::unique_ptr<vulkan::ImageView> view;
    NrdExternalResource external;
};

NrdVulkanDenoiser::NrdVulkanDenoiser(
    const vulkan::Device& device, vulkan::MemoryAllocator& allocator,
    VkExtent2D extent, VkPipelineCache pipelineCache,
    std::uint32_t queuedFrameCount, nrd::Denoiser denoiser)
    : device_(device), extent_(extent), queuedFrameCount_(queuedFrameCount),
      context_(denoiser) {
    if (extent_.width == 0 || extent_.height == 0 ||
        extent_.width > std::numeric_limits<std::uint16_t>::max() ||
        extent_.height > std::numeric_limits<std::uint16_t>::max()) {
        throw std::invalid_argument("NRD Vulkan extent is invalid");
    }
    if (queuedFrameCount_ == 0) {
        throw std::invalid_argument("NRD requires at least one queued frame");
    }
    createTexturePool(allocator);
    createDescriptors(allocator);
    createPipelines(pipelineCache);
}

NrdVulkanDenoiser::~NrdVulkanDenoiser() = default;

std::uint32_t NrdVulkanDenoiser::pipelineCount() const {
    return static_cast<std::uint32_t>(pipelines_.size());
}

void NrdVulkanDenoiser::createTexturePool(vulkan::MemoryAllocator& allocator) {
    const nrd::InstanceDesc& instance = context_.instanceDescription();
    const auto createPoolTexture = [&](const nrd::TextureDesc& description,
                                       std::uint64_t& byteCounter) {
        const std::uint32_t factor = std::max<std::uint32_t>(
            description.downsampleFactor, 1U);
        const VkExtent3D textureExtent{
            .width = divideUp(extent_.width, factor),
            .height = divideUp(extent_.height, factor),
            .depth = 1,
        };
        auto texture = std::make_unique<PoolTexture>();
        texture->image = std::make_unique<vulkan::Image>(
            device_, allocator,
            vulkan::ImageCreateInfo{
                .extent = textureExtent,
                .format = nrdFormatToVulkan(description.format),
                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
            });
        texture->view =
            std::make_unique<vulkan::ImageView>(device_, *texture->image);
        texture->external = {
            .image = texture->image->handle(),
            .view = texture->view->handle(),
            .format = texture->image->format(),
        };
        byteCounter += static_cast<std::uint64_t>(textureExtent.width) *
                       textureExtent.height *
                       nrdFormatBytesPerTexel(description.format);
        return texture;
    };

    permanentPool_.reserve(instance.permanentPoolSize);
    for (std::uint32_t index = 0; index < instance.permanentPoolSize; ++index) {
        permanentPool_.push_back(
            createPoolTexture(instance.permanentPool[index], permanentPoolBytes_));
    }
    transientPool_.reserve(instance.transientPoolSize);
    for (std::uint32_t index = 0; index < instance.transientPoolSize; ++index) {
        transientPool_.push_back(
            createPoolTexture(instance.transientPool[index], transientPoolBytes_));
    }
}

void NrdVulkanDenoiser::createDescriptors(vulkan::MemoryAllocator& allocator) {
    const nrd::LibraryDesc& library = context_.libraryDescription();
    const nrd::InstanceDesc& instance = context_.instanceDescription();
    setsPerFrame_ = instance.descriptorPoolDesc.setsMaxNum;
    if (setsPerFrame_ == 0 ||
        setsPerFrame_ > std::numeric_limits<std::uint32_t>::max() /
                            queuedFrameCount_) {
        throw std::runtime_error("NRD returned an invalid descriptor-set count");
    }
    const std::uint32_t totalSetCount = setsPerFrame_ * queuedFrameCount_;

    std::vector<vulkan::DescriptorBinding> resourceBindings;
    resourceBindings.reserve(
        instance.descriptorPoolDesc.perSetTexturesMaxNum +
        instance.descriptorPoolDesc.perSetStorageTexturesMaxNum);
    for (std::uint32_t index = 0;
         index < instance.descriptorPoolDesc.perSetTexturesMaxNum; ++index) {
        resourceBindings.push_back({
            .set = instance.resourcesSpaceIndex,
            .binding = library.spirvBindingOffsets.textureOffset +
                       instance.resourcesBaseRegisterIndex + index,
            .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .count = 1,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
        });
    }
    for (std::uint32_t index = 0;
         index < instance.descriptorPoolDesc.perSetStorageTexturesMaxNum; ++index) {
        resourceBindings.push_back({
            .set = instance.resourcesSpaceIndex,
            .binding = library.spirvBindingOffsets.storageTextureAndBufferOffset +
                       instance.resourcesBaseRegisterIndex + index,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .count = 1,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
        });
    }
    resourceLayout_ = std::make_unique<vulkan::DescriptorSetLayout>(
        device_, instance.resourcesSpaceIndex, resourceBindings);
    resourceDescriptorPool_ = std::make_unique<vulkan::DescriptorPool>(
        device_, *resourceLayout_, totalSetCount);

    std::vector<vulkan::DescriptorBinding> constantBindings;
    constantBindings.reserve(instance.samplersNum + 1U);
    constantBindings.push_back({
        .set = instance.constantBufferAndSamplersSpaceIndex,
        .binding = library.spirvBindingOffsets.constantBufferOffset +
                   instance.constantBufferRegisterIndex,
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .count = 1,
        .stages = VK_SHADER_STAGE_COMPUTE_BIT,
    });
    for (std::uint32_t index = 0; index < instance.samplersNum; ++index) {
        constantBindings.push_back({
            .set = instance.constantBufferAndSamplersSpaceIndex,
            .binding = library.spirvBindingOffsets.samplerOffset +
                       instance.samplersBaseRegisterIndex + index,
            .type = VK_DESCRIPTOR_TYPE_SAMPLER,
            .count = 1,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
        });
    }
    constantsLayout_ = std::make_unique<vulkan::DescriptorSetLayout>(
        device_, instance.constantBufferAndSamplersSpaceIndex, constantBindings);
    constantsDescriptorPool_ = std::make_unique<vulkan::DescriptorPool>(
        device_, *constantsLayout_, totalSetCount);

    nearestSampler_ = std::make_unique<vulkan::Sampler>(
        device_, samplerDescription(VK_FILTER_NEAREST));
    linearSampler_ = std::make_unique<vulkan::Sampler>(
        device_, samplerDescription(VK_FILTER_LINEAR));

    const VkDeviceSize alignment = std::max<VkDeviceSize>(
        device_.properties().limits.minUniformBufferOffsetAlignment, 1);
    constantStride_ = alignUp(instance.constantBufferMaxDataSize, alignment);
    constantsBuffer_ = std::make_unique<vulkan::Buffer>(
        device_, allocator,
        vulkan::BufferCreateInfo{
            .size = constantStride_ * totalSetCount,
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            .requiredMemory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            .preferredMemory = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        });

    std::vector<VkDescriptorImageInfo> samplerInfos(instance.samplersNum);
    std::vector<VkWriteDescriptorSet> samplerWrites;
    samplerWrites.reserve(static_cast<std::size_t>(totalSetCount) *
                          instance.samplersNum);
    for (std::uint32_t index = 0; index < instance.samplersNum; ++index) {
        const nrd::Sampler sampler = instance.samplers[index];
        samplerInfos[index].sampler =
            sampler == nrd::Sampler::NEAREST_CLAMP ? nearestSampler_->handle()
                                                   : linearSampler_->handle();
    }
    for (std::uint32_t setIndex = 0; setIndex < totalSetCount; ++setIndex) {
        for (std::uint32_t samplerIndex = 0; samplerIndex < instance.samplersNum;
             ++samplerIndex) {
            samplerWrites.push_back({
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = constantsDescriptorPool_->set(setIndex),
                .dstBinding = library.spirvBindingOffsets.samplerOffset +
                              instance.samplersBaseRegisterIndex + samplerIndex,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                .pImageInfo = &samplerInfos[samplerIndex],
            });
        }
    }
    vkUpdateDescriptorSets(device_.handle(),
                           static_cast<std::uint32_t>(samplerWrites.size()),
                           samplerWrites.data(), 0, nullptr);
}

void NrdVulkanDenoiser::createPipelines(VkPipelineCache pipelineCache) {
    const nrd::InstanceDesc& instance = context_.instanceDescription();
    const std::array layouts{resourceLayout_->handle(), constantsLayout_->handle()};
    shaders_.reserve(instance.pipelinesNum);
    pipelines_.reserve(instance.pipelinesNum);
    for (std::uint32_t index = 0; index < instance.pipelinesNum; ++index) {
        const nrd::ComputeShaderDesc& shader =
            instance.pipelines[index].computeShaderSPIRV;
        if (shader.bytecode == nullptr || shader.size == 0 ||
            shader.size % sizeof(std::uint32_t) != 0) {
            throw std::runtime_error("NRD provided invalid embedded SPIR-V");
        }
        const auto* words = static_cast<const std::uint32_t*>(shader.bytecode);
        shaders_.push_back(std::make_unique<vulkan::ShaderModule>(
            device_, std::span(words, static_cast<std::size_t>(
                                         shader.size / sizeof(std::uint32_t)))));
        pipelines_.push_back(std::make_unique<vulkan::ComputePipeline>(
            device_, vulkan::ComputePipelineCreateInfo{
                         .shader = shaders_.back().get(),
                         .cache = pipelineCache,
                         .descriptorSetLayouts = layouts,
                         .entryPoint = instance.shaderEntryPoint,
                     }));
    }
}

const NrdExternalResource& NrdVulkanDenoiser::resolveResource(
    const nrd::ResourceDesc& resource,
    const NrdExternalResourceTable& externalResources) const {
    if (resource.type == nrd::ResourceType::PERMANENT_POOL) {
        if (resource.indexInPool >= permanentPool_.size()) {
            throw std::out_of_range("NRD permanent-pool resource index is invalid");
        }
        return permanentPool_[resource.indexInPool]->external;
    }
    if (resource.type == nrd::ResourceType::TRANSIENT_POOL) {
        if (resource.indexInPool >= transientPool_.size()) {
            throw std::out_of_range("NRD transient-pool resource index is invalid");
        }
        return transientPool_[resource.indexInPool]->external;
    }
    const std::size_t index = static_cast<std::size_t>(resource.type);
    if (index >= externalResources.size()) {
        throw std::out_of_range("NRD external resource type is invalid");
    }
    const NrdExternalResource& external = externalResources[index];
    if (external.image == VK_NULL_HANDLE || external.view == VK_NULL_HANDLE ||
        external.format == VK_FORMAT_UNDEFINED) {
        throw std::runtime_error(std::string("Missing NRD external resource: ") +
                                 nrd::GetResourceTypeString(resource.type));
    }
    return external;
}

void NrdVulkanDenoiser::transitionTexturePool(VkCommandBuffer commandBuffer) {
    if (texturePoolInitialized_) {
        const VkMemoryBarrier2 memoryBarrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        };
        const VkDependencyInfo dependency{
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &memoryBarrier,
        };
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
        return;
    }
    std::vector<VkImageMemoryBarrier2> barriers;
    barriers.reserve(permanentPool_.size() + transientPool_.size());
    const auto append = [&](const auto& pool) {
        for (const auto& texture : pool) {
            barriers.push_back({
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .image = texture->external.image,
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            });
        }
    };
    append(permanentPool_);
    append(transientPool_);
    const VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = static_cast<std::uint32_t>(barriers.size()),
        .pImageMemoryBarriers = barriers.data(),
    };
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
    texturePoolInitialized_ = true;
}

void NrdVulkanDenoiser::record(
    VkCommandBuffer commandBuffer, std::uint32_t frameSlot,
    const nrd::CommonSettings& settings,
    const NrdExternalResourceTable& externalResources) {
    if (commandBuffer == VK_NULL_HANDLE || frameSlot >= queuedFrameCount_) {
        throw std::invalid_argument("Invalid NRD Vulkan recording parameters");
    }
    const std::span<const nrd::DispatchDesc> dispatches =
        context_.prepareFrame(settings);
    if (dispatches.size() > setsPerFrame_) {
        throw std::runtime_error("NRD dispatches exceed the descriptor-set budget");
    }

    const nrd::LibraryDesc& library = context_.libraryDescription();
    const nrd::InstanceDesc& instance = context_.instanceDescription();
    const std::uint32_t frameSetBase = frameSlot * setsPerFrame_;
    std::vector<std::byte> constantUpload(
        static_cast<std::size_t>(constantStride_ * setsPerFrame_));
    std::vector<VkDeviceSize> constantOffsets(dispatches.size());
    VkDeviceSize previousConstantOffset = constantStride_ * frameSetBase;
    for (std::size_t index = 0; index < dispatches.size(); ++index) {
        const nrd::DispatchDesc& dispatch = dispatches[index];
        if (dispatch.constantBufferDataSize > instance.constantBufferMaxDataSize) {
            throw std::runtime_error("NRD constant data exceeds its advertised limit");
        }
        const VkDeviceSize slotOffset = constantStride_ * index;
        if (dispatch.constantBufferDataSize > 0 &&
            !dispatch.constantBufferDataMatchesPreviousDispatch) {
            if (dispatch.constantBufferData == nullptr) {
                throw std::runtime_error("NRD constant data pointer is null");
            }
            std::memcpy(constantUpload.data() + slotOffset,
                        dispatch.constantBufferData,
                        dispatch.constantBufferDataSize);
            previousConstantOffset = constantStride_ * frameSetBase + slotOffset;
        }
        constantOffsets[index] = previousConstantOffset;
    }
    constantsBuffer_->write(constantUpload, constantStride_ * frameSetBase);

    transitionTexturePool(commandBuffer);
    for (std::size_t dispatchIndex = 0; dispatchIndex < dispatches.size();
         ++dispatchIndex) {
        const nrd::DispatchDesc& dispatch = dispatches[dispatchIndex];
        if (dispatch.pipelineIndex >= pipelines_.size()) {
            throw std::runtime_error("NRD dispatch pipeline index is invalid");
        }
        const nrd::PipelineDesc& pipelineDescription =
            instance.pipelines[dispatch.pipelineIndex];
        const std::uint32_t setIndex =
            frameSetBase + static_cast<std::uint32_t>(dispatchIndex);

        std::vector<VkDescriptorImageInfo> imageInfos;
        std::vector<VkWriteDescriptorSet> writes;
        imageInfos.reserve(dispatch.resourcesNum);
        writes.reserve(dispatch.resourcesNum + 1U);
        std::uint32_t flatResourceIndex = 0;
        for (std::uint32_t rangeIndex = 0;
             rangeIndex < pipelineDescription.resourceRangesNum; ++rangeIndex) {
            const nrd::ResourceRangeDesc& range =
                pipelineDescription.resourceRanges[rangeIndex];
            const std::uint32_t bindingOffset =
                range.descriptorType == nrd::DescriptorType::TEXTURE
                    ? library.spirvBindingOffsets.textureOffset
                    : library.spirvBindingOffsets.storageTextureAndBufferOffset;
            for (std::uint32_t rangeResourceIndex = 0;
                 rangeResourceIndex < range.descriptorsNum; ++rangeResourceIndex) {
                if (flatResourceIndex >= dispatch.resourcesNum) {
                    throw std::runtime_error("NRD descriptor ranges exceed resources");
                }
                const nrd::ResourceDesc& resource =
                    dispatch.resources[flatResourceIndex++];
                if (resource.descriptorType != range.descriptorType) {
                    throw std::runtime_error("NRD resource descriptor type mismatch");
                }
                const NrdExternalResource& resolved =
                    resolveResource(resource, externalResources);
                imageInfos.push_back({
                    .imageView = resolved.view,
                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                });
                writes.push_back({
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = resourceDescriptorPool_->set(setIndex),
                    .dstBinding = bindingOffset + instance.resourcesBaseRegisterIndex +
                                  rangeResourceIndex,
                    .descriptorCount = 1,
                    .descriptorType = descriptorType(range.descriptorType),
                    .pImageInfo = &imageInfos.back(),
                });
            }
        }
        if (flatResourceIndex != dispatch.resourcesNum) {
            throw std::runtime_error("NRD resources exceed descriptor ranges");
        }

        const VkDescriptorBufferInfo constantsInfo{
            .buffer = constantsBuffer_->handle(),
            .offset = constantOffsets[dispatchIndex],
            .range = instance.constantBufferMaxDataSize,
        };
        writes.push_back({
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = constantsDescriptorPool_->set(setIndex),
            .dstBinding = library.spirvBindingOffsets.constantBufferOffset +
                          instance.constantBufferRegisterIndex,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &constantsInfo,
        });
        vkUpdateDescriptorSets(device_.handle(),
                               static_cast<std::uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);

        if (dispatchIndex > 0) {
            const VkMemoryBarrier2 memoryBarrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            };
            const VkDependencyInfo dependency{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &memoryBarrier,
            };
            vkCmdPipelineBarrier2(commandBuffer, &dependency);
        }

        const vulkan::ComputePipeline& pipeline =
            *pipelines_[dispatch.pipelineIndex];
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline.handle());
        const std::array descriptorSets{
            resourceDescriptorPool_->set(setIndex),
            constantsDescriptorPool_->set(setIndex),
        };
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline.layout(), 0,
                                static_cast<std::uint32_t>(descriptorSets.size()),
                                descriptorSets.data(), 0, nullptr);
        vkCmdDispatch(commandBuffer, dispatch.gridWidth, dispatch.gridHeight, 1);
    }
}

} // namespace renderer::nvidia
