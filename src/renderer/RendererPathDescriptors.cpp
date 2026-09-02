#include "renderer/renderer/Renderer.hpp"

#include "renderer/vulkan/Descriptors.hpp"
#include "renderer/vulkan/Device.hpp"
#include "renderer/vulkan/Image.hpp"
#include "renderer/vulkan/Sampler.hpp"

#include <array>
#include <stdexcept>

namespace renderer {

void Renderer::updatePathHistoryDescriptors() {
    if (!pathTracingEnabled() || !pathHistoryDescriptorPool_ || !hiZSampler_ ||
        !pathTracingNormalView_ || !pathTracingMotionDepthView_ || !pathTracingDepthView_ ||
        !pathHistoryValidityView_) {
        throw std::runtime_error(
            "Path-history descriptors require complete reconstruction resources");
    }
    for (std::size_t writeIndex = 0U; writeIndex < framesInFlight; ++writeIndex) {
        const std::size_t readIndex = 1U - writeIndex;
        if (!pathHistoryNormalViews_[readIndex] || !pathHistoryNormalViews_[writeIndex] ||
            !pathHistoryDepthViews_[readIndex] || !pathHistoryDepthViews_[writeIndex]) {
            throw std::runtime_error("Path-history ping-pong resources are incomplete");
        }
        const std::array sampledInfos{
            VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                                  .imageView = pathTracingNormalView_->handle(),
                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                                  .imageView = pathTracingMotionDepthView_->handle(),
                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                                  .imageView = pathTracingDepthView_->handle(),
                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                                  .imageView = pathHistoryNormalViews_[readIndex]->handle(),
                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                                  .imageView = pathHistoryDepthViews_[readIndex]->handle(),
                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
        };
        const std::array storageInfos{
            VkDescriptorImageInfo{.imageView = pathHistoryNormalViews_[writeIndex]->handle(),
                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{.imageView = pathHistoryDepthViews_[writeIndex]->handle(),
                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{.imageView = pathHistoryValidityView_->handle(),
                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
        };
        std::array<VkWriteDescriptorSet, 8> writes{};
        for (std::size_t binding = 0U; binding < writes.size(); ++binding) {
            const bool sampled = binding < sampledInfos.size();
            writes[binding] = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = pathHistoryDescriptorPool_->set(writeIndex),
                .dstBinding = static_cast<std::uint32_t>(binding),
                .descriptorCount = 1U,
                .descriptorType = sampled ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                          : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo =
                    sampled ? &sampledInfos[binding] : &storageInfos[binding - sampledInfos.size()],
            };
        }
        vkUpdateDescriptorSets(device_.handle(), static_cast<std::uint32_t>(writes.size()),
                               writes.data(), 0U, nullptr);
    }
}

void Renderer::updatePathReconstructionDescriptors() {
    if (!pathTracingEnabled() || !pathReconstructionDescriptorPool_ || !brdfLutSampler_ ||
        !hiZSampler_ || !pathTracingAccumulationView_ || !pathTracingMotionDepthView_ ||
        !pathHistoryValidityView_ || !pathDenoisedView_) {
        throw std::runtime_error(
            "Path reconstruction descriptors require complete temporal resources");
    }
    for (std::size_t writeIndex = 0U; writeIndex < framesInFlight; ++writeIndex) {
        const std::size_t readIndex = 1U - writeIndex;
        if (!pathReconstructionViews_[writeIndex] || !pathMomentViews_[readIndex] ||
            !pathMomentViews_[writeIndex]) {
            throw std::runtime_error("Path reconstruction ping-pong resources are incomplete");
        }
        const std::array sampledInfos{
            VkDescriptorImageInfo{.sampler = brdfLutSampler_->handle(),
                                  .imageView = pathTracingAccumulationView_->handle(),
                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                                  .imageView = pathTracingMotionDepthView_->handle(),
                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                                  .imageView = pathHistoryValidityView_->handle(),
                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{.sampler = brdfLutSampler_->handle(),
                                  .imageView = pathDenoisedView_->handle(),
                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
            VkDescriptorImageInfo{.sampler = brdfLutSampler_->handle(),
                                  .imageView = pathMomentViews_[readIndex]->handle(),
                                  .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
        };
        const std::array outputInfos{
            VkDescriptorImageInfo{
                .imageView = pathReconstructionViews_[writeIndex]->handle(),
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            },
            VkDescriptorImageInfo{
                .imageView = pathMomentViews_[writeIndex]->handle(),
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            },
        };
        std::array<VkWriteDescriptorSet, 7> writes{};
        for (std::size_t binding = 0U; binding < writes.size(); ++binding) {
            const bool sampled = binding < sampledInfos.size();
            writes[binding] = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = pathReconstructionDescriptorPool_->set(writeIndex),
                .dstBinding = static_cast<std::uint32_t>(binding),
                .descriptorCount = 1U,
                .descriptorType = sampled ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                          : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo =
                    sampled ? &sampledInfos[binding] : &outputInfos[binding - sampledInfos.size()],
            };
        }
        vkUpdateDescriptorSets(device_.handle(), static_cast<std::uint32_t>(writes.size()),
                               writes.data(), 0U, nullptr);
    }
}

void Renderer::updatePathDenoisingDescriptors() {
    if (!pathTracingEnabled() || !pathDenoisingDescriptorPool_ || !hiZSampler_ ||
        !pathTracingDirectView_ || !pathTracingSurfaceView_ || !pathTracingNormalView_ ||
        !pathTracingMotionDepthView_ || !pathTracingDepthView_ ||
        !pathDenoisingDirectViews_.front() || !pathDenoisingDirectViews_.back() ||
        !pathDenoisingIndirectViews_.front() || !pathDenoisingIndirectViews_.back() ||
        !pathDenoisedView_) {
        throw std::runtime_error(
            "Path denoising descriptors require complete reconstruction resources");
    }
    for (std::size_t writeIndex = 0U; writeIndex < framesInFlight; ++writeIndex) {
        if (!pathReconstructionViews_[writeIndex] || !pathMomentViews_[writeIndex]) {
            throw std::runtime_error("Path denoising frame resources are incomplete");
        }
        for (std::size_t passIndex = 0U; passIndex < pathDenoisingPassCount; ++passIndex) {
            const vulkan::ImageView& directInput = passIndex == 0U
                                                       ? *pathTracingDirectView_
                                                       : *pathDenoisingDirectViews_[passIndex - 1U];
            const vulkan::ImageView& indirectOrBeautyInput =
                passIndex == 0U ? *pathReconstructionViews_[writeIndex]
                                : *pathDenoisingIndirectViews_[passIndex - 1U];
            const std::array sampledInfos{
                VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                                      .imageView = directInput.handle(),
                                      .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
                VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                                      .imageView = indirectOrBeautyInput.handle(),
                                      .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
                VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                                      .imageView = pathTracingSurfaceView_->handle(),
                                      .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
                VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                                      .imageView = pathTracingNormalView_->handle(),
                                      .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
                VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                                      .imageView = pathTracingMotionDepthView_->handle(),
                                      .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
                VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                                      .imageView = pathTracingDepthView_->handle(),
                                      .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
                VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                                      .imageView = pathMomentViews_[writeIndex]->handle(),
                                      .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
            };
            const bool finalPass = passIndex + 1U == pathDenoisingPassCount;
            const std::size_t scratchOutput = finalPass ? 0U : passIndex;
            const VkImageView directOutput =
                finalPass ? pathDenoisedView_->handle()
                          : pathDenoisingDirectViews_[scratchOutput]->handle();
            const VkImageView indirectOutput =
                finalPass ? pathDenoisedView_->handle()
                          : pathDenoisingIndirectViews_[scratchOutput]->handle();
            // All statically reachable storage descriptors must name an image
            // transitioned by this graph pass, even though the push-constant
            // branch writes only scratch outputs or only final radiance.
            const VkImageView radianceOutput =
                finalPass ? pathDenoisedView_->handle() : directOutput;
            const std::array outputInfos{
                VkDescriptorImageInfo{
                    .imageView = directOutput,
                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                },
                VkDescriptorImageInfo{
                    .imageView = indirectOutput,
                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                },
                VkDescriptorImageInfo{
                    .imageView = radianceOutput,
                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                },
            };
            const std::size_t descriptorIndex = writeIndex * pathDenoisingPassCount + passIndex;
            std::array<VkWriteDescriptorSet, 10> writes{};
            for (std::size_t binding = 0U; binding < writes.size(); ++binding) {
                const bool sampled = binding < sampledInfos.size();
                writes[binding] = {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = pathDenoisingDescriptorPool_->set(descriptorIndex),
                    .dstBinding = static_cast<std::uint32_t>(binding),
                    .descriptorCount = 1U,
                    .descriptorType = sampled ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                              : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .pImageInfo = sampled ? &sampledInfos[binding]
                                          : &outputInfos[binding - sampledInfos.size()],
                };
            }
            vkUpdateDescriptorSets(device_.handle(), static_cast<std::uint32_t>(writes.size()),
                                   writes.data(), 0U, nullptr);
        }
    }
}

void Renderer::updateNrdDescriptors() {
#if RENDERER_ENABLE_NRD
    if (!pathTracingEnabled() || !pathNrdPrepareDescriptorPool_ ||
        !pathNrdResolveDescriptorPool_ || !hiZSampler_ || !pathTracingAccumulationView_ ||
        !pathTracingSurfaceView_ || !pathTracingNormalView_ || !pathTracingMotionDepthView_ ||
        !pathTracingDepthView_ || !nrdMotionView_ || !nrdNormalRoughnessView_ || !nrdViewZView_ ||
        !nrdDiffuseInputView_ || !nrdSpecularInputView_ || !nrdMaterialView_ ||
        !nrdDiffuseOutputView_ || !nrdSpecularOutputView_ || !pathDenoisedView_) {
        throw std::runtime_error("NVIDIA NRD descriptors require complete guide and signal resources");
    }
    const std::array prepareSampledInfos{
        VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                              .imageView = pathTracingSurfaceView_->handle(),
                              .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
        VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                              .imageView = pathTracingNormalView_->handle(),
                              .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
        VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                              .imageView = pathTracingMotionDepthView_->handle(),
                              .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
        VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                              .imageView = pathTracingDepthView_->handle(),
                              .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
    };
    const std::array prepareStorageInfos{
        VkDescriptorImageInfo{.imageView = nrdMotionView_->handle(),
                              .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
        VkDescriptorImageInfo{.imageView = nrdNormalRoughnessView_->handle(),
                              .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
        VkDescriptorImageInfo{.imageView = nrdViewZView_->handle(),
                              .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
    };
    const std::array resolveSampledInfos{
        VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                              .imageView = nrdDiffuseOutputView_->handle(),
                              .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
        VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                              .imageView = nrdSpecularOutputView_->handle(),
                              .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
        VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                              .imageView = nrdMaterialView_->handle(),
                              .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
        VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                              .imageView = pathTracingMotionDepthView_->handle(),
                              .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
        VkDescriptorImageInfo{.sampler = hiZSampler_->handle(),
                              .imageView = pathTracingAccumulationView_->handle(),
                              .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
    };
    const VkDescriptorImageInfo resolveStorageInfo{
        .imageView = pathDenoisedView_->handle(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    for (std::size_t frameIndex = 0U; frameIndex < framesInFlight; ++frameIndex) {
        std::array<VkWriteDescriptorSet, 7> prepareWrites{};
        for (std::size_t binding = 0U; binding < prepareWrites.size(); ++binding) {
            const bool sampled = binding < prepareSampledInfos.size();
            prepareWrites[binding] = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = pathNrdPrepareDescriptorPool_->set(frameIndex),
                .dstBinding = static_cast<std::uint32_t>(binding),
                .descriptorCount = 1U,
                .descriptorType = sampled ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                          : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = sampled
                                  ? &prepareSampledInfos[binding]
                                  : &prepareStorageInfos[binding - prepareSampledInfos.size()],
            };
        }
        vkUpdateDescriptorSets(device_.handle(),
                               static_cast<std::uint32_t>(prepareWrites.size()),
                               prepareWrites.data(), 0U, nullptr);

        std::array<VkWriteDescriptorSet, 6> resolveWrites{};
        for (std::size_t binding = 0U; binding < resolveWrites.size(); ++binding) {
            const bool sampled = binding < resolveSampledInfos.size();
            resolveWrites[binding] = {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = pathNrdResolveDescriptorPool_->set(frameIndex),
                .dstBinding = static_cast<std::uint32_t>(binding),
                .descriptorCount = 1U,
                .descriptorType = sampled ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                          : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = sampled ? &resolveSampledInfos[binding] : &resolveStorageInfo,
            };
        }
        vkUpdateDescriptorSets(device_.handle(),
                               static_cast<std::uint32_t>(resolveWrites.size()),
                               resolveWrites.data(), 0U, nullptr);
    }
#endif
}

} // namespace renderer
