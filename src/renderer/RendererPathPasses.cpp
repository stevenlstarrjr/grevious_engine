#include "renderer/renderer/Renderer.hpp"

#if RENDERER_ENABLE_NRD
#include "renderer/nvidia/NrdVulkanDenoiser.hpp"
#endif

#include "renderer/vulkan/Buffer.hpp"
#include "renderer/vulkan/Descriptors.hpp"
#include "renderer/vulkan/Image.hpp"
#include "renderer/vulkan/Pipeline.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/matrix.hpp>

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace renderer {

void Renderer::recordAtmosphereEnvironmentPass(VkCommandBuffer commandBuffer,
                                               const GraphFrameContext& context) {
    if (!atmosphereEnvironmentUpdatePending_) {
        return;
    }
    if (!atmosphereEnvironmentPipeline_ || !atmosphereEnvironmentDescriptorPool_ ||
        !environmentImage_) {
        throw std::runtime_error(
            "Atmosphere environment dispatch requires a complete compute pipeline");
    }
    writeGpuTimestamp(commandBuffer, context.frameIndex,
                      GpuTimingPass::atmosphereEnvironment, false);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      atmosphereEnvironmentPipeline_->handle());
    const VkDescriptorSet descriptorSet =
        atmosphereEnvironmentDescriptorPool_->set(context.frameIndex);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            atmosphereEnvironmentPipeline_->layout(), 0U, 1U,
                            &descriptorSet, 0U, nullptr);
    constexpr std::uint32_t workgroupSize = 8U;
    const VkExtent3D extent = environmentImage_->extent();
    vkCmdDispatch(commandBuffer, (extent.width + workgroupSize - 1U) / workgroupSize,
                  (extent.height + workgroupSize - 1U) / workgroupSize, 1U);
    writeGpuTimestamp(commandBuffer, context.frameIndex,
                      GpuTimingPass::atmosphereEnvironment, true);
    atmosphereEnvironmentUpdatePending_ = false;
}

void Renderer::recordPathTracingPass(VkCommandBuffer commandBuffer,
                                     const GraphFrameContext& context) {
    if (!pathTracingPipeline_ || !pathTracingDescriptorPool_) {
        throw std::runtime_error("Path-tracing dispatch requires a complete ray pipeline");
    }
    writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::pathTracing, false);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                      pathTracingPipeline_->handle());
    const std::array descriptorSets{
        pathTracingDescriptorPool_->set(context.frameIndex),
        bindlessDescriptorPool_->set(0),
    };
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pathTracingPipeline_->layout(), 0U,
        static_cast<std::uint32_t>(descriptorSets.size()), descriptorSets.data(), 0U, nullptr);
    const VkExtent2D extent = renderExtent();
    const VkShaderStageFlags pathTracingStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                                 VK_SHADER_STAGE_MISS_BIT_KHR |
                                                 VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    for (std::uint32_t sampleInFrame = 0U; sampleInFrame < pathSamplesThisFrame_; ++sampleInFrame) {
        const PathTracingControls controls{
            .sampleIndex = rayReconstructionEnabled()
                               ? static_cast<std::uint32_t>(renderedFrameCount_) + sampleInFrame
                               : pathTracingSampleIndex_ + sampleInFrame,
            .sampleInFrame = sampleInFrame,
            .samplesThisFrame = pathSamplesThisFrame_,
            .fixedPrimarySample = rayReconstructionEnabled()
                                      ? 2U
                                      : (temporalUpscalingEnabled() ? 1U : 0U),
        };
        vkCmdPushConstants(commandBuffer, pathTracingPipeline_->layout(), pathTracingStages, 0U,
                           sizeof(controls), &controls);
        vkCmdTraceRaysKHR(commandBuffer, &pathTracingPipeline_->rayGenerationRegion(),
                          &pathTracingPipeline_->missRegion(), &pathTracingPipeline_->hitRegion(),
                          &pathTracingPipeline_->callableRegion(), extent.width, extent.height, 1U);
        if (sampleInFrame + 1U < pathSamplesThisFrame_) {
            const VkMemoryBarrier2 sampleBarrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                .dstAccessMask =
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            };
            const VkDependencyInfo dependency{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .memoryBarrierCount = 1U,
                .pMemoryBarriers = &sampleBarrier,
            };
            vkCmdPipelineBarrier2(commandBuffer, &dependency);
        }
    }
    writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::pathTracing, true);
}

void Renderer::recordPathHistoryPass(VkCommandBuffer commandBuffer,
                                     const GraphFrameContext& context) {
    if (!pathHistoryPipeline_ || !pathHistoryDescriptorPool_) {
        throw std::runtime_error("Path-history dispatch requires a complete compute pipeline");
    }
    writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::pathHistory, false);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pathHistoryPipeline_->handle());
    const VkDescriptorSet descriptorSet = pathHistoryDescriptorPool_->set(context.frameIndex);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pathHistoryPipeline_->layout(), 0U, 1U, &descriptorSet, 0U, nullptr);
    const std::uint32_t historyValid =
        pathHistoryLayoutsInitialized_ && temporalState_.historyValid() ? 1U : 0U;
    vkCmdPushConstants(commandBuffer, pathHistoryPipeline_->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                       0U, sizeof(historyValid), &historyValid);
    constexpr std::uint32_t workgroupSize = 8U;
    const VkExtent2D extent = renderExtent();
    vkCmdDispatch(commandBuffer, (extent.width + workgroupSize - 1U) / workgroupSize,
                  (extent.height + workgroupSize - 1U) / workgroupSize, 1U);
    writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::pathHistory, true);
}

void Renderer::recordPathReconstructionPass(VkCommandBuffer commandBuffer,
                                            const GraphFrameContext& context) {
    if (!pathReconstructionPipeline_ || !pathReconstructionDescriptorPool_) {
        throw std::runtime_error(
            "Path reconstruction dispatch requires a complete compute pipeline");
    }
    writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::pathReconstruction, false);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pathReconstructionPipeline_->handle());
    const VkDescriptorSet descriptorSet =
        pathReconstructionDescriptorPool_->set(context.frameIndex);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pathReconstructionPipeline_->layout(), 0U, 1U, &descriptorSet, 0U,
                            nullptr);
    const PathReconstructionControls controls{
        .sampleIndex = pathTracingSampleIndex_,
        .samplesThisFrame = pathSamplesThisFrame_,
    };
    vkCmdPushConstants(commandBuffer, pathReconstructionPipeline_->layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(controls), &controls);
    constexpr std::uint32_t workgroupSize = 8U;
    const VkExtent2D extent = renderExtent();
    vkCmdDispatch(commandBuffer, (extent.width + workgroupSize - 1U) / workgroupSize,
                  (extent.height + workgroupSize - 1U) / workgroupSize, 1U);
    writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::pathReconstruction, true);
}

void Renderer::recordPathDenoisingPass(VkCommandBuffer commandBuffer,
                                       const GraphFrameContext& context, std::size_t passIndex) {
    if (!pathDenoisingPipeline_ || !pathDenoisingDescriptorPool_ ||
        passIndex >= pathDenoisingPassCount) {
        throw std::runtime_error("Path denoising dispatch requires a complete compute pipeline");
    }
    if (passIndex == 0U) {
        writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::pathDenoising, false);
    }
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pathDenoisingPipeline_->handle());
    const std::size_t descriptorIndex = context.frameIndex * pathDenoisingPassCount + passIndex;
    const VkDescriptorSet descriptorSet = pathDenoisingDescriptorPool_->set(descriptorIndex);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pathDenoisingPipeline_->layout(), 0U, 1U, &descriptorSet, 0U, nullptr);
    const PathDenoisingControls controls{
        .sampleIndex = pathTracingSampleIndex_,
        .stepWidth = 1U << static_cast<std::uint32_t>(passIndex),
        .passIndex = static_cast<std::uint32_t>(passIndex),
        .passCount = static_cast<std::uint32_t>(pathDenoisingPassCount),
        .samplesThisFrame = pathSamplesThisFrame_,
    };
    vkCmdPushConstants(commandBuffer, pathDenoisingPipeline_->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                       0U, sizeof(controls), &controls);
    constexpr std::uint32_t workgroupSize = 8U;
    const VkExtent2D extent = renderExtent();
    vkCmdDispatch(commandBuffer, (extent.width + workgroupSize - 1U) / workgroupSize,
                  (extent.height + workgroupSize - 1U) / workgroupSize, 1U);
    if (passIndex + 1U == pathDenoisingPassCount) {
        writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::pathDenoising, true);
    }
}

void Renderer::recordNrdPreparePass(VkCommandBuffer commandBuffer,
                                    const GraphFrameContext& context) {
#if RENDERER_ENABLE_NRD
    if (!pathNrdPreparePipeline_ || !pathNrdPrepareDescriptorPool_) {
        throw std::runtime_error("NRD guide preparation requires a complete compute pipeline");
    }
    writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::nrdPrepare, false);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pathNrdPreparePipeline_->handle());
    const VkDescriptorSet descriptorSet = pathNrdPrepareDescriptorPool_->set(context.frameIndex);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pathNrdPreparePipeline_->layout(), 0U, 1U, &descriptorSet, 0U,
                            nullptr);
    const glm::mat4 inverseProjection =
        glm::inverse(temporalState_.current().unjitteredProjection);
    vkCmdPushConstants(commandBuffer, pathNrdPreparePipeline_->layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0U, sizeof(inverseProjection),
                       glm::value_ptr(inverseProjection));
    constexpr std::uint32_t workgroupSize = 8U;
    const VkExtent2D extent = renderExtent();
    vkCmdDispatch(commandBuffer, (extent.width + workgroupSize - 1U) / workgroupSize,
                  (extent.height + workgroupSize - 1U) / workgroupSize, 1U);
    writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::nrdPrepare, true);
#else
    (void)commandBuffer;
    (void)context;
#endif
}

void Renderer::recordNrdDenoisingPass(VkCommandBuffer commandBuffer,
                                      const GraphFrameContext& context) {
#if RENDERER_ENABLE_NRD
    if (!nrdDenoiser_ || !nrdMotionImage_ || !nrdMotionView_ || !nrdNormalRoughnessImage_ ||
        !nrdNormalRoughnessView_ || !nrdViewZImage_ || !nrdViewZView_ ||
        !nrdDiffuseInputImage_ || !nrdDiffuseInputView_ || !nrdSpecularInputImage_ ||
        !nrdSpecularInputView_ || !nrdDiffuseOutputImage_ || !nrdDiffuseOutputView_ ||
        !nrdSpecularOutputImage_ || !nrdSpecularOutputView_) {
        throw std::runtime_error("NRD denoising requires complete external resources");
    }
    const auto matrixArray = [](const glm::mat4& matrix) {
        std::array<float, 16> result{};
        std::memcpy(result.data(), glm::value_ptr(matrix), sizeof(matrix));
        return result;
    };
    const temporal::CameraMatrices& currentCamera = temporalState_.current();
    const temporal::CameraMatrices& previousCamera = temporalState_.previousForRendering();
    const VkExtent2D extent = renderExtent();
    if (extent.width > std::numeric_limits<std::uint16_t>::max() ||
        extent.height > std::numeric_limits<std::uint16_t>::max()) {
        throw std::runtime_error("NRD render extent exceeds its 16-bit API limit");
    }
    const nrd::CommonSettings settings = nvidia::makeNrdCommonSettings(
        static_cast<std::uint16_t>(extent.width), static_cast<std::uint16_t>(extent.height),
        static_cast<std::uint32_t>(renderedFrameCount_),
        matrixArray(currentCamera.unjitteredProjection),
        matrixArray(previousCamera.unjitteredProjection), matrixArray(currentCamera.view),
        matrixArray(previousCamera.view), !temporalState_.historyValid());

    nvidia::NrdExternalResourceTable externalResources{};
    const auto bind = [&externalResources](nrd::ResourceType type, const vulkan::Image& image,
                                           const vulkan::ImageView& view) {
        externalResources[static_cast<std::size_t>(type)] = {
            .image = image.handle(),
            .view = view.handle(),
            .format = image.format(),
        };
    };
    bind(nrd::ResourceType::IN_MV, *nrdMotionImage_, *nrdMotionView_);
    bind(nrd::ResourceType::IN_NORMAL_ROUGHNESS, *nrdNormalRoughnessImage_,
         *nrdNormalRoughnessView_);
    bind(nrd::ResourceType::IN_VIEWZ, *nrdViewZImage_, *nrdViewZView_);
    bind(nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST, *nrdDiffuseInputImage_,
         *nrdDiffuseInputView_);
    bind(nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST, *nrdSpecularInputImage_,
         *nrdSpecularInputView_);
    bind(nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST, *nrdDiffuseOutputImage_,
         *nrdDiffuseOutputView_);
    bind(nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST, *nrdSpecularOutputImage_,
         *nrdSpecularOutputView_);

    writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::nrdDenoising, false);
    nrdDenoiser_->record(commandBuffer, static_cast<std::uint32_t>(context.frameIndex), settings,
                         externalResources);
    writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::nrdDenoising, true);
#else
    (void)commandBuffer;
    (void)context;
#endif
}

void Renderer::recordNrdResolvePass(VkCommandBuffer commandBuffer,
                                    const GraphFrameContext& context) {
#if RENDERER_ENABLE_NRD
    if (!pathNrdResolvePipeline_ || !pathNrdResolveDescriptorPool_) {
        throw std::runtime_error("NRD resolve requires a complete compute pipeline");
    }
    writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::nrdResolve, false);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      pathNrdResolvePipeline_->handle());
    const VkDescriptorSet descriptorSet = pathNrdResolveDescriptorPool_->set(context.frameIndex);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pathNrdResolvePipeline_->layout(), 0U, 1U, &descriptorSet, 0U,
                            nullptr);
    constexpr std::uint32_t workgroupSize = 8U;
    const VkExtent2D extent = renderExtent();
    vkCmdDispatch(commandBuffer, (extent.width + workgroupSize - 1U) / workgroupSize,
                  (extent.height + workgroupSize - 1U) / workgroupSize, 1U);
    writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::nrdResolve, true);
#else
    (void)commandBuffer;
    (void)context;
#endif
}

void Renderer::recordPathAovReadbackPass(VkCommandBuffer commandBuffer,
                                         const GraphFrameContext& context) {
    if (!pathAovReadbackBuffer_ || context.historyWriteIndex >= framesInFlight) {
        throw std::runtime_error("Path AOV readback requires complete frame resources");
    }
    const std::array<VkImage, pathAovReadbackTargetCount> images{
        pathTracingAccumulationImage_->handle(),
        pathTracingDirectImage_->handle(),
        pathTracingSurfaceImage_->handle(),
        pathTracingNormalImage_->handle(),
        pathTracingMotionDepthImage_->handle(),
        pathTracingDepthImage_->handle(),
        pathHistoryValidityImage_->handle(),
        pathReconstructionImages_[context.historyWriteIndex]->handle(),
        pathDenoisedImage_->handle(),
    };
    const VkExtent2D extent = renderExtent();
    writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::pathAovReadback, false);
    for (std::size_t index = 0U; index < images.size(); ++index) {
        const VkBufferImageCopy copy{
            .bufferOffset = pathAovReadbackOffsets_[index],
            .bufferRowLength = 0U,
            .bufferImageHeight = 0U,
            .imageSubresource =
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .mipLevel = 0U,
                    .baseArrayLayer = 0U,
                    .layerCount = 1U,
                },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent.width, extent.height, 1U},
        };
        vkCmdCopyImageToBuffer(commandBuffer, images[index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               pathAovReadbackBuffer_->handle(), 1U, &copy);
    }
    writeGpuTimestamp(commandBuffer, context.frameIndex, GpuTimingPass::pathAovReadback, true);
}

} // namespace renderer
