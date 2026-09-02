#include "renderer/renderer/Renderer.hpp"

#include "renderer/vulkan/Device.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>
#include <stdexcept>

namespace renderer {
namespace {

constexpr glm::mat4 identityMatrix{1.0F};

} // namespace

void Renderer::setEditorOverlay(EditorOverlayFrame overlay) {
    if (!config_.editorOverlayEnabled) {
        throw std::logic_error("Editor overlay data requires editorOverlayEnabled");
    }
    editorOverlay_ = std::move(overlay);
}

void Renderer::createEditorOverlayResources() {
    if (!config_.editorOverlayEnabled) {
        return;
    }
    if (pathTracingEnabled()) {
        throw std::logic_error("Editor overlays are disabled for the clean path-traced view");
    }

    for (auto& buffer : editorOverlayVertexBuffers_) {
        buffer = std::make_unique<vulkan::Buffer>(
            device_, memoryAllocator_,
            vulkan::BufferCreateInfo{
                .size = editorOverlayBufferSize,
                .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                .requiredMemory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                .preferredMemory = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                .randomHostAccess = false,
            });
    }

    const std::filesystem::path shaderDirectory(RENDERER_SHADER_DIR);
    editorOverlayVertexShader_ = std::make_unique<vulkan::ShaderModule>(
        device_, shaderDirectory / "editor_overlay.vert.spv");
    editorOverlayFragmentShader_ = std::make_unique<vulkan::ShaderModule>(
        device_, shaderDirectory / "editor_overlay.frag.spv");

    constexpr VkVertexInputBindingDescription binding{
        .binding = 0U,
        .stride = sizeof(EditorOverlayVertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    constexpr std::array attributes{
        VkVertexInputAttributeDescription{
            .location = 0U,
            .binding = 0U,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(EditorOverlayVertex, position),
        },
        VkVertexInputAttributeDescription{
            .location = 1U,
            .binding = 0U,
            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
            .offset = offsetof(EditorOverlayVertex, color),
        },
    };
    constexpr VkPushConstantRange matrixPush{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0U,
        .size = sizeof(glm::mat4),
    };
    const auto makePipeline = [&](VkPrimitiveTopology topology, bool depthTest,
                                  bool depthWrite, VkCompareOp comparison,
                                  float depthBiasConstant = 0.0F,
                                  float depthBiasSlope = 0.0F) {
        const bool depthBias = depthBiasConstant != 0.0F || depthBiasSlope != 0.0F;
        return std::make_unique<vulkan::GraphicsPipeline>(
            device_, vulkan::GraphicsPipelineCreateInfo{
                         .vertexShader = editorOverlayVertexShader_.get(),
                         .fragmentShader = editorOverlayFragmentShader_.get(),
                         .colorFormat = swapchain_.format(),
                         .depthFormat = depthFormat_,
                         .depthTestEnable = depthTest,
                         .depthWriteEnable = depthWrite,
                         .depthCompareOp = comparison,
                         .depthBiasEnable = depthBias,
                         .depthBiasConstantFactor = depthBiasConstant,
                         .depthBiasSlopeFactor = depthBiasSlope,
                         .cullMode = VK_CULL_MODE_NONE,
                         .topology = topology,
                         .lineWidth = 1.0F,
                         .alphaBlendEnable = true,
                         .cache = pipelineCache_.handle(),
                         .pushConstantRanges = std::span(&matrixPush, 1),
                         .vertexBindings = std::span(&binding, 1),
                         .vertexAttributes = attributes,
                     });
    };

    editorSceneTrianglePipeline_ =
        makePipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, true, true,
                     VK_COMPARE_OP_LESS);
    // Selected-face tint is intentionally coplanar with the PBR mesh. Pull it
    // toward the camera in depth only; moving the actual vertices would make
    // the highlight change size or detach at grazing angles.
    editorSelectionTrianglePipeline_ =
        makePipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, true, false,
                     VK_COMPARE_OP_LESS_OR_EQUAL, -8.0F, -2.0F);
    editorDepthLinePipeline_ =
        makePipeline(VK_PRIMITIVE_TOPOLOGY_LINE_LIST, true, false,
                     VK_COMPARE_OP_LESS_OR_EQUAL, -2.0F, -1.0F);
    editorDepthTrianglePipeline_ =
        makePipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, true, false,
                     VK_COMPARE_OP_LESS_OR_EQUAL);
    editorOverlayTrianglePipeline_ =
        makePipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, false, false,
                     VK_COMPARE_OP_ALWAYS);
    editorOverlayLinePipeline_ =
        makePipeline(VK_PRIMITIVE_TOPOLOGY_LINE_LIST, false, false,
                     VK_COMPARE_OP_ALWAYS);
}

void Renderer::updateEditorOverlayBuffer(std::size_t frameIndex) {
    if (!config_.editorOverlayEnabled) {
        return;
    }
    if (frameIndex >= editorOverlayVertexBuffers_.size() ||
        !editorOverlayVertexBuffers_[frameIndex]) {
        throw std::logic_error("Editor overlay frame buffer is unavailable");
    }

    std::vector<EditorOverlayVertex> packed;
    const auto totalVertexCount =
        editorOverlay_.sceneTriangles.size() +
        editorOverlay_.selectedFaceTriangles.size() +
        editorOverlay_.gridLines.size() +
        editorOverlay_.xRayMeshLines.size() +
        editorOverlay_.xRayDepthScreenTriangles.size() +
        editorOverlay_.xRayTriangles.size() + editorOverlay_.meshLines.size() +
        editorOverlay_.overlayWorldLines.size() +
        editorOverlay_.depthScreenTriangles.size() +
        editorOverlay_.overlayScreenTriangles.size() +
        editorOverlay_.overlayScreenLines.size();
    packed.reserve(totalVertexCount);

    const auto append = [&packed](std::vector<EditorOverlayVertex> const& vertices) {
        const EditorOverlayRange range{
            .first = static_cast<std::uint32_t>(packed.size()),
            .count = static_cast<std::uint32_t>(vertices.size()),
        };
        packed.insert(packed.end(), vertices.begin(), vertices.end());
        return range;
    };
    editorOverlayRanges_ = {
        .sceneTriangles = append(editorOverlay_.sceneTriangles),
        .selectedFaceTriangles = append(editorOverlay_.selectedFaceTriangles),
        .gridLines = append(editorOverlay_.gridLines),
        .xRayMeshLines = append(editorOverlay_.xRayMeshLines),
        .xRayDepthScreenTriangles = append(editorOverlay_.xRayDepthScreenTriangles),
        .xRayTriangles = append(editorOverlay_.xRayTriangles),
        .meshLines = append(editorOverlay_.meshLines),
        .overlayWorldLines = append(editorOverlay_.overlayWorldLines),
        .depthScreenTriangles = append(editorOverlay_.depthScreenTriangles),
        .overlayScreenTriangles = append(editorOverlay_.overlayScreenTriangles),
        .overlayScreenLines = append(editorOverlay_.overlayScreenLines),
    };

    const auto bytes = std::as_bytes(std::span(packed));
    if (bytes.size() > editorOverlayBufferSize) {
        throw std::runtime_error("Editor overlay exceeds the 4 MiB per-frame vertex budget");
    }
    if (!bytes.empty()) {
        editorOverlayVertexBuffers_[frameIndex]->write(bytes);
    }
}

void Renderer::recordEditorOverlayPass(VkCommandBuffer commandBuffer,
                                       const render_graph::RenderGraph& graph,
                                       const GraphFrameContext& context) {
    if (!config_.editorOverlayEnabled ||
        context.frameIndex >= editorOverlayVertexBuffers_.size()) {
        return;
    }

    const VkClearValue clearColor{.color = {{0.0307F, 0.0369F, 0.0369F, 1.0F}}};
    const VkClearValue clearDepth{.depthStencil = {1.0F, 0U}};
    const VkRenderingAttachmentInfo colorAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = swapchain_.imageView(context.imageIndex),
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = editorOverlay_.replaceScene ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                              : VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = clearColor,
    };
    const VkRenderingAttachmentInfo depthAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = graph.imageView(graphDepthImage_),
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp = editorOverlay_.replaceScene ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                              : VK_ATTACHMENT_LOAD_OP_LOAD,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = clearDepth,
    };
    const VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {.offset = {0, 0}, .extent = swapchain_.extent()},
        .layerCount = 1U,
        .colorAttachmentCount = 1U,
        .pColorAttachments = &colorAttachment,
        .pDepthAttachment = &depthAttachment,
    };
    vkCmdBeginRendering(commandBuffer, &renderingInfo);
    const VkViewport viewport{
        .x = 0.0F,
        .y = 0.0F,
        .width = static_cast<float>(swapchain_.extent().width),
        .height = static_cast<float>(swapchain_.extent().height),
        .minDepth = 0.0F,
        .maxDepth = 1.0F,
    };
    const VkRect2D scissor{.offset = {0, 0}, .extent = swapchain_.extent()};
    vkCmdSetViewport(commandBuffer, 0U, 1U, &viewport);
    vkCmdSetScissor(commandBuffer, 0U, 1U, &scissor);
    const VkBuffer vertexBuffer = editorOverlayVertexBuffers_[context.frameIndex]->handle();
    constexpr VkDeviceSize offset = 0U;
    vkCmdBindVertexBuffers(commandBuffer, 0U, 1U, &vertexBuffer, &offset);

    const auto draw = [&](vulkan::GraphicsPipeline const& pipeline,
                          EditorOverlayRange range, glm::mat4 const& matrix) {
        if (range.count == 0U) {
            return;
        }
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipeline.handle());
        vkCmdPushConstants(commandBuffer, pipeline.layout(), VK_SHADER_STAGE_VERTEX_BIT,
                           0U, sizeof(matrix), &matrix);
        vkCmdDraw(commandBuffer, range.count, 1U, range.first, 0U);
    };

    draw(*editorDepthLinePipeline_, editorOverlayRanges_.gridLines,
         editorOverlay_.viewProjection);
    draw(*editorSceneTrianglePipeline_, editorOverlayRanges_.sceneTriangles,
         editorOverlay_.viewProjection);
    draw(*editorSelectionTrianglePipeline_,
         editorOverlayRanges_.selectedFaceTriangles,
         editorOverlay_.viewProjection);
    draw(*editorDepthLinePipeline_, editorOverlayRanges_.xRayMeshLines,
         editorOverlay_.viewProjection);
    draw(*editorDepthTrianglePipeline_, editorOverlayRanges_.xRayDepthScreenTriangles,
         identityMatrix);
    draw(*editorDepthTrianglePipeline_, editorOverlayRanges_.xRayTriangles,
         editorOverlay_.viewProjection);
    draw(*editorDepthLinePipeline_, editorOverlayRanges_.meshLines,
         editorOverlay_.viewProjection);
    draw(*editorOverlayLinePipeline_, editorOverlayRanges_.overlayWorldLines,
         editorOverlay_.viewProjection);
    draw(*editorDepthTrianglePipeline_, editorOverlayRanges_.depthScreenTriangles,
         identityMatrix);
    draw(*editorOverlayTrianglePipeline_, editorOverlayRanges_.overlayScreenTriangles,
         identityMatrix);
    draw(*editorOverlayLinePipeline_, editorOverlayRanges_.overlayScreenLines,
         identityMatrix);
    vkCmdEndRendering(commandBuffer);
}

} // namespace renderer
