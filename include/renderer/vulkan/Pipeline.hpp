#pragma once

#include <volk.h>

#include <filesystem>
#include <span>

namespace renderer::vulkan {

class Device;
class ShaderModule;

class PipelineCache final {
  public:
    PipelineCache(const Device& device, std::filesystem::path path);
    ~PipelineCache();

    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;

    [[nodiscard]] VkPipelineCache handle() const { return cache_; }

  private:
    const Device& device_;
    std::filesystem::path path_;
    VkPipelineCache cache_ = VK_NULL_HANDLE;
};

struct GraphicsPipelineCreateInfo {
    const ShaderModule* vertexShader = nullptr;
    const ShaderModule* fragmentShader = nullptr;
    VkFormat colorFormat = VK_FORMAT_UNDEFINED;
    // Additional native Vulkan dynamic-rendering color formats. The primary
    // format remains explicit for the common single-target path.
    std::span<const VkFormat> additionalColorFormats{};
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;
    bool depthTestEnable = false;
    bool depthWriteEnable = false;
    VkCompareOp depthCompareOp = VK_COMPARE_OP_LESS;
    bool depthBiasEnable = false;
    float depthBiasConstantFactor = 0.0F;
    float depthBiasSlopeFactor = 0.0F;
    VkCullModeFlags cullMode = VK_CULL_MODE_BACK_BIT;
    // The renderer uses counter-clockwise mesh winding and flips projection Y
    // for Vulkan clip space, preserving CCW front faces in framebuffer space.
    VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkPipelineCache cache = VK_NULL_HANDLE;
    std::span<const VkDescriptorSetLayout> descriptorSetLayouts{};
    std::span<const VkPushConstantRange> pushConstantRanges{};
    std::span<const VkVertexInputBindingDescription> vertexBindings{};
    std::span<const VkVertexInputAttributeDescription> vertexAttributes{};
};

class GraphicsPipeline final {
  public:
    GraphicsPipeline(const Device& device, const GraphicsPipelineCreateInfo& createInfo);
    ~GraphicsPipeline();

    GraphicsPipeline(const GraphicsPipeline&) = delete;
    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;

    [[nodiscard]] VkPipeline handle() const { return pipeline_; }
    [[nodiscard]] VkPipelineLayout layout() const { return layout_; }

  private:
    const Device& device_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

struct ComputePipelineCreateInfo {
    const ShaderModule* shader = nullptr;
    VkPipelineCache cache = VK_NULL_HANDLE;
    std::span<const VkDescriptorSetLayout> descriptorSetLayouts{};
    std::span<const VkPushConstantRange> pushConstantRanges{};
};

class ComputePipeline final {
  public:
    ComputePipeline(const Device& device, const ComputePipelineCreateInfo& createInfo);
    ~ComputePipeline();

    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;

    [[nodiscard]] VkPipeline handle() const { return pipeline_; }
    [[nodiscard]] VkPipelineLayout layout() const { return layout_; }

  private:
    const Device& device_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

} // namespace renderer::vulkan
