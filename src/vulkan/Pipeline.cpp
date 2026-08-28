#include "renderer/vulkan/Pipeline.hpp"

#include "renderer/core/Log.hpp"
#include "renderer/vulkan/Device.hpp"
#include "renderer/vulkan/Shader.hpp"
#include "renderer/vulkan/VulkanError.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <ranges>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace renderer::vulkan {
namespace {

std::vector<std::byte> readCache(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return {};
    }
    const std::streamsize size = input.tellg();
    if (size <= 0) {
        return {};
    }
    input.seekg(0);
    std::vector<std::byte> data(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(data.data()), size);
    return input ? data : std::vector<std::byte>{};
}

} // namespace

PipelineCache::PipelineCache(const Device& device, std::filesystem::path path)
    : device_(device), path_(std::move(path)) {
    auto data = readCache(path_);
    VkPipelineCacheCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .initialDataSize = data.size(),
        .pInitialData = data.data(),
    };
    VkResult result = vkCreatePipelineCache(device_.handle(), &createInfo, nullptr, &cache_);
    if (result != VK_SUCCESS && !data.empty()) {
        core::log(core::LogLevel::warning,
                  "Discarding incompatible pipeline cache: ", path_.string());
        createInfo.initialDataSize = 0;
        createInfo.pInitialData = nullptr;
        result = vkCreatePipelineCache(device_.handle(), &createInfo, nullptr, &cache_);
    }
    check(result, "vkCreatePipelineCache");
}

PipelineCache::~PipelineCache() {
    if (cache_ == VK_NULL_HANDLE) {
        return;
    }
    std::size_t size = 0;
    if (vkGetPipelineCacheData(device_.handle(), cache_, &size, nullptr) == VK_SUCCESS &&
        size > 0) {
        std::vector<std::byte> data(size);
        if (vkGetPipelineCacheData(device_.handle(), cache_, &size, data.data()) == VK_SUCCESS) {
            std::error_code error;
            std::filesystem::create_directories(path_.parent_path(), error);
            const auto temporaryPath = path_.string() + ".tmp";
            std::ofstream output(temporaryPath, std::ios::binary | std::ios::trunc);
            if (output) {
                output.write(reinterpret_cast<const char*>(data.data()),
                             static_cast<std::streamsize>(size));
                output.close();
                if (output) {
                    std::filesystem::rename(temporaryPath, path_, error);
                }
            }
            if (error) {
                core::log(core::LogLevel::warning,
                          "Could not persist pipeline cache: ", error.message());
            }
        }
    }
    vkDestroyPipelineCache(device_.handle(), cache_, nullptr);
}

GraphicsPipeline::GraphicsPipeline(const Device& device,
                                   const GraphicsPipelineCreateInfo& createInfo)
    : device_(device) {
    if (createInfo.vertexShader == nullptr ||
        (createInfo.colorFormat == VK_FORMAT_UNDEFINED &&
         createInfo.depthFormat == VK_FORMAT_UNDEFINED) ||
        (createInfo.depthFormat == VK_FORMAT_UNDEFINED && createInfo.fragmentShader == nullptr)) {
        throw std::runtime_error("Graphics pipeline create info is incomplete");
    }
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<std::uint32_t>(createInfo.descriptorSetLayouts.size()),
        .pSetLayouts = createInfo.descriptorSetLayouts.data(),
        .pushConstantRangeCount = static_cast<std::uint32_t>(createInfo.pushConstantRanges.size()),
        .pPushConstantRanges = createInfo.pushConstantRanges.data(),
    };
    check(vkCreatePipelineLayout(device_.handle(), &layoutInfo, nullptr, &layout_),
          "vkCreatePipelineLayout");

    std::vector<VkPipelineShaderStageCreateInfo> stages{VkPipelineShaderStageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = createInfo.vertexShader->handle(),
        .pName = "main",
    }};
    if (createInfo.fragmentShader != nullptr) {
        stages.push_back(VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = createInfo.fragmentShader->handle(),
            .pName = "main",
        });
    }
    const VkPipelineVertexInputStateCreateInfo vertexInput{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount =
            static_cast<std::uint32_t>(createInfo.vertexBindings.size()),
        .pVertexBindingDescriptions = createInfo.vertexBindings.data(),
        .vertexAttributeDescriptionCount =
            static_cast<std::uint32_t>(createInfo.vertexAttributes.size()),
        .pVertexAttributeDescriptions = createInfo.vertexAttributes.data(),
    };
    const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
    const VkPipelineViewportStateCreateInfo viewportState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };
    const VkPipelineRasterizationStateCreateInfo rasterization{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = createInfo.cullMode,
        .frontFace = createInfo.frontFace,
        .depthBiasEnable = createInfo.depthBiasEnable,
        .depthBiasConstantFactor = createInfo.depthBiasConstantFactor,
        .depthBiasSlopeFactor = createInfo.depthBiasSlopeFactor,
        .lineWidth = 1.0F,
    };
    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    const VkPipelineColorBlendAttachmentState colorAttachment{
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const bool hasColorAttachment = createInfo.colorFormat != VK_FORMAT_UNDEFINED;
    std::vector<VkPipelineColorBlendAttachmentState> colorAttachments;
    if (hasColorAttachment) {
        colorAttachments.assign(createInfo.additionalColorFormats.size() + 1U, colorAttachment);
    }
    const VkPipelineColorBlendStateCreateInfo colorBlend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = static_cast<std::uint32_t>(colorAttachments.size()),
        .pAttachments = colorAttachments.data(),
    };
    const VkPipelineDepthStencilStateCreateInfo depthStencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = createInfo.depthTestEnable,
        .depthWriteEnable = createInfo.depthWriteEnable,
        .depthCompareOp = createInfo.depthCompareOp,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
    };
    constexpr std::array dynamicStates{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamicState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };
    std::vector<VkFormat> colorFormats;
    if (hasColorAttachment) {
        colorFormats.reserve(createInfo.additionalColorFormats.size() + 1U);
        colorFormats.push_back(createInfo.colorFormat);
        colorFormats.insert(colorFormats.end(), createInfo.additionalColorFormats.begin(),
                            createInfo.additionalColorFormats.end());
    }
    if (std::ranges::any_of(colorFormats,
                            [](VkFormat format) { return format == VK_FORMAT_UNDEFINED; })) {
        vkDestroyPipelineLayout(device_.handle(), layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
        throw std::runtime_error("Graphics pipeline contains an undefined color format");
    }
    const VkPipelineRenderingCreateInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = static_cast<std::uint32_t>(colorFormats.size()),
        .pColorAttachmentFormats = colorFormats.data(),
        .depthAttachmentFormat = createInfo.depthFormat,
    };
    const VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &renderingInfo,
        .stageCount = static_cast<std::uint32_t>(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = createInfo.depthFormat != VK_FORMAT_UNDEFINED ||
                                      createInfo.depthTestEnable || createInfo.depthWriteEnable
                                  ? &depthStencil
                                  : nullptr,
        .pColorBlendState = &colorBlend,
        .pDynamicState = &dynamicState,
        .layout = layout_,
    };
    try {
        check(vkCreateGraphicsPipelines(device_.handle(), createInfo.cache, 1, &pipelineInfo,
                                        nullptr, &pipeline_),
              "vkCreateGraphicsPipelines");
    } catch (...) {
        vkDestroyPipelineLayout(device_.handle(), layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
        throw;
    }
}

GraphicsPipeline::~GraphicsPipeline() {
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_.handle(), pipeline_, nullptr);
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_.handle(), layout_, nullptr);
    }
}

ComputePipeline::ComputePipeline(const Device& device, const ComputePipelineCreateInfo& createInfo)
    : device_(device) {
    if (createInfo.shader == nullptr ||
        createInfo.shader->reflection().stage != VK_SHADER_STAGE_COMPUTE_BIT) {
        throw std::runtime_error("Compute pipeline requires a compute shader");
    }
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<std::uint32_t>(createInfo.descriptorSetLayouts.size()),
        .pSetLayouts = createInfo.descriptorSetLayouts.data(),
        .pushConstantRangeCount = static_cast<std::uint32_t>(createInfo.pushConstantRanges.size()),
        .pPushConstantRanges = createInfo.pushConstantRanges.data(),
    };
    check(vkCreatePipelineLayout(device_.handle(), &layoutInfo, nullptr, &layout_),
          "vkCreatePipelineLayout(compute)");
    const VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = createInfo.shader->handle(),
                .pName = "main",
            },
        .layout = layout_,
    };
    try {
        check(vkCreateComputePipelines(device_.handle(), createInfo.cache, 1, &pipelineInfo,
                                       nullptr, &pipeline_),
              "vkCreateComputePipelines");
    } catch (...) {
        vkDestroyPipelineLayout(device_.handle(), layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
        throw;
    }
}

ComputePipeline::~ComputePipeline() {
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_.handle(), pipeline_, nullptr);
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_.handle(), layout_, nullptr);
    }
}

} // namespace renderer::vulkan
