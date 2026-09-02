#include "renderer/vulkan/Pipeline.hpp"

#include "renderer/core/Log.hpp"
#include "renderer/vulkan/Buffer.hpp"
#include "renderer/vulkan/Device.hpp"
#include "renderer/vulkan/Shader.hpp"
#include "renderer/vulkan/VulkanError.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <ranges>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace renderer::vulkan {
namespace {

constexpr VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1U) & ~(alignment - 1U);
}

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
        .topology = createInfo.topology,
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
        .lineWidth = createInfo.lineWidth,
    };
    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    const VkPipelineColorBlendAttachmentState colorAttachment{
        .blendEnable = createInfo.alphaBlendEnable,
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .alphaBlendOp = VK_BLEND_OP_ADD,
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
        createInfo.shader->reflection().stage != VK_SHADER_STAGE_COMPUTE_BIT ||
        createInfo.entryPoint == nullptr || createInfo.entryPoint[0] == '\0') {
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
                .pName = createInfo.entryPoint,
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

RayTracingPipeline::RayTracingPipeline(const Device& device, MemoryAllocator& allocator,
                                       const RayTracingPipelineCreateInfo& createInfo)
    : device_(device) {
    if (createInfo.rayGenerationShader == nullptr || createInfo.missShaders.empty() ||
        createInfo.closestHitShaders.empty() || createInfo.maxRecursionDepth == 0) {
        throw std::runtime_error("Ray-tracing pipeline create info is incomplete");
    }
    if (!createInfo.anyHitShaders.empty() &&
        createInfo.anyHitShaders.size() != createInfo.closestHitShaders.size()) {
        throw std::runtime_error("Ray-tracing any-hit groups do not match closest-hit groups");
    }
    if (createInfo.rayGenerationShader->reflection().stage != VK_SHADER_STAGE_RAYGEN_BIT_KHR) {
        throw std::runtime_error("Ray-tracing pipeline requires a ray-generation shader");
    }
    if (std::ranges::any_of(createInfo.missShaders,
                            [](const ShaderModule* shader) {
                                return shader == nullptr ||
                                       shader->reflection().stage != VK_SHADER_STAGE_MISS_BIT_KHR;
                            }) ||
        std::ranges::any_of(createInfo.closestHitShaders,
                            [](const ShaderModule* shader) {
                                return shader != nullptr && shader->reflection().stage !=
                                                                VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
                            }) ||
        std::ranges::any_of(createInfo.anyHitShaders, [](const ShaderModule* shader) {
            return shader != nullptr &&
                   shader->reflection().stage != VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
        })) {
        throw std::runtime_error("Ray-tracing shader stage does not match its group");
    }

    const auto& capabilities = device_.optionalCapabilities();
    if (!capabilities.rayTracingPipeline ||
        createInfo.maxRecursionDepth > capabilities.maxRayRecursionDepth) {
        throw std::runtime_error("Requested ray-tracing pipeline is not supported by the device");
    }

    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<std::uint32_t>(createInfo.descriptorSetLayouts.size()),
        .pSetLayouts = createInfo.descriptorSetLayouts.data(),
        .pushConstantRangeCount = static_cast<std::uint32_t>(createInfo.pushConstantRanges.size()),
        .pPushConstantRanges = createInfo.pushConstantRanges.data(),
    };
    check(vkCreatePipelineLayout(device_.handle(), &layoutInfo, nullptr, &layout_),
          "vkCreatePipelineLayout(ray tracing)");

    std::vector<VkPipelineShaderStageCreateInfo> stages;
    stages.reserve(1U + createInfo.missShaders.size() + createInfo.closestHitShaders.size() +
                   createInfo.anyHitShaders.size());
    const auto addStage = [&stages](const ShaderModule& shader, VkShaderStageFlagBits stage) {
        const std::uint32_t index = static_cast<std::uint32_t>(stages.size());
        stages.push_back({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = stage,
            .module = shader.handle(),
            .pName = "main",
        });
        return index;
    };

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;
    groups.reserve(1U + createInfo.missShaders.size() + createInfo.closestHitShaders.size());
    const auto addGeneralGroup = [&groups](std::uint32_t shaderIndex) {
        groups.push_back({
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            .generalShader = shaderIndex,
            .closestHitShader = VK_SHADER_UNUSED_KHR,
            .anyHitShader = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        });
    };

    addGeneralGroup(addStage(*createInfo.rayGenerationShader, VK_SHADER_STAGE_RAYGEN_BIT_KHR));
    for (const ShaderModule* shader : createInfo.missShaders) {
        addGeneralGroup(addStage(*shader, VK_SHADER_STAGE_MISS_BIT_KHR));
    }
    for (std::size_t groupIndex = 0U; groupIndex < createInfo.closestHitShaders.size();
         ++groupIndex) {
        const ShaderModule* closestHitShader = createInfo.closestHitShaders[groupIndex];
        const ShaderModule* anyHitShader =
            createInfo.anyHitShaders.empty() ? nullptr : createInfo.anyHitShaders[groupIndex];
        const std::uint32_t closestHitShaderIndex =
            closestHitShader != nullptr
                ? addStage(*closestHitShader, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
                : VK_SHADER_UNUSED_KHR;
        const std::uint32_t anyHitShaderIndex =
            anyHitShader != nullptr ? addStage(*anyHitShader, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)
                                    : VK_SHADER_UNUSED_KHR;
        groups.push_back({
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            .type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
            .generalShader = VK_SHADER_UNUSED_KHR,
            .closestHitShader = closestHitShaderIndex,
            .anyHitShader = anyHitShaderIndex,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        });
    }

    const VkRayTracingPipelineCreateInfoKHR pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .stageCount = static_cast<std::uint32_t>(stages.size()),
        .pStages = stages.data(),
        .groupCount = static_cast<std::uint32_t>(groups.size()),
        .pGroups = groups.data(),
        .maxPipelineRayRecursionDepth = createInfo.maxRecursionDepth,
        .layout = layout_,
    };
    try {
        check(vkCreateRayTracingPipelinesKHR(device_.handle(), VK_NULL_HANDLE, createInfo.cache, 1,
                                             &pipelineInfo, nullptr, &pipeline_),
              "vkCreateRayTracingPipelinesKHR");

        const VkDeviceSize handleSize = capabilities.shaderGroupHandleSize;
        const VkDeviceSize recordStride =
            alignUp(handleSize, capabilities.shaderGroupHandleAlignment);
        if (recordStride > capabilities.maxShaderGroupStride) {
            throw std::runtime_error("Aligned shader-group record exceeds the device stride limit");
        }
        const VkDeviceSize baseAlignment = capabilities.shaderGroupBaseAlignment;
        const VkDeviceSize rayGenerationOffset = 0;
        const VkDeviceSize missOffset = alignUp(recordStride, baseAlignment);
        const VkDeviceSize missSize = recordStride * createInfo.missShaders.size();
        const VkDeviceSize hitOffset = alignUp(missOffset + missSize, baseAlignment);
        const VkDeviceSize hitSize = recordStride * createInfo.closestHitShaders.size();
        const VkDeviceSize tableSize = hitOffset + hitSize;

        shaderBindingTable_ =
            std::make_unique<Buffer>(device_, allocator,
                                     BufferCreateInfo{
                                         .size = tableSize,
                                         .usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                         .requiredMemory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                         .preferredMemory = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                     });
        std::vector<std::byte> handles(handleSize * groups.size());
        check(vkGetRayTracingShaderGroupHandlesKHR(device_.handle(), pipeline_, 0,
                                                   static_cast<std::uint32_t>(groups.size()),
                                                   handles.size(), handles.data()),
              "vkGetRayTracingShaderGroupHandlesKHR");
        std::vector<std::byte> table(static_cast<std::size_t>(tableSize));
        const auto copyHandle = [&](std::size_t group, VkDeviceSize offset) {
            std::memcpy(table.data() + static_cast<std::size_t>(offset),
                        handles.data() + group * static_cast<std::size_t>(handleSize),
                        static_cast<std::size_t>(handleSize));
        };
        copyHandle(0, rayGenerationOffset);
        for (std::size_t index = 0; index < createInfo.missShaders.size(); ++index) {
            copyHandle(1U + index, missOffset + recordStride * index);
        }
        const std::size_t hitGroupBase = 1U + createInfo.missShaders.size();
        for (std::size_t index = 0; index < createInfo.closestHitShaders.size(); ++index) {
            copyHandle(hitGroupBase + index, hitOffset + recordStride * index);
        }
        shaderBindingTable_->write(table);

        const VkDeviceAddress address = shaderBindingTable_->deviceAddress();
        rayGenerationRegion_ = {address + rayGenerationOffset, recordStride, recordStride};
        missRegion_ = {address + missOffset, recordStride, missSize};
        hitRegion_ = {address + hitOffset, recordStride, hitSize};
    } catch (...) {
        shaderBindingTable_.reset();
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_.handle(), pipeline_, nullptr);
            pipeline_ = VK_NULL_HANDLE;
        }
        vkDestroyPipelineLayout(device_.handle(), layout_, nullptr);
        layout_ = VK_NULL_HANDLE;
        throw;
    }
}

RayTracingPipeline::~RayTracingPipeline() {
    shaderBindingTable_.reset();
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_.handle(), pipeline_, nullptr);
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_.handle(), layout_, nullptr);
    }
}

} // namespace renderer::vulkan
