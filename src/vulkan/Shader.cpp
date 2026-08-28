#include "renderer/vulkan/Shader.hpp"

#include "renderer/vulkan/Device.hpp"
#include "renderer/vulkan/VulkanError.hpp"

#include <spirv/unified1/spirv.hpp>

#include <algorithm>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

namespace renderer::vulkan {
namespace {

struct SpirvId {
    spv::Op operation = spv::OpNop;
    std::uint32_t typeId = 0;
    std::uint32_t storageClass = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t elementType = 0;
    std::uint32_t lengthId = 0;
    std::uint32_t constantValue = 0;
    std::uint32_t imageDimension = 0;
    std::uint32_t imageSampled = 0;
    std::optional<std::uint32_t> set;
    std::optional<std::uint32_t> binding;
    std::optional<std::uint32_t> location;
    bool block = false;
    bool bufferBlock = false;
    bool builtIn = false;
};

VkShaderStageFlagBits stageForExecutionModel(spv::ExecutionModel model) {
    switch (model) {
    case spv::ExecutionModelVertex:
        return VK_SHADER_STAGE_VERTEX_BIT;
    case spv::ExecutionModelTessellationControl:
        return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    case spv::ExecutionModelTessellationEvaluation:
        return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    case spv::ExecutionModelGeometry:
        return VK_SHADER_STAGE_GEOMETRY_BIT;
    case spv::ExecutionModelFragment:
        return VK_SHADER_STAGE_FRAGMENT_BIT;
    case spv::ExecutionModelGLCompute:
        return VK_SHADER_STAGE_COMPUTE_BIT;
    case spv::ExecutionModelRayGenerationKHR:
        return VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    case spv::ExecutionModelIntersectionKHR:
        return VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
    case spv::ExecutionModelAnyHitKHR:
        return VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    case spv::ExecutionModelClosestHitKHR:
        return VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    case spv::ExecutionModelMissKHR:
        return VK_SHADER_STAGE_MISS_BIT_KHR;
    case spv::ExecutionModelCallableKHR:
        return VK_SHADER_STAGE_CALLABLE_BIT_KHR;
    case spv::ExecutionModelTaskEXT:
        return VK_SHADER_STAGE_TASK_BIT_EXT;
    case spv::ExecutionModelMeshEXT:
        return VK_SHADER_STAGE_MESH_BIT_EXT;
    default:
        throw std::runtime_error("Unsupported SPIR-V execution model");
    }
}

std::uint32_t unwrapDescriptorArray(std::uint32_t typeId, const std::vector<SpirvId>& ids,
                                    std::uint32_t& count) {
    while (typeId < ids.size() && ids[typeId].operation == spv::OpTypeArray) {
        const SpirvId& array = ids[typeId];
        if (array.lengthId >= ids.size() || ids[array.lengthId].operation != spv::OpConstant) {
            throw std::runtime_error("SPIR-V descriptor array has a non-constant length");
        }
        count *= ids[array.lengthId].constantValue;
        typeId = array.elementType;
    }
    if (typeId < ids.size() && ids[typeId].operation == spv::OpTypeRuntimeArray) {
        count = 0;
        typeId = ids[typeId].elementType;
    }
    return typeId;
}

VkDescriptorType descriptorTypeForVariable(const SpirvId& variable, const std::vector<SpirvId>& ids,
                                           std::uint32_t& count) {
    if (variable.typeId >= ids.size() || ids[variable.typeId].operation != spv::OpTypePointer) {
        throw std::runtime_error("SPIR-V descriptor variable does not use a pointer type");
    }
    std::uint32_t typeId = ids[variable.typeId].elementType;
    typeId = unwrapDescriptorArray(typeId, ids, count);
    if (typeId >= ids.size()) {
        throw std::runtime_error("SPIR-V descriptor type ID is out of bounds");
    }
    const SpirvId& type = ids[typeId];
    const auto storage = static_cast<spv::StorageClass>(variable.storageClass);
    if (storage == spv::StorageClassUniform) {
        if (type.bufferBlock) {
            return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        if (type.block) {
            return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        }
    }
    if (storage == spv::StorageClassStorageBuffer) {
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    if (storage != spv::StorageClassUniformConstant) {
        return VK_DESCRIPTOR_TYPE_MAX_ENUM;
    }
    if (type.operation == spv::OpTypeSampler) {
        return VK_DESCRIPTOR_TYPE_SAMPLER;
    }
    if (type.operation == spv::OpTypeSampledImage) {
        return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    if (type.operation == spv::OpTypeAccelerationStructureKHR) {
        return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    }
    if (type.operation == spv::OpTypeImage) {
        const bool storageImage = type.imageSampled == 2;
        if (type.imageDimension == spv::DimBuffer) {
            return storageImage ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
                                : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
        }
        return storageImage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    }
    throw std::runtime_error("Unsupported SPIR-V descriptor type");
}

} // namespace

ShaderReflection reflectSpirv(std::span<const std::uint32_t> words) {
    if (words.size() < 5 || words[0] != spv::MagicNumber) {
        throw std::runtime_error("Shader is not a valid SPIR-V module");
    }
    const std::uint32_t bound = words[3];
    if (bound == 0 || bound > 1'000'000) {
        throw std::runtime_error("SPIR-V ID bound is invalid");
    }
    std::vector<SpirvId> ids(bound);
    ShaderReflection reflection;

    std::size_t offset = 5;
    while (offset < words.size()) {
        const std::uint32_t instruction = words[offset];
        const std::uint32_t wordCount = instruction >> 16U;
        const auto operation = static_cast<spv::Op>(instruction & 0xffffU);
        if (wordCount == 0 || offset + wordCount > words.size()) {
            throw std::runtime_error("Malformed SPIR-V instruction stream");
        }
        const auto word = [&](std::size_t operand) { return words[offset + operand]; };
        switch (operation) {
        case spv::OpEntryPoint:
            if (reflection.stage == VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM) {
                reflection.stage =
                    stageForExecutionModel(static_cast<spv::ExecutionModel>(word(1)));
            }
            break;
        case spv::OpDecorate: {
            const std::uint32_t target = word(1);
            if (target >= ids.size()) {
                throw std::runtime_error("SPIR-V decoration target is out of bounds");
            }
            const auto decoration = static_cast<spv::Decoration>(word(2));
            if (decoration == spv::DecorationBinding) {
                ids[target].binding = word(3);
            } else if (decoration == spv::DecorationDescriptorSet) {
                ids[target].set = word(3);
            } else if (decoration == spv::DecorationLocation) {
                ids[target].location = word(3);
            } else if (decoration == spv::DecorationBlock) {
                ids[target].block = true;
            } else if (decoration == spv::DecorationBufferBlock) {
                ids[target].bufferBlock = true;
            } else if (decoration == spv::DecorationBuiltIn) {
                ids[target].builtIn = true;
            }
            break;
        }
        case spv::OpTypePointer:
            ids.at(word(1)).operation = operation;
            ids.at(word(1)).storageClass = word(2);
            ids.at(word(1)).elementType = word(3);
            break;
        case spv::OpTypeArray:
            ids.at(word(1)).operation = operation;
            ids.at(word(1)).elementType = word(2);
            ids.at(word(1)).lengthId = word(3);
            break;
        case spv::OpTypeRuntimeArray:
        case spv::OpTypeSampledImage:
            ids.at(word(1)).operation = operation;
            ids.at(word(1)).elementType = word(2);
            break;
        case spv::OpTypeStruct:
        case spv::OpTypeSampler:
        case spv::OpTypeAccelerationStructureKHR:
            ids.at(word(1)).operation = operation;
            break;
        case spv::OpTypeImage:
            ids.at(word(1)).operation = operation;
            ids.at(word(1)).imageDimension = word(3);
            ids.at(word(1)).imageSampled = word(7);
            break;
        case spv::OpConstant:
            ids.at(word(2)).operation = operation;
            ids.at(word(2)).typeId = word(1);
            ids.at(word(2)).constantValue = word(3);
            break;
        case spv::OpVariable:
            ids.at(word(2)).operation = operation;
            ids.at(word(2)).typeId = word(1);
            ids.at(word(2)).storageClass = word(3);
            break;
        default:
            break;
        }
        offset += wordCount;
    }
    if (reflection.stage == VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM) {
        throw std::runtime_error("SPIR-V module has no supported entry point");
    }

    for (const SpirvId& id : ids) {
        if (id.operation != spv::OpVariable) {
            continue;
        }
        const auto storage = static_cast<spv::StorageClass>(id.storageClass);
        if (storage == spv::StorageClassInput && id.location.has_value() && !id.builtIn) {
            reflection.inputLocations.push_back(*id.location);
        }
        if (!id.set.has_value() || !id.binding.has_value()) {
            continue;
        }
        std::uint32_t count = 1;
        const VkDescriptorType type = descriptorTypeForVariable(id, ids, count);
        if (type != VK_DESCRIPTOR_TYPE_MAX_ENUM) {
            reflection.descriptorBindings.push_back({
                .set = *id.set,
                .binding = *id.binding,
                .type = type,
                .count = count,
                .stages = reflection.stage,
            });
        }
    }
    std::ranges::sort(reflection.inputLocations);
    std::ranges::sort(reflection.descriptorBindings, [](const auto& lhs, const auto& rhs) {
        return lhs.set < rhs.set || (lhs.set == rhs.set && lhs.binding < rhs.binding);
    });
    return reflection;
}

ShaderModule::ShaderModule(const Device& device, const std::filesystem::path& path)
    : device_(device) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Unable to open SPIR-V shader: " + path.string());
    }
    const std::streamsize byteCount = input.tellg();
    if (byteCount <= 0 || byteCount % static_cast<std::streamsize>(sizeof(std::uint32_t)) != 0) {
        throw std::runtime_error("SPIR-V shader size is invalid: " + path.string());
    }
    input.seekg(0);
    std::vector<std::uint32_t> words(static_cast<std::size_t>(byteCount) / sizeof(std::uint32_t));
    input.read(reinterpret_cast<char*>(words.data()), byteCount);
    if (!input) {
        throw std::runtime_error("Unable to read SPIR-V shader: " + path.string());
    }
    reflection_ = reflectSpirv(words);
    const VkShaderModuleCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = static_cast<std::size_t>(byteCount),
        .pCode = words.data(),
    };
    check(vkCreateShaderModule(device_.handle(), &createInfo, nullptr, &module_),
          "vkCreateShaderModule");
}

ShaderModule::~ShaderModule() {
    if (module_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_.handle(), module_, nullptr);
    }
}

std::vector<DescriptorBinding>
mergeDescriptorBindings(std::span<const ShaderReflection* const> reflections) {
    std::vector<DescriptorBinding> merged;
    for (const ShaderReflection* reflection : reflections) {
        if (reflection == nullptr) {
            continue;
        }
        for (const DescriptorBinding& incoming : reflection->descriptorBindings) {
            const auto found = std::ranges::find_if(merged, [&](const auto& binding) {
                return binding.set == incoming.set && binding.binding == incoming.binding;
            });
            if (found == merged.end()) {
                merged.push_back(incoming);
            } else {
                if (found->type != incoming.type || found->count != incoming.count) {
                    throw std::runtime_error("Shader descriptor declarations disagree");
                }
                found->stages |= incoming.stages;
            }
        }
    }
    std::ranges::sort(merged, [](const auto& lhs, const auto& rhs) {
        return lhs.set < rhs.set || (lhs.set == rhs.set && lhs.binding < rhs.binding);
    });
    return merged;
}

} // namespace renderer::vulkan
