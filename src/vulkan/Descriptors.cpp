#include "renderer/vulkan/Descriptors.hpp"

#include "renderer/vulkan/Device.hpp"
#include "renderer/vulkan/VulkanError.hpp"

#include <map>
#include <stdexcept>
#include <vector>

namespace renderer::vulkan {

DescriptorSetLayout::DescriptorSetLayout(const Device& device, std::uint32_t set,
                                         std::span<const DescriptorBinding> bindings,
                                         std::uint32_t variableDescriptorCount)
    : device_(device) {
    std::vector<VkDescriptorSetLayoutBinding> vulkanBindings;
    std::vector<VkDescriptorBindingFlags> bindingFlags;
    for (const DescriptorBinding& binding : bindings) {
        if (binding.set != set) {
            continue;
        }
        DescriptorBinding storedBinding = binding;
        VkDescriptorBindingFlags flags = 0;
        if (storedBinding.count == 0) {
            if (variableDescriptorCount == 0 || variableDescriptorCount_ != 0) {
                throw std::runtime_error(
                    "Runtime descriptor array requires one nonzero variable descriptor count");
            }
            storedBinding.count = variableDescriptorCount;
            variableDescriptorCount_ = variableDescriptorCount;
            updateAfterBind_ = true;
            flags = VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT |
                    VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                    VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
        }
        bindings_.push_back(storedBinding);
        vulkanBindings.push_back({
            .binding = storedBinding.binding,
            .descriptorType = storedBinding.type,
            .descriptorCount = storedBinding.count,
            .stageFlags = storedBinding.stages,
        });
        bindingFlags.push_back(flags);
    }
    if (variableDescriptorCount_ != 0 &&
        (bindingFlags.empty() ||
         (bindingFlags.back() & VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT) == 0)) {
        throw std::runtime_error("Variable descriptor binding must be the highest binding in its set");
    }
    const VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .bindingCount = static_cast<std::uint32_t>(bindingFlags.size()),
        .pBindingFlags = bindingFlags.data(),
    };
    const VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = bindingFlags.empty() ? nullptr : &bindingFlagsInfo,
        .flags = updateAfterBind_
                     ? VkDescriptorSetLayoutCreateFlags{
                           VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT}
                     : VkDescriptorSetLayoutCreateFlags{},
        .bindingCount = static_cast<std::uint32_t>(vulkanBindings.size()),
        .pBindings = vulkanBindings.data(),
    };
    check(vkCreateDescriptorSetLayout(device_.handle(), &layoutInfo, nullptr, &layout_),
          "vkCreateDescriptorSetLayout");
}

DescriptorSetLayout::~DescriptorSetLayout() {
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_.handle(), layout_, nullptr);
    }
}

DescriptorPool::DescriptorPool(const Device& device, const DescriptorSetLayout& layout,
                               std::uint32_t setCount)
    : device_(device) {
    if (setCount == 0) {
        throw std::runtime_error("Descriptor pool set count cannot be zero");
    }
    std::map<VkDescriptorType, std::uint32_t> counts;
    for (const DescriptorBinding& binding : layout.bindings()) {
        counts[binding.type] += binding.count * setCount;
    }
    std::vector<VkDescriptorPoolSize> poolSizes;
    poolSizes.reserve(counts.size());
    for (const auto& [type, count] : counts) {
        poolSizes.push_back({.type = type, .descriptorCount = count});
    }
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = layout.updateAfterBind()
                     ? VkDescriptorPoolCreateFlags{
                           VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT}
                     : VkDescriptorPoolCreateFlags{},
        .maxSets = setCount,
        .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    check(vkCreateDescriptorPool(device_.handle(), &poolInfo, nullptr, &pool_),
          "vkCreateDescriptorPool");

    const std::vector layouts(setCount, layout.handle());
    const std::vector<std::uint32_t> variableCounts(setCount, layout.variableDescriptorCount());
    const VkDescriptorSetVariableDescriptorCountAllocateInfo variableInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
        .descriptorSetCount = setCount,
        .pDescriptorCounts = variableCounts.data(),
    };
    const VkDescriptorSetAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = layout.variableDescriptorCount() == 0 ? nullptr : &variableInfo,
        .descriptorPool = pool_,
        .descriptorSetCount = setCount,
        .pSetLayouts = layouts.data(),
    };
    sets_.resize(setCount);
    try {
        check(vkAllocateDescriptorSets(device_.handle(), &allocateInfo, sets_.data()),
              "vkAllocateDescriptorSets");
    } catch (...) {
        vkDestroyDescriptorPool(device_.handle(), pool_, nullptr);
        pool_ = VK_NULL_HANDLE;
        throw;
    }
}

DescriptorPool::~DescriptorPool() {
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_.handle(), pool_, nullptr);
    }
}

} // namespace renderer::vulkan
