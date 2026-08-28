#include "renderer/vulkan/Sampler.hpp"

#include "renderer/vulkan/Device.hpp"
#include "renderer/vulkan/VulkanError.hpp"

namespace renderer::vulkan {

Sampler::Sampler(const Device& device, const VkSamplerCreateInfo& createInfo) : device_(device) {
    check(vkCreateSampler(device_.handle(), &createInfo, nullptr, &sampler_), "vkCreateSampler");
}

Sampler::~Sampler() {
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_.handle(), sampler_, nullptr);
    }
}

} // namespace renderer::vulkan
