#pragma once

#include <volk.h>

namespace renderer::vulkan {

class Device;

class Sampler final {
  public:
    Sampler(const Device& device, const VkSamplerCreateInfo& createInfo);
    ~Sampler();

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    [[nodiscard]] VkSampler handle() const { return sampler_; }

  private:
    const Device& device_;
    VkSampler sampler_ = VK_NULL_HANDLE;
};

} // namespace renderer::vulkan
