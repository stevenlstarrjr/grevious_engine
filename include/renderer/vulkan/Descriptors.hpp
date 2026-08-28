#pragma once

#include "renderer/vulkan/Shader.hpp"

#include <volk.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace renderer::vulkan {

class Device;

class DescriptorSetLayout final {
  public:
    DescriptorSetLayout(const Device& device, std::uint32_t set,
                        std::span<const DescriptorBinding> bindings,
                        std::uint32_t variableDescriptorCount = 0);
    ~DescriptorSetLayout();

    DescriptorSetLayout(const DescriptorSetLayout&) = delete;
    DescriptorSetLayout& operator=(const DescriptorSetLayout&) = delete;

    [[nodiscard]] VkDescriptorSetLayout handle() const { return layout_; }
    [[nodiscard]] std::span<const DescriptorBinding> bindings() const { return bindings_; }
    [[nodiscard]] std::uint32_t variableDescriptorCount() const {
        return variableDescriptorCount_;
    }
    [[nodiscard]] bool updateAfterBind() const { return updateAfterBind_; }

  private:
    const Device& device_;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    std::vector<DescriptorBinding> bindings_;
    std::uint32_t variableDescriptorCount_ = 0;
    bool updateAfterBind_ = false;
};

class DescriptorPool final {
  public:
    DescriptorPool(const Device& device, const DescriptorSetLayout& layout, std::uint32_t setCount);
    ~DescriptorPool();

    DescriptorPool(const DescriptorPool&) = delete;
    DescriptorPool& operator=(const DescriptorPool&) = delete;

    [[nodiscard]] VkDescriptorSet set(std::size_t index) const { return sets_.at(index); }
    [[nodiscard]] std::size_t size() const { return sets_.size(); }

  private:
    const Device& device_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sets_;
};

} // namespace renderer::vulkan
