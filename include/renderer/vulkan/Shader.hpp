#pragma once

#include <volk.h>

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace renderer::vulkan {

class Device;

struct DescriptorBinding {
    std::uint32_t set = 0;
    std::uint32_t binding = 0;
    VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    std::uint32_t count = 0;
    VkShaderStageFlags stages = 0;
};

struct ShaderReflection {
    VkShaderStageFlagBits stage = VK_SHADER_STAGE_FLAG_BITS_MAX_ENUM;
    std::vector<DescriptorBinding> descriptorBindings;
    std::vector<std::uint32_t> inputLocations;
};

class ShaderModule final {
  public:
    ShaderModule(const Device& device, const std::filesystem::path& path);
    ~ShaderModule();

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;

    [[nodiscard]] VkShaderModule handle() const { return module_; }
    [[nodiscard]] const ShaderReflection& reflection() const { return reflection_; }

  private:
    const Device& device_;
    VkShaderModule module_ = VK_NULL_HANDLE;
    ShaderReflection reflection_{};
};

[[nodiscard]] ShaderReflection reflectSpirv(std::span<const std::uint32_t> words);
[[nodiscard]] std::vector<DescriptorBinding>
mergeDescriptorBindings(std::span<const ShaderReflection* const> reflections);

} // namespace renderer::vulkan
