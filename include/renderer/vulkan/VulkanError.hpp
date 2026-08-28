#pragma once

#include <volk.h>

#include <stdexcept>
#include <string>
#include <string_view>

namespace renderer::vulkan {

inline void check(VkResult result, std::string_view operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
    }
}

} // namespace renderer::vulkan
