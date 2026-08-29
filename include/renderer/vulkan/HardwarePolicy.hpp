#pragma once

#include <cstdint>
#include <string_view>

namespace renderer::vulkan {

inline constexpr std::uint32_t nvidiaVendorId = 0x10DEU;

enum class HardwarePolicy {
    nvidiaRtx,
    portable,
};

struct HardwareProfile {
    std::uint32_t vendorId = 0U;
    bool accelerationStructure = false;
    bool rayTracingPipeline = false;
};

[[nodiscard]] constexpr bool acceptsHardware(HardwarePolicy policy,
                                             const HardwareProfile& profile) {
    if (policy == HardwarePolicy::portable) {
        return true;
    }
    return profile.vendorId == nvidiaVendorId && profile.accelerationStructure &&
           profile.rayTracingPipeline;
}

[[nodiscard]] constexpr std::string_view hardwarePolicyName(HardwarePolicy policy) {
    return policy == HardwarePolicy::nvidiaRtx ? "NVIDIA RTX" : "portable Vulkan";
}

} // namespace renderer::vulkan
