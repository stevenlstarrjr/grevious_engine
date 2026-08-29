#include "renderer/vulkan/HardwarePolicy.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void testNvidiaRtxPolicy() {
    using namespace renderer::vulkan;
    require(acceptsHardware(HardwarePolicy::nvidiaRtx,
                            {.vendorId = nvidiaVendorId,
                             .accelerationStructure = true,
                             .rayTracingPipeline = true}),
            "NVIDIA RTX policy rejected a complete RTX profile");
    require(!acceptsHardware(HardwarePolicy::nvidiaRtx,
                             {.vendorId = 0x1002U,
                              .accelerationStructure = true,
                              .rayTracingPipeline = true}),
            "NVIDIA RTX policy accepted a non-NVIDIA device");
    require(!acceptsHardware(HardwarePolicy::nvidiaRtx,
                             {.vendorId = nvidiaVendorId,
                              .accelerationStructure = true,
                              .rayTracingPipeline = false}),
            "NVIDIA RTX policy accepted a device without an RT pipeline");
}

void testPortablePolicy() {
    using namespace renderer::vulkan;
    require(acceptsHardware(HardwarePolicy::portable, {}),
            "Portable policy rejected a base Vulkan device");
    require(hardwarePolicyName(HardwarePolicy::nvidiaRtx) == "NVIDIA RTX" &&
                hardwarePolicyName(HardwarePolicy::portable) == "portable Vulkan",
            "Hardware policy names are incomplete");
}

} // namespace

int main() {
    try {
        testNvidiaRtxPolicy();
        testPortablePolicy();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
