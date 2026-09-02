#pragma once

#include <cstdint>

namespace renderer::lighting {

struct DirectionalLightControls {
    float intensity = 0.0F;
    std::uint32_t activeShadowCascadeCount = 0U;
};

// Light energy and shadow visibility are independent controls. Disabling
// shadows must retain an unoccluded sun; disabling the sun avoids shadow work.
[[nodiscard]] constexpr DirectionalLightControls makeDirectionalLightControls(
    float requestedIntensity, bool shadowsEnabled, std::uint32_t shadowCascadeCount) {
    const float intensity = requestedIntensity > 0.0F ? requestedIntensity : 0.0F;
    return {
        .intensity = intensity,
        .activeShadowCascadeCount =
            shadowsEnabled && intensity > 0.0F ? shadowCascadeCount : 0U,
    };
}

} // namespace renderer::lighting
