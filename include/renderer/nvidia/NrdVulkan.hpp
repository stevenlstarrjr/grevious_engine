#pragma once

#include <NRD.h>
#include <volk.h>

#include <cstdint>

namespace renderer::nvidia {

[[nodiscard]] VkFormat nrdFormatToVulkan(nrd::Format format);
[[nodiscard]] std::uint32_t nrdFormatBytesPerTexel(nrd::Format format);

} // namespace renderer::nvidia
