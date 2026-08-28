#pragma once

#include <cstdint>
#include <filesystem>
#include <span>

namespace renderer::assets {

void writeRgba8Png(const std::filesystem::path& path, std::uint32_t width,
                   std::uint32_t height, std::span<const std::uint8_t> rgba);

} // namespace renderer::assets
