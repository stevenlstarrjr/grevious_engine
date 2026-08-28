#include "renderer/assets/ImageWriter.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace renderer::assets {

void writeRgba8Png(const std::filesystem::path& path, std::uint32_t width,
                   std::uint32_t height, std::span<const std::uint8_t> rgba) {
    if (path.empty() || width == 0 || height == 0 ||
        static_cast<std::uint64_t>(width) * height * 4U != rgba.size() ||
        width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        height > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("PNG output requires a valid path and RGBA8 image");
    }
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code error;
        std::filesystem::create_directories(parent, error);
        if (error) {
            throw std::runtime_error("Could not create PNG output directory: " +
                                     parent.string());
        }
    }
    const std::string nativePath = path.string();
    const int stride = static_cast<int>(static_cast<std::uint64_t>(width) * 4U);
    if (stbi_write_png(nativePath.c_str(), static_cast<int>(width), static_cast<int>(height), 4,
                       rgba.data(), stride) == 0) {
        throw std::runtime_error("Could not write PNG output: " + nativePath);
    }
}

} // namespace renderer::assets
