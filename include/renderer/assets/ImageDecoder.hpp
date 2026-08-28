#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace renderer::assets {

enum class ImageComponentType : std::uint8_t {
    unorm8,
    float32,
};

enum class ImageColorSpace : std::uint8_t {
    linear,
    srgb,
};

enum class ImageMipSemantic : std::uint8_t {
    colorOrData,
    tangentSpaceNormal,
};

struct DecodedImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    ImageComponentType componentType = ImageComponentType::unorm8;
    ImageColorSpace colorSpace = ImageColorSpace::linear;
    std::vector<std::uint8_t> unorm8;
    std::vector<float> float32;

    [[nodiscard]] std::size_t texelCount() const;
    [[nodiscard]] std::size_t byteSize() const;
    [[nodiscard]] bool valid() const;
};

struct DecodedMipChain {
    std::vector<DecodedImage> levels;

    [[nodiscard]] bool valid() const;
};

struct ImageDecodeLimits {
    std::uint32_t maxDimension = 16'384;
    std::size_t maxEncodedBytes = 256U * 1024U * 1024U;
    std::size_t maxDecodedBytes = 512U * 1024U * 1024U;
};

// PNG/JPEG inputs use the caller-selected color space. Radiance HDR inputs are
// always returned as unclamped linear RGBA32F, irrespective of that selection.
[[nodiscard]] DecodedImage decodeImage(
    std::span<const std::byte> encoded,
    ImageColorSpace ldrColorSpace = ImageColorSpace::srgb,
    const ImageDecodeLimits& limits = {});

[[nodiscard]] DecodedImage loadDecodedImage(
    const std::filesystem::path& path,
    ImageColorSpace ldrColorSpace = ImageColorSpace::srgb,
    const ImageDecodeLimits& limits = {});

// Builds the complete 2D mip pyramid down to 1x1. sRGB images are averaged in
// linear light; tangent-space normals are decoded, averaged, and renormalized.
[[nodiscard]] DecodedMipChain generateMipChain(
    DecodedImage baseLevel,
    ImageMipSemantic semantic = ImageMipSemantic::colorOrData);

} // namespace renderer::assets
