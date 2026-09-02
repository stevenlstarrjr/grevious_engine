#include "renderer/assets/ImageDecoder.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

std::vector<std::byte> makeTinyHdr() {
    constexpr std::string_view header =
        "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n";
    std::vector<std::byte> encoded;
    encoded.reserve(header.size() + 8);
    for (const char character : header) {
        encoded.push_back(static_cast<std::byte>(character));
    }
    constexpr std::uint8_t pixels[]{128, 64, 32, 132, 32, 64, 128, 131};
    for (const std::uint8_t value : pixels) {
        encoded.push_back(static_cast<std::byte>(value));
    }
    return encoded;
}

renderer::assets::DecodedImage makeUnormImage(
    std::uint32_t width, std::uint32_t height,
    renderer::assets::ImageColorSpace colorSpace,
    std::initializer_list<std::array<std::uint8_t, 4>> texels) {
    renderer::assets::DecodedImage result{
        .width = width,
        .height = height,
        .componentType = renderer::assets::ImageComponentType::unorm8,
        .colorSpace = colorSpace,
        .unorm8 = {},
        .float32 = {},
    };
    require(texels.size() == result.texelCount(), "Test image texel count is invalid");
    result.unorm8.reserve(texels.size() * 4U);
    for (const auto& texel : texels) {
        result.unorm8.insert(result.unorm8.end(), texel.begin(), texel.end());
    }
    return result;
}

} // namespace

int main() {
    try {
        namespace ra = renderer::assets;
        const std::filesystem::path samples(RENDERER_GLTF_SAMPLE_ASSETS_DIR);
        const ra::DecodedImage png = ra::loadDecodedImage(
            samples / "Models/BoxTextured/glTF/CesiumLogoFlat.png");
        require(png.valid() && png.componentType == ra::ImageComponentType::unorm8 &&
                    png.colorSpace == ra::ImageColorSpace::srgb && png.width == 256 &&
                    png.height == 256,
                "PNG decode did not preserve dimensions or sRGB byte data");

        const ra::DecodedImage jpeg = ra::loadDecodedImage(
            samples / "Models/CompareMetallic/glTF/Compare_Metallic_img0.jpg",
            ra::ImageColorSpace::linear);
        require(jpeg.valid() && jpeg.componentType == ra::ImageComponentType::unorm8 &&
                    jpeg.colorSpace == ra::ImageColorSpace::linear,
                "JPEG decode did not honor the requested linear interpretation");

        const std::vector<std::byte> hdrBytes = makeTinyHdr();
        const ra::DecodedImage hdr = ra::decodeImage(hdrBytes);
        require(hdr.valid() && hdr.width == 2 && hdr.height == 1 &&
                    hdr.componentType == ra::ImageComponentType::float32 &&
                    hdr.colorSpace == ra::ImageColorSpace::linear,
                "Radiance HDR did not decode to linear RGBA32F");
        require(*std::max_element(hdr.float32.begin(), hdr.float32.end()) > 1.0F,
                "HDR decode clamped radiance to display range");

        bool rejectedBudget = false;
        try {
            (void)ra::decodeImage(hdrBytes, ra::ImageColorSpace::srgb,
                                  {.maxDimension = 1,
                                   .maxEncodedBytes = 1024,
                                   .maxDecodedBytes = 1024});
        } catch (const std::runtime_error&) {
            rejectedBudget = true;
        }
        require(rejectedBudget, "Image decoder ignored its dimension budget");

        const ra::DecodedMipChain srgbMips = ra::generateMipChain(makeUnormImage(
            2, 2, ra::ImageColorSpace::srgb,
            {{0, 0, 0, 255}, {255, 255, 255, 255},
             {0, 0, 0, 255}, {255, 255, 255, 255}}));
        require(srgbMips.valid() && srgbMips.levels.size() == 2 &&
                    srgbMips.levels.back().unorm8[0] >= 187 &&
                    srgbMips.levels.back().unorm8[0] <= 188,
                "sRGB mip generation did not average in linear light");
        require(std::abs(ra::srgbToLinearChannel(0.5F) - 0.21404114F) < 1.0e-6F &&
                    std::abs(ra::srgbToLinearChannel(0.04045F) - 0.0031308F) < 1.0e-6F &&
                    std::abs(ra::linearToSrgbChannel(0.21404114F) - 0.5F) < 1.0e-6F,
                "sRGB transfer functions do not match the glTF/Vulkan reference values");

        const ra::DecodedMipChain oddMips = ra::generateMipChain(makeUnormImage(
            3, 1, ra::ImageColorSpace::linear,
            {{0, 0, 0, 255}, {128, 128, 128, 255}, {255, 255, 255, 255}}));
        require(oddMips.valid() && oddMips.levels.size() == 2 &&
                    oddMips.levels.back().unorm8[0] >= 127 &&
                    oddMips.levels.back().unorm8[0] <= 128,
                "Odd-sized mip generation discarded source texels");

        const ra::DecodedMipChain normalMips = ra::generateMipChain(
            makeUnormImage(2, 2, ra::ImageColorSpace::linear,
                           {{255, 128, 128, 255}, {255, 128, 128, 255},
                            {128, 255, 128, 255}, {128, 255, 128, 255}}),
            ra::ImageMipSemantic::tangentSpaceNormal);
        const std::vector<std::uint8_t>& filteredNormal = normalMips.levels.back().unorm8;
        require(normalMips.valid() && filteredNormal[0] >= 217 && filteredNormal[0] <= 219 &&
                    filteredNormal[1] >= 217 && filteredNormal[1] <= 219 &&
                    filteredNormal[2] >= 127 && filteredNormal[2] <= 129,
                "Normal-map mip generation did not renormalize tangent-space vectors");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
