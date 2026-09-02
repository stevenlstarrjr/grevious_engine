#include "renderer/assets/ImageDecoder.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#define STBI_ONLY_HDR
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace renderer::assets {
float srgbToLinearChannel(float value) {
    return value <= 0.04045F ? value / 12.92F
                             : std::pow((value + 0.055F) / 1.055F, 2.4F);
}

float linearToSrgbChannel(float value) {
    const float clamped = std::clamp(value, 0.0F, 1.0F);
    return clamped <= 0.0031308F ? clamped * 12.92F
                                : 1.055F * std::pow(clamped, 1.0F / 2.4F) - 0.055F;
}

namespace {

constexpr std::size_t channelCount = 4;

std::uint8_t quantizeUnorm(float value) {
    return static_cast<std::uint8_t>(
        std::lround(std::clamp(value, 0.0F, 1.0F) * 255.0F));
}

struct StbiDeleter {
    void operator()(void* pixels) const { stbi_image_free(pixels); }
};

void validateEncodedSize(std::size_t size, const ImageDecodeLimits& limits) {
    if (size == 0) {
        throw std::runtime_error("Image input is empty");
    }
    if (size > limits.maxEncodedBytes ||
        size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Encoded image exceeds the configured byte budget");
    }
    if (limits.maxDimension == 0 || limits.maxDecodedBytes == 0) {
        throw std::runtime_error("Image decode limits must be nonzero");
    }
}

std::size_t checkedElementCount(int width, int height, std::size_t componentSize,
                                const ImageDecodeLimits& limits) {
    if (width <= 0 || height <= 0 ||
        static_cast<std::uint64_t>(width) > limits.maxDimension ||
        static_cast<std::uint64_t>(height) > limits.maxDimension) {
        throw std::runtime_error("Decoded image dimensions exceed the configured limit");
    }
    const std::uint64_t texels = static_cast<std::uint64_t>(width) *
                                 static_cast<std::uint64_t>(height);
    const std::uint64_t elements = texels * channelCount;
    const std::uint64_t bytes = elements * componentSize;
    if (elements > std::numeric_limits<std::size_t>::max() ||
        bytes > limits.maxDecodedBytes) {
        throw std::runtime_error("Decoded image exceeds the configured byte budget");
    }
    return static_cast<std::size_t>(elements);
}

std::runtime_error decodeError() {
    const char* reason = stbi_failure_reason();
    return std::runtime_error("Image decode failed: " +
                              std::string(reason != nullptr ? reason : "unknown format error"));
}

} // namespace

std::size_t DecodedImage::texelCount() const {
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}

std::size_t DecodedImage::byteSize() const {
    return componentType == ImageComponentType::float32
               ? float32.size() * sizeof(float)
               : unorm8.size() * sizeof(std::uint8_t);
}

bool DecodedImage::valid() const {
    if (width == 0 || height == 0) {
        return false;
    }
    const std::size_t expected = texelCount() * channelCount;
    if (componentType == ImageComponentType::float32) {
        return float32.size() == expected && unorm8.empty();
    }
    return unorm8.size() == expected && float32.empty();
}

bool DecodedMipChain::valid() const {
    if (levels.empty() || !levels.front().valid()) {
        return false;
    }
    const ImageComponentType componentType = levels.front().componentType;
    const ImageColorSpace colorSpace = levels.front().colorSpace;
    std::uint32_t expectedWidth = levels.front().width;
    std::uint32_t expectedHeight = levels.front().height;
    for (const DecodedImage& level : levels) {
        if (!level.valid() || level.width != expectedWidth || level.height != expectedHeight ||
            level.componentType != componentType || level.colorSpace != colorSpace) {
            return false;
        }
        expectedWidth = std::max(1U, expectedWidth / 2U);
        expectedHeight = std::max(1U, expectedHeight / 2U);
    }
    return levels.back().width == 1 && levels.back().height == 1;
}

DecodedImage downsample(const DecodedImage& source, ImageMipSemantic semantic) {
    const std::uint32_t destinationWidth = std::max(1U, source.width / 2U);
    const std::uint32_t destinationHeight = std::max(1U, source.height / 2U);
    DecodedImage destination{
        .width = destinationWidth,
        .height = destinationHeight,
        .componentType = source.componentType,
        .colorSpace = source.colorSpace,
        .unorm8 = {},
        .float32 = {},
    };
    const std::size_t destinationElements = destination.texelCount() * channelCount;
    if (source.componentType == ImageComponentType::float32) {
        destination.float32.resize(destinationElements);
    } else {
        destination.unorm8.resize(destinationElements);
    }

    const auto read = [&](std::uint32_t x, std::uint32_t y, std::size_t channel) {
        x = std::min(x, source.width - 1U);
        y = std::min(y, source.height - 1U);
        const std::size_t offset =
            (static_cast<std::size_t>(y) * source.width + x) * channelCount + channel;
        return source.componentType == ImageComponentType::float32
                   ? source.float32[offset]
                   : static_cast<float>(source.unorm8[offset]) / 255.0F;
    };
    const auto write = [&](std::uint32_t x, std::uint32_t y, std::size_t channel, float value) {
        const std::size_t offset =
            (static_cast<std::size_t>(y) * destinationWidth + x) * channelCount + channel;
        if (destination.componentType == ImageComponentType::float32) {
            destination.float32[offset] = value;
        } else {
            destination.unorm8[offset] = quantizeUnorm(value);
        }
    };

    for (std::uint32_t y = 0; y < destinationHeight; ++y) {
        for (std::uint32_t x = 0; x < destinationWidth; ++x) {
            std::array<float, channelCount> average{};
            const std::uint32_t sourceXBegin = x * source.width / destinationWidth;
            const std::uint32_t sourceXEnd = (x + 1U) * source.width / destinationWidth;
            const std::uint32_t sourceYBegin = y * source.height / destinationHeight;
            const std::uint32_t sourceYEnd = (y + 1U) * source.height / destinationHeight;
            const std::uint32_t sampleCount =
                (sourceXEnd - sourceXBegin) * (sourceYEnd - sourceYBegin);
            const float sampleWeight = 1.0F / static_cast<float>(sampleCount);
            for (std::uint32_t sourceY = sourceYBegin; sourceY < sourceYEnd; ++sourceY) {
                for (std::uint32_t sourceX = sourceXBegin; sourceX < sourceXEnd; ++sourceX) {
                    for (std::size_t channel = 0; channel < channelCount; ++channel) {
                        float value = read(sourceX, sourceY, channel);
                        if (source.colorSpace == ImageColorSpace::srgb && channel < 3U) {
                            value = srgbToLinearChannel(value);
                        }
                        average[channel] += value * sampleWeight;
                    }
                }
            }
            if (semantic == ImageMipSemantic::tangentSpaceNormal) {
                float normalX = average[0] * 2.0F - 1.0F;
                float normalY = average[1] * 2.0F - 1.0F;
                float normalZ = average[2] * 2.0F - 1.0F;
                const float lengthSquared =
                    normalX * normalX + normalY * normalY + normalZ * normalZ;
                if (lengthSquared > 1.0e-12F) {
                    const float inverseLength = 1.0F / std::sqrt(lengthSquared);
                    normalX *= inverseLength;
                    normalY *= inverseLength;
                    normalZ *= inverseLength;
                } else {
                    normalX = 0.0F;
                    normalY = 0.0F;
                    normalZ = 1.0F;
                }
                average[0] = normalX * 0.5F + 0.5F;
                average[1] = normalY * 0.5F + 0.5F;
                average[2] = normalZ * 0.5F + 0.5F;
            } else if (source.colorSpace == ImageColorSpace::srgb) {
                for (std::size_t channel = 0; channel < 3U; ++channel) {
                    average[channel] = linearToSrgbChannel(average[channel]);
                }
            }
            for (std::size_t channel = 0; channel < channelCount; ++channel) {
                write(x, y, channel, average[channel]);
            }
        }
    }
    return destination;
}

DecodedImage decodeImage(std::span<const std::byte> encoded, ImageColorSpace ldrColorSpace,
                         const ImageDecodeLimits& limits) {
    validateEncodedSize(encoded.size(), limits);
    const auto* bytes = reinterpret_cast<const stbi_uc*>(encoded.data());
    const int byteCount = static_cast<int>(encoded.size());
    int width = 0;
    int height = 0;
    int sourceChannels = 0;
    if (stbi_info_from_memory(bytes, byteCount, &width, &height, &sourceChannels) == 0) {
        throw decodeError();
    }
    (void)sourceChannels;

    const bool hdr = stbi_is_hdr_from_memory(bytes, byteCount) != 0;
    const std::size_t elementCount =
        checkedElementCount(width, height, hdr ? sizeof(float) : sizeof(stbi_uc), limits);
    DecodedImage result{
        .width = static_cast<std::uint32_t>(width),
        .height = static_cast<std::uint32_t>(height),
        .componentType = hdr ? ImageComponentType::float32 : ImageComponentType::unorm8,
        .colorSpace = hdr ? ImageColorSpace::linear : ldrColorSpace,
        .unorm8 = {},
        .float32 = {},
    };
    if (hdr) {
        const std::unique_ptr<float, StbiDeleter> pixels(
            stbi_loadf_from_memory(bytes, byteCount, &width, &height, &sourceChannels,
                                   static_cast<int>(channelCount)));
        if (!pixels) {
            throw decodeError();
        }
        result.float32.assign(pixels.get(), pixels.get() + elementCount);
    } else {
        const std::unique_ptr<stbi_uc, StbiDeleter> pixels(
            stbi_load_from_memory(bytes, byteCount, &width, &height, &sourceChannels,
                                  static_cast<int>(channelCount)));
        if (!pixels) {
            throw decodeError();
        }
        result.unorm8.assign(pixels.get(), pixels.get() + elementCount);
    }
    return result;
}

DecodedImage loadDecodedImage(const std::filesystem::path& path,
                              ImageColorSpace ldrColorSpace,
                              const ImageDecodeLimits& limits) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Could not open image: " + path.string());
    }
    const std::streamsize length = input.tellg();
    if (length <= 0 || static_cast<std::uint64_t>(length) > limits.maxEncodedBytes) {
        throw std::runtime_error("Encoded image exceeds the configured byte budget: " +
                                 path.string());
    }
    input.seekg(0);
    std::vector<std::byte> encoded(static_cast<std::size_t>(length));
    input.read(reinterpret_cast<char*>(encoded.data()), length);
    if (!input) {
        throw std::runtime_error("Could not read image: " + path.string());
    }
    return decodeImage(encoded, ldrColorSpace, limits);
}

DecodedMipChain generateMipChain(DecodedImage baseLevel, ImageMipSemantic semantic) {
    if (!baseLevel.valid()) {
        throw std::runtime_error("Cannot generate mipmaps for an invalid image");
    }
    if (semantic == ImageMipSemantic::tangentSpaceNormal &&
        (baseLevel.componentType != ImageComponentType::unorm8 ||
         baseLevel.colorSpace != ImageColorSpace::linear)) {
        throw std::runtime_error("Normal-map mip generation requires linear UNORM8 input");
    }
    DecodedMipChain result;
    result.levels.push_back(std::move(baseLevel));
    while (result.levels.back().width > 1 || result.levels.back().height > 1) {
        result.levels.push_back(downsample(result.levels.back(), semantic));
    }
    return result;
}

} // namespace renderer::assets
