#include "renderer/nvidia/NrdVulkan.hpp"

#include <stdexcept>

namespace renderer::nvidia {

VkFormat nrdFormatToVulkan(nrd::Format format) {
    switch (format) {
    case nrd::Format::R8_UNORM: return VK_FORMAT_R8_UNORM;
    case nrd::Format::R8_SNORM: return VK_FORMAT_R8_SNORM;
    case nrd::Format::R8_UINT: return VK_FORMAT_R8_UINT;
    case nrd::Format::R8_SINT: return VK_FORMAT_R8_SINT;
    case nrd::Format::RG8_UNORM: return VK_FORMAT_R8G8_UNORM;
    case nrd::Format::RG8_SNORM: return VK_FORMAT_R8G8_SNORM;
    case nrd::Format::RG8_UINT: return VK_FORMAT_R8G8_UINT;
    case nrd::Format::RG8_SINT: return VK_FORMAT_R8G8_SINT;
    case nrd::Format::RGBA8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case nrd::Format::RGBA8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
    case nrd::Format::RGBA8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
    case nrd::Format::RGBA8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
    case nrd::Format::RGBA8_SRGB: return VK_FORMAT_R8G8B8A8_SRGB;
    case nrd::Format::R16_UNORM: return VK_FORMAT_R16_UNORM;
    case nrd::Format::R16_SNORM: return VK_FORMAT_R16_SNORM;
    case nrd::Format::R16_UINT: return VK_FORMAT_R16_UINT;
    case nrd::Format::R16_SINT: return VK_FORMAT_R16_SINT;
    case nrd::Format::R16_SFLOAT: return VK_FORMAT_R16_SFLOAT;
    case nrd::Format::RG16_UNORM: return VK_FORMAT_R16G16_UNORM;
    case nrd::Format::RG16_SNORM: return VK_FORMAT_R16G16_SNORM;
    case nrd::Format::RG16_UINT: return VK_FORMAT_R16G16_UINT;
    case nrd::Format::RG16_SINT: return VK_FORMAT_R16G16_SINT;
    case nrd::Format::RG16_SFLOAT: return VK_FORMAT_R16G16_SFLOAT;
    case nrd::Format::RGBA16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
    case nrd::Format::RGBA16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
    case nrd::Format::RGBA16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
    case nrd::Format::RGBA16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
    case nrd::Format::RGBA16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case nrd::Format::R32_UINT: return VK_FORMAT_R32_UINT;
    case nrd::Format::R32_SINT: return VK_FORMAT_R32_SINT;
    case nrd::Format::R32_SFLOAT: return VK_FORMAT_R32_SFLOAT;
    case nrd::Format::RG32_UINT: return VK_FORMAT_R32G32_UINT;
    case nrd::Format::RG32_SINT: return VK_FORMAT_R32G32_SINT;
    case nrd::Format::RG32_SFLOAT: return VK_FORMAT_R32G32_SFLOAT;
    case nrd::Format::RGB32_UINT: return VK_FORMAT_R32G32B32_UINT;
    case nrd::Format::RGB32_SINT: return VK_FORMAT_R32G32B32_SINT;
    case nrd::Format::RGB32_SFLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
    case nrd::Format::RGBA32_UINT: return VK_FORMAT_R32G32B32A32_UINT;
    case nrd::Format::RGBA32_SINT: return VK_FORMAT_R32G32B32A32_SINT;
    case nrd::Format::RGBA32_SFLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case nrd::Format::R10_G10_B10_A2_UNORM:
        return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case nrd::Format::R10_G10_B10_A2_UINT:
        return VK_FORMAT_A2B10G10R10_UINT_PACK32;
    case nrd::Format::R11_G11_B10_UFLOAT:
        return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case nrd::Format::R9_G9_B9_E5_UFLOAT:
        return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    case nrd::Format::MAX_NUM:
        break;
    }
    throw std::invalid_argument("Unsupported NRD texture format");
}

std::uint32_t nrdFormatBytesPerTexel(nrd::Format format) {
    switch (format) {
    case nrd::Format::R8_UNORM:
    case nrd::Format::R8_SNORM:
    case nrd::Format::R8_UINT:
    case nrd::Format::R8_SINT:
        return 1;
    case nrd::Format::RG8_UNORM:
    case nrd::Format::RG8_SNORM:
    case nrd::Format::RG8_UINT:
    case nrd::Format::RG8_SINT:
    case nrd::Format::R16_UNORM:
    case nrd::Format::R16_SNORM:
    case nrd::Format::R16_UINT:
    case nrd::Format::R16_SINT:
    case nrd::Format::R16_SFLOAT:
        return 2;
    case nrd::Format::RGBA8_UNORM:
    case nrd::Format::RGBA8_SNORM:
    case nrd::Format::RGBA8_UINT:
    case nrd::Format::RGBA8_SINT:
    case nrd::Format::RGBA8_SRGB:
    case nrd::Format::RG16_UNORM:
    case nrd::Format::RG16_SNORM:
    case nrd::Format::RG16_UINT:
    case nrd::Format::RG16_SINT:
    case nrd::Format::RG16_SFLOAT:
    case nrd::Format::R32_UINT:
    case nrd::Format::R32_SINT:
    case nrd::Format::R32_SFLOAT:
    case nrd::Format::R10_G10_B10_A2_UNORM:
    case nrd::Format::R10_G10_B10_A2_UINT:
    case nrd::Format::R11_G11_B10_UFLOAT:
    case nrd::Format::R9_G9_B9_E5_UFLOAT:
        return 4;
    case nrd::Format::RGBA16_UNORM:
    case nrd::Format::RGBA16_SNORM:
    case nrd::Format::RGBA16_UINT:
    case nrd::Format::RGBA16_SINT:
    case nrd::Format::RGBA16_SFLOAT:
    case nrd::Format::RG32_UINT:
    case nrd::Format::RG32_SINT:
    case nrd::Format::RG32_SFLOAT:
        return 8;
    case nrd::Format::RGB32_UINT:
    case nrd::Format::RGB32_SINT:
    case nrd::Format::RGB32_SFLOAT:
        return 12;
    case nrd::Format::RGBA32_UINT:
    case nrd::Format::RGBA32_SINT:
    case nrd::Format::RGBA32_SFLOAT:
        return 16;
    case nrd::Format::MAX_NUM:
        break;
    }
    throw std::invalid_argument("Unsupported NRD texture format");
}

} // namespace renderer::nvidia
