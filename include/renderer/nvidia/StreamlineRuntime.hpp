#pragma once

#include <volk.h>

#include <cstdint>
#include <array>
#include <filesystem>
#include <string_view>

namespace renderer::nvidia {

enum class StreamlineFeature : std::uint32_t {
    dlssSuperResolution,
    dlssRayReconstruction,
};

enum class DlssQuality : std::uint32_t { quality, balanced, performance };

struct StreamlineImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t mipLevels = 1U;
    std::uint32_t arrayLayers = 1U;
    VkImageCreateFlags flags = 0U;
    VkImageUsageFlags usage = 0U;
};

struct DlssFrame {
    std::uint32_t frameIndex = 0U;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    DlssQuality quality = DlssQuality::quality;
    StreamlineImage inputColor;
    StreamlineImage outputColor;
    StreamlineImage depth;
    StreamlineImage motionVectors;
    std::array<float, 16> cameraViewToClip{};
    std::array<float, 16> clipToCameraView{};
    std::array<float, 16> clipToPrevClip{};
    std::array<float, 16> prevClipToClip{};
    std::array<float, 3> cameraPosition{};
    std::array<float, 3> cameraUp{};
    std::array<float, 3> cameraRight{};
    std::array<float, 3> cameraForward{};
    float jitterX = 0.0F;
    float jitterY = 0.0F;
    float cameraNear = 0.0F;
    float cameraFar = 0.0F;
    float cameraFovRadians = 0.0F;
    float cameraAspectRatio = 1.0F;
    float exposureScale = 1.0F;
    bool resetHistory = true;
};

struct RayReconstructionFrame {
    DlssFrame dlss;
    StreamlineImage diffuseAlbedo;
    StreamlineImage specularAlbedo;
    StreamlineImage normalRoughness;
    StreamlineImage specularHitDistance;
    std::array<float, 16> worldToCameraView{};
    std::array<float, 16> cameraViewToWorld{};
};

// Owns the signed NVIDIA Streamline interposer. This object must be created
// before the first Vulkan API call and must outlive the Vulkan instance.
class StreamlineRuntime final {
  public:
    explicit StreamlineRuntime(std::filesystem::path runtimeDirectory);
    ~StreamlineRuntime();

    StreamlineRuntime(const StreamlineRuntime&) = delete;
    StreamlineRuntime& operator=(const StreamlineRuntime&) = delete;

    [[nodiscard]] PFN_vkGetInstanceProcAddr vulkanGetInstanceProcAddr() const;
    [[nodiscard]] bool featureSupported(StreamlineFeature feature,
                                        VkPhysicalDevice physicalDevice) const;
    void evaluateDlss(const DlssFrame& frame);
    void evaluateRayReconstruction(const RayReconstructionFrame& frame);
    void releaseDlssResources();
    void shutdown();
    [[nodiscard]] std::string_view sdkVersion() const { return "2.12.0"; }
    [[nodiscard]] const std::filesystem::path& runtimeDirectory() const {
        return runtimeDirectory_;
    }

  private:
    struct Functions;

    std::filesystem::path runtimeDirectory_;
    void* module_ = nullptr;
    Functions* functions_ = nullptr;
    bool initialized_ = false;
};

} // namespace renderer::nvidia
