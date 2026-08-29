#pragma once

#include <NRD.h>

#include <array>
#include <cstdint>
#include <span>

namespace renderer::nvidia {

// Owns NRD's CPU-side denoiser instance. The instance is deliberately kept
// separate from Vulkan resource ownership so its exact texture, pipeline, and
// dispatch contract can be validated before the render graph allocates it.
class NrdContext final {
  public:
    static constexpr nrd::Identifier denoiserIdentifier = 1;

    explicit NrdContext(
        nrd::Denoiser denoiser = nrd::Denoiser::RELAX_DIFFUSE_SPECULAR);
    ~NrdContext();

    NrdContext(const NrdContext&) = delete;
    NrdContext& operator=(const NrdContext&) = delete;
    NrdContext(NrdContext&&) = delete;
    NrdContext& operator=(NrdContext&&) = delete;

    [[nodiscard]] nrd::Denoiser denoiser() const { return denoiser_; }
    [[nodiscard]] const nrd::LibraryDesc& libraryDescription() const;
    [[nodiscard]] const nrd::InstanceDesc& instanceDescription() const;

    // Applies the common per-frame state and returns the exact compute work
    // selected by NRD for this frame. Returned descriptors remain owned by NRD
    // and are valid until the next prepareFrame call.
    [[nodiscard]] std::span<const nrd::DispatchDesc>
    prepareFrame(const nrd::CommonSettings& settings);

  private:
    void configureDenoiser();

    nrd::Denoiser denoiser_;
    nrd::Instance* instance_ = nullptr;
};

[[nodiscard]] nrd::CommonSettings makeNrdCommonSettings(
    std::uint16_t width, std::uint16_t height, std::uint32_t frameIndex,
    const std::array<float, 16>& viewToClip,
    const std::array<float, 16>& previousViewToClip,
    const std::array<float, 16>& worldToView,
    const std::array<float, 16>& previousWorldToView, bool resetHistory);

} // namespace renderer::nvidia
