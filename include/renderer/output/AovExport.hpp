#pragma once

#include <cstdint>
#include <filesystem>
#include <string_view>
#include <vector>

namespace renderer::output {

enum class AovSemantic : std::uint32_t {
    beauty,
    rawBeauty,
    temporalBeauty,
    directLighting,
    indirectLighting,
    albedo,
    roughness,
    geometricNormal,
    shadingNormal,
    motion,
    expectedPreviousDepth,
    depth,
    hitDistance,
    historyValidity,
    variance,
};

[[nodiscard]] std::string_view aovName(AovSemantic semantic);
[[nodiscard]] std::string_view aovDescription(AovSemantic semantic);

struct AovPlane {
    AovSemantic semantic = AovSemantic::beauty;
    std::uint32_t channels = 0;
    std::vector<float> pixels;
};

struct AovFrame {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t frameNumber = 0;
    std::uint32_t accumulatedSamples = 0;
    std::vector<AovPlane> planes;
};

void validateAovFrame(const AovFrame& frame);

class AovFrameSink {
  public:
    virtual ~AovFrameSink() = default;
    virtual void consume(AovFrame frame) = 0;
};

class PfmAovDirectorySink final : public AovFrameSink {
  public:
    explicit PfmAovDirectorySink(std::filesystem::path rootDirectory);
    void consume(AovFrame frame) override;

  private:
    std::filesystem::path rootDirectory_;
};

} // namespace renderer::output
