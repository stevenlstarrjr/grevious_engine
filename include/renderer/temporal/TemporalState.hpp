#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>

#include <cstdint>

namespace renderer::temporal {

struct CameraMatrices {
    glm::mat4 view{1.0F};
    glm::mat4 unjitteredProjection{1.0F};
    glm::mat4 jitteredProjection{1.0F};
    glm::mat4 unjitteredViewProjection{1.0F};
    glm::mat4 jitteredViewProjection{1.0F};
    glm::vec2 jitterPixels{0.0F};
};

// Owns frame-to-frame camera state only. Persistent Vulkan images remain under
// Renderer ownership and are imported into the render graph separately.
class TemporalState final {
  public:
    void beginFrame(const glm::mat4& view, const glm::mat4& unjitteredProjection,
                    std::uint32_t width, std::uint32_t height, std::uint64_t frameIndex,
                    bool jitterEnabled);
    void commitFrame();
    void invalidateHistory();
    void reset();

    [[nodiscard]] const CameraMatrices& current() const;
    [[nodiscard]] const CameraMatrices& previousForRendering() const;
    [[nodiscard]] bool previousCameraValid() const { return previousCameraValid_; }
    [[nodiscard]] bool historyValid() const { return historyValid_; }

  private:
    CameraMatrices current_{};
    CameraMatrices previous_{};
    bool currentReady_ = false;
    bool previousCameraValid_ = false;
    bool historyValid_ = false;
};

} // namespace renderer::temporal
