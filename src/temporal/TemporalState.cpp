#include "renderer/temporal/TemporalState.hpp"

#include "renderer/temporal/TemporalSampling.hpp"

#include <stdexcept>

namespace renderer::temporal {

void TemporalState::beginFrame(const glm::mat4& view, const glm::mat4& unjitteredProjection,
                               std::uint32_t width, std::uint32_t height,
                               std::uint64_t frameIndex, bool jitterEnabled) {
    if (width == 0 || height == 0) {
        throw std::invalid_argument("Temporal camera state requires a nonzero render extent");
    }
    const glm::vec2 jitter = jitterEnabled ? taaJitterPixels(frameIndex) : glm::vec2(0.0F);
    glm::mat4 jitteredProjection = unjitteredProjection;
    // GLM is column-major. With the renderer's RH projection and positive
    // Vulkan viewport, subtracting these terms shifts raster samples by the
    // requested positive pixel offset after perspective division.
    jitteredProjection[2][0] -= 2.0F * jitter.x / static_cast<float>(width);
    jitteredProjection[2][1] -= 2.0F * jitter.y / static_cast<float>(height);

    current_ = {
        .view = view,
        .unjitteredProjection = unjitteredProjection,
        .jitteredProjection = jitteredProjection,
        .unjitteredViewProjection = unjitteredProjection * view,
        .jitteredViewProjection = jitteredProjection * view,
        .jitterPixels = jitter,
    };
    currentReady_ = true;
}

void TemporalState::commitFrame() {
    if (!currentReady_) {
        throw std::logic_error("Cannot commit temporal state before beginning a frame");
    }
    previous_ = current_;
    previousCameraValid_ = true;
    historyValid_ = true;
    currentReady_ = false;
}

void TemporalState::invalidateHistory() { historyValid_ = false; }

void TemporalState::reset() {
    currentReady_ = false;
    previousCameraValid_ = false;
    historyValid_ = false;
}

const CameraMatrices& TemporalState::current() const {
    if (!currentReady_) {
        throw std::logic_error("Current temporal camera state is unavailable");
    }
    return current_;
}

const CameraMatrices& TemporalState::previousForRendering() const {
    if (!currentReady_) {
        throw std::logic_error("Temporal rendering state is unavailable");
    }
    return previousCameraValid_ ? previous_ : current_;
}

} // namespace renderer::temporal
