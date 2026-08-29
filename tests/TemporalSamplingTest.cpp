#include "renderer/temporal/TemporalSampling.hpp"
#include "renderer/temporal/MotionVectors.hpp"
#include "renderer/temporal/TemporalResolve.hpp"
#include "renderer/temporal/TemporalState.hpp"

#include <glm/common.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/geometric.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

bool close(float lhs, float rhs) { return std::abs(lhs - rhs) <= 1.0e-6F; }

void testPbrRadicalInverseReferenceValues() {
    using renderer::temporal::radicalInverse;
    require(close(radicalInverse(2, 0), 0.0F) && close(radicalInverse(2, 1), 0.5F) &&
                close(radicalInverse(2, 2), 0.25F) && close(radicalInverse(2, 3), 0.75F) &&
                close(radicalInverse(3, 1), 1.0F / 3.0F) &&
                close(radicalInverse(3, 2), 2.0F / 3.0F) &&
                close(radicalInverse(3, 3), 1.0F / 9.0F),
            "Radical inverse differs from PBRT's base-2/base-3 reference sequence");

    bool rejected = false;
    try {
        (void)radicalInverse(1, 1);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Radical inverse accepted an invalid base");
}

void testFiniteCenteredJitterCycle() {
    glm::vec2 sum(0.0F);
    std::array<glm::vec2, renderer::temporal::taaJitterPeriod> samples{};
    for (std::uint32_t index = 0; index < samples.size(); ++index) {
        samples[index] = renderer::temporal::taaJitterPixels(index);
        sum += samples[index];
        require(std::isfinite(samples[index].x) && std::isfinite(samples[index].y) &&
                    glm::all(glm::lessThan(glm::abs(samples[index]), glm::vec2(0.5F))),
                "TAA jitter escaped its subpixel bounds");
        for (std::uint32_t previous = 0; previous < index; ++previous) {
            require(glm::length(samples[index] - samples[previous]) > 1.0e-6F,
                    "TAA jitter cycle contains a duplicate sample");
        }
    }
    require(glm::length(sum) < 1.0e-6F, "Finite TAA jitter cycle is not zero mean");
    require(renderer::temporal::taaJitterPixels(renderer::temporal::taaJitterPeriod) == samples[0],
            "TAA jitter cycle is not deterministic");
}

void testExplicitFrameSequencingAndReset() {
    renderer::temporal::TemporalState state;
    glm::mat4 projection = glm::perspectiveRH_ZO(glm::radians(60.0F), 16.0F / 9.0F, 0.1F, 100.0F);
    projection[1][1] *= -1.0F;
    const glm::mat4 firstView(1.0F);
    state.beginFrame(firstView, projection, 1920, 1080, 0, true);
    require(!state.previousCameraValid() && !state.historyValid() &&
                state.previousForRendering().jitteredViewProjection ==
                    state.current().jitteredViewProjection &&
                state.current().unjitteredProjection == projection &&
                state.current().jitteredProjection != projection,
            "First-frame temporal camera fallback is invalid");
    const glm::mat4 committed = state.current().jitteredViewProjection;
    state.commitFrame();

    const glm::mat4 secondView = glm::translate(glm::mat4(1.0F), {-0.1F, 0.0F, 0.0F});
    state.beginFrame(secondView, projection, 1920, 1080, 1, true);
    require(state.previousCameraValid() && state.historyValid() &&
                state.previousForRendering().jitteredViewProjection == committed &&
                state.current().jitteredViewProjection != committed,
            "Previous camera state was overwritten before frame commit");
    state.reset();
    require(!state.previousCameraValid() && !state.historyValid(),
            "Temporal reset retained invalid previous state or history");
}

void testRigidMotionConventions() {
    glm::mat4 projection = glm::perspectiveRH_ZO(glm::radians(60.0F), 1.0F, 0.1F, 10.0F);
    projection[1][1] *= -1.0F;
    const glm::vec3 point(0.0F, 0.0F, -2.0F);
    const glm::mat4 identity(1.0F);

    const auto staticSample =
        renderer::temporal::projectRigidMotion(point, identity, identity, projection, projection);
    require(staticSample.valid && glm::length(staticSample.motionUv) < 1.0e-7F,
            "Static camera/static object produced motion");

    const glm::mat4 cameraMoved = glm::translate(glm::mat4(1.0F), {-0.1F, 0.0F, 0.0F});
    const auto cameraSample = renderer::temporal::projectRigidMotion(
        point, identity, identity, projection * cameraMoved, projection);
    require(cameraSample.valid && cameraSample.motionUv.x < 0.0F,
            "Moving-camera motion has the wrong direction");

    const glm::mat4 objectMoved = glm::translate(glm::mat4(1.0F), {0.1F, 0.0F, 0.0F});
    const auto objectSample = renderer::temporal::projectRigidMotion(point, objectMoved, identity,
                                                                     projection, projection);
    require(objectSample.valid && objectSample.motionUv.x > 0.0F,
            "Moving-object motion has the wrong direction");

    const glm::mat4 objectMovedFurther = glm::translate(glm::mat4(1.0F), {0.2F, 0.0F, 0.0F});
    const auto combinedSample = renderer::temporal::projectRigidMotion(
        point, objectMovedFurther, identity, projection * cameraMoved, projection);
    require(combinedSample.valid && combinedSample.motionUv.x > 0.0F &&
                combinedSample.motionUv.x < objectSample.motionUv.x * 1.01F,
            "Combined camera/object motion is not composed in clip space");

    const auto invalidSample = renderer::temporal::projectRigidMotion(
        {0.0F, 0.0F, 0.0F}, identity, identity, projection, projection);
    require(!invalidSample.valid && std::isfinite(invalidSample.motionUv.x) &&
                std::isfinite(invalidSample.motionUv.y),
            "Invalid perspective division did not fail to finite zero motion");
}

void testJitterCorrectedHistoryReprojection() {
    constexpr std::uint32_t width = 1920;
    constexpr std::uint32_t height = 1080;
    const glm::vec2 currentUnjitteredUv(0.61F, 0.37F);
    const glm::vec2 previousUnjitteredUv(0.56F, 0.40F);
    const glm::vec2 currentJitter(0.25F, -0.25F);
    const glm::vec2 previousJitter(-0.25F, 0.125F);
    const glm::vec2 extent(static_cast<float>(width), static_cast<float>(height));
    const glm::vec2 currentRasterUv = currentUnjitteredUv + currentJitter / extent;
    const glm::vec2 motion = currentUnjitteredUv - previousUnjitteredUv;
    const glm::vec2 reprojected = renderer::temporal::reprojectHistoryUv(
        currentRasterUv, motion, currentJitter, previousJitter, width, height);
    const glm::vec2 expected = previousUnjitteredUv + previousJitter / extent;
    require(glm::length(reprojected - expected) < 1.0e-6F,
            "Jitter-corrected history UV has the wrong sign");

    bool rejected = false;
    try {
        (void)renderer::temporal::reprojectHistoryUv({}, {}, {}, {}, 0, height);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Temporal reprojection accepted a zero render extent");
}

void testTemporalRejectionAndClipping() {
    renderer::temporal::HistoryRejectionInput sample{
        .historyValid = true,
        .motionValid = true,
        .previousUv = {0.5F, 0.5F},
        .expectedPreviousDepth = 0.65F,
        .historyDepth = 0.6505F,
        .depthThreshold = 0.001F,
        .currentNormal = {0.0F, 0.0F, 1.0F},
        .historyNormal = {0.0F, 0.1F, 0.995F},
        .currentSurface = true,
        .historySurface = true,
    };
    require(renderer::temporal::acceptsHistory(sample),
            "Stable matching surface history was rejected");

    sample.historyDepth = 0.66F;
    require(!renderer::temporal::acceptsHistory(sample), "Disoccluded depth history was accepted");
    sample.historyDepth = 0.6505F;
    sample.historyNormal = {1.0F, 0.0F, 0.0F};
    require(!renderer::temporal::acceptsHistory(sample),
            "Discontinuous normal history was accepted");
    sample.historyNormal = {0.0F, 0.0F, 1.0F};
    sample.historySurface = false;
    require(!renderer::temporal::acceptsHistory(sample),
            "Surface-to-sky history transition was accepted");
    sample.historySurface = true;
    sample.previousUv.x = 1.01F;
    require(!renderer::temporal::acceptsHistory(sample), "Out-of-bounds history was accepted");
    sample.previousUv.x = 0.5F;
    sample.historyDepth = std::numeric_limits<float>::quiet_NaN();
    require(!renderer::temporal::acceptsHistory(sample), "Non-finite history data was accepted");

    require(close(renderer::temporal::temporalHistoryWeight({}, 1920, 1080), 0.92F) &&
                close(renderer::temporal::temporalHistoryWeight({1.0F, 0.0F}, 1920, 1080), 0.10F),
            "Motion-adaptive history weights differ from the shader convention");
    require(close(renderer::temporal::pathReconstructionHistoryWeight({}, 1920, 1080, 0U),
                  0.80F) &&
                close(renderer::temporal::pathReconstructionHistoryWeight({}, 1920, 1080, 15U),
                      0.20F) &&
                close(renderer::temporal::pathReconstructionHistoryWeight(
                          {1.0F, 0.0F}, 1920, 1080, 0U),
                      0.10F),
            "Path reconstruction weights differ from the shader convention");
    require(glm::length(renderer::temporal::pathTemporalMoments(4.0F, {2.0F, 5.0F}, 0.0F) -
                        glm::vec2(4.0F, 16.0F)) < 1.0e-6F &&
                glm::length(renderer::temporal::pathTemporalMoments(
                                4.0F, {2.0F, 5.0F}, 0.5F) -
                            glm::vec2(3.0F, 10.5F)) < 1.0e-6F &&
                close(renderer::temporal::pathTemporalVariance({3.0F, 10.5F}), 1.5F),
            "Path temporal moment tracking differs from the shader convention");
    require(close(renderer::temporal::pathSpatialFilterStrength(0U), 1.0F) &&
                close(renderer::temporal::pathSpatialFilterStrength(12U), 0.25F) &&
                renderer::temporal::pathSpatialFilterStrength(1024U) < 0.004F,
            "Path spatial filter strength does not fade with progressive convergence");
    const glm::vec3 clipped = renderer::temporal::clipHistoryColor(
        {4.0F, -1.0F, 0.5F}, {0.1F, 0.2F, 0.3F}, {1.0F, 1.5F, 0.8F});
    require(glm::length(clipped - glm::vec3(1.0F, 0.2F, 0.5F)) < 1.0e-6F,
            "Neighborhood color clipping is incorrect");
}

void testPathHistoryRejection() {
    renderer::temporal::PathHistoryRejectionInput sample{
        .historyValid = true,
        .motionValid = true,
        .previousUv = {0.5F, 0.5F},
        .expectedPreviousDepth = 0.65F,
        .historyDepth = 0.6505F,
        .depthThreshold = 0.001F,
        .currentGeometricNormal = {0.0F, 0.0F, 1.0F},
        .historyGeometricNormal = {0.0F, 0.1F, 0.995F},
        .currentSurface = true,
        .historySurface = true,
    };
    require(renderer::temporal::acceptsPathHistory(sample),
            "Stable path surface history was rejected");

    sample.historyDepth = 0.66F;
    require(!renderer::temporal::acceptsPathHistory(sample),
            "Path history accepted a depth disocclusion");
    sample.historyDepth = 0.6505F;
    sample.historyGeometricNormal = {1.0F, 0.0F, 0.0F};
    require(!renderer::temporal::acceptsPathHistory(sample),
            "Path history accepted a geometric-normal discontinuity");
    sample.historyGeometricNormal = {0.0F, 0.0F, 1.0F};
    sample.historySurface = false;
    require(!renderer::temporal::acceptsPathHistory(sample),
            "Path history accepted a surface-to-sky transition");

    sample.currentSurface = false;
    sample.historySurface = false;
    sample.expectedPreviousDepth = std::numeric_limits<float>::quiet_NaN();
    sample.historyDepth = std::numeric_limits<float>::quiet_NaN();
    sample.currentGeometricNormal = {};
    sample.historyGeometricNormal = {};
    require(renderer::temporal::acceptsPathHistory(sample),
            "Valid reprojected sky history required surface metadata");
    sample.previousUv.x = -0.01F;
    require(!renderer::temporal::acceptsPathHistory(sample),
            "Path history accepted out-of-bounds sky reprojection");
    sample.previousUv.x = 0.5F;
    sample.motionValid = false;
    require(!renderer::temporal::acceptsPathHistory(sample),
            "Path history accepted an invalid projection");
}

} // namespace

int main() {
    try {
        testPbrRadicalInverseReferenceValues();
        testFiniteCenteredJitterCycle();
        testExplicitFrameSequencingAndReset();
        testRigidMotionConventions();
        testJitterCorrectedHistoryReprojection();
        testTemporalRejectionAndClipping();
        testPathHistoryRejection();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
