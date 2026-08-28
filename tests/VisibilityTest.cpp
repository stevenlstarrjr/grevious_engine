#include "renderer/visibility/Visibility.hpp"
#include "renderer/gpu_scene/GpuScene.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

namespace rg = renderer::gpu_scene;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

rg::InstanceRecord instance(renderer::RenderLayerMask mask) {
    rg::InstanceRecord result;
    const glm::uvec2 packed = rg::packRenderLayerMask(mask);
    result.visibilityLayer = {packed, 0U, 0U};
    return result;
}

void testMaskSemantics() {
    using renderer::RenderLayer;
    using renderer::renderLayerMask;

    const rg::InstanceRecord defaultInstance;
    require(rg::instanceVisible(defaultInstance, renderer::defaultViewVisibilityMask),
            "A default instance is not visible to the default view");

    const rg::InstanceRecord world = instance(renderLayerMask(RenderLayer::world));
    require(rg::instanceVisible(world, renderLayerMask(RenderLayer::world)),
            "One matching layer was rejected");
    require(!rg::instanceVisible(world, renderLayerMask(RenderLayer::characters)),
            "One non-matching layer was accepted");

    renderer::RenderLayerMask worldAndEffects = RenderLayer::world | RenderLayer::effects;
    require(rg::instanceVisible(instance(worldAndEffects), renderLayerMask(RenderLayer::effects)),
            "A multi-layer instance did not match one view layer");

    const renderer::RenderLayerMask charactersAndVegetation =
        RenderLayer::characters | RenderLayer::vegetation;
    require(rg::instanceVisible(instance(renderLayerMask(RenderLayer::vegetation)),
                                charactersAndVegetation),
            "A multi-layer view did not match one instance layer");
    require(!rg::instanceVisible(instance(0), renderer::allRenderLayersMask),
            "A zero instance mask must always remain hidden");
    require(!rg::instanceVisible(world, 0), "A zero view mask must reject every instance");
    require(rg::instanceVisible(instance(std::uint64_t{1} << 63U), renderer::allRenderLayersMask),
            "The all-layer view did not admit a high-bit layer");
}

void testHelpersAndGpuRepresentation() {
    using renderer::RenderLayer;
    renderer::RenderLayerMask mask = 0;
    renderer::setRenderLayer(mask, RenderLayer::world);
    renderer::setRenderLayer(mask, RenderLayer::debug);
    require(renderer::hasRenderLayer(mask, RenderLayer::world) &&
                renderer::hasRenderLayer(mask, RenderLayer::debug),
            "Layer mutation or membership helpers failed");
    renderer::setRenderLayer(mask, RenderLayer::debug, false);
    require(!renderer::hasRenderLayer(mask, RenderLayer::debug), "Layer removal helper failed");

    constexpr renderer::RenderLayerMask wideMask =
        (std::uint64_t{1} << 3U) | (std::uint64_t{1} << 40U) | (std::uint64_t{1} << 63U);
    constexpr glm::uvec2 packed = rg::packRenderLayerMask(wideMask);
    static_assert(rg::unpackRenderLayerMask(packed) == wideMask);
    require(rg::unpackRenderLayerMask(packed) == wideMask,
            "CPU and GPU 64-bit layer representations disagree");
}

void testDeterministicVisibilitySmokeCounts() {
    using renderer::RenderLayer;
    const std::array instances{
        instance(renderer::renderLayerMask(RenderLayer::world)),
        instance(renderer::renderLayerMask(RenderLayer::characters)),
        instance(renderer::renderLayerMask(RenderLayer::debug)),
    };
    const renderer::RenderLayerMask mainMask = RenderLayer::world | RenderLayer::characters |
                                               RenderLayer::vegetation | RenderLayer::effects |
                                               RenderLayer::sky;
    require(rg::countLayerVisibleInstances(instances, mainMask) == 2,
            "Main-view GPU-scene eligibility count should contain A and B only");
    const renderer::RenderLayerMask debugMask = mainMask | RenderLayer::debug;
    require(rg::countLayerVisibleInstances(instances, debugMask) == 3,
            "Debug-enabled GPU-scene eligibility count should contain A, B, and C");
}

} // namespace

int main() {
    try {
        testMaskSemantics();
        testHelpersAndGpuRepresentation();
        testDeterministicVisibilitySmokeCounts();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
