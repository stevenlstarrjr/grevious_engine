#pragma once

#include <cstdint>

namespace renderer {

using RenderLayerMask = std::uint64_t;

// These names are useful defaults for renderer tests and applications. The
// residency and culling systems operate only on masks; they do not maintain a
// string registry or attach gameplay meaning to a bit.
enum class RenderLayer : RenderLayerMask {
    world = 1ULL << 0U,
    characters = 1ULL << 1U,
    vegetation = 1ULL << 2U,
    effects = 1ULL << 3U,
    sky = 1ULL << 4U,
    ui3D = 1ULL << 5U,
    debug = 1ULL << 6U,
};

[[nodiscard]] constexpr RenderLayerMask renderLayerMask(RenderLayer layer) {
    return static_cast<RenderLayerMask>(layer);
}

[[nodiscard]] constexpr RenderLayerMask operator|(RenderLayer lhs, RenderLayer rhs) {
    return renderLayerMask(lhs) | renderLayerMask(rhs);
}

[[nodiscard]] constexpr RenderLayerMask operator|(RenderLayerMask lhs, RenderLayer rhs) {
    return lhs | renderLayerMask(rhs);
}

[[nodiscard]] constexpr RenderLayerMask operator&(RenderLayerMask lhs, RenderLayer rhs) {
    return lhs & renderLayerMask(rhs);
}

inline constexpr RenderLayerMask defaultInstanceRenderLayerMask =
    renderLayerMask(RenderLayer::world);
inline constexpr RenderLayerMask defaultViewVisibilityMask =
    RenderLayer::world | RenderLayer::characters | RenderLayer::vegetation | RenderLayer::effects |
    RenderLayer::sky | RenderLayer::ui3D;
inline constexpr RenderLayerMask allRenderLayersMask = ~RenderLayerMask{0};

[[nodiscard]] constexpr bool hasRenderLayer(RenderLayerMask mask, RenderLayer layer) {
    return (mask & layer) != 0;
}

constexpr void setRenderLayer(RenderLayerMask& mask, RenderLayer layer, bool enabled = true) {
    if (enabled) {
        mask |= renderLayerMask(layer);
    } else {
        mask &= ~renderLayerMask(layer);
    }
}

[[nodiscard]] constexpr RenderLayerMask withRenderLayer(RenderLayerMask mask, RenderLayer layer) {
    setRenderLayer(mask, layer);
    return mask;
}

[[nodiscard]] constexpr RenderLayerMask withoutRenderLayer(RenderLayerMask mask,
                                                           RenderLayer layer) {
    setRenderLayer(mask, layer, false);
    return mask;
}

[[nodiscard]] constexpr bool renderLayersIntersect(RenderLayerMask instanceMask,
                                                   RenderLayerMask viewMask) {
    return (instanceMask & viewMask) != 0;
}

// Camera and projection data remain owned by the renderer today. This is the
// stable view-level visibility contract they will move beside as multi-view
// rendering is introduced.
struct RenderView {
    RenderLayerMask visibilityMask = defaultViewVisibilityMask;
};

static_assert(sizeof(RenderLayerMask) == 8);

} // namespace renderer
