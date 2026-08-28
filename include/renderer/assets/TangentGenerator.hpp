#pragma once

#include <cstdint>

namespace renderer::scene {
struct MeshPrimitive;
}

namespace renderer::assets {

// Generates glTF-compatible tangent frames with the unmodified upstream
// MikkTSpace implementation. Indexed vertices are split wherever MikkTSpace
// returns distinct per-corner tangent frames.
void generateMikkTangents(scene::MeshPrimitive& primitive, std::uint32_t texCoordSet = 0);

} // namespace renderer::assets
