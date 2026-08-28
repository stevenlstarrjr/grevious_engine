#pragma once

#include "renderer/scene/Scene.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>

namespace renderer::assets {

// Import limits are checked before attribute extraction. They make failures
// deterministic and give future scripting bindings one policy object to expose
// instead of granting a parser an unbounded allocation surface.
struct GltfImportLimits {
    std::uintmax_t maxSourceFileBytes = 512ULL * 1024ULL * 1024ULL;
    std::uintmax_t maxExternalFileBytes = 512ULL * 1024ULL * 1024ULL;
    std::uintmax_t maxTotalBufferBytes = 1024ULL * 1024ULL * 1024ULL;
    std::uintmax_t maxTotalImageBytes = 512ULL * 1024ULL * 1024ULL;
    std::size_t maxAccessors = 1'000'000;
    std::size_t maxImages = 65'536;
    std::size_t maxMaterials = 1'000'000;
    std::size_t maxMeshes = 250'000;
    std::size_t maxPrimitives = 1'000'000;
    std::size_t maxNodes = 1'000'000;
    std::size_t maxVertices = 25'000'000;
    std::size_t maxIndices = 75'000'000;
};

struct GltfImportOptions {
    // If unset, use the declared default scene, or scene zero when no default
    // exists. Assets without a scenes array import every parentless hierarchy.
    std::optional<std::size_t> sceneIndex;
    GltfImportLimits limits;
    bool allowExternalFilesOutsideSourceDirectory = false;
};

class GltfImportError final : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// Imports source data only. Images remain encoded and no GPU objects are
// created. The public API intentionally contains no fastgltf types.
[[nodiscard]] scene::Scene importGltf(const std::filesystem::path& sourcePath,
                                      const GltfImportOptions& options = {});

} // namespace renderer::assets
