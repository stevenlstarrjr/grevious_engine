#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <vector>

namespace renderer {

// Host-neutral editor geometry drawn after tone mapping. Qt, SDL, and other
// window systems remain outside the renderer; a DCC host only supplies packed
// colored vertices and the matrix used by its editing tools.
struct EditorOverlayVertex {
    glm::vec3 position{0.0F};
    glm::vec4 color{1.0F};
};

struct EditorOverlayFrame {
    glm::mat4 viewProjection{1.0F};

    // Workbench-only opaque scene fill. When replaceScene is true, the overlay
    // pass clears the PBR result and depth before drawing these triangles.
    std::vector<EditorOverlayVertex> sceneTriangles;
    // Material-preview selection tint, depth-tested against the PBR surface.
    std::vector<EditorOverlayVertex> selectedFaceTriangles;
    std::vector<EditorOverlayVertex> gridLines;
    std::vector<EditorOverlayVertex> xRayMeshLines;
    std::vector<EditorOverlayVertex> xRayDepthScreenTriangles;
    std::vector<EditorOverlayVertex> xRayTriangles;
    std::vector<EditorOverlayVertex> meshLines;
    std::vector<EditorOverlayVertex> overlayWorldLines;
    std::vector<EditorOverlayVertex> depthScreenTriangles;
    std::vector<EditorOverlayVertex> overlayScreenTriangles;
    std::vector<EditorOverlayVertex> overlayScreenLines;
    bool replaceScene = false;
};

} // namespace renderer
