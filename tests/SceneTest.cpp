#include "renderer/scene/Scene.hpp"

#include <glm/ext/matrix_transform.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

bool close(float lhs, float rhs) { return std::abs(lhs - rhs) < 0.0001F; }

void testValidSceneAndEvaluation() {
    namespace rs = renderer::scene;
    rs::Scene scene("Deterministic scene", "memory://scene-test");

    const rs::ImageHandle image = scene.addImage(
        {.name = "Base color", .sourceUri = "base-color.png", .mimeType = "image/png"});
    const rs::SamplerHandle sampler = scene.addSampler({.name = "Linear repeat"});
    const rs::TextureHandle texture =
        scene.addTexture({.name = "Base color", .image = image, .sampler = sampler});
    const rs::MaterialHandle material = scene.addMaterial({
        .name = "Metal rough",
        .baseColorFactor = {0.8F, 0.6F, 0.4F, 1.0F},
        .metallicFactor = 0.75F,
        .roughnessFactor = 0.25F,
        .baseColorTexture = {.texture = texture, .texCoord = 0},
    });
    rs::MeshPrimitive triangle{
        .name = "Triangle",
        .positions = {{-1.0F, -2.0F, 0.0F}, {2.0F, -2.0F, 0.0F}, {0.0F, 3.0F, 0.0F}},
        .normals = {{0.0F, 0.0F, 1.0F}, {0.0F, 0.0F, 1.0F}, {0.0F, 0.0F, 1.0F}},
        .indices = {0, 1, 2},
        .material = material,
    };
    triangle.texCoords[0] = {{0.0F, 0.0F}, {1.0F, 0.0F}, {0.5F, 1.0F}};
    const rs::PrimitiveHandle primitive = scene.addPrimitive(std::move(triangle));
    const rs::MeshHandle mesh = scene.addMesh({.name = "Mesh", .primitives = {primitive}});
    const rs::CameraHandle camera = scene.addCamera({.name = "Camera"});
    const rs::LightHandle light =
        scene.addLight({.name = "Key", .type = rs::LightType::directional, .intensity = 3.0F});

    const rs::NodeHandle root = scene.addNode({
        .name = "Root",
        .localTransform = glm::translate(glm::mat4(1.0F), {1.0F, 2.0F, 3.0F}),
        .camera = camera,
        .light = light,
    });
    const rs::NodeHandle child = scene.addNode({
        .name = "Mesh node",
        .localTransform = glm::translate(glm::mat4(1.0F), {4.0F, 0.0F, 0.0F}),
        .mesh = mesh,
        .visibilityMask = renderer::RenderLayer::characters | renderer::RenderLayer::effects,
    });
    scene.addChild(root, child);
    scene.addRoot(root);

    require(root.index() == 0 && child.index() == 1,
            "Scene handles are not deterministic append-only indices");
    require(scene.validate().ok(), scene.validate().describe());

    const rs::AxisAlignedBounds bounds = rs::computeBounds(scene.primitive(primitive));
    require(bounds.valid && close(bounds.minimum.x, -1.0F) && close(bounds.minimum.y, -2.0F) &&
                close(bounds.maximum.x, 2.0F) && close(bounds.maximum.y, 3.0F),
            "Primitive bounds are incorrect");

    const rs::SceneEvaluation evaluation = rs::evaluate(scene);
    require(evaluation.meshInstances.size() == 1, "Expected one derived mesh instance");
    const glm::mat4& transform = evaluation.meshInstances.front().worldTransform;
    require(close(transform[3].x, 5.0F) && close(transform[3].y, 2.0F) &&
                close(transform[3].z, 3.0F),
            "Node hierarchy did not produce the expected world transform");
    require(evaluation.meshInstances.front().visibilityMask ==
                (renderer::RenderLayer::characters | renderer::RenderLayer::effects),
            "Scene evaluation did not propagate renderer visibility layers");
}

void testInvalidGeometryDiagnostics() {
    namespace rs = renderer::scene;
    rs::Scene scene("Invalid geometry");
    const rs::PrimitiveHandle primitive = scene.addPrimitive({
        .positions = {{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}},
        .normals = {{0.0F, 0.0F, 1.0F}},
        .indices = {0, 1, 9},
    });
    const rs::MeshHandle mesh = scene.addMesh({.primitives = {primitive}});
    const rs::NodeHandle root = scene.addNode({.mesh = mesh});
    scene.addRoot(root);

    const rs::ValidationReport report = scene.validate();
    require(!report.ok(), "Invalid primitive unexpectedly passed validation");
    require(report.describe().find("primitives[0].normals") != std::string::npos,
            "Attribute-count diagnostic is not actionable");
    require(report.describe().find("primitives[0].indices[2]") != std::string::npos,
            "Out-of-range index diagnostic is not actionable");

    bool rejected = false;
    try {
        (void)rs::evaluate(scene);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "Scene evaluation accepted invalid geometry");
}

void testHierarchyCycleDetection() {
    namespace rs = renderer::scene;
    rs::Scene scene("Cycle");
    const rs::NodeHandle first = scene.addNode({.name = "First"});
    const rs::NodeHandle second = scene.addNode({.name = "Second"});
    scene.addChild(first, second);
    scene.addChild(second, first);

    const rs::ValidationReport report = scene.validate();
    require(!report.ok() &&
                report.describe().find("hierarchy contains a cycle") != std::string::npos,
            "Hierarchy cycle was not diagnosed");
}

void testInvalidHandleAccess() {
    renderer::scene::Scene scene;
    bool rejected = false;
    try {
        (void)scene.node(renderer::scene::NodeHandle::fromIndex(42));
    } catch (const std::out_of_range&) {
        rejected = true;
    }
    require(rejected, "Invalid scene handle access was not rejected");
}

} // namespace

int main() {
    try {
        testValidSceneAndEvaluation();
        testInvalidGeometryDiagnostics();
        testHierarchyCycleDetection();
        testInvalidHandleAccess();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
