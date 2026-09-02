#include "renderer/assets/GltfImporter.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

namespace fs = std::filesystem;
namespace ra = renderer::assets;
namespace rs = renderer::scene;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

bool close(float lhs, float rhs) { return std::abs(lhs - rhs) < 0.0001F; }

fs::path sample(std::string_view relative) {
    return fs::path(RENDERER_GLTF_SAMPLE_ASSETS_DIR) / relative;
}

void testExternalGltfTriangle() {
    const rs::Scene scene = ra::importGltf(sample("Models/Triangle/glTF/Triangle.gltf"));
    require(scene.validate().ok(), scene.validate().describe());
    require(scene.meshes().size() == 1 && scene.primitives().size() == 1,
            "Triangle did not import one mesh primitive");
    require(scene.nodes().size() == 1 && scene.roots().size() == 1,
            "Triangle hierarchy was not imported");
    require(scene.nodes().front().visibilityMask == renderer::defaultInstanceRenderLayerMask,
            "glTF nodes did not receive the renderer's default visibility layer");

    const rs::MeshPrimitive& primitive = scene.primitives().front();
    require(primitive.positions.size() == 3 && primitive.indices.size() == 3,
            "Triangle geometry cardinality is incorrect");
    require(close(primitive.positions[0].x, 0.0F) && close(primitive.positions[1].x, 1.0F) &&
                close(primitive.positions[2].y, 1.0F),
            "Triangle accessor data was decoded incorrectly");
    require(rs::evaluate(scene).meshInstances.size() == 1,
            "Triangle did not evaluate to one mesh instance");
}

void testBinaryTexturedBox() {
    const rs::Scene scene =
        ra::importGltf(sample("Models/BoxTextured/glTF-Binary/BoxTextured.glb"));
    require(scene.validate().ok(), scene.validate().describe());
    require(scene.images().size() == 1 && !scene.images().front().encodedData.empty(),
            "GLB image bufferView was not retained as encoded bytes");
    require(scene.textures().size() == 1 && scene.samplers().size() == 1,
            "GLB texture or sampler was not imported");
    require(scene.materials().size() == 1 &&
                scene.materials().front().baseColorTexture.texture.valid(),
            "Base-color texture binding was not imported");
    require(close(scene.materials().front().metallicFactor, 0.0F),
            "Metallic factor was not imported");

    const rs::MeshPrimitive& primitive = scene.primitives().front();
    require(primitive.positions.size() == 24 && primitive.normals.size() == 24 &&
                primitive.texCoords[0].size() == 24 && primitive.indices.size() == 36,
            "Textured box attributes were not decoded completely");
    require(scene.nodes().size() == 2 && rs::evaluate(scene).meshInstances.size() == 1,
            "Textured box hierarchy was not preserved");
}

void testMetallicRoughnessMaterialGrid() {
    const rs::Scene scene = ra::importGltf(
        sample("Models/MetalRoughSpheresNoTextures/glTF/MetalRoughSpheresNoTextures.gltf"));
    require(scene.validate().ok(), scene.validate().describe());
    require(scene.materials().size() == 98, "Material grid did not import all materials");
    require(close(scene.materials().front().metallicFactor, 0.0F) &&
                close(scene.materials().front().roughnessFactor, 0.0F),
            "First metallic/roughness grid endpoint is incorrect");
    require(close(scene.materials().back().metallicFactor, 1.0F) &&
                close(scene.materials().back().roughnessFactor, 1.0F),
            "Last metallic/roughness grid endpoint is incorrect");
    require(!rs::evaluate(scene).meshInstances.empty(),
            "Material grid did not produce mesh instances");
}

void testEmissiveStrength() {
    const rs::Scene scene =
        ra::importGltf(sample("Models/CompareEmissiveStrength/glTF/CompareEmissiveStrength.gltf"));
    require(scene.validate().ok(), scene.validate().describe());
    require(scene.materials().size() == 2,
            "Emissive-strength comparison did not import both materials");

    const rs::Material* regular = nullptr;
    const rs::Material* strengthened = nullptr;
    for (const rs::Material& material : scene.materials()) {
        if (material.name == "glTF Logo Emissive") {
            regular = &material;
        } else if (material.name == "glTF Logo Emissive Strength") {
            strengthened = &material;
        }
    }
    require(regular != nullptr && strengthened != nullptr,
            "Emissive-strength comparison material names were not preserved");
    require(close(strengthened->emissiveFactor.r, regular->emissiveFactor.r * 3.0F) &&
                close(strengthened->emissiveFactor.g, regular->emissiveFactor.g * 3.0F) &&
                close(strengthened->emissiveFactor.b, regular->emissiveFactor.b * 3.0F),
            "KHR_materials_emissive_strength was not applied to the emissive factor");
}

void testVertexColors() {
    const rs::Scene scene =
        ra::importGltf(sample("Models/BoxVertexColors/glTF/BoxVertexColors.gltf"));
    require(scene.validate().ok(), scene.validate().describe());
    require(scene.primitives().size() == 1U,
            "BoxVertexColors did not import exactly one primitive");
    const rs::MeshPrimitive& primitive = scene.primitives().front();
    require(primitive.colors.size() == primitive.positions.size() &&
                primitive.colors.size() == 24U,
            "COLOR_0 did not retain one value per imported vertex");
    bool foundNonWhite = false;
    for (const glm::vec4& color : primitive.colors) {
        foundNonWhite = foundNonWhite || color != glm::vec4(1.0F);
        require(close(color.a, 1.0F),
                "VEC3 COLOR_0 did not receive the required default alpha of one");
    }
    require(foundNonWhite, "Imported COLOR_0 values were replaced by the white default");
}

void testPolicyFailuresAreActionable() {
    ra::GltfImportOptions smallSource;
    smallSource.limits.maxSourceFileBytes = 32;
    bool limitRejected = false;
    try {
        (void)ra::importGltf(sample("Models/Triangle/glTF/Triangle.gltf"), smallSource);
    } catch (const ra::GltfImportError& error) {
        limitRejected =
            std::string_view(error.what()).find("source file size") != std::string_view::npos;
    }
    require(limitRejected, "Source-size limit did not produce an actionable rejection");

    bool animationRejected = false;
    try {
        (void)ra::importGltf(sample("Models/BoxAnimated/glTF/BoxAnimated.gltf"));
    } catch (const ra::GltfImportError& error) {
        animationRejected = std::string_view(error.what()).find("animations are not supported") !=
                            std::string_view::npos;
    }
    require(animationRejected, "Unsupported animation was not explicitly rejected");

    ra::GltfImportOptions staticPose;
    staticPose.allowAnimationStaticPose = true;
    const rs::Scene animatedInitialPose =
        ra::importGltf(sample("Models/BoxAnimated/glTF/BoxAnimated.gltf"), staticPose);
    require(animatedInitialPose.validate().ok(),
            "Explicit animation static-pose import did not produce a valid scene");
    require(!rs::evaluate(animatedInitialPose).meshInstances.empty(),
            "Explicit animation static-pose import lost the animated mesh instance");

    ra::GltfImportOptions invalidScene;
    invalidScene.sceneIndex = 99;
    bool sceneRejected = false;
    try {
        (void)ra::importGltf(sample("Models/Triangle/glTF/Triangle.gltf"), invalidScene);
    } catch (const ra::GltfImportError& error) {
        sceneRejected =
            std::string_view(error.what()).find("selected scene index") != std::string_view::npos;
    }
    require(sceneRejected, "Out-of-range scene selection was not rejected");
}

} // namespace

int main() {
    try {
        testExternalGltfTriangle();
        testBinaryTexturedBox();
        testMetallicRoughnessMaterialGrid();
        testEmissiveStrength();
        testVertexColors();
        testPolicyFailuresAreActionable();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
