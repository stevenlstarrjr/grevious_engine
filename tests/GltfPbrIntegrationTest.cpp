#include "renderer/assets/GltfImporter.hpp"
#include "renderer/gpu_scene/GpuScene.hpp"
#include "renderer/materials/PbrMath.hpp"

#include <glm/geometric.hpp>
#include <glm/vec3.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace ra = renderer::assets;
namespace rg = renderer::gpu_scene;
namespace rm = renderer::materials;
namespace rs = renderer::scene;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

fs::path sample(std::string_view relative) {
    return fs::path(RENDERER_GLTF_SAMPLE_ASSETS_DIR) / relative;
}

bool finiteNonNegative(const glm::vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z) &&
           value.x >= 0.0F && value.y >= 0.0F && value.z >= 0.0F;
}

bool validTangentFrame(const rs::MeshPrimitive& primitive) {
    if (primitive.tangents.size() != primitive.positions.size() ||
        primitive.normals.size() != primitive.positions.size()) {
        return false;
    }
    for (std::size_t index = 0; index < primitive.positions.size(); ++index) {
        const glm::vec3 tangent(primitive.tangents[index]);
        const glm::vec3 normal = glm::normalize(primitive.normals[index]);
        if (!std::isfinite(tangent.x) || !std::isfinite(tangent.y) || !std::isfinite(tangent.z) ||
            std::abs(glm::length(tangent) - 1.0F) > 2.0e-4F ||
            std::abs(glm::dot(tangent, normal)) > 2.0e-4F ||
            std::abs(std::abs(primitive.tangents[index].w) - 1.0F) > 1.0e-6F) {
            return false;
        }
    }
    return true;
}

void evaluateImportedMaterialSweep() {
    const rs::Scene scene = ra::importGltf(
        sample("Models/MetalRoughSpheresNoTextures/glTF/MetalRoughSpheresNoTextures.gltf"));
    require(scene.materials().size() == 98,
            "Khronos metallic/roughness factor sweep has an unexpected material count");
    const rg::PackedScene residentScene = rg::packScene(scene);
    const std::vector<rg::MeshletWorkRecord> meshletWork =
        rg::makeMeshletWorkRecords(residentScene.primitiveLods, residentScene.lods,
                                   residentScene.meshlets, residentScene.instances);
    const std::vector<rg::MaterialRecord>& packedMaterials = residentScene.materials;
    require(packedMaterials.size() >= scene.materials().size() &&
                packedMaterials.size() <= scene.materials().size() + 1,
            "GPU material packing did not preserve source materials and one optional default");
    const rs::SceneEvaluation evaluation = rs::evaluate(scene);
    std::size_t expectedInstanceCount = 0;
    for (const rs::MeshInstance& meshInstance : evaluation.meshInstances) {
        expectedInstanceCount += scene.mesh(meshInstance.mesh).primitives.size();
    }
    require(!residentScene.vertices.empty() && !residentScene.indices.empty() &&
                residentScene.primitives.size() == scene.primitives().size() &&
                residentScene.primitiveLods.size() == residentScene.primitives.size() &&
                !residentScene.meshlets.empty() && !meshletWork.empty() &&
                residentScene.transforms.size() == evaluation.meshInstances.size() &&
                residentScene.instances.size() == expectedInstanceCount,
            "Scene IR geometry or hierarchy did not pack into GPU batches");
    for (std::size_t instanceIndex = 0; instanceIndex < residentScene.instances.size();
         ++instanceIndex) {
        const rg::InstanceRecord& instance = residentScene.instances[instanceIndex];
        require(instance.centerRadius.w > 0.0F &&
                    instance.meshMaterial.x < residentScene.primitives.size() &&
                    instance.meshMaterial.y < residentScene.materials.size() &&
                    instance.meshMaterial.z < residentScene.transforms.size() &&
                    rg::unpackRenderLayerMask(glm::uvec2(instance.visibilityLayer)) ==
                        renderer::defaultInstanceRenderLayerMask,
                "Packed GPU instance has invalid resident indices");
    }
    std::vector<std::size_t> workPerInstance(residentScene.instances.size(), 0U);
    for (const rg::MeshletWorkRecord& work : meshletWork) {
        require(work.instanceIndex < residentScene.instances.size() && work.indexCount > 0U &&
                    work.indexCount <= rg::maxMeshletTriangles * 3U && work.indexCount % 3U == 0U,
                "Packed meshlet work has invalid draw or instance bounds");
        ++workPerInstance[work.instanceIndex];
    }
    for (const std::size_t workCount : workPerInstance) {
        require(workCount > 0U, "A resident instance has no meshlet work candidate");
    }

    constexpr std::array viewDirections{
        glm::vec3(0.0F, 0.0F, 1.0F),
        glm::vec3(0.35F, 0.0F, 0.93675F),
        glm::vec3(-0.65F, 0.2F, 0.73314F),
    };
    constexpr std::array lightDirections{
        glm::vec3(0.0F, 0.0F, 1.0F),
        glm::vec3(-0.25F, 0.45F, 0.85732F),
        glm::vec3(0.7F, -0.1F, 0.70710F),
    };
    const glm::vec3 normal(0.0F, 0.0F, 1.0F);
    for (std::size_t materialIndex = 0; materialIndex < scene.materials().size(); ++materialIndex) {
        const rs::Material& imported = scene.materials()[materialIndex];
        const rg::MaterialRecord& packed = packedMaterials[materialIndex];
        require(packed.baseColorFactor == imported.baseColorFactor &&
                    packed.metallicRoughnessOcclusionNormal.x == imported.metallicFactor &&
                    packed.metallicRoughnessOcclusionNormal.y == imported.roughnessFactor &&
                    packed.textureState.y == 0U,
                "Scene IR material factors changed at the GPU residency boundary");
        const rm::MetallicRoughnessSample material{
            .baseColor = glm::vec3(imported.baseColorFactor),
            .metallic = imported.metallicFactor,
            .perceptualRoughness = imported.roughnessFactor,
        };
        for (const glm::vec3& view : viewDirections) {
            for (const glm::vec3& light : lightDirections) {
                const glm::vec3 brdf = rm::evaluateMetallicRoughnessBrdf(
                    material, normal, glm::normalize(view), glm::normalize(light));
                require(finiteNonNegative(brdf),
                        "A Khronos material produced a non-finite or negative BRDF");
            }
        }
    }
}

void verifyTextureAndTangentCompatibilityAssets() {
    const rs::Scene metallic =
        ra::importGltf(sample("Models/CompareMetallic/glTF/CompareMetallic.gltf"));
    require(metallic.materials().size() == 2 &&
                metallic.materials()[0].baseColorTexture.texture.valid() &&
                !metallic.materials()[0].metallicRoughnessTexture.texture.valid() &&
                metallic.materials()[1].baseColorTexture.texture.valid() &&
                metallic.materials()[1].metallicRoughnessTexture.texture.valid(),
            "CompareMetallic lost metallic/roughness texture bindings");
    std::vector<std::uint32_t> descriptorIndices(metallic.textures().size());
    std::iota(descriptorIndices.begin(), descriptorIndices.end(), 3U);
    const std::vector<rg::MaterialRecord> packedMetallic =
        rg::packMaterials(metallic, descriptorIndices);
    const auto descriptorFor = [&](rs::TextureHandle texture) {
        return descriptorIndices[texture.index()];
    };
    require((packedMetallic[0].textureState.y & rg::baseColorTextureBit) != 0U &&
                (packedMetallic[0].textureState.y & rg::metallicRoughnessTextureBit) == 0U &&
                (packedMetallic[1].textureState.y & rg::metallicRoughnessTextureBit) != 0U &&
                packedMetallic[0].textureIndices.x ==
                    descriptorFor(metallic.materials()[0].baseColorTexture.texture) &&
                packedMetallic[1].textureIndices.y ==
                    descriptorFor(metallic.materials()[1].metallicRoughnessTexture.texture),
            "GPU material packing lost bindless texture residency or presence flags");

    const rs::Scene roughness =
        ra::importGltf(sample("Models/CompareRoughness/glTF/CompareRoughness.gltf"));
    require(roughness.materials().size() == 2 &&
                roughness.materials()[0].baseColorTexture.texture.valid() &&
                !roughness.materials()[0].metallicRoughnessTexture.texture.valid() &&
                roughness.materials()[1].baseColorTexture.texture.valid() &&
                roughness.materials()[1].metallicRoughnessTexture.texture.valid(),
            "CompareRoughness lost metallic/roughness texture bindings");

    const rs::Scene normal = ra::importGltf(sample("Models/CompareNormal/glTF/CompareNormal.gltf"));
    require(normal.materials().size() == 2 &&
                !normal.materials()[0].normalTexture.texture.valid() &&
                normal.materials()[1].normalTexture.texture.valid(),
            "CompareNormal lost a normal texture binding");

    const rs::Scene tangent =
        ra::importGltf(sample("Models/NormalTangentTest/glTF/NormalTangentTest.gltf"));
    require(!tangent.primitives().empty() && validTangentFrame(tangent.primitives().front()) &&
                tangent.primitives().front().normals.size() ==
                    tangent.primitives().front().positions.size() &&
                tangent.primitives().front().texCoords[0].size() ==
                    tangent.primitives().front().positions.size() &&
                tangent.materials().front().normalTexture.texture.valid(),
            "NormalTangentTest did not receive a valid MikkTSpace tangent frame");

    const rs::Scene helmet = ra::importGltf(sample("Models/DamagedHelmet/glTF/DamagedHelmet.gltf"));
    require(helmet.primitives().size() == 1,
            "DamagedHelmet imported an unexpected primitive count");
    require(helmet.materials().size() == 1 &&
                helmet.materials().front().normalTexture.texture.valid(),
            "DamagedHelmet lost its normal-mapped material");
    require(helmet.primitives().front().tangents.size() ==
                helmet.primitives().front().positions.size(),
            "DamagedHelmet tangent generation produced the wrong cardinality");
    require(validTangentFrame(helmet.primitives().front()),
            "DamagedHelmet generated tangent vectors violate frame invariants");
    std::vector<std::uint32_t> helmetDescriptors(helmet.textures().size());
    std::iota(helmetDescriptors.begin(), helmetDescriptors.end(), 1U);
    const rg::PackedScene packedHelmet = rg::packScene(helmet, helmetDescriptors);
    require(!packedHelmet.vertices.empty() && !packedHelmet.indices.empty() &&
                packedHelmet.primitives.size() == 1 && packedHelmet.instances.size() == 1,
            "DamagedHelmet did not cross the Scene IR to GPU residency boundary");
    require(std::abs(packedHelmet.instances.front().centerRadius.w -
                     packedHelmet.primitives.front().centerRadius.w) < 1.0e-5F,
            "Identity/TRS instance bounds were inflated instead of using exact basis scale");

    std::vector<std::uint32_t> baseColorDescriptors(helmet.textures().size());
    std::vector<std::uint32_t> metallicRoughnessDescriptors(helmet.textures().size());
    std::vector<std::uint32_t> normalDescriptors(helmet.textures().size());
    std::vector<std::uint32_t> occlusionDescriptors(helmet.textures().size());
    std::vector<std::uint32_t> emissiveDescriptors(helmet.textures().size());
    std::iota(baseColorDescriptors.begin(), baseColorDescriptors.end(), 10U);
    std::iota(metallicRoughnessDescriptors.begin(), metallicRoughnessDescriptors.end(), 20U);
    std::iota(normalDescriptors.begin(), normalDescriptors.end(), 30U);
    std::iota(occlusionDescriptors.begin(), occlusionDescriptors.end(), 40U);
    std::iota(emissiveDescriptors.begin(), emissiveDescriptors.end(), 50U);
    const rg::PackedScene usageSpecificHelmet =
        rg::packScene(helmet, rg::MaterialTextureDescriptorTable{
                                  .baseColor = baseColorDescriptors,
                                  .metallicRoughness = metallicRoughnessDescriptors,
                                  .normal = normalDescriptors,
                                  .occlusion = occlusionDescriptors,
                                  .emissive = emissiveDescriptors,
                              });
    const rs::Material& helmetMaterial = helmet.materials().front();
    const rg::MaterialRecord& packedHelmetMaterial = usageSpecificHelmet.materials.front();
    require(packedHelmetMaterial.textureIndices.x ==
                    baseColorDescriptors[helmetMaterial.baseColorTexture.texture.index()] &&
                packedHelmetMaterial.textureIndices.y ==
                    metallicRoughnessDescriptors[helmetMaterial.metallicRoughnessTexture.texture
                                                     .index()] &&
                packedHelmetMaterial.textureIndices.z ==
                    normalDescriptors[helmetMaterial.normalTexture.texture.index()] &&
                packedHelmetMaterial.textureIndices.w ==
                    occlusionDescriptors[helmetMaterial.occlusionTexture.texture.index()] &&
                packedHelmetMaterial.textureState.x ==
                    emissiveDescriptors[helmetMaterial.emissiveTexture.texture.index()],
            "GPU material packing did not preserve usage-specific texture descriptors");

    const rs::Scene texturedGrid =
        ra::importGltf(sample("Models/MetalRoughSpheres/glTF/MetalRoughSpheres.gltf"));
    require(texturedGrid.materials().size() == 1 &&
                texturedGrid.materials().front().baseColorTexture.texture.valid() &&
                texturedGrid.materials().front().metallicRoughnessTexture.texture.valid() &&
                !rs::evaluate(texturedGrid).meshInstances.empty(),
            "MetalRoughSpheres lost PBR texture bindings or mesh instances");
}

} // namespace

int main() {
    try {
        evaluateImportedMaterialSweep();
        verifyTextureAndTangentCompatibilityAssets();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
