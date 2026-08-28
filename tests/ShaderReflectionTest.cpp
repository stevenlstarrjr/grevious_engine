#include "renderer/vulkan/Shader.hpp"

#include <volk.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::vector<std::uint32_t> readWords(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Could not open " + path.string());
    }
    const std::streamsize byteCount = input.tellg();
    if (byteCount <= 0 || byteCount % static_cast<std::streamsize>(sizeof(std::uint32_t)) != 0) {
        throw std::runtime_error("Invalid SPIR-V size for " + path.string());
    }
    input.seekg(0);
    std::vector<std::uint32_t> words(static_cast<std::size_t>(byteCount) / sizeof(std::uint32_t));
    input.read(reinterpret_cast<char*>(words.data()), byteCount);
    if (!input) {
        throw std::runtime_error("Could not read " + path.string());
    }
    return words;
}

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

} // namespace

int main() {
    try {
        const std::filesystem::path shaderDirectory(RENDERER_SHADER_DIR);
        const auto vertex =
            renderer::vulkan::reflectSpirv(readWords(shaderDirectory / "mesh.vert.spv"));
        const auto fragment =
            renderer::vulkan::reflectSpirv(readWords(shaderDirectory / "mesh.frag.spv"));
        const auto shadowVertex =
            renderer::vulkan::reflectSpirv(readWords(shaderDirectory / "shadow.vert.spv"));
        const auto shadowFragment =
            renderer::vulkan::reflectSpirv(readWords(shaderDirectory / "shadow.frag.spv"));
        const auto shadowCull =
            renderer::vulkan::reflectSpirv(readWords(shaderDirectory / "shadow_cull.comp.spv"));
        const auto frustumCull =
            renderer::vulkan::reflectSpirv(readWords(shaderDirectory / "frustum_cull.comp.spv"));
        const auto occlusionCull =
            renderer::vulkan::reflectSpirv(readWords(shaderDirectory / "occlusion_cull.comp.spv"));
        const auto hiZCopy =
            renderer::vulkan::reflectSpirv(readWords(shaderDirectory / "hiz_copy.comp.spv"));
        const auto hiZReduce =
            renderer::vulkan::reflectSpirv(readWords(shaderDirectory / "hiz_reduce.comp.spv"));
        const auto skyFragment =
            renderer::vulkan::reflectSpirv(readWords(shaderDirectory / "sky.frag.spv"));
        const auto postTestFragment =
            renderer::vulkan::reflectSpirv(readWords(shaderDirectory / "post_test.frag.spv"));
        const auto bloomDownsample = renderer::vulkan::reflectSpirv(
            readWords(shaderDirectory / "bloom_downsample.frag.spv"));
        const auto bloomHorizontal =
            renderer::vulkan::reflectSpirv(readWords(shaderDirectory / "bloom_blur_h.frag.spv"));
        const auto bloomVertical =
            renderer::vulkan::reflectSpirv(readWords(shaderDirectory / "bloom_blur_v.frag.spv"));
        const auto toneMappingVertex =
            renderer::vulkan::reflectSpirv(readWords(shaderDirectory / "tonemap.vert.spv"));
        const auto toneMappingFragment =
            renderer::vulkan::reflectSpirv(readWords(shaderDirectory / "tonemap.frag.spv"));
        const auto taaFragment =
            renderer::vulkan::reflectSpirv(readWords(shaderDirectory / "taa.frag.spv"));

        require(vertex.stage == VK_SHADER_STAGE_VERTEX_BIT, "Vertex stage reflection failed");
        require(vertex.inputLocations == std::vector<std::uint32_t>{0, 1, 2, 3, 4},
                "Vertex input reflection failed");
        require(vertex.descriptorBindings.size() == 4,
                "Vertex descriptor binding count is incorrect");
        const auto& binding = vertex.descriptorBindings.front();
        require(binding.set == 0 && binding.binding == 0 &&
                    binding.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && binding.count == 1,
                "Uniform-buffer reflection failed");
        require(vertex.descriptorBindings[1].binding == 6 &&
                    vertex.descriptorBindings[1].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER &&
                    vertex.descriptorBindings[2].binding == 7 &&
                    vertex.descriptorBindings[2].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER &&
                    vertex.descriptorBindings[3].binding == 9 &&
                    vertex.descriptorBindings[3].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                "GPU scene vertex-buffer reflection failed");
        require(fragment.stage == VK_SHADER_STAGE_FRAGMENT_BIT, "Fragment stage reflection failed");
        require(fragment.descriptorBindings.size() == 7 &&
                    fragment.descriptorBindings[0].binding == 0 &&
                    fragment.descriptorBindings[0].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
                    fragment.descriptorBindings[1].binding == 2 &&
                    fragment.descriptorBindings[3].binding == 4 &&
                    fragment.descriptorBindings[4].binding == 8 &&
                    fragment.descriptorBindings[4].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER &&
                    fragment.descriptorBindings[5].binding == 10 &&
                    fragment.descriptorBindings[5].type ==
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                    fragment.descriptorBindings[5].count == 3 &&
                    fragment.descriptorBindings[6].set == 1 &&
                    fragment.descriptorBindings[6].binding == 0 &&
                    fragment.descriptorBindings[6].count == 0,
                "PBR fragment descriptor reflection failed");
        for (std::size_t index = 1; index < fragment.descriptorBindings.size(); ++index) {
            if (index == 4) {
                continue;
            }
            require(fragment.descriptorBindings[index].type ==
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    "PBR image descriptor reflection failed");
        }
        require(shadowVertex.stage == VK_SHADER_STAGE_VERTEX_BIT &&
                    shadowVertex.inputLocations == std::vector<std::uint32_t>{0, 2, 4} &&
                    shadowVertex.descriptorBindings.size() == 4 &&
                    shadowVertex.descriptorBindings[0].binding == 0 &&
                    shadowVertex.descriptorBindings[0].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
                    shadowVertex.descriptorBindings[1].binding == 6 &&
                    shadowVertex.descriptorBindings[2].binding == 7 &&
                    shadowVertex.descriptorBindings[3].binding == 9,
                "Directional-shadow vertex reflection failed");
        require(shadowFragment.stage == VK_SHADER_STAGE_FRAGMENT_BIT &&
                    shadowFragment.inputLocations == std::vector<std::uint32_t>{0, 1, 2} &&
                    shadowFragment.descriptorBindings.size() == 2 &&
                    shadowFragment.descriptorBindings[0].set == 0 &&
                    shadowFragment.descriptorBindings[0].binding == 8 &&
                    shadowFragment.descriptorBindings[0].type ==
                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER &&
                    shadowFragment.descriptorBindings[1].set == 1 &&
                    shadowFragment.descriptorBindings[1].binding == 0 &&
                    shadowFragment.descriptorBindings[1].type ==
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                    shadowFragment.descriptorBindings[1].count == 0,
                "Directional-shadow alpha-mask reflection failed");
        require(shadowCull.stage == VK_SHADER_STAGE_COMPUTE_BIT &&
                    shadowCull.descriptorBindings.size() == 6 &&
                    shadowCull.descriptorBindings[0].binding == 0 &&
                    shadowCull.descriptorBindings[0].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                "Directional-shadow culling reflection failed");
        for (std::size_t index = 1; index < shadowCull.descriptorBindings.size(); ++index) {
            require(shadowCull.descriptorBindings[index].binding == index &&
                        shadowCull.descriptorBindings[index].type ==
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    "Directional-shadow culling buffer reflection failed");
        }
        require(skyFragment.stage == VK_SHADER_STAGE_FRAGMENT_BIT &&
                    skyFragment.descriptorBindings.size() == 2 &&
                    skyFragment.descriptorBindings[0].binding == 0 &&
                    skyFragment.descriptorBindings[0].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
                    skyFragment.descriptorBindings[1].binding == 5 &&
                    skyFragment.descriptorBindings[1].type ==
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                "Sky fragment descriptor reflection failed");
        require(postTestFragment.stage == VK_SHADER_STAGE_FRAGMENT_BIT &&
                    postTestFragment.descriptorBindings.size() == 1 &&
                    postTestFragment.descriptorBindings[0].set == 0 &&
                    postTestFragment.descriptorBindings[0].binding == 0 &&
                    postTestFragment.descriptorBindings[0].type ==
                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                "Procedural post-test descriptor reflection failed");
        require(frustumCull.stage == VK_SHADER_STAGE_COMPUTE_BIT &&
                    frustumCull.descriptorBindings.size() == 6 &&
                    frustumCull.descriptorBindings[0].binding == 0 &&
                    frustumCull.descriptorBindings[0].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                "Frustum-culling uniform reflection failed");
        for (std::size_t index = 1; index < frustumCull.descriptorBindings.size(); ++index) {
            require(frustumCull.descriptorBindings[index].binding == index &&
                        frustumCull.descriptorBindings[index].type ==
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    "Frustum-culling storage-buffer reflection failed");
        }
        require(occlusionCull.stage == VK_SHADER_STAGE_COMPUTE_BIT &&
                    occlusionCull.descriptorBindings.size() == 9 &&
                    occlusionCull.descriptorBindings[0].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
                    occlusionCull.descriptorBindings[4].binding == 4 &&
                    occlusionCull.descriptorBindings[4].type ==
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                    occlusionCull.descriptorBindings[4].count == 16,
                "Hi-Z occlusion-culling descriptor reflection failed");
        require(taaFragment.stage == VK_SHADER_STAGE_FRAGMENT_BIT &&
                    taaFragment.descriptorBindings.size() == 8,
                "Temporal-resolve descriptor reflection failed");
        for (std::size_t index = 0; index < taaFragment.descriptorBindings.size(); ++index) {
            const VkDescriptorType expected = index == 7
                                                  ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                  : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            require(taaFragment.descriptorBindings[index].binding == index &&
                        taaFragment.descriptorBindings[index].type == expected,
                    "Temporal-resolve descriptor interface is incorrect");
        }
        for (std::size_t index = 1; index < occlusionCull.descriptorBindings.size(); ++index) {
            if (index == 4) {
                continue;
            }
            require(occlusionCull.descriptorBindings[index].binding == index &&
                        occlusionCull.descriptorBindings[index].type ==
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER &&
                        occlusionCull.descriptorBindings[index].count == 1,
                    "Hi-Z occlusion-culling buffer reflection failed");
        }
        for (const auto* hiZ : {&hiZCopy, &hiZReduce}) {
            require(
                hiZ->stage == VK_SHADER_STAGE_COMPUTE_BIT && hiZ->descriptorBindings.size() == 2 &&
                    hiZ->descriptorBindings[0].binding == 0 &&
                    hiZ->descriptorBindings[0].type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                    hiZ->descriptorBindings[1].binding == 1 &&
                    hiZ->descriptorBindings[1].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                "Hi-Z construction descriptor reflection failed");
        }
        for (const auto* bloom : {&bloomDownsample, &bloomHorizontal, &bloomVertical}) {
            require(bloom->stage == VK_SHADER_STAGE_FRAGMENT_BIT &&
                        bloom->descriptorBindings.size() == 2 &&
                        bloom->descriptorBindings[0].binding == 0 &&
                        bloom->descriptorBindings[0].type ==
                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                        bloom->descriptorBindings[1].binding == 1 &&
                        bloom->descriptorBindings[1].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    "Bloom descriptor reflection failed");
        }
        require(toneMappingVertex.stage == VK_SHADER_STAGE_VERTEX_BIT &&
                    toneMappingVertex.inputLocations.empty() &&
                    toneMappingVertex.descriptorBindings.empty(),
                "Tone-mapping vertex reflection failed");
        require(toneMappingFragment.stage == VK_SHADER_STAGE_FRAGMENT_BIT &&
                    toneMappingFragment.descriptorBindings.size() == 4 &&
                    toneMappingFragment.descriptorBindings[0].binding == 0 &&
                    toneMappingFragment.descriptorBindings[0].type ==
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                    toneMappingFragment.descriptorBindings[1].binding == 1 &&
                    toneMappingFragment.descriptorBindings[1].type ==
                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
                    toneMappingFragment.descriptorBindings[2].binding == 2 &&
                    toneMappingFragment.descriptorBindings[2].type ==
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                    toneMappingFragment.descriptorBindings[3].binding == 3 &&
                    toneMappingFragment.descriptorBindings[3].type ==
                        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                "Tone-mapping fragment reflection failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
