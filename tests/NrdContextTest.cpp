#include "renderer/nvidia/NrdContext.hpp"
#include "renderer/nvidia/NrdVulkan.hpp"

#include <NRD.h>

#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

constexpr std::array<float, 16> identityMatrix{
    1.0F, 0.0F, 0.0F, 0.0F,
    0.0F, 1.0F, 0.0F, 0.0F,
    0.0F, 0.0F, 1.0F, 0.0F,
    0.0F, 0.0F, 0.0F, 1.0F,
};

void validate(nrd::Denoiser denoiser) {
    renderer::nvidia::NrdContext context(denoiser);
    const nrd::LibraryDesc& library = context.libraryDescription();
    require(library.versionMajor >= 4, "Unexpectedly old NRD library");
    require(library.normalEncoding == nrd::NormalEncoding::R10_G10_B10_A2_UNORM,
            "NRD normal encoding does not match the Genesis guide-buffer contract");
    require(library.roughnessEncoding == nrd::RoughnessEncoding::LINEAR,
            "NRD roughness encoding does not match the Genesis guide-buffer contract");

    const nrd::InstanceDesc& instance = context.instanceDescription();
    require(instance.pipelinesNum > 0, "NRD instance contains no pipelines");
    require(instance.pipelines != nullptr, "NRD pipeline table is null");
    require(instance.permanentPoolSize > 0, "NRD permanent texture pool is empty");
    require(instance.transientPoolSize > 0, "NRD transient texture pool is empty");
    require(instance.constantBufferMaxDataSize > 0,
            "NRD did not publish a constant-buffer size");
    for (std::uint32_t index = 0; index < instance.pipelinesNum; ++index) {
        const nrd::PipelineDesc& pipeline = instance.pipelines[index];
        require(pipeline.computeShaderSPIRV.bytecode != nullptr,
                "NRD pipeline is missing embedded SPIR-V");
        require(pipeline.computeShaderSPIRV.size > 0,
                "NRD pipeline has empty embedded SPIR-V");
        require(pipeline.computeShaderDXIL.bytecode == nullptr,
                "Vulkan-only NRD build unexpectedly contains DXIL");
        require(pipeline.computeShaderDXBC.bytecode == nullptr,
                "Vulkan-only NRD build unexpectedly contains DXBC");
    }
    for (std::uint32_t index = 0; index < instance.permanentPoolSize; ++index) {
        require(renderer::nvidia::nrdFormatToVulkan(instance.permanentPool[index].format) !=
                    VK_FORMAT_UNDEFINED,
                "NRD permanent-pool format has no Vulkan mapping");
        require(renderer::nvidia::nrdFormatBytesPerTexel(
                    instance.permanentPool[index].format) > 0,
                "NRD permanent-pool format has no byte-size mapping");
    }
    for (std::uint32_t index = 0; index < instance.transientPoolSize; ++index) {
        require(renderer::nvidia::nrdFormatToVulkan(instance.transientPool[index].format) !=
                    VK_FORMAT_UNDEFINED,
                "NRD transient-pool format has no Vulkan mapping");
        require(renderer::nvidia::nrdFormatBytesPerTexel(
                    instance.transientPool[index].format) > 0,
                "NRD transient-pool format has no byte-size mapping");
    }

    const nrd::CommonSettings frame = renderer::nvidia::makeNrdCommonSettings(
        800, 450, 0, identityMatrix, identityMatrix, identityMatrix,
        identityMatrix, true);
    const std::span<const nrd::DispatchDesc> dispatches = context.prepareFrame(frame);
    require(dispatches.size() >= 5, "NRD produced too few compute dispatches");
    for (const nrd::DispatchDesc& dispatch : dispatches) {
        require(dispatch.pipelineIndex < instance.pipelinesNum,
                "NRD dispatch references an invalid pipeline");
        require(dispatch.resources != nullptr && dispatch.resourcesNum > 0,
                "NRD dispatch has no resources");
        require(dispatch.gridWidth > 0 && dispatch.gridHeight > 0,
                "NRD dispatch grid is empty");
        require(dispatch.constantBufferDataSize <= instance.constantBufferMaxDataSize,
                "NRD dispatch constants exceed the advertised limit");
    }
}

} // namespace

int main() {
    try {
        validate(nrd::Denoiser::RELAX_DIFFUSE_SPECULAR);
        validate(nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR);
        std::cout << "NRD SPIR-V contracts validated for RELAX and REBLUR\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "NRD context test failed: " << error.what() << '\n';
        return 1;
    }
}
