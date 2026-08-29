#include "renderer/nvidia/NrdContext.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace renderer::nvidia {
namespace {

[[noreturn]] void throwNrdError(const char* operation, nrd::Result result) {
    throw std::runtime_error(std::string(operation) + " failed with NRD result " +
                             std::to_string(static_cast<std::uint32_t>(result)));
}

void requireSuccess(const char* operation, nrd::Result result) {
    if (result != nrd::Result::SUCCESS) {
        throwNrdError(operation, result);
    }
}

void copyMatrix(float (&destination)[16], const std::array<float, 16>& source) {
    std::copy(source.begin(), source.end(), destination);
}

} // namespace

NrdContext::NrdContext(nrd::Denoiser denoiser) : denoiser_(denoiser) {
    if (denoiser_ != nrd::Denoiser::RELAX_DIFFUSE_SPECULAR &&
        denoiser_ != nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR) {
        throw std::invalid_argument(
            "Genesis NRD context currently requires a diffuse/specular denoiser");
    }

    const nrd::DenoiserDesc denoiserDescription{
        .identifier = denoiserIdentifier,
        .denoiser = denoiser_,
    };
    const nrd::InstanceCreationDesc creationDescription{
        .denoisers = &denoiserDescription,
        .denoisersNum = 1,
    };
    requireSuccess("nrd::CreateInstance",
                   nrd::CreateInstance(creationDescription, instance_));
    try {
        configureDenoiser();
    } catch (...) {
        nrd::DestroyInstance(*instance_);
        instance_ = nullptr;
        throw;
    }
}

NrdContext::~NrdContext() {
    if (instance_ != nullptr) {
        nrd::DestroyInstance(*instance_);
    }
}

const nrd::LibraryDesc& NrdContext::libraryDescription() const {
    const nrd::LibraryDesc* description = nrd::GetLibraryDesc();
    if (description == nullptr) {
        throw std::runtime_error("nrd::GetLibraryDesc returned null");
    }
    return *description;
}

const nrd::InstanceDesc& NrdContext::instanceDescription() const {
    const nrd::InstanceDesc* description = nrd::GetInstanceDesc(*instance_);
    if (description == nullptr) {
        throw std::runtime_error("nrd::GetInstanceDesc returned null");
    }
    return *description;
}

std::span<const nrd::DispatchDesc>
NrdContext::prepareFrame(const nrd::CommonSettings& settings) {
    requireSuccess("nrd::SetCommonSettings",
                   nrd::SetCommonSettings(*instance_, settings));
    const nrd::DispatchDesc* dispatches = nullptr;
    std::uint32_t dispatchCount = 0;
    const nrd::Identifier identifier = denoiserIdentifier;
    requireSuccess("nrd::GetComputeDispatches",
                   nrd::GetComputeDispatches(*instance_, &identifier, 1,
                                             dispatches, dispatchCount));
    if (dispatches == nullptr || dispatchCount == 0) {
        throw std::runtime_error("NRD returned an empty compute dispatch list");
    }
    return {dispatches, dispatchCount};
}

void NrdContext::configureDenoiser() {
    if (denoiser_ == nrd::Denoiser::RELAX_DIFFUSE_SPECULAR) {
        nrd::RelaxSettings settings{};
        settings.diffuseMaxAccumulatedFrameNum = 30;
        settings.specularMaxAccumulatedFrameNum = 30;
        settings.diffuseMaxFastAccumulatedFrameNum = 6;
        settings.specularMaxFastAccumulatedFrameNum = 6;
        settings.hitDistanceReconstructionMode =
            nrd::HitDistanceReconstructionMode::AREA_3X3;
        settings.enableAntiFirefly = true;
        requireSuccess("nrd::SetDenoiserSettings(RELAX)",
                       nrd::SetDenoiserSettings(*instance_, denoiserIdentifier,
                                                &settings));
        return;
    }

    nrd::ReblurSettings settings{};
    settings.hitDistanceReconstructionMode =
        nrd::HitDistanceReconstructionMode::AREA_3X3;
    settings.enableAntiFirefly = true;
    requireSuccess("nrd::SetDenoiserSettings(REBLUR)",
                   nrd::SetDenoiserSettings(*instance_, denoiserIdentifier,
                                            &settings));
}

nrd::CommonSettings makeNrdCommonSettings(
    std::uint16_t width, std::uint16_t height, std::uint32_t frameIndex,
    const std::array<float, 16>& viewToClip,
    const std::array<float, 16>& previousViewToClip,
    const std::array<float, 16>& worldToView,
    const std::array<float, 16>& previousWorldToView, bool resetHistory) {
    if (width == 0 || height == 0) {
        throw std::invalid_argument("NRD render extent must be nonzero");
    }

    nrd::CommonSettings settings{};
    copyMatrix(settings.viewToClipMatrix, viewToClip);
    copyMatrix(settings.viewToClipMatrixPrev, previousViewToClip);
    copyMatrix(settings.worldToViewMatrix, worldToView);
    copyMatrix(settings.worldToViewMatrixPrev, previousWorldToView);
    settings.resourceSize[0] = width;
    settings.resourceSize[1] = height;
    settings.resourceSizePrev[0] = width;
    settings.resourceSizePrev[1] = height;
    settings.rectSize[0] = width;
    settings.rectSize[1] = height;
    settings.rectSizePrev[0] = width;
    settings.rectSizePrev[1] = height;
    settings.frameIndex = frameIndex;
    settings.accumulationMode = resetHistory
                                    ? nrd::AccumulationMode::CLEAR_AND_RESTART
                                    : nrd::AccumulationMode::CONTINUE;
    settings.isMotionVectorInWorldSpace = false;
    settings.motionVectorScale[0] = 1.0F;
    settings.motionVectorScale[1] = 1.0F;
    settings.motionVectorScale[2] = 0.0F;
    return settings;
}

} // namespace renderer::nvidia
