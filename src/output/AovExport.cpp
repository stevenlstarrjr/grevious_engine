#include "renderer/output/AovExport.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace renderer::output {
namespace {

void createDirectory(const std::filesystem::path& directory) {
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        throw std::runtime_error("Could not create AOV output directory: " +
                                 directory.string());
    }
}

std::filesystem::path frameDirectory(const std::filesystem::path& root,
                                     std::uint64_t frameNumber) {
    std::ostringstream name;
    name << "frame_" << std::setfill('0') << std::setw(8) << frameNumber;
    return root / name.str();
}

void writePfm(const std::filesystem::path& path, const AovFrame& frame,
              const AovPlane& plane) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Could not write AOV plane: " + path.string());
    }
    output << (plane.channels == 3U ? "PF\n" : "Pf\n") << frame.width << ' '
           << frame.height << '\n'
           << (std::endian::native == std::endian::little ? "-1.0\n" : "1.0\n");
    const std::size_t rowValues = static_cast<std::size_t>(frame.width) * plane.channels;
    for (std::uint32_t row = frame.height; row > 0U; --row) {
        const float* values = plane.pixels.data() +
                              static_cast<std::size_t>(row - 1U) * rowValues;
        output.write(reinterpret_cast<const char*>(values),
                     static_cast<std::streamsize>(rowValues * sizeof(float)));
    }
    if (!output) {
        throw std::runtime_error("Could not finish AOV plane: " + path.string());
    }
}

} // namespace

std::string_view aovName(AovSemantic semantic) {
    switch (semantic) {
    case AovSemantic::beauty:
        return "beauty";
    case AovSemantic::rawBeauty:
        return "raw_beauty";
    case AovSemantic::temporalBeauty:
        return "temporal_beauty";
    case AovSemantic::directLighting:
        return "direct_lighting";
    case AovSemantic::indirectLighting:
        return "indirect_lighting";
    case AovSemantic::albedo:
        return "albedo";
    case AovSemantic::roughness:
        return "roughness";
    case AovSemantic::geometricNormal:
        return "geometric_normal";
    case AovSemantic::shadingNormal:
        return "shading_normal";
    case AovSemantic::motion:
        return "motion";
    case AovSemantic::expectedPreviousDepth:
        return "expected_previous_depth";
    case AovSemantic::depth:
        return "depth";
    case AovSemantic::hitDistance:
        return "hit_distance";
    case AovSemantic::historyValidity:
        return "history_validity";
    case AovSemantic::variance:
        return "variance";
    }
    throw std::runtime_error("Unknown AOV semantic");
}

std::string_view aovDescription(AovSemantic semantic) {
    switch (semantic) {
    case AovSemantic::beauty:
        return "Linear scene-referred spatially reconstructed beauty radiance";
    case AovSemantic::rawBeauty:
        return "Linear scene-referred progressive beauty accumulator";
    case AovSemantic::temporalBeauty:
        return "Linear scene-referred temporal reconstruction before spatial filtering";
    case AovSemantic::directLighting:
        return "Linear direct radiance accumulator";
    case AovSemantic::indirectLighting:
        return "Nonnegative raw beauty minus direct radiance";
    case AovSemantic::albedo:
        return "First-hit linear base color";
    case AovSemantic::roughness:
        return "First-hit perceptual roughness";
    case AovSemantic::geometricNormal:
        return "First-hit world-space geometric normal";
    case AovSemantic::shadingNormal:
        return "First-hit world-space shading normal";
    case AovSemantic::motion:
        return "Current UV minus previous UV in RG; B is zero";
    case AovSemantic::expectedPreviousDepth:
        return "Expected previous Vulkan device depth; negative marks invalid projection";
    case AovSemantic::depth:
        return "Current Vulkan device depth";
    case AovSemantic::hitDistance:
        return "First-hit world distance; zero identifies the infinite environment";
    case AovSemantic::historyValidity:
        return "Motion-aware temporal history confidence";
    case AovSemantic::variance:
        return "Temporal luminance variance used by spatial reconstruction";
    }
    throw std::runtime_error("Unknown AOV semantic");
}

void validateAovFrame(const AovFrame& frame) {
    if (frame.width == 0U || frame.height == 0U || frame.planes.empty()) {
        throw std::invalid_argument("AOV frame requires an extent and at least one plane");
    }
    const std::uint64_t pixelCount =
        static_cast<std::uint64_t>(frame.width) * frame.height;
    if (pixelCount > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("AOV frame extent exceeds host address space");
    }
    std::vector<AovSemantic> semantics;
    semantics.reserve(frame.planes.size());
    for (const AovPlane& plane : frame.planes) {
        if (plane.channels != 1U && plane.channels != 3U) {
            throw std::invalid_argument("PFM AOV planes require one or three channels");
        }
        const std::uint64_t expectedValues = pixelCount * plane.channels;
        if (expectedValues != plane.pixels.size()) {
            throw std::invalid_argument("AOV plane size does not match the frame extent");
        }
        for (float value : plane.pixels) {
            if (!std::isfinite(value)) {
                throw std::invalid_argument("AOV planes require finite float values");
            }
        }
        if (std::find(semantics.begin(), semantics.end(), plane.semantic) !=
            semantics.end()) {
            throw std::invalid_argument("AOV frame contains a duplicate semantic");
        }
        semantics.push_back(plane.semantic);
    }
}

PfmAovDirectorySink::PfmAovDirectorySink(std::filesystem::path rootDirectory)
    : rootDirectory_(std::move(rootDirectory)) {
    if (rootDirectory_.empty()) {
        throw std::invalid_argument("AOV directory sink requires a non-empty root path");
    }
}

void PfmAovDirectorySink::consume(AovFrame frame) {
    validateAovFrame(frame);
    const std::filesystem::path directory =
        frameDirectory(rootDirectory_, frame.frameNumber);
    createDirectory(directory);
    for (const AovPlane& plane : frame.planes) {
        writePfm(directory / (std::string(aovName(plane.semantic)) + ".pfm"), frame,
                 plane);
    }

    const std::filesystem::path manifestPath = directory / "manifest.json";
    std::ofstream manifest(manifestPath);
    if (!manifest) {
        throw std::runtime_error("Could not write AOV manifest: " + manifestPath.string());
    }
    manifest << "{\n"
             << "  \"schema\": \"bloxel.aov-frame.v1\",\n"
             << "  \"producer\": \"Genesis Render\",\n"
             << "  \"frame\": " << frame.frameNumber << ",\n"
             << "  \"accumulated_samples\": " << frame.accumulatedSamples << ",\n"
             << "  \"width\": " << frame.width << ",\n"
             << "  \"height\": " << frame.height << ",\n"
             << "  \"encoding\": \"float32-pfm-linear\",\n"
             << "  \"planes\": [\n";
    for (std::size_t index = 0U; index < frame.planes.size(); ++index) {
        const AovPlane& plane = frame.planes[index];
        const std::string_view name = aovName(plane.semantic);
        manifest << "    {\"name\": \"" << name << "\", \"file\": \"" << name
                 << ".pfm\", \"channels\": " << plane.channels
                 << ", \"description\": \"" << aovDescription(plane.semantic) << "\"}"
                 << (index + 1U == frame.planes.size() ? "\n" : ",\n");
    }
    manifest << "  ]\n}\n";
    if (!manifest) {
        throw std::runtime_error("Could not finish AOV manifest: " + manifestPath.string());
    }
}

} // namespace renderer::output
