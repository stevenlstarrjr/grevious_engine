#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

struct PlaneSpec {
    std::string_view name;
    std::uint32_t channels;
};

struct Plane {
    std::uint32_t width = 0U;
    std::uint32_t height = 0U;
    std::uint32_t channels = 0U;
    std::vector<float> values;
};

constexpr std::array expectedPlanes{
    PlaneSpec{"beauty", 3U},
    PlaneSpec{"raw_beauty", 3U},
    PlaneSpec{"temporal_beauty", 3U},
    PlaneSpec{"direct_lighting", 3U},
    PlaneSpec{"indirect_lighting", 3U},
    PlaneSpec{"albedo", 3U},
    PlaneSpec{"roughness", 1U},
    PlaneSpec{"geometric_normal", 3U},
    PlaneSpec{"shading_normal", 3U},
    PlaneSpec{"motion", 3U},
    PlaneSpec{"expected_previous_depth", 1U},
    PlaneSpec{"depth", 1U},
    PlaneSpec{"hit_distance", 1U},
    PlaneSpec{"history_validity", 1U},
    PlaneSpec{"variance", 1U},
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::uint32_t parseUnsigned(const char* value, std::string_view label) {
    const std::string text(value);
    std::size_t parsed = 0U;
    const unsigned long result = std::stoul(text, &parsed);
    require(parsed == text.size() && result <= std::numeric_limits<std::uint32_t>::max(),
            "Invalid " + std::string(label) + ": " + text);
    return static_cast<std::uint32_t>(result);
}

Plane readPfm(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(static_cast<bool>(input), "Missing AOV plane: " + path.string());

    std::string magic;
    std::uint32_t width = 0U;
    std::uint32_t height = 0U;
    float scale = 0.0F;
    input >> magic >> width >> height >> scale;
    input.get();
    require((magic == "PF" || magic == "Pf") && width > 0U && height > 0U,
            "Invalid PFM header: " + path.string());
    require(scale < 0.0F && std::endian::native == std::endian::little,
            "AOV validation requires little-endian PFM payloads");

    Plane plane{
        .width = width,
        .height = height,
        .channels = magic == "PF" ? 3U : 1U,
        .values = {},
    };
    const std::uint64_t valueCount = static_cast<std::uint64_t>(width) * height * plane.channels;
    require(valueCount <= std::numeric_limits<std::size_t>::max(),
            "PFM plane exceeds the host address space");
    plane.values.resize(static_cast<std::size_t>(valueCount));
    const auto bytes = static_cast<std::streamsize>(plane.values.size() * sizeof(float));
    input.read(reinterpret_cast<char*>(plane.values.data()), bytes);
    require(input.gcount() == bytes, "Truncated PFM payload: " + path.string());
    require(input.peek() == std::char_traits<char>::eof(),
            "Unexpected trailing PFM payload: " + path.string());
    for (float value : plane.values) {
        require(std::isfinite(value), "Non-finite value in " + path.string());
    }
    return plane;
}

void requireRange(const Plane& plane, float minimum, float maximum,
                  std::string_view name) {
    const auto [low, high] = std::minmax_element(plane.values.begin(), plane.values.end());
    require(*low >= minimum && *high <= maximum,
            std::string(name) + " values fall outside [" + std::to_string(minimum) + ", " +
                std::to_string(maximum) + "]");
}

double mean(const Plane& plane) {
    double sum = 0.0;
    for (float value : plane.values) {
        sum += static_cast<double>(value);
    }
    return sum / static_cast<double>(plane.values.size());
}

std::size_t countOccurrences(std::string_view text, std::string_view token) {
    std::size_t count = 0U;
    for (std::size_t offset = 0U;
         (offset = text.find(token, offset)) != std::string_view::npos;
         offset += token.size()) {
        ++count;
    }
    return count;
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 6,
                "Usage: aov_export_metrics FRAME_DIRECTORY WIDTH HEIGHT FRAME_NUMBER "
                "ACCUMULATED_SAMPLES");
        const std::filesystem::path directory(argv[1]);
        const std::uint32_t expectedWidth = parseUnsigned(argv[2], "width");
        const std::uint32_t expectedHeight = parseUnsigned(argv[3], "height");
        const std::uint32_t expectedFrame = parseUnsigned(argv[4], "frame number");
        const std::uint32_t expectedSamples = parseUnsigned(argv[5], "sample count");

        std::ifstream manifestFile(directory / "manifest.json");
        require(static_cast<bool>(manifestFile), "Missing AOV manifest");
        const std::string manifest((std::istreambuf_iterator<char>(manifestFile)),
                                   std::istreambuf_iterator<char>());
        require(manifest.find("\"schema\": \"bloxel.aov-frame.v1\"") != std::string::npos,
                "AOV manifest schema is missing or unsupported");
        require(manifest.find("\"producer\": \"Genesis Render\"") != std::string::npos,
                "AOV manifest producer is missing or unsupported");
        require(manifest.find("\"frame\": " + std::to_string(expectedFrame)) !=
                        std::string::npos &&
                    manifest.find("\"width\": " + std::to_string(expectedWidth)) !=
                    std::string::npos &&
                    manifest.find("\"height\": " + std::to_string(expectedHeight)) !=
                        std::string::npos &&
                    manifest.find("\"accumulated_samples\": " +
                                  std::to_string(expectedSamples)) != std::string::npos,
                "AOV manifest dimensions or sample count do not match the requested frame");
        require(countOccurrences(manifest, "\"name\": ") == expectedPlanes.size(),
                "AOV manifest does not contain exactly the expected semantic planes");

        std::unordered_map<std::string, Plane> planes;
        for (const PlaneSpec& spec : expectedPlanes) {
            require(manifest.find("\"name\": \"" + std::string(spec.name) + "\"") !=
                        std::string::npos,
                    "AOV manifest is missing plane " + std::string(spec.name));
            Plane plane = readPfm(directory / (std::string(spec.name) + ".pfm"));
            require(plane.width == expectedWidth && plane.height == expectedHeight &&
                        plane.channels == spec.channels,
                    "AOV plane layout does not match its contract: " +
                        std::string(spec.name));
            planes.emplace(std::string(spec.name), std::move(plane));
        }

        for (std::string_view name : {"beauty", "raw_beauty", "temporal_beauty",
                                     "direct_lighting", "indirect_lighting", "albedo"}) {
            requireRange(planes.at(std::string(name)), -1.0e-5F,
                         std::numeric_limits<float>::max(), name);
        }
        require(mean(planes.at("beauty")) > 1.0e-5, "Beauty AOV is effectively black");
        requireRange(planes.at("albedo"), -1.0e-5F, 1.001F, "albedo");
        requireRange(planes.at("roughness"), -1.0e-5F, 1.001F, "roughness");
        requireRange(planes.at("depth"), -1.0e-5F, 1.001F, "depth");
        requireRange(planes.at("expected_previous_depth"), -1.001F, 1.001F,
                     "expected_previous_depth");
        requireRange(planes.at("hit_distance"), -1.0e-5F,
                     std::numeric_limits<float>::max(), "hit_distance");
        requireRange(planes.at("history_validity"), -1.0e-5F, 1.001F,
                     "history_validity");
        requireRange(planes.at("variance"), -1.0e-5F,
                     std::numeric_limits<float>::max(), "variance");
        requireRange(planes.at("geometric_normal"), -1.001F, 1.001F,
                     "geometric_normal");
        requireRange(planes.at("shading_normal"), -1.001F, 1.001F,
                     "shading_normal");

        const Plane& raw = planes.at("raw_beauty");
        const Plane& direct = planes.at("direct_lighting");
        const Plane& indirect = planes.at("indirect_lighting");
        float maximumResidual = 0.0F;
        for (std::size_t index = 0U; index < raw.values.size(); ++index) {
            const float expected = std::max(raw.values[index] - direct.values[index], 0.0F);
            maximumResidual = std::max(maximumResidual,
                                       std::abs(indirect.values[index] - expected));
        }
        require(maximumResidual <= 1.0e-6F,
                "Indirect-lighting decomposition residual exceeds tolerance");

        const Plane& hitDistance = planes.at("hit_distance");
        const Plane& geometricNormal = planes.at("geometric_normal");
        const Plane& shadingNormal = planes.at("shading_normal");
        std::size_t surfacePixels = 0U;
        for (std::size_t pixel = 0U; pixel < hitDistance.values.size(); ++pixel) {
            const std::size_t offset = pixel * 3U;
            const auto length = [offset](const Plane& normal) {
                const float x = normal.values[offset];
                const float y = normal.values[offset + 1U];
                const float z = normal.values[offset + 2U];
                return std::sqrt(x * x + y * y + z * z);
            };
            if (hitDistance.values[pixel] > 0.0F) {
                ++surfacePixels;
                require(std::abs(length(geometricNormal) - 1.0F) <= 1.0e-4F &&
                            std::abs(length(shadingNormal) - 1.0F) <= 1.0e-4F,
                        "Surface normal is not unit length");
            } else {
                require(length(geometricNormal) <= 1.0e-6F &&
                            length(shadingNormal) <= 1.0e-6F,
                        "Environment pixel contains a surface normal");
            }
        }
        require(surfacePixels > 0U, "AOV frame contains no surface pixels");

        const Plane& motion = planes.at("motion");
        const Plane& historyValidity = planes.at("history_validity");
        double motionLengthSum = 0.0;
        float motionLengthMaximum = 0.0F;
        double movingHistorySum = 0.0;
        std::size_t movingPixels = 0U;
        std::size_t rejectedPixels = 0U;
        std::size_t rejectedSurfacePixels = 0U;
        std::size_t rejectedMovingPixels = 0U;
        for (std::size_t index = 2U; index < motion.values.size(); index += 3U) {
            require(std::abs(motion.values[index]) <= 1.0e-7F,
                    "Motion AOV blue channel must be zero");
            const float x = motion.values[index - 2U];
            const float y = motion.values[index - 1U];
            const float length = std::sqrt(x * x + y * y);
            motionLengthSum += static_cast<double>(length);
            motionLengthMaximum = std::max(motionLengthMaximum, length);
            if (length > 1.0e-6F) {
                const std::size_t pixel = index / 3U;
                movingHistorySum += static_cast<double>(historyValidity.values[pixel]);
                ++movingPixels;
                if (historyValidity.values[pixel] < 0.5F) {
                    ++rejectedMovingPixels;
                }
            }
            const std::size_t pixel = index / 3U;
            if (historyValidity.values[pixel] < 0.5F) {
                ++rejectedPixels;
                if (hitDistance.values[pixel] > 0.0F) {
                    ++rejectedSurfacePixels;
                }
            }
        }
        const double motionLengthMean =
            motionLengthSum / static_cast<double>(hitDistance.values.size());
        const double movingHistoryMean =
            movingPixels > 0U
                ? movingHistorySum / static_cast<double>(movingPixels)
                : 0.0;
        const double historyMean = mean(historyValidity);

        std::cout << "AOV export metrics: planes=" << expectedPlanes.size()
                  << " extent=" << expectedWidth << 'x' << expectedHeight
                  << " frame=" << expectedFrame
                  << " samples=" << expectedSamples
                  << " surface_pixels=" << surfacePixels
                  << " beauty_mean=" << mean(planes.at("beauty"))
                  << " decomposition_max=" << maximumResidual
                  << " motion_mean=" << motionLengthMean
                  << " motion_max=" << motionLengthMaximum
                  << " moving_pixels=" << movingPixels
                  << " moving_history_mean=" << movingHistoryMean
                  << " rejected_pixels=" << rejectedPixels
                  << " rejected_surface_pixels=" << rejectedSurfacePixels
                  << " rejected_moving_pixels=" << rejectedMovingPixels
                  << " history_mean=" << historyMean << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
