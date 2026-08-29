#include "renderer/assets/ImageDecoder.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

using renderer::assets::DecodedImage;

std::array<double, 3> rgb(const DecodedImage& image, std::uint32_t x, std::uint32_t y) {
    const std::size_t offset = (static_cast<std::size_t>(y) * image.width + x) * 4U;
    if (image.componentType == renderer::assets::ImageComponentType::unorm8) {
        return {static_cast<double>(image.unorm8[offset]) / 255.0,
                static_cast<double>(image.unorm8[offset + 1U]) / 255.0,
                static_cast<double>(image.unorm8[offset + 2U]) / 255.0};
    }
    return {image.float32[offset], image.float32[offset + 1U], image.float32[offset + 2U]};
}

double luminance(const std::array<double, 3>& color) {
    return color[0] * 0.2126 + color[1] * 0.7152 + color[2] * 0.0722;
}

DecodedImage load(std::string_view path) {
    DecodedImage image = renderer::assets::loadDecodedImage(
        std::filesystem::path(path), renderer::assets::ImageColorSpace::linear);
    if (!image.valid()) {
        throw std::runtime_error("Decoded image is invalid");
    }
    return image;
}

void stats(const DecodedImage& image) {
    double total = 0.0;
    std::array<double, 3> channelTotals{};
    double maximum = 0.0;
    double weightedX = 0.0;
    double weightedY = 0.0;
    bool finite = true;
    for (std::uint32_t y = 0; y < image.height; ++y) {
        for (std::uint32_t x = 0; x < image.width; ++x) {
            const auto color = rgb(image, x, y);
            const double value = luminance(color);
            finite = finite && std::isfinite(color[0]) && std::isfinite(color[1]) &&
                     std::isfinite(color[2]);
            total += value;
            for (std::size_t channel = 0U; channel < channelTotals.size(); ++channel) {
                channelTotals[channel] += color[channel];
            }
            maximum = std::max(maximum, value);
            weightedX += value * static_cast<double>(x);
            weightedY += value * static_cast<double>(y);
        }
    }
    const double count = static_cast<double>(image.width) * image.height;
    std::cout << "width=" << image.width << " height=" << image.height
              << " finite=" << (finite ? 1 : 0) << " mean=" << total / count
              << " mean_r=" << channelTotals[0] / count
              << " mean_g=" << channelTotals[1] / count
              << " mean_b=" << channelTotals[2] / count
              << " max=" << maximum << " total=" << total
              << " centroid_x=" << (total > 0.0 ? weightedX / total : 0.0)
              << " centroid_y=" << (total > 0.0 ? weightedY / total : 0.0) << '\n';
}

void compare(const DecodedImage& first, const DecodedImage& second) {
    if (first.width != second.width || first.height != second.height) {
        throw std::runtime_error("Image comparison dimensions differ");
    }
    double squaredError = 0.0;
    double maximumError = 0.0;
    const double count = static_cast<double>(first.width) * first.height * 3.0;
    for (std::uint32_t y = 0; y < first.height; ++y) {
        for (std::uint32_t x = 0; x < first.width; ++x) {
            const auto firstColor = rgb(first, x, y);
            const auto secondColor = rgb(second, x, y);
            for (std::size_t channel = 0; channel < 3U; ++channel) {
                const double error = std::abs(firstColor[channel] - secondColor[channel]);
                squaredError += error * error;
                maximumError = std::max(maximumError, error);
            }
        }
    }
    std::cout << "rmse=" << std::sqrt(squaredError / count)
              << " max_abs=" << maximumError << '\n';
}

std::uint32_t parseCoordinate(std::string_view text) {
    std::uint32_t value = 0U;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::runtime_error("Image sample coordinate is not an unsigned integer");
    }
    return value;
}

void sample(const DecodedImage& image, std::uint32_t x, std::uint32_t y) {
    if (x >= image.width || y >= image.height) {
        throw std::runtime_error("Image sample coordinate is out of bounds");
    }
    const auto color = rgb(image, x, y);
    std::cout << "r=" << color[0] << " g=" << color[1] << " b=" << color[2]
              << " luminance=" << luminance(color) << '\n';
}

void symmetry(const DecodedImage& image) {
    double horizontal = 0.0;
    double vertical = 0.0;
    double axis = 0.0;
    const double count = static_cast<double>(image.width) * image.height * 3.0;
    for (std::uint32_t y = 0; y < image.height; ++y) {
        for (std::uint32_t x = 0; x < image.width; ++x) {
            const auto color = rgb(image, x, y);
            const auto mirrorX = rgb(image, image.width - 1U - x, y);
            const auto mirrorY = rgb(image, x, image.height - 1U - y);
            for (std::size_t channel = 0; channel < 3U; ++channel) {
                const double errorX = color[channel] - mirrorX[channel];
                const double errorY = color[channel] - mirrorY[channel];
                horizontal += errorX * errorX;
                vertical += errorY * errorY;
            }
            if (image.width == image.height) {
                const auto transpose = rgb(image, y, x);
                for (std::size_t channel = 0; channel < 3U; ++channel) {
                    const double error = color[channel] - transpose[channel];
                    axis += error * error;
                }
            }
        }
    }
    std::cout << "horizontal_rmse=" << std::sqrt(horizontal / count)
              << " vertical_rmse=" << std::sqrt(vertical / count);
    if (image.width == image.height) {
        std::cout << " axis_rmse=" << std::sqrt(axis / count);
    }
    std::cout << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(12);
        if (argc == 3 && std::string_view(argv[1]) == "stats") {
            stats(load(argv[2]));
            return EXIT_SUCCESS;
        }
        if (argc == 4 && std::string_view(argv[1]) == "compare") {
            compare(load(argv[2]), load(argv[3]));
            return EXIT_SUCCESS;
        }
        if (argc == 5 && std::string_view(argv[1]) == "sample") {
            sample(load(argv[2]), parseCoordinate(argv[3]), parseCoordinate(argv[4]));
            return EXIT_SUCCESS;
        }
        if (argc == 3 && std::string_view(argv[1]) == "symmetry") {
            symmetry(load(argv[2]));
            return EXIT_SUCCESS;
        }
        std::cerr << "Usage: post_image_metrics stats IMAGE | compare A B | "
                     "sample IMAGE X Y | symmetry IMAGE\n";
        return EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
