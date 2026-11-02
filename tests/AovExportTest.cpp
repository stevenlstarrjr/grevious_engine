#include "renderer/output/AovExport.hpp"

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

class TemporaryDirectory final {
  public:
    TemporaryDirectory()
        : path_(std::filesystem::temp_directory_path() /
                ("genesis_render_aov_export_" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {}

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

  private:
    std::filesystem::path path_;
};

} // namespace

int main() {
    try {
        require(std::endian::native == std::endian::little,
                "AOV export test requires the supported little-endian host");
        renderer::output::AovFrame frame{
            .width = 1U,
            .height = 2U,
            .frameNumber = 7U,
            .accumulatedSamples = 8U,
            .planes = {
                renderer::output::AovPlane{
                    .semantic = renderer::output::AovSemantic::beauty,
                    .channels = 3U,
                    .pixels = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                },
                renderer::output::AovPlane{
                    .semantic = renderer::output::AovSemantic::depth,
                    .channels = 1U,
                    .pixels = {0.25F, 0.75F},
                },
            },
        };
        renderer::output::validateAovFrame(frame);

        TemporaryDirectory temporary;
        renderer::output::PfmAovDirectorySink sink(temporary.path());
        sink.consume(frame);
        const std::filesystem::path directory = temporary.path() / "frame_00000007";

        std::ifstream beauty(directory / "beauty.pfm", std::ios::binary);
        require(static_cast<bool>(beauty), "Beauty PFM was not written");
        std::string magic;
        std::uint32_t width = 0U;
        std::uint32_t height = 0U;
        float scale = 0.0F;
        beauty >> magic >> width >> height >> scale;
        beauty.get();
        require(magic == "PF" && width == 1U && height == 2U && scale < 0.0F,
                "Beauty PFM header is incorrect");
        std::array<float, 6> values{};
        beauty.read(reinterpret_cast<char*>(values.data()),
                    static_cast<std::streamsize>(sizeof(values)));
        require(beauty.gcount() == static_cast<std::streamsize>(sizeof(values)) &&
                    std::abs(values[0] - 4.0F) < 1.0e-6F &&
                    std::abs(values[3] - 1.0F) < 1.0e-6F,
                "PFM rows are not stored in standard bottom-to-top order");

        std::ifstream manifest(directory / "manifest.json");
        const std::string manifestText((std::istreambuf_iterator<char>(manifest)),
                                       std::istreambuf_iterator<char>());
        require(manifestText.find("\"schema\": \"bloxel.aov-frame.v1\"") !=
                        std::string::npos &&
                    manifestText.find("\"producer\": \"Genesis Render\"") !=
                        std::string::npos &&
                    manifestText.find("\"accumulated_samples\": 8") !=
                        std::string::npos &&
                    manifestText.find("\"name\": \"depth\"") != std::string::npos,
                "AOV manifest metadata is incomplete");

        frame.planes.push_back({renderer::output::AovSemantic::objectId, 1U, {0.0F, 1.0F}});
        frame.objectLabels.push_back({1U, "Wheel, \"left\""});
        sink.consume(frame);
        require(std::filesystem::file_size(directory / "object_id.png") > 8U,
                "Lossless ID color image was not written");
        std::ifstream legend(directory / "id_legend.csv");
        const std::string legendText((std::istreambuf_iterator<char>(legend)), {});
        require(legendText.find("object_id,1,158,55,121,\"Wheel, \"\"left\"\"\"") !=
                    std::string::npos,
                "ID legend did not preserve the object name and color");
        frame.planes.back().pixels[0] = 0.5F;
        bool rejectedFractionalId = false;
        try {
            renderer::output::validateAovFrame(frame);
        } catch (const std::invalid_argument&) {
            rejectedFractionalId = true;
        }
        require(rejectedFractionalId, "Fractional object ID was accepted");
        frame.planes.pop_back();
        frame.planes.push_back(frame.planes.front());
        bool rejectedDuplicate = false;
        try {
            renderer::output::validateAovFrame(frame);
        } catch (const std::invalid_argument&) {
            rejectedDuplicate = true;
        }
        require(rejectedDuplicate, "Duplicate AOV semantics were accepted");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
