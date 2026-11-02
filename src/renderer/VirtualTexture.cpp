#include "VirtualTexture.hpp"
#include "renderer/vulkan/Device.hpp"
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace renderer {
namespace {
constexpr uint32_t edge = 128, tileBytes = edge * edge * 4, batchSize = 64;
template <class T> T readWord(std::istream& input) {
    T value{};
    if (!input.read(reinterpret_cast<char*>(&value), sizeof(value)))
        throw std::runtime_error("Truncated virtual texture file");
    return value;
}
} // namespace

VirtualTexture::VirtualTexture(const std::filesystem::path& path, const vulkan::Device& device,
                               vulkan::MemoryAllocator& allocator, uint32_t poolMiB)
    : file_(path, std::ios::binary) {
    static_assert(std::endian::native == std::endian::little);
    char magic[8]{};
    if (!file_.read(magic, 8) || std::memcmp(magic, "GRVT001", 8) != 0)
        throw std::runtime_error("Invalid virtual texture header");
    const auto width = readWord<uint32_t>(file_), height = readWord<uint32_t>(file_);
    const auto size = readWord<uint32_t>(file_), levels = readWord<uint32_t>(file_);
    if (!std::has_single_bit(width) || !std::has_single_bit(height) ||
        std::max(width, height) > 131072 || size != edge ||
        levels != std::bit_width(std::max(width, height)) || poolMiB < 1 || poolMiB > 192)
        throw std::runtime_error("Unsupported virtual texture dimensions or pool size");
    for (uint32_t i = 0; i < levels; ++i) {
        levelStarts_.push_back(pageCount_);
        pageCount_ += ((std::max(1u, width >> i) + edge - 1) / edge) *
                      ((std::max(1u, height >> i) + edge - 1) / edge);
    }
    const uint64_t bytes = std::filesystem::file_size(path);
    const uint64_t firstPayload = 24ull + uint64_t(pageCount_) * 12;
    for (uint32_t i = 0; i < pageCount_; ++i) {
        const Entry e{readWord<uint64_t>(file_), readWord<uint32_t>(file_)};
        if (!e.length || e.length > 2 * tileBytes || e.offset < firstPayload || e.offset > bytes ||
            e.length > bytes - e.offset)
            throw std::runtime_error("Invalid virtual texture tile index");
        index_.push_back(e);
    }
    slots_.resize(poolMiB * 1024u * 1024u / tileBytes);
    // u32 header, mip start indices, page->(slot+1), then ray feedback bits.
    slotStart_ = 8 + levels;
    requestStart_ = slotStart_ + pageCount_;
    std::vector<uint32_t> state(requestStart_ + (pageCount_ + 31) / 32);
    state[0] = width;
    state[1] = height;
    state[2] = levels;
    state[3] = slotStart_;
    state[4] = requestStart_;
    std::copy(levelStarts_.begin(), levelStarts_.end(), state.begin() + 8);
    const VkBufferUsageFlags access =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    state_ = std::make_unique<vulkan::Buffer>(
        device, allocator,
        vulkan::BufferCreateInfo{.size = state.size() * sizeof(uint32_t),
                                 .usage = access,
                                 .requiredMemory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 .preferredMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                 .randomHostAccess = true});
    state_->write(std::as_bytes(std::span(state)));
    std::vector<uint32_t> queues{device.queueFamilies().graphics};
    if (device.queueFamilies().transfer != queues[0])
        queues.push_back(device.queueFamilies().transfer);
    pixels_ = std::make_unique<vulkan::Buffer>(
        device, allocator,
        vulkan::BufferCreateInfo{.size = uint64_t(slots_.size()) * tileBytes,
                                 .usage = access | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 .requiredMemory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                 .queueFamilies = queues});
    std::cout << "VT_CONFIG virtual=" << width << "x" << height << " levels=" << levels
              << " pages=" << pageCount_ << " poolBytes=" << pixels_->size()
              << " tableFeedbackBytes=" << state_->size()
              << " maxStagingBytes=" << batchSize * tileBytes << " fileBytes=" << bytes << '\n';
}

std::optional<vulkan::UploadTicket> VirtualTexture::service(vulkan::UploadContext& uploads,
                                                            uint64_t frame) {
    auto* state = static_cast<uint32_t*>(state_->mappedData());
    auto* table = state + slotStart_;
    auto* feedback = state + requestStart_;
    std::vector<uint32_t> missing;
    uint32_t requested = 0, finest = UINT32_MAX;
    const uint32_t root = pageCount_ - 1;
    if (!table[root])
        missing.push_back(root);
    for (uint32_t word = 0; word < (pageCount_ + 31) / 32; ++word) {
        uint32_t bits = feedback[word];
        feedback[word] = 0;
        while (bits) {
            const uint32_t page = word * 32 + std::countr_zero(bits);
            bits &= bits - 1;
            if (page >= pageCount_)
                continue;
            ++requested;
            const auto level =
                uint32_t(std::upper_bound(levelStarts_.begin(), levelStarts_.end(), page) -
                         levelStarts_.begin() - 1);
            finest = std::min(finest, level);
            if (table[page])
                slots_[table[page] - 1].lastUsed = frame;
            else if (page != root)
                missing.push_back(page);
        }
    }
    // Coarse requests win if the view changes faster than pages can arrive.
    std::sort(missing.rbegin(), missing.rend());
    std::vector<std::vector<std::byte>> decoded;
    std::vector<vulkan::BufferUpload> copies;
    for (const auto page : missing) {
        if (copies.size() == batchSize)
            break;
        auto candidate = slots_.end();
        for (auto it = slots_.begin(); it != slots_.end(); ++it) {
            if (it->page == UINT32_MAX) {
                candidate = it;
                break;
            }
            if (it->page != root && it->lastUsed < frame &&
                (candidate == slots_.end() || it->lastUsed < candidate->lastUsed))
                candidate = it;
        }
        if (candidate == slots_.end())
            break;
        const auto entry = index_[page];
        std::vector<char> compressed(entry.length);
        file_.seekg(static_cast<std::streamoff>(entry.offset));
        if (!file_.read(compressed.data(), entry.length))
            throw std::runtime_error("Failed to read requested VT tile");
        auto& pixels = decoded.emplace_back(tileBytes);
        if (stbi_zlib_decode_buffer(reinterpret_cast<char*>(pixels.data()), tileBytes,
                                    compressed.data(), int(entry.length)) != tileBytes)
            throw std::runtime_error("Damaged compressed VT tile");
        const auto slot = uint32_t(candidate - slots_.begin());
        if (candidate->page != UINT32_MAX) {
            table[candidate->page] = 0;
            ++evictions_;
        }
        *candidate = {page, frame};
        table[page] = slot + 1;
        copies.push_back(
            {pixels_.get(), uint64_t(slot) * tileBytes, std::span<const std::byte>(pixels)});
        diskBytes_ += entry.length;
        ++loads_;
    }
    if (!copies.empty() || frame % 16 == 0) {
        const auto resident = std::count_if(slots_.begin(), slots_.end(),
                                            [](const Slot& s) { return s.page != UINT32_MAX; });
        std::cout << "VT_FRAME frame=" << frame << " requested=" << requested
                  << " missing=" << missing.size() << " loaded=" << copies.size()
                  << " resident=" << resident << " evictions=" << evictions_
                  << " totalLoads=" << loads_ << " diskPayloadBytes=" << diskBytes_
                  << " finestMip=" << (finest == UINT32_MAX ? -1 : int(finest)) << '\n';
    }
    if (copies.empty())
        return std::nullopt;
    return uploads.upload(copies);
}
} // namespace renderer
