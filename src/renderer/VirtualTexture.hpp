#pragma once

#include "renderer/vulkan/Buffer.hpp"
#include "renderer/vulkan/UploadContext.hpp"
#include <filesystem>
#include <fstream>
#include <optional>

namespace renderer {

// Diagnostic path for one opaque base-color map. The renderer synchronizes the
// previous graphics frame before service() reads its ray-hit feedback.
class VirtualTexture final {
  public:
    VirtualTexture(const std::filesystem::path&, const vulkan::Device&, vulkan::MemoryAllocator&,
                   uint32_t poolMiB);
    std::optional<vulkan::UploadTicket> service(vulkan::UploadContext&, uint64_t frame);
    VkDeviceAddress stateAddress() const { return state_->deviceAddress(); }
    VkDeviceAddress pixelsAddress() const { return pixels_->deviceAddress(); }

  private:
    struct Entry {
        uint64_t offset;
        uint32_t length;
    };
    struct Slot {
        uint32_t page = UINT32_MAX;
        uint64_t lastUsed = 0;
    };
    std::ifstream file_;
    std::vector<Entry> index_;
    std::vector<uint32_t> levelStarts_;
    std::vector<Slot> slots_;
    std::unique_ptr<vulkan::Buffer> state_, pixels_;
    uint32_t pageCount_ = 0, slotStart_ = 0, requestStart_ = 0;
    uint64_t diskBytes_ = 0, loads_ = 0, evictions_ = 0;
};
} // namespace renderer
