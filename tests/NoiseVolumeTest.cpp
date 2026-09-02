#include "renderer/procedural/NoiseVolume.hpp"

#include <algorithm>
#include <cassert>

int main() {
    const renderer::procedural::CloudNoiseVolumes first =
        renderer::procedural::makeCloudNoiseVolumes(16U, 8U, 1234U);
    const renderer::procedural::CloudNoiseVolumes second =
        renderer::procedural::makeCloudNoiseVolumes(16U, 8U, 1234U);
    const renderer::procedural::CloudNoiseVolumes different =
        renderer::procedural::makeCloudNoiseVolumes(16U, 8U, 4321U);

    assert(first.base.valid());
    assert(first.detail.valid());
    assert(first.base.levels.size() == 5U);
    assert(first.detail.levels.size() == 4U);
    assert(first.base.levels.front().texels == second.base.levels.front().texels);
    assert(first.base.levels.front().texels != different.base.levels.front().texels);

    const auto [minimum, maximum] = std::minmax_element(
        first.base.levels.front().texels.begin(),
        first.base.levels.front().texels.end());
    assert(*minimum < 64U);
    assert(*maximum > 192U);
    return 0;
}
