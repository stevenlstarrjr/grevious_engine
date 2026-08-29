#!/usr/bin/env bash
set -euo pipefail

renderer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${renderer_root}/build/debug}"
output_dir="${2:-${renderer_root}/captures/current/temporal}"
static_capture="${output_dir}/static_history.png"
teleport_capture="${output_dir}/teleport_rejection.png"

cmake --build "${build_dir}" -j4
mkdir -p "${output_dir}"

env SDL_VIDEODRIVER=wayland "${build_dir}/bin/genesis_render" \
    --taa on \
    --debug-view history-validity \
    --frames 8 \
    --capture "${static_capture}" \
    --capture-frame 8 \
    --validation required

env SDL_VIDEODRIVER=wayland "${build_dir}/bin/genesis_render" \
    --taa on \
    --camera-script teleport \
    --debug-view history-validity \
    --frames 3 \
    --capture "${teleport_capture}" \
    --capture-frame 3 \
    --validation required

env SDL_VIDEODRIVER=wayland "${build_dir}/bin/genesis_render" \
    --taa on \
    --resize-script \
    --frames 12 \
    --validation required

static_green="$(magick "${static_capture}" -format '%[fx:mean.g]' info:)"
teleport_red="$(magick "${teleport_capture}" -format '%[fx:mean.r]' info:)"
awk -v value="${static_green}" 'BEGIN { if (value < 0.95) exit 1 }' || {
    echo "Temporal validation failed: static history acceptance ${static_green} < 0.95" >&2
    exit 1
}
awk -v value="${teleport_red}" 'BEGIN { if (value < 0.99) exit 1 }' || {
    echo "Temporal validation failed: teleport rejection ${teleport_red} < 0.99" >&2
    exit 1
}

echo "Temporal validation passed: static acceptance ${static_green}, teleport rejection ${teleport_red}"
