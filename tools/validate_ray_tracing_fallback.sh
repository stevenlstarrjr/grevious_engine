#!/usr/bin/env bash
set -euo pipefail

renderer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${renderer_root}/build/debug}"
output_dir="${2:-${renderer_root}/captures/current/ray_tracing_fallback}"
renderer_demo="${build_dir}/bin/renderer_demo"
metrics="${build_dir}/post_image_metrics"

cmake --build "${build_dir}" -j4
mkdir -p "${output_dir}"

capture() {
    local mode="$1"
    local output="$2"
    local log="$3"
    env SDL_VIDEODRIVER=wayland "${renderer_demo}" \
        --validation required \
        --demo directional-shadow \
        --shadows on \
        --ray-tracing "${mode}" \
        --taa off \
        --bloom off \
        --resolution 1280 720 \
        --frames 8 \
        --capture "${output}" \
        --capture-frame 8 2>&1 | tee "${log}"
}

capture auto "${output_dir}/auto.png" "${output_dir}/auto.log"
capture off "${output_dir}/off.png" "${output_dir}/off.log"

grep -F "Shadow backend: raster" "${output_dir}/auto.log" >/dev/null
grep -F "Shadow backend: raster" "${output_dir}/off.log" >/dev/null

comparison="$(${metrics} compare "${output_dir}/off.png" "${output_dir}/auto.png")"
rmse="$(sed -n 's/.*rmse=\([^ ]*\).*/\1/p' <<<"${comparison}")"
if [[ -z "${rmse}" ]]; then
    echo "Could not parse ray-tracing fallback RMSE" >&2
    exit 1
fi
awk -v actual="${rmse}" 'BEGIN { if (actual > 0.000001) exit 1 }'

echo "Ray-tracing raster fallback validation passed:"
echo "  automatic/off image RMSE: ${rmse} (maximum 0.000001)"
echo "  automatic mode selected the certified raster shadow backend"
