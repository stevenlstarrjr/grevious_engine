#!/usr/bin/env bash
set -euo pipefail

renderer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${renderer_root}/build/debug}"
output_dir="${2:-${renderer_root}/captures/current/directional_shadow}"
genesis_render="${build_dir}/bin/genesis_render"
metrics="${build_dir}/post_image_metrics"

cmake --build "${build_dir}" -j4
mkdir -p "${output_dir}"

run_capture() {
    local shadow_mode="$1"
    local output="$2"
    local frames="$3"
    shift 3
    env SDL_VIDEODRIVER=wayland "${genesis_render}" \
        --demo directional-shadow \
        --ray-tracing off \
        --shadows "${shadow_mode}" \
        --taa off \
        --bloom off \
        --frames "${frames}" \
        --capture "${output}" \
        --capture-frame "${frames}" \
        --validation required \
        "$@"
}

run_capture on "${output_dir}/on.png" 8 \
    --timing-output "${output_dir}/gpu_timings.txt"
run_capture on "${output_dir}/on_repeat.png" 8
run_capture off "${output_dir}/off.png" 8
run_capture on "${output_dir}/moving_camera.png" 40 --camera-script translate
run_capture on "${output_dir}/moving_camera_repeat.png" 40 --camera-script translate

on_stats="$(${metrics} stats "${output_dir}/on.png")"
off_stats="$(${metrics} stats "${output_dir}/off.png")"
repeat_comparison="$(${metrics} compare "${output_dir}/on.png" "${output_dir}/on_repeat.png")"
toggle_comparison="$(${metrics} compare "${output_dir}/off.png" "${output_dir}/on.png")"
moving_comparison="$(${metrics} compare "${output_dir}/moving_camera.png" \
                                      "${output_dir}/moving_camera_repeat.png")"

value() {
    local line="$1"
    local key="$2"
    sed -n "s/.*${key}=\([^ ]*\).*/\1/p" <<<"${line}"
}

on_mean="$(value "${on_stats}" mean)"
off_mean="$(value "${off_stats}" mean)"
repeat_rmse="$(value "${repeat_comparison}" rmse)"
toggle_rmse="$(value "${toggle_comparison}" rmse)"
moving_rmse="$(value "${moving_comparison}" rmse)"

[[ "${on_stats}" == *"finite=1"* && "${off_stats}" == *"finite=1"* ]]
awk -v actual="${repeat_rmse}" 'BEGIN { if (actual > 0.000001) exit 1 }'
awk -v actual="${moving_rmse}" 'BEGIN { if (actual > 0.000001) exit 1 }'
awk -v actual="${toggle_rmse}" 'BEGIN { if (actual < 0.02) exit 1 }'
awk -v shadowed="${on_mean}" -v unshadowed="${off_mean}" \
    'BEGIN { if (unshadowed - shadowed < 0.005) exit 1 }'

echo "Directional-shadow regression passed"
echo "  repeat RMSE: ${repeat_rmse} (maximum 0.000001)"
echo "  moving-camera repeat RMSE: ${moving_rmse} (maximum 0.000001)"
echo "  on/off RMSE: ${toggle_rmse} (minimum 0.02)"
echo "  mean luminance: shadows on ${on_mean}, shadows off ${off_mean}"
