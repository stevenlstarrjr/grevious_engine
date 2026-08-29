#!/usr/bin/env bash
set -euo pipefail

renderer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${renderer_root}/build/release}"
output_dir="${2:-${renderer_root}/captures/current/temporal_upscaling}"
genesis_render="${build_dir}/bin/genesis_render"
metrics="${build_dir}/post_image_metrics"
maximum_temporal_ms="${TEMPORAL_UPSCALING_MAX_MS:-0.10}"

cmake --build "${build_dir}" --target genesis_render upscaling_test \
    post_image_metrics -j4
mkdir -p "${output_dir}"
"${build_dir}/upscaling_test"

run_renderer() {
    local name="$1"
    shift
    env SDL_VIDEODRIVER=wayland "${genesis_render}" \
        --validation required \
        --camera-input off \
        "$@" >"${output_dir}/${name}.log" 2>&1
}

capture_raster() {
    local name="$1"
    local frames="$2"
    shift 2
    run_renderer "${name}" \
        --resolution 320 180 \
        --frames "${frames}" \
        --capture "${output_dir}/${name}.png" \
        --capture-frame "${frames}" \
        --ray-tracing off \
        --taa on \
        --bloom off \
        --post-test checkerboard \
        --post-value 4 \
        --upscaler temporal \
        --upscaler-quality performance \
        "$@"
}

capture_raster repeat_a 6 --timing-output "${output_dir}/raster_timing.txt"
capture_raster repeat_b 6
capture_raster history_invalid 1 --debug-view history-validity
capture_raster history_static 4 --debug-view history-validity
capture_raster history_teleport 3 --debug-view history-validity \
    --camera-script teleport

run_renderer resize \
    --resolution 640 360 \
    --frames 10 \
    --ray-tracing off \
    --taa on \
    --bloom on \
    --upscaler temporal \
    --upscaler-quality performance \
    --resize-script odd

run_renderer animated_path \
    --resolution 320 180 \
    --frames 8 \
    --render-mode path-traced \
    --ray-tracing required \
    --demo directional-shadow \
    --rt-transform-script oscillate \
    --upscaler temporal \
    --upscaler-quality performance \
    --timing-output "${output_dir}/path_timing.txt"

for name in repeat_a repeat_b history_invalid history_static history_teleport resize \
    animated_path; do
    if grep -E '\[(WARN|ERROR)\]|VUID-|Validation Error' "${output_dir}/${name}.log"; then
        echo "Temporal upscaling validation log contains a warning or error: ${name}" >&2
        exit 1
    fi
done

for name in repeat_a repeat_b history_invalid history_static history_teleport animated_path; do
    grep -F "Upscaler: temporal, quality performance, render 160x90 -> output 320x180" \
        "${output_dir}/${name}.log" >/dev/null
    grep -F "Upscaler capabilities: reconstruction=temporal, history=backend, output=renderer, reactive-mask=0, sharpening=0, dynamic-resolution=0" \
        "${output_dir}/${name}.log" >/dev/null
    grep -F "Temporal upscaling history at 320x180: HDR color 0.878906 MiB, depth 0.439453 MiB; persistent total 1.31836 MiB" \
        "${output_dir}/${name}.log" >/dev/null
done
grep -F "Raster TAA bypassed because the temporal upscaler owns output history" \
    "${output_dir}/repeat_a.log" >/dev/null
grep -F "Temporal upscaling" "${output_dir}/repeat_a.log" >/dev/null
grep -F "Upscaler resize: temporal, quality performance, render 640x360 -> output 1279x719; reconstruction history reset" \
    "${output_dir}/resize.log" >/dev/null
grep -F "Upscaler resize: temporal, quality performance, render 320x180 -> output 640x360; reconstruction history reset" \
    "${output_dir}/resize.log" >/dev/null
grep -F "Recorded 8 synchronized in-place TLAS update(s), including 8 with scripted changing transforms" \
    "${output_dir}/animated_path.log" >/dev/null
grep -F "Path tracing accumulated 1 sample(s) since the last reset across 7 reset(s)" \
    "${output_dir}/animated_path.log" >/dev/null

repeat_comparison="$("${metrics}" compare "${output_dir}/repeat_a.png" \
    "${output_dir}/repeat_b.png")"
repeat_rmse="$(sed -n 's/.*rmse=\([^ ]*\).*/\1/p' <<<"${repeat_comparison}")"
if [[ -z "${repeat_rmse}" ]]; then
    echo "Could not parse temporal repeat RMSE" >&2
    exit 1
fi
awk -v value="${repeat_rmse}" 'BEGIN { if (value != value || value > 0.000001) exit 1 }'

channel_mean() {
    local image="$1"
    local channel="$2"
    "${metrics}" stats "${image}" | sed -n "s/.*mean_${channel}=\([^ ]*\).*/\1/p"
}

invalid_red="$(channel_mean "${output_dir}/history_invalid.png" r)"
invalid_green="$(channel_mean "${output_dir}/history_invalid.png" g)"
static_red="$(channel_mean "${output_dir}/history_static.png" r)"
static_green="$(channel_mean "${output_dir}/history_static.png" g)"
teleport_red="$(channel_mean "${output_dir}/history_teleport.png" r)"
teleport_green="$(channel_mean "${output_dir}/history_teleport.png" g)"
awk -v red="${invalid_red}" -v green="${invalid_green}" \
    'BEGIN { if (red < 0.99 || green > 0.01) exit 1 }'
awk -v red="${static_red}" -v green="${static_green}" \
    'BEGIN { if (red > 0.01 || green < 0.99) exit 1 }'
awk -v red="${teleport_red}" -v green="${teleport_green}" \
    'BEGIN { if (red < 0.99 || green > 0.01) exit 1 }'

timing_value() {
    sed -n 's/^upscaling: \([^ ]*\) ms.*/\1/p' "$1"
}

raster_ms="$(timing_value "${output_dir}/raster_timing.txt")"
path_ms="$(timing_value "${output_dir}/path_timing.txt")"
if [[ -z "${raster_ms}" || -z "${path_ms}" ]]; then
    echo "Could not parse temporal upscaling timing" >&2
    exit 1
fi
for value in "${raster_ms}" "${path_ms}"; do
    awk -v value="${value}" -v maximum="${maximum_temporal_ms}" \
        'BEGIN { if (value != value || value > maximum) exit 1 }'
done

{
    echo "Temporal upscaling validation passed:"
    echo "  deterministic repeat RMSE: ${repeat_rmse}"
    echo "  first-frame validity red/green: ${invalid_red}/${invalid_green}"
    echo "  stationary validity red/green: ${static_red}/${static_green}"
    echo "  teleport validity red/green: ${teleport_red}/${teleport_green}"
    echo "  raster temporal reconstruction: ${raster_ms} ms (maximum ${maximum_temporal_ms} ms)"
    echo "  animated-path temporal reconstruction: ${path_ms} ms (maximum ${maximum_temporal_ms} ms)"
    echo "  output history: 1.31836 MiB at 320x180"
    echo "  odd resize, bloom, animated TLAS updates, and Vulkan validation: clean"
} | tee "${output_dir}/summary.txt"
