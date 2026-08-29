#!/usr/bin/env bash
set -euo pipefail

renderer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${renderer_root}/build/release}"
output_dir="${2:-${renderer_root}/captures/current/upscaling}"
genesis_render="${build_dir}/bin/genesis_render"
metrics="${build_dir}/post_image_metrics"
maximum_native_ms="${UPSCALING_NATIVE_MAX_MS:-0.05}"

cmake --build "${build_dir}" --target genesis_render upscaling_test post_image_metrics -j4
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
    shift
    run_renderer "${name}" \
        --resolution 320 180 \
        --frames 4 \
        --capture "${output_dir}/${name}.png" \
        --capture-frame 4 \
        --ray-tracing off \
        --taa off \
        --bloom off \
        --post-test checkerboard \
        --post-value 4 \
        "$@"
}

capture_raster implicit
capture_raster explicit --upscaler native \
    --timing-output "${output_dir}/raster_timing.txt"

run_renderer resize \
    --resolution 640 360 \
    --frames 10 \
    --ray-tracing off \
    --taa on \
    --bloom on \
    --upscaler native \
    --resize-script odd

run_renderer animated_path \
    --resolution 320 180 \
    --frames 8 \
    --render-mode path-traced \
    --ray-tracing required \
    --demo directional-shadow \
    --rt-transform-script oscillate \
    --upscaler native \
    --timing-output "${output_dir}/path_timing.txt"

for name in implicit explicit resize animated_path; do
    if grep -E '\[(WARN|ERROR)\]|VUID-|Validation Error' "${output_dir}/${name}.log"; then
        echo "Upscaling validation log contains a warning or error: ${name}" >&2
        exit 1
    fi
    grep -F "Upscaler: native identity" "${output_dir}/${name}.log" >/dev/null
    grep -F "Native upscaling" "${output_dir}/${name}.log" >/dev/null
done

grep -F "Upscaler resize: native identity, render 1279x719 -> output 1279x719; reconstruction history reset" \
    "${output_dir}/resize.log" >/dev/null
grep -F "Upscaler resize: native identity, render 640x360 -> output 640x360; reconstruction history reset" \
    "${output_dir}/resize.log" >/dev/null
grep -F "Recorded 8 synchronized in-place TLAS update(s), including 8 with scripted changing transforms" \
    "${output_dir}/animated_path.log" >/dev/null
grep -F "Path tracing accumulated 1 sample(s) since the last reset across 7 reset(s)" \
    "${output_dir}/animated_path.log" >/dev/null

comparison="$("${metrics}" compare "${output_dir}/implicit.png" "${output_dir}/explicit.png")"
rmse="$(sed -n 's/.*rmse=\([^ ]*\).*/\1/p' <<<"${comparison}")"
if [[ -z "${rmse}" ]]; then
    echo "Could not parse native identity RMSE" >&2
    exit 1
fi
awk -v value="${rmse}" 'BEGIN { if (value != value || value > 0.000001) exit 1 }'

timing_value() {
    local report="$1"
    sed -n 's/^native upscaling boundary: \([^ ]*\) ms.*/\1/p' "${report}"
}

raster_ms="$(timing_value "${output_dir}/raster_timing.txt")"
path_ms="$(timing_value "${output_dir}/path_timing.txt")"
if [[ -z "${raster_ms}" || -z "${path_ms}" ]]; then
    echo "Could not parse native upscaling timing" >&2
    exit 1
fi
awk -v value="${raster_ms}" -v maximum="${maximum_native_ms}" \
    'BEGIN { if (value != value || value > maximum) exit 1 }'
awk -v value="${path_ms}" -v maximum="${maximum_native_ms}" \
    'BEGIN { if (value != value || value > maximum) exit 1 }'

{
    echo "Native upscaling validation passed:"
    echo "  implicit/explicit native RMSE: ${rmse}"
    echo "  raster timestamp boundary: ${raster_ms} ms (maximum ${maximum_native_ms} ms)"
    echo "  animated-path timestamp boundary: ${path_ms} ms (maximum ${maximum_native_ms} ms)"
    echo "  odd 1279x719 resize and 640x360 restore: clean"
    echo "  animated path frame-slot updates: 8, resets: 7"
} | tee "${output_dir}/summary.txt"
