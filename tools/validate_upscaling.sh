#!/usr/bin/env bash
set -euo pipefail

renderer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${renderer_root}/build/release}"
output_dir="${2:-${renderer_root}/captures/current/upscaling}"
genesis_render="${build_dir}/bin/genesis_render"
metrics="${build_dir}/post_image_metrics"
maximum_native_ms="${UPSCALING_NATIVE_MAX_MS:-0.05}"
maximum_spatial_ms="${UPSCALING_SPATIAL_MAX_MS:-0.10}"

cmake --build "${build_dir}" --target genesis_render upscaling_test post_image_metrics -j4
mkdir -p "${output_dir}"
"${build_dir}/upscaling_test"

expect_cli_failure() {
    local name="$1"
    local expected="$2"
    shift 2
    if "${genesis_render}" "$@" >"${output_dir}/${name}.log" 2>&1; then
        echo "Upscaling CLI unexpectedly accepted invalid options: ${name}" >&2
        exit 1
    fi
    grep -F -- "${expected}" "${output_dir}/${name}.log" >/dev/null
}

expect_cli_failure quality_scale_conflict \
    "--upscaler-quality and --render-scale are mutually exclusive" \
    --upscaler spatial --upscaler-quality quality --render-scale 0.5
expect_cli_failure native_quality \
    "--upscaler-quality requires --upscaler spatial or temporal" \
    --upscaler native --upscaler-quality performance

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
capture_raster spatial_quality --upscaler spatial --upscaler-quality quality \
    --timing-output "${output_dir}/spatial_quality_timing.txt"
capture_raster spatial_balanced --upscaler spatial --upscaler-quality balanced \
    --timing-output "${output_dir}/spatial_balanced_timing.txt"
capture_raster spatial_performance_a --upscaler spatial --upscaler-quality performance \
    --timing-output "${output_dir}/spatial_performance_timing.txt"
capture_raster spatial_performance_b --upscaler spatial --upscaler-quality performance
capture_raster spatial_custom --upscaler spatial --render-scale 0.75 \
    --timing-output "${output_dir}/spatial_custom_timing.txt"

run_renderer resize \
    --resolution 640 360 \
    --frames 10 \
    --ray-tracing off \
    --taa on \
    --bloom on \
    --upscaler spatial \
    --upscaler-quality performance \
    --resize-script odd

run_renderer animated_path_native \
    --resolution 320 180 \
    --frames 8 \
    --render-mode path-traced \
    --ray-tracing required \
    --demo directional-shadow \
    --rt-transform-script oscillate \
    --upscaler native \
    --timing-output "${output_dir}/native_path_timing.txt"

run_renderer animated_path_spatial \
    --resolution 320 180 \
    --frames 8 \
    --render-mode path-traced \
    --ray-tracing required \
    --demo directional-shadow \
    --rt-transform-script oscillate \
    --upscaler spatial \
    --upscaler-quality performance \
    --timing-output "${output_dir}/spatial_path_timing.txt"

for name in implicit explicit spatial_quality spatial_balanced spatial_performance_a \
    spatial_performance_b spatial_custom resize animated_path_native animated_path_spatial; do
    if grep -E '\[(WARN|ERROR)\]|VUID-|Validation Error' "${output_dir}/${name}.log"; then
        echo "Upscaling validation log contains a warning or error: ${name}" >&2
        exit 1
    fi
done

grep -F "Upscaler: native, quality native, render 320x180 -> output 320x180" \
    "${output_dir}/implicit.log" >/dev/null
grep -F "Native upscaling" "${output_dir}/explicit.log" >/dev/null
grep -F "Upscaler: spatial, quality quality, render 213x120 -> output 320x180" \
    "${output_dir}/spatial_quality.log" >/dev/null
grep -F "Upscaler: spatial, quality balanced, render 188x106 -> output 320x180" \
    "${output_dir}/spatial_balanced.log" >/dev/null
for name in spatial_performance_a spatial_performance_b animated_path_spatial; do
    grep -F "Upscaler: spatial, quality performance, render 160x90 -> output 320x180" \
        "${output_dir}/${name}.log" >/dev/null
    grep -F "Spatial upscaling" "${output_dir}/${name}.log" >/dev/null
done
grep -F "Upscaler: spatial, quality custom, render 240x135 -> output 320x180" \
    "${output_dir}/spatial_custom.log" >/dev/null
for name in spatial_quality spatial_balanced spatial_performance_a spatial_performance_b \
    spatial_custom animated_path_spatial; do
    grep -F "Spatial upscaling HDR output at 320x180: 0.439453 MiB transient" \
        "${output_dir}/${name}.log" >/dev/null
    grep -F "Upscaler capabilities: reconstruction=spatial, history=none, output=renderer, reactive-mask=0, sharpening=0, dynamic-resolution=0" \
        "${output_dir}/${name}.log" >/dev/null
    grep -F "Upscaler input conventions: color=linear-hdr-before-exposure, depth=vulkan-device-0-to-1, motion=current-minus-previous-uv, jitter=render-pixel-offsets, exposure=linear-multiplier, motion-excludes-jitter=1" \
        "${output_dir}/${name}.log" >/dev/null
done
grep -F "Upscaler capabilities: reconstruction=identity, history=none, output=renderer, reactive-mask=0, sharpening=0, dynamic-resolution=0" \
    "${output_dir}/implicit.log" >/dev/null
grep -F "Upscaler resize: spatial, quality performance, render 640x360 -> output 1279x719; reconstruction history reset" \
    "${output_dir}/resize.log" >/dev/null
grep -F "Upscaler resize: spatial, quality performance, render 320x180 -> output 640x360; reconstruction history reset" \
    "${output_dir}/resize.log" >/dev/null
for name in animated_path_native animated_path_spatial; do
    grep -F "Recorded 8 synchronized in-place TLAS update(s), including 8 with scripted changing transforms" \
        "${output_dir}/${name}.log" >/dev/null
    grep -F "Path tracing accumulated 1 sample(s) since the last reset across 7 reset(s)" \
        "${output_dir}/${name}.log" >/dev/null
done

comparison="$("${metrics}" compare "${output_dir}/implicit.png" "${output_dir}/explicit.png")"
rmse="$(sed -n 's/.*rmse=\([^ ]*\).*/\1/p' <<<"${comparison}")"
if [[ -z "${rmse}" ]]; then
    echo "Could not parse native identity RMSE" >&2
    exit 1
fi
awk -v value="${rmse}" 'BEGIN { if (value != value || value > 0.000001) exit 1 }'

spatial_comparison="$("${metrics}" compare "${output_dir}/spatial_performance_a.png" "${output_dir}/spatial_performance_b.png")"
spatial_repeat_rmse="$(sed -n 's/.*rmse=\([^ ]*\).*/\1/p' <<<"${spatial_comparison}")"
native_spatial_comparison="$("${metrics}" compare "${output_dir}/implicit.png" "${output_dir}/spatial_performance_a.png")"
native_spatial_rmse="$(sed -n 's/.*rmse=\([^ ]*\).*/\1/p' <<<"${native_spatial_comparison}")"
spatial_stats="$("${metrics}" stats "${output_dir}/spatial_performance_a.png")"
if [[ -z "${spatial_repeat_rmse}" || -z "${native_spatial_rmse}" ]] ||
   [[ "${spatial_stats}" != width=320\ height=180\ * ]]; then
    echo "Could not validate spatial output dimensions or RMSE" >&2
    exit 1
fi
awk -v value="${spatial_repeat_rmse}" 'BEGIN { if (value != value || value > 0.000001) exit 1 }'
awk -v value="${native_spatial_rmse}" 'BEGIN { if (value != value || value < 0.001) exit 1 }'

timing_value() {
    local report="$1"
    sed -n 's/^upscaling: \([^ ]*\) ms.*/\1/p' "${report}"
}

raster_ms="$(timing_value "${output_dir}/raster_timing.txt")"
spatial_quality_ms="$(timing_value "${output_dir}/spatial_quality_timing.txt")"
spatial_balanced_ms="$(timing_value "${output_dir}/spatial_balanced_timing.txt")"
spatial_performance_ms="$(timing_value "${output_dir}/spatial_performance_timing.txt")"
spatial_custom_ms="$(timing_value "${output_dir}/spatial_custom_timing.txt")"
native_path_upscaling_ms="$(timing_value "${output_dir}/native_path_timing.txt")"
spatial_path_upscaling_ms="$(timing_value "${output_dir}/spatial_path_timing.txt")"
native_path_ms="$(sed -n 's/^path tracing: \([^ ]*\) ms.*/\1/p' "${output_dir}/native_path_timing.txt")"
spatial_path_ms="$(sed -n 's/^path tracing: \([^ ]*\) ms.*/\1/p' "${output_dir}/spatial_path_timing.txt")"
if [[ -z "${raster_ms}" || -z "${spatial_quality_ms}" ||
      -z "${spatial_balanced_ms}" || -z "${spatial_performance_ms}" ||
      -z "${spatial_custom_ms}" ||
      -z "${native_path_upscaling_ms}" || -z "${spatial_path_upscaling_ms}" ||
      -z "${native_path_ms}" || -z "${spatial_path_ms}" ]]; then
    echo "Could not parse upscaling or path timing" >&2
    exit 1
fi
awk -v value="${raster_ms}" -v maximum="${maximum_native_ms}" \
    'BEGIN { if (value != value || value > maximum) exit 1 }'
awk -v value="${native_path_upscaling_ms}" -v maximum="${maximum_native_ms}" \
    'BEGIN { if (value != value || value > maximum) exit 1 }'
for value in "${spatial_quality_ms}" "${spatial_balanced_ms}" \
    "${spatial_performance_ms}" "${spatial_custom_ms}" "${spatial_path_upscaling_ms}"; do
    awk -v value="${value}" -v maximum="${maximum_spatial_ms}" \
        'BEGIN { if (value != value || value > maximum) exit 1 }'
done
awk -v native="${native_path_ms}" -v spatial="${spatial_path_ms}" \
    'BEGIN { if (native != native || spatial != spatial || spatial >= native) exit 1 }'

{
    echo "Native and spatial upscaling validation passed:"
    echo "  implicit/explicit native RMSE: ${rmse}"
    echo "  spatial repeat RMSE: ${spatial_repeat_rmse}"
    echo "  native/spatial checkerboard RMSE: ${native_spatial_rmse}"
    echo "  named render extents at 320x180 output: quality 213x120, balanced 188x106, performance 160x90"
    echo "  custom render extent at 0.75 scale: 240x135"
    echo "  raster timestamp boundary: ${raster_ms} ms (maximum ${maximum_native_ms} ms)"
    echo "  spatial quality dispatch: ${spatial_quality_ms} ms (maximum ${maximum_spatial_ms} ms)"
    echo "  spatial balanced dispatch: ${spatial_balanced_ms} ms (maximum ${maximum_spatial_ms} ms)"
    echo "  spatial performance dispatch: ${spatial_performance_ms} ms (maximum ${maximum_spatial_ms} ms)"
    echo "  spatial custom dispatch: ${spatial_custom_ms} ms (maximum ${maximum_spatial_ms} ms)"
    echo "  native animated-path boundary: ${native_path_upscaling_ms} ms"
    echo "  spatial animated-path dispatch: ${spatial_path_upscaling_ms} ms"
    echo "  native/spatial path tracing: ${native_path_ms}/${spatial_path_ms} ms"
    echo "  odd 1279x719 output -> 640x360 render and 640x360 restore -> 320x180 render: clean"
    echo "  native and spatial animated path frame-slot updates: 8 each, resets: 7 each"
} | tee "${output_dir}/summary.txt"
