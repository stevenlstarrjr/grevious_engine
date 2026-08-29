#!/usr/bin/env bash
set -euo pipefail

renderer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${renderer_root}/build/debug}"
output_dir="${2:-${renderer_root}/captures/current/ray_query_shadows}"
genesis_render="${build_dir}/bin/genesis_render"
metrics="${build_dir}/post_image_metrics"

cmake --build "${build_dir}" -j4
mkdir -p "${output_dir}"

capture() {
    local name="$1"
    local ray_mode="$2"
    local shadow_mode="$3"
    shift 3
    env SDL_VIDEODRIVER=wayland "${genesis_render}" \
        --validation required \
        --demo directional-shadow \
        --ray-tracing "${ray_mode}" \
        --shadows "${shadow_mode}" \
        --taa off \
        --bloom off \
        --resolution 1280 720 \
        --frames 8 \
        --capture "${output_dir}/${name}.png" \
        --capture-frame 8 \
        "$@" 2>&1 | tee "${output_dir}/${name}.log"
}

capture_alpha() {
    local name="$1"
    local ray_mode="$2"
    local shadow_mode="$3"
    env SDL_VIDEODRIVER=wayland "${genesis_render}" \
        --validation required \
        --demo alpha-shadow \
        --ray-tracing "${ray_mode}" \
        --shadows "${shadow_mode}" \
        --camera-script rotate \
        --taa off \
        --bloom off \
        --resolution 1280 720 \
        --frames 8 \
        --capture "${output_dir}/${name}.png" \
        --capture-frame 8 2>&1 | tee "${output_dir}/${name}.log"
}

field() {
    local line="$1"
    local key="$2"
    sed -n "s/.*${key}=\([^ ]*\).*/\1/p" <<<"${line}"
}

pixel_luminance() {
    local image="$1"
    local x="$2"
    local y="$3"
    local sample
    sample="$("${metrics}" sample "${image}" "${x}" "${y}")"
    field "${sample}" luminance
}

capture automatic auto on
capture automatic_repeat auto on
capture required required on
capture unshadowed auto off
capture raster off on

for log in automatic automatic_repeat required unshadowed; do
    grep -F "Shadow backend: ray query" "${output_dir}/${log}.log" >/dev/null
    grep -F "Recorded 8 synchronized in-place TLAS update(s)" \
        "${output_dir}/${log}.log" >/dev/null
done
grep -F "Shadow backend: raster" "${output_dir}/raster.log" >/dev/null
grep -F "directional shadow culling skipped" "${output_dir}/automatic.log" >/dev/null
grep -F "directional shadow map skipped" "${output_dir}/automatic.log" >/dev/null

if grep -E '\[(WARN|ERROR)\]|VUID-|Validation Error' \
    "${output_dir}/automatic.log" \
    "${output_dir}/automatic_repeat.log" \
    "${output_dir}/required.log" \
    "${output_dir}/unshadowed.log" \
    "${output_dir}/raster.log"; then
    echo "Ray-query validation logs contain a warning or error" >&2
    exit 1
fi

automatic_stats="$("${metrics}" stats "${output_dir}/automatic.png")"
unshadowed_stats="$("${metrics}" stats "${output_dir}/unshadowed.png")"
repeat_compare="$("${metrics}" compare "${output_dir}/automatic.png" \
                                      "${output_dir}/automatic_repeat.png")"
required_compare="$("${metrics}" compare "${output_dir}/automatic.png" \
                                        "${output_dir}/required.png")"
toggle_compare="$("${metrics}" compare "${output_dir}/unshadowed.png" \
                                      "${output_dir}/automatic.png")"
raster_compare="$("${metrics}" compare "${output_dir}/raster.png" \
                                      "${output_dir}/automatic.png")"

automatic_mean="$(field "${automatic_stats}" mean)"
unshadowed_mean="$(field "${unshadowed_stats}" mean)"
repeat_rmse="$(field "${repeat_compare}" rmse)"
required_rmse="$(field "${required_compare}" rmse)"
toggle_rmse="$(field "${toggle_compare}" rmse)"
raster_rmse="$(field "${raster_compare}" rmse)"

[[ "${automatic_stats}" == *"finite=1"* && "${unshadowed_stats}" == *"finite=1"* ]]
awk -v actual="${repeat_rmse}" 'BEGIN { if (actual > 0.000001) exit 1 }'
awk -v actual="${required_rmse}" 'BEGIN { if (actual > 0.000001) exit 1 }'
awk -v actual="${toggle_rmse}" 'BEGIN { if (actual < 0.02) exit 1 }'
awk -v shadowed="${automatic_mean}" -v unshadowed="${unshadowed_mean}" \
    'BEGIN { if (unshadowed - shadowed < 0.005) exit 1 }'

capture_alpha alpha_automatic auto on
capture_alpha alpha_automatic_repeat auto on
capture_alpha alpha_required required on
capture_alpha alpha_unshadowed auto off
capture_alpha alpha_raster off on

for log in alpha_automatic alpha_automatic_repeat alpha_required alpha_unshadowed; do
    grep -F "Shadow backend: ray query" "${output_dir}/${log}.log" >/dev/null
    grep -F "Recorded 8 synchronized in-place TLAS update(s)" \
        "${output_dir}/${log}.log" >/dev/null
done
grep -F "Shadow backend: raster" "${output_dir}/alpha_raster.log" >/dev/null
grep -F "directional shadow culling skipped" "${output_dir}/alpha_automatic.log" >/dev/null
grep -F "directional shadow map skipped" "${output_dir}/alpha_automatic.log" >/dev/null
if grep -E '\[(WARN|ERROR)\]|VUID-|Validation Error' \
    "${output_dir}/alpha_automatic.log" \
    "${output_dir}/alpha_automatic_repeat.log" \
    "${output_dir}/alpha_required.log" \
    "${output_dir}/alpha_unshadowed.log" \
    "${output_dir}/alpha_raster.log"; then
    echo "Alpha ray-query validation logs contain a warning or error" >&2
    exit 1
fi

alpha_repeat_compare="$("${metrics}" compare "${output_dir}/alpha_automatic.png" \
    "${output_dir}/alpha_automatic_repeat.png")"
alpha_required_compare="$("${metrics}" compare "${output_dir}/alpha_automatic.png" \
    "${output_dir}/alpha_required.png")"
alpha_toggle_compare="$("${metrics}" compare "${output_dir}/alpha_unshadowed.png" \
    "${output_dir}/alpha_automatic.png")"
alpha_raster_compare="$("${metrics}" compare "${output_dir}/alpha_raster.png" \
    "${output_dir}/alpha_automatic.png")"
alpha_automatic_stats="$("${metrics}" stats "${output_dir}/alpha_automatic.png")"
alpha_unshadowed_stats="$("${metrics}" stats "${output_dir}/alpha_unshadowed.png")"
alpha_repeat_rmse="$(field "${alpha_repeat_compare}" rmse)"
alpha_required_rmse="$(field "${alpha_required_compare}" rmse)"
alpha_toggle_rmse="$(field "${alpha_toggle_compare}" rmse)"
alpha_raster_rmse="$(field "${alpha_raster_compare}" rmse)"

[[ "${alpha_automatic_stats}" == *"finite=1"* && \
   "${alpha_unshadowed_stats}" == *"finite=1"* ]]
awk -v actual="${alpha_repeat_rmse}" 'BEGIN { if (actual > 0.000001) exit 1 }'
awk -v actual="${alpha_required_rmse}" 'BEGIN { if (actual > 0.000001) exit 1 }'
awk -v actual="${alpha_toggle_rmse}" 'BEGIN { if (actual < 0.005) exit 1 }'

opaque_coordinates=("416 380" "435 399")
hole_coordinates=("435 380" "416 399")
for coordinate in "${opaque_coordinates[@]}"; do
    read -r x y <<<"${coordinate}"
    shadowed="$(pixel_luminance "${output_dir}/alpha_automatic.png" "${x}" "${y}")"
    unshadowed="$(pixel_luminance "${output_dir}/alpha_unshadowed.png" "${x}" "${y}")"
    awk -v shadowed="${shadowed}" -v unshadowed="${unshadowed}" \
        'BEGIN { if (unshadowed - shadowed < 0.05) exit 1 }'
done
for coordinate in "${hole_coordinates[@]}"; do
    read -r x y <<<"${coordinate}"
    shadowed="$(pixel_luminance "${output_dir}/alpha_automatic.png" "${x}" "${y}")"
    unshadowed="$(pixel_luminance "${output_dir}/alpha_unshadowed.png" "${x}" "${y}")"
    awk -v shadowed="${shadowed}" -v unshadowed="${unshadowed}" \
        'BEGIN { difference = unshadowed - shadowed; if (difference < 0) difference = -difference; if (difference > 0.01) exit 1 }'
done

echo "Ray-query shadow validation passed:"
echo "  repeat RMSE: ${repeat_rmse} (maximum 0.000001)"
echo "  automatic/required RMSE: ${required_rmse} (maximum 0.000001)"
echo "  shadows on/off RMSE: ${toggle_rmse} (minimum 0.02)"
echo "  mean luminance: ray shadows ${automatic_mean}, shadows off ${unshadowed_mean}"
echo "  ray-query/raster reference RMSE: ${raster_rmse}"
echo "  alpha repeat RMSE: ${alpha_repeat_rmse} (maximum 0.000001)"
echo "  alpha automatic/required RMSE: ${alpha_required_rmse} (maximum 0.000001)"
echo "  alpha shadows on/off RMSE: ${alpha_toggle_rmse} (minimum 0.005)"
echo "  alpha ray-query/raster reference RMSE: ${alpha_raster_rmse}"
echo "  alpha checker opaque cells darken while transparent holes stay unshadowed"
