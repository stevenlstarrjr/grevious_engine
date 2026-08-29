#!/usr/bin/env bash
set -euo pipefail

renderer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${renderer_root}/build/debug}"
output_dir="${2:-${renderer_root}/captures/current/alpha_shadows}"
genesis_render="${build_dir}/bin/genesis_render"
metrics="${build_dir}/post_image_metrics"

cmake --build "${build_dir}" -j4
mkdir -p "${output_dir}"

capture() {
    local name="$1"
    local shadows="$2"
    env SDL_VIDEODRIVER=wayland "${genesis_render}" \
        --validation required \
        --demo alpha-shadow \
        --ray-tracing off \
        --shadows "${shadows}" \
        --camera-script rotate \
        --taa off \
        --bloom off \
        --resolution 1280 720 \
        --frames 8 \
        --capture "${output_dir}/${name}.png" \
        --capture-frame 8 2>&1 | tee "${output_dir}/${name}.log"
}

normalized_rmse() {
    local reference="$1"
    local candidate="$2"
    local comparison
    comparison="$("${metrics}" compare "${reference}" "${candidate}")"
    sed -n 's/.*rmse=\([^ ]*\).*/\1/p' <<<"${comparison}"
}

pixel_luminance() {
    local image="$1"
    local x="$2"
    local y="$3"
    local sample
    sample="$("${metrics}" sample "${image}" "${x}" "${y}")"
    sed -n 's/.*luminance=\([^ ]*\).*/\1/p' <<<"${sample}"
}

capture shadow_on on
capture shadow_on_repeat on
capture shadow_off off

for log in shadow_on shadow_on_repeat shadow_off; do
    grep -F "Shadow backend: raster (ray tracing was explicitly disabled)" \
        "${output_dir}/${log}.log" >/dev/null
done
if grep -E '\[(WARN|ERROR)\]|VUID-|Validation Error' \
    "${output_dir}/shadow_on.log" \
    "${output_dir}/shadow_on_repeat.log" \
    "${output_dir}/shadow_off.log"; then
    echo "Alpha-shadow validation logs contain a warning or error" >&2
    exit 1
fi

repeat_rmse="$(normalized_rmse "${output_dir}/shadow_on.png" \
    "${output_dir}/shadow_on_repeat.png")"
toggle_rmse="$(normalized_rmse "${output_dir}/shadow_off.png" \
    "${output_dir}/shadow_on.png")"

if [[ -z "${repeat_rmse}" || -z "${toggle_rmse}" ]]; then
    echo "Could not parse alpha-shadow RMSE metrics" >&2
    exit 1
fi
awk -v actual="${repeat_rmse}" 'BEGIN { if (actual > 0.000001) exit 1 }'
awk -v actual="${toggle_rmse}" 'BEGIN { if (actual < 0.005) exit 1 }'

opaque_coordinates=("416 380" "435 399")
hole_coordinates=("435 380" "416 399")

for coordinate in "${opaque_coordinates[@]}"; do
    read -r x y <<<"${coordinate}"
    shadowed="$(pixel_luminance "${output_dir}/shadow_on.png" "${x}" "${y}")"
    unshadowed="$(pixel_luminance "${output_dir}/shadow_off.png" "${x}" "${y}")"
    awk -v shadowed="${shadowed}" -v unshadowed="${unshadowed}" \
        'BEGIN { if (unshadowed - shadowed < 0.05) exit 1 }'
done

for coordinate in "${hole_coordinates[@]}"; do
    read -r x y <<<"${coordinate}"
    shadowed="$(pixel_luminance "${output_dir}/shadow_on.png" "${x}" "${y}")"
    unshadowed="$(pixel_luminance "${output_dir}/shadow_off.png" "${x}" "${y}")"
    awk -v shadowed="${shadowed}" -v unshadowed="${unshadowed}" \
        'BEGIN { difference = unshadowed - shadowed; if (difference < 0) difference = -difference; if (difference > 0.01) exit 1 }'
done

echo "Alpha-cutout shadow validation passed:"
echo "  repeat RMSE: ${repeat_rmse} (maximum 0.000001)"
echo "  shadow on/off RMSE: ${toggle_rmse} (minimum 0.005)"
echo "  opaque checker cells darken by at least 0.05 luminance"
echo "  transparent checker holes remain within 0.01 luminance of shadows-off"
