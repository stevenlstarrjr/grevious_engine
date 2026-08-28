#!/usr/bin/env bash
set -euo pipefail

renderer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${renderer_root}/build/debug}"
output_dir="${2:-${renderer_root}/captures/current/alpha_shadows}"
renderer_demo="${build_dir}/bin/renderer_demo"

cmake --build "${build_dir}" -j4
mkdir -p "${output_dir}"

capture() {
    local name="$1"
    local shadows="$2"
    env SDL_VIDEODRIVER=wayland "${renderer_demo}" \
        --validation required \
        --demo alpha-shadow \
        --shadows "${shadows}" \
        --taa off \
        --bloom off \
        --resolution 1280 720 \
        --frames 8 \
        --capture "${output_dir}/${name}.png" \
        --capture-frame 8
}

normalized_rmse() {
    local reference="$1"
    local candidate="$2"
    local metric
    metric="$(magick compare -metric RMSE "${reference}" "${candidate}" null: 2>&1)" || true
    sed -n 's/.*(\([^)]*\)).*/\1/p' <<<"${metric}"
}

pixel_luminance() {
    local image="$1"
    local x="$2"
    local y="$3"
    magick "${image}" -crop "1x1+${x}+${y}" -colorspace Gray -format '%[fx:mean]' info:
}

capture shadow_on on
capture shadow_on_repeat on
capture shadow_off off

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

opaque_coordinates=("602 400" "618 416")
hole_coordinates=("618 400" "602 416")

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
