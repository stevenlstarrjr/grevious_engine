#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-build/debug}"
output_dir="${2:-captures/current/occlusion_camera_motion}"
renderer_demo="${build_dir}/bin/renderer_demo"

mkdir -p "${output_dir}"

capture() {
    local name="$1"
    local camera_script="$2"
    local frame="$3"
    local occlusion="$4"
    env SDL_VIDEODRIVER=wayland "${renderer_demo}" \
        --validation required \
        --demo directional-shadow \
        --taa off \
        --camera-script "${camera_script}" \
        --occlusion-culling "${occlusion}" \
        --frames "${frame}" \
        --capture "${output_dir}/${name}.png" \
        --capture-frame "${frame}"
}

compare_exact() {
    local reference="$1"
    local candidate="$2"
    local label="$3"
    local metric
    metric="$(magick compare -metric RMSE "${reference}" "${candidate}" null: 2>&1)" || true
    local normalized
    normalized="$(sed -n 's/.*(\([^)]*\)).*/\1/p' <<<"${metric}")"
    if [[ -z "${normalized}" ]]; then
        echo "Could not parse ${label} RMSE: ${metric}" >&2
        exit 1
    fi
    awk -v actual="${normalized}" 'BEGIN { if (actual > 0.000001) exit 1 }'
    echo "  ${label}: RMSE ${normalized} (maximum 0.000001)"
}

for frame in 8 16 24; do
    capture "rotate_f${frame}_reference" rotate "${frame}" off
    capture "rotate_f${frame}_hiz" rotate "${frame}" on
done

for frame in 24 36; do
    capture "approach_f${frame}_reference" approach "${frame}" off
    capture "approach_f${frame}_hiz" approach "${frame}" on
done

capture teleport_f3_reference teleport 3 off
capture teleport_f3_hiz teleport 3 on

echo "Occlusion camera-motion comparisons:"
for frame in 8 16 24; do
    compare_exact "${output_dir}/rotate_f${frame}_reference.png" \
        "${output_dir}/rotate_f${frame}_hiz.png" "rotating camera frame ${frame}"
done
for frame in 24 36; do
    compare_exact "${output_dir}/approach_f${frame}_reference.png" \
        "${output_dir}/approach_f${frame}_hiz.png" "approaching camera frame ${frame}"
done
compare_exact "${output_dir}/teleport_f3_reference.png" \
    "${output_dir}/teleport_f3_hiz.png" "camera teleport frame 3"

echo "Occlusion camera-motion validation passed"
