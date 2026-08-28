#!/usr/bin/env bash
set -euo pipefail

renderer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd -- "${renderer_root}/.." && pwd)"
build_dir="${1:-${renderer_root}/build/release}"
output_dir="${2:-${renderer_root}/captures/current}"
baseline="${renderer_root}/captures/regression/damaged_helmet_hangar.png"
capture="${output_dir}/damaged_helmet_hangar.png"
maximum_rmse="${PBR_MAX_RMSE:-0.025}"

cmake --build "${build_dir}" -j4
mkdir -p "${output_dir}"
env SDL_VIDEODRIVER=wayland "${build_dir}/bin/renderer_demo" \
    --scene "${workspace_root}/glTF-Sample-Assets/Models/DamagedHelmet/glTF/DamagedHelmet.gltf" \
    --environment "${workspace_root}/hdri/hangar_interior_4k.hdr" \
    --environment-rotation 0 \
    --exposure-ev -1 \
    --taa off \
    --shadows off \
    --frames 8 \
    --capture "${capture}" \
    --capture-frame 8 \
    --validation required

if [[ ! -f "${baseline}" ]]; then
    echo "Capture written to ${capture}; no baseline exists at ${baseline}" >&2
    exit 0
fi

comparison_status=0
metric="$(magick compare -metric RMSE "${baseline}" "${capture}" null: 2>&1)" || \
    comparison_status=$?
if (( comparison_status > 1 )); then
    echo "ImageMagick could not compare the PBR capture: ${metric}" >&2
    exit 1
fi
normalized_rmse="$(sed -n 's/.*(\([^)]*\)).*/\1/p' <<<"${metric}")"
if [[ -z "${normalized_rmse}" ]]; then
    echo "Could not parse ImageMagick RMSE output: ${metric}" >&2
    exit 1
fi
awk -v actual="${normalized_rmse}" -v maximum="${maximum_rmse}" \
    'BEGIN { if (actual > maximum) exit 1 }' || {
        echo "PBR regression failed: normalized RMSE ${normalized_rmse} > ${maximum_rmse}" >&2
        exit 1
    }
echo "PBR regression passed: normalized RMSE ${normalized_rmse} <= ${maximum_rmse}"
