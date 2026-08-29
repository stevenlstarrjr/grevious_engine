#!/usr/bin/env bash
set -euo pipefail

renderer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${renderer_root}/build/debug}"
output_dir="${2:-${renderer_root}/captures/current/tlas_updates}"
genesis_render="${build_dir}/bin/genesis_render"
metrics="${build_dir}/post_image_metrics"

cmake --build "${build_dir}" -j4
mkdir -p "${output_dir}"

capture() {
    local name="$1"
    shift
    env SDL_VIDEODRIVER=wayland "${genesis_render}" \
        --validation required \
        --demo directional-shadow \
        --shadows off \
        --taa off \
        --bloom off \
        --resolution 1280 720 \
        --frames 8 \
        --capture "${output_dir}/${name}.png" \
        --capture-frame 8 \
        "$@" 2>&1 | tee "${output_dir}/${name}.log"
}

normalized_rmse() {
    local reference="$1"
    local candidate="$2"
    local comparison
    comparison="$("${metrics}" compare "${reference}" "${candidate}")"
    sed -n 's/.*rmse=\([^ ]*\).*/\1/p' <<<"${comparison}"
}

capture static --ray-tracing auto
capture changing --ray-tracing auto --rt-transform-script oscillate
capture off --ray-tracing off

grep -F "Recorded 8 synchronized in-place TLAS update(s), including 0 with scripted changing transforms" \
    "${output_dir}/static.log" >/dev/null
grep -F "Recorded 8 synchronized in-place TLAS update(s), including 8 with scripted changing transforms" \
    "${output_dir}/changing.log" >/dev/null
grep -F "RT transform script: deterministic TLAS-only oscillation enabled" \
    "${output_dir}/changing.log" >/dev/null
grep -F "Shadow backend: ray query" "${output_dir}/static.log" >/dev/null
grep -F "Shadow backend: ray query" "${output_dir}/changing.log" >/dev/null
grep -F "Ray-tracing acceleration-structure foundation: skipped" \
    "${output_dir}/off.log" >/dev/null
if grep -E '\[(WARN|ERROR)\]|VUID-|Validation Error' \
    "${output_dir}/static.log" "${output_dir}/changing.log" "${output_dir}/off.log"; then
    echo "TLAS-update validation logs contain a warning or error" >&2
    exit 1
fi

static_rmse="$(normalized_rmse "${output_dir}/off.png" "${output_dir}/static.png")"
changing_rmse="$(normalized_rmse "${output_dir}/off.png" "${output_dir}/changing.png")"
if [[ -z "${static_rmse}" || -z "${changing_rmse}" ]]; then
    echo "Could not parse disabled-shadow TLAS-update RMSE" >&2
    exit 1
fi
awk -v actual="${static_rmse}" 'BEGIN { if (actual > 0.000001) exit 1 }'
awk -v actual="${changing_rmse}" 'BEGIN { if (actual > 0.000001) exit 1 }'

echo "Per-frame TLAS update validation passed:"
echo "  static update/RT-off image RMSE with shadows disabled: ${static_rmse} (maximum 0.000001)"
echo "  changing update/RT-off image RMSE with shadows disabled: ${changing_rmse} (maximum 0.000001)"
echo "  static and changing runs each recorded 8 synchronized TLAS updates"
