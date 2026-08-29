#!/usr/bin/env bash
set -euo pipefail

renderer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${renderer_root}/build/release}"
output_dir="${2:-${renderer_root}/captures/current/aov_export}"
maximum_readback_ms="${PATH_AOV_READBACK_MAX_MS:-4.0}"
frame_number="${PATH_AOV_EXPORT_FRAME:-8}"
width="${PATH_AOV_EXPORT_WIDTH:-640}"
height="${PATH_AOV_EXPORT_HEIGHT:-360}"
genesis_render="${build_dir}/bin/genesis_render"
metrics="${build_dir}/aov_export_metrics"
frame_directory="${output_dir}/frames/$(printf 'frame_%08d' "${frame_number}")"

cmake --build "${build_dir}" --target genesis_render aov_export_metrics -j4
mkdir -p "${output_dir}"

env SDL_VIDEODRIVER=wayland "${genesis_render}" \
    --validation required \
    --render-mode path-traced \
    --ray-tracing required \
    --demo directional-shadow \
    --taa off \
    --bloom off \
    --occlusion-culling off \
    --camera-input off \
    --resolution "${width}" "${height}" \
    --frames "${frame_number}" \
    --capture "${output_dir}/preview.png" \
    --capture-frame "${frame_number}" \
    --aov-export "${output_dir}/frames" \
    --aov-frame "${frame_number}" \
    --timing-output "${output_dir}/timing.txt" \
    2>&1 | tee "${output_dir}/validation.log"

"${metrics}" "${frame_directory}" "${width}" "${height}" "${frame_number}" \
    "${frame_number}" |
    tee "${output_dir}/metrics.txt"

grep -F "Exported path AOV frame ${frame_number} with ${frame_number} accumulated sample(s)" \
    "${output_dir}/validation.log" >/dev/null
grep -F "Captured frame ${frame_number}" "${output_dir}/validation.log" >/dev/null
grep -F "path AOV readback:" "${output_dir}/timing.txt" >/dev/null

if grep -E '\[(WARN|ERROR)\]|VUID-|Validation Error' "${output_dir}/validation.log"; then
    echo "AOV export validation log contains a warning or error" >&2
    exit 1
fi

readback_ms="$(sed -n 's/^path AOV readback: \([^ ]*\) ms.*/\1/p' \
    "${output_dir}/timing.txt")"
if [[ -z "${readback_ms}" ]] ||
   ! awk -v actual="${readback_ms}" -v maximum="${maximum_readback_ms}" \
       'BEGIN { if (actual != actual || actual <= 0.0 || actual > maximum) exit 1 }'; then
    echo "Path AOV readback GPU time ${readback_ms:-missing} ms exceeds ${maximum_readback_ms} ms" >&2
    exit 1
fi

{
    echo "Path AOV export validation passed:"
    echo "  frame: ${frame_number} (${width}x${height}, ${frame_number} accumulated samples)"
    echo "  renderer-neutral planes: 15 float32 PFM files plus manifest"
    echo "  simultaneous tone-mapped PNG capture: passed"
    echo "  binary plane contract and decomposition checks: passed"
    echo "  RTX 2060 SUPER AOV GPU readback: ${readback_ms} ms (maximum ${maximum_readback_ms} ms)"
} | tee "${output_dir}/summary.txt"
