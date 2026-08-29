#!/usr/bin/env bash
set -euo pipefail

renderer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${renderer_root}/build/release}"
output_dir="${2:-${renderer_root}/captures/current/ray_query_timings}"
frame_count="${RAY_QUERY_TIMING_FRAMES:-240}"
genesis_render="${build_dir}/bin/genesis_render"

cmake --build "${build_dir}" -j4
mkdir -p "${output_dir}"

run_timing() {
    local name="$1"
    local shadows="$2"
    env SDL_VIDEODRIVER=wayland "${genesis_render}" \
        --validation required \
        --demo directional-shadow \
        --ray-tracing required \
        --shadows "${shadows}" \
        --taa off \
        --bloom off \
        --resolution 1280 720 \
        --frames "${frame_count}" \
        --timing-output "${output_dir}/${name}.txt" \
        2>&1 | tee "${output_dir}/${name}.log"
}

timing_value() {
    local report="$1"
    local pass="$2"
    sed -n "s/^${pass}: \([^ ]*\) ms.*/\1/p" "${report}"
}

run_timing shadows_off off
run_timing shadows_on on

for name in shadows_off shadows_on; do
    grep -F "Directional shadow backend: ray query" "${output_dir}/${name}.txt" >/dev/null
    grep -F "Recorded ${frame_count} synchronized in-place TLAS update(s)" \
        "${output_dir}/${name}.log" >/dev/null
    grep -F "TLAS update:" "${output_dir}/${name}.txt" >/dev/null
    grep -F "PBR geometry draw:" "${output_dir}/${name}.txt" >/dev/null
done
grep -F "Ray-query shadow traversal: disabled" \
    "${output_dir}/shadows_off.txt" >/dev/null
grep -F "Ray-query shadow traversal: enabled" \
    "${output_dir}/shadows_on.txt" >/dev/null

if grep -E '\[(WARN|ERROR)\]|VUID-|Validation Error' \
    "${output_dir}/shadows_off.log" "${output_dir}/shadows_on.log"; then
    echo "Ray-query timing validation logs contain a warning or error" >&2
    exit 1
fi

off_tlas_ms="$(timing_value "${output_dir}/shadows_off.txt" "TLAS update")"
on_tlas_ms="$(timing_value "${output_dir}/shadows_on.txt" "TLAS update")"
off_pbr_ms="$(timing_value "${output_dir}/shadows_off.txt" "PBR geometry draw")"
on_pbr_ms="$(timing_value "${output_dir}/shadows_on.txt" "PBR geometry draw")"

for measurement in "${off_tlas_ms}" "${on_tlas_ms}" "${off_pbr_ms}" "${on_pbr_ms}"; do
    if [[ -z "${measurement}" ]] ||
       ! awk -v actual="${measurement}" \
           'BEGIN { if (actual != actual || actual <= 0.0 || actual >= 100.0) exit 1 }'; then
        echo "Invalid GPU timing measurement: ${measurement:-missing}" >&2
        exit 1
    fi
done

ray_query_ms="$(awk -v enabled="${on_pbr_ms}" -v disabled="${off_pbr_ms}" \
    'BEGIN { printf "%.9f", enabled - disabled }')"
ray_query_percent="$(awk -v delta="${ray_query_ms}" -v disabled="${off_pbr_ms}" \
    'BEGIN { printf "%.3f", 100.0 * delta / disabled }')"
awk -v delta="${ray_query_ms}" 'BEGIN { if (delta <= 0.0) exit 1 }'

{
    echo "Ray-query GPU timing validation passed:"
    echo "  frames per matched run: ${frame_count}"
    echo "  TLAS update, shadows off: ${off_tlas_ms} ms"
    echo "  TLAS update, shadows on: ${on_tlas_ms} ms"
    echo "  PBR geometry, traversal disabled: ${off_pbr_ms} ms"
    echo "  PBR geometry, traversal enabled: ${on_pbr_ms} ms"
    echo "  estimated ray-query traversal cost: ${ray_query_ms} ms (${ray_query_percent}%)"
} | tee "${output_dir}/summary.txt"
