#!/usr/bin/env bash
set -euo pipefail

renderer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${renderer_root}/build/release}"
output_dir="${2:-${renderer_root}/captures/current/path_animation}"
genesis_render="${build_dir}/bin/genesis_render"
metrics="${build_dir}/aov_export_metrics"
width="${PATH_ANIMATION_WIDTH:-320}"
height="${PATH_ANIMATION_HEIGHT:-180}"
maximum_tlas_ms="${PATH_ANIMATION_TLAS_MAX_MS:-0.25}"
maximum_history_ms="${PATH_ANIMATION_HISTORY_MAX_MS:-0.10}"
stress_frames="${PATH_ANIMATION_STRESS_FRAMES:-120}"

cmake --build "${build_dir}" --target genesis_render aov_export_metrics -j4
mkdir -p "${output_dir}"

export_frame() {
    local name="$1"
    local frame="$2"
    local samples="$3"
    shift 3
    local root="${output_dir}/${name}"
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
        --frames "${frame}" \
        --aov-export "${root}" \
        --aov-frame "${frame}" \
        "$@" \
        2>&1 | tee "${output_dir}/${name}.log"
    "${metrics}" "${root}/$(printf 'frame_%08d' "${frame}")" \
        "${width}" "${height}" "${frame}" "${samples}" |
        tee "${output_dir}/${name}_metrics.txt"
}

field() {
    local report="$1"
    local key="$2"
    awk -v key="${key}" '{
        for (i = 1; i <= NF; ++i) {
            split($i, field, "=")
            if (field[1] == key) {
                print field[2]
            }
        }
    }' "${report}"
}

export_frame static 3 3
export_frame animated_first 1 1 --rt-transform-script oscillate
export_frame animated 3 1 --rt-transform-script oscillate \
    --timing-output "${output_dir}/animated_timing.txt"
export_frame animated_repeat 3 1 --rt-transform-script oscillate

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
    --frames "${stress_frames}" \
    --rt-transform-script oscillate \
    2>&1 | tee "${output_dir}/stress.log"

for name in static animated_first animated animated_repeat stress; do
    if grep -E '\[(WARN|ERROR)\]|VUID-|Validation Error' "${output_dir}/${name}.log"; then
        echo "Path-animation validation log contains a warning or error: ${name}" >&2
        exit 1
    fi
done

grep -F "deterministic path instance animation enabled" \
    "${output_dir}/animated.log" >/dev/null
grep -F "Recorded 3 synchronized in-place TLAS update(s), including 3 with scripted changing transforms" \
    "${output_dir}/animated.log" >/dev/null
grep -F "Path tracing accumulated 1 sample(s) since the last reset across 2 reset(s)" \
    "${output_dir}/animated.log" >/dev/null
grep -F "Recorded ${stress_frames} synchronized in-place TLAS update(s), including ${stress_frames} with scripted changing transforms" \
    "${output_dir}/stress.log" >/dev/null
grep -F "Path tracing accumulated 1 sample(s) since the last reset across $((stress_frames - 1)) reset(s)" \
    "${output_dir}/stress.log" >/dev/null

static_metrics="${output_dir}/static_metrics.txt"
first_metrics="${output_dir}/animated_first_metrics.txt"
animated_metrics="${output_dir}/animated_metrics.txt"
static_motion_max="$(field "${static_metrics}" motion_max)"
static_history_mean="$(field "${static_metrics}" history_mean)"
first_history_mean="$(field "${first_metrics}" history_mean)"
animated_motion_max="$(field "${animated_metrics}" motion_max)"
animated_moving_pixels="$(field "${animated_metrics}" moving_pixels)"
animated_moving_history="$(field "${animated_metrics}" moving_history_mean)"
animated_rejected_pixels="$(field "${animated_metrics}" rejected_pixels)"
animated_rejected_surface="$(field "${animated_metrics}" rejected_surface_pixels)"
animated_rejected_moving="$(field "${animated_metrics}" rejected_moving_pixels)"

awk -v value="${static_motion_max}" 'BEGIN { if (value != value || value > 0.000001) exit 1 }'
awk -v value="${static_history_mean}" 'BEGIN { if (value < 0.98 || value > 1.001) exit 1 }'
awk -v value="${first_history_mean}" 'BEGIN { if (value < 0.0 || value > 0.000001) exit 1 }'
awk -v value="${animated_motion_max}" 'BEGIN { if (value < 0.0001) exit 1 }'
awk -v value="${animated_moving_pixels}" 'BEGIN { if (value < 32) exit 1 }'
awk -v value="${animated_moving_history}" 'BEGIN { if (value < 0.50 || value > 1.001) exit 1 }'
awk -v value="${animated_rejected_pixels}" 'BEGIN { if (value < 1) exit 1 }'
awk -v value="${animated_rejected_surface}" 'BEGIN { if (value < 1) exit 1 }'
awk -v value="${animated_rejected_moving}" 'BEGIN { if (value < 1) exit 1 }'

animated_directory="${output_dir}/animated/frame_00000003"
repeat_directory="${output_dir}/animated_repeat/frame_00000003"
file_count=0
for source in "${animated_directory}"/*; do
    candidate="${repeat_directory}/$(basename "${source}")"
    if ! cmp -s "${source}" "${candidate}"; then
        echo "Animated path AOV repeat differs: $(basename "${source}")" >&2
        exit 1
    fi
    file_count=$((file_count + 1))
done
if [[ "${file_count}" -ne 16 ]]; then
    echo "Animated path AOV repeat did not contain 15 planes plus one manifest" >&2
    exit 1
fi
if cmp -s "${output_dir}/static/frame_00000003/beauty.pfm" \
    "${animated_directory}/beauty.pfm"; then
    echo "Animated instance did not change the path-traced beauty output" >&2
    exit 1
fi

tlas_ms="$(sed -n 's/^TLAS update: \([^ ]*\) ms.*/\1/p' \
    "${output_dir}/animated_timing.txt")"
history_ms="$(sed -n 's/^path history validation: \([^ ]*\) ms.*/\1/p' \
    "${output_dir}/animated_timing.txt")"
if [[ -z "${tlas_ms}" ]] ||
   ! awk -v actual="${tlas_ms}" -v maximum="${maximum_tlas_ms}" \
       'BEGIN { if (actual != actual || actual <= 0.0 || actual > maximum) exit 1 }'; then
    echo "Animated TLAS update ${tlas_ms:-missing} ms exceeds ${maximum_tlas_ms} ms" >&2
    exit 1
fi
if [[ -z "${history_ms}" ]] ||
   ! awk -v actual="${history_ms}" -v maximum="${maximum_history_ms}" \
       'BEGIN { if (actual != actual || actual <= 0.0 || actual > maximum) exit 1 }'; then
    echo "Animated history validation ${history_ms:-missing} ms exceeds ${maximum_history_ms} ms" >&2
    exit 1
fi

{
    echo "Path-animation validation passed:"
    echo "  static motion maximum: ${static_motion_max}"
    echo "  first-frame history mean: ${first_history_mean}"
    echo "  animated motion maximum: ${animated_motion_max} across ${animated_moving_pixels} pixels"
    echo "  accepted moving-surface history mean: ${animated_moving_history}"
    echo "  rejected pixels: ${animated_rejected_pixels} total, ${animated_rejected_surface} surface, ${animated_rejected_moving} moving"
    echo "  accumulated samples at animated frame 3: 1 after two deliberate resets"
    echo "  deterministic 15-plane AOV repeat: exact"
    echo "  frame-local transform/TLAS reuse stress: ${stress_frames} frames"
    echo "  RTX 2060 SUPER TLAS update: ${tlas_ms} ms (maximum ${maximum_tlas_ms} ms)"
    echo "  RTX 2060 SUPER history validation: ${history_ms} ms (maximum ${maximum_history_ms} ms)"
} | tee "${output_dir}/summary.txt"
