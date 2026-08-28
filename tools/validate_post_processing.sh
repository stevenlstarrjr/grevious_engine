#!/usr/bin/env bash
set -euo pipefail

renderer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd -- "${renderer_root}/.." && pwd)"
build_dir="${1:-${renderer_root}/build/debug}"
output_dir="${2:-${renderer_root}/captures/current/post_processing}"
renderer="${build_dir}/bin/renderer_demo"
metrics="${build_dir}/post_image_metrics"
report="${output_dir}/certification_results.txt"

env CCACHE_DISABLE=1 cmake --build "${build_dir}" -j4
mkdir -p "${output_dir}"
: >"${report}"

run_renderer() {
    local name="$1"
    shift
    if ! env SDL_VIDEODRIVER=wayland "${renderer}" --validation required "$@" \
        >"${output_dir}/${name}.log" 2>&1; then
        cat "${output_dir}/${name}.log" >&2
        exit 1
    fi
}

capture() {
    local name="$1"
    local frame="$2"
    shift 2
    run_renderer "${name}" "$@" --frames "${frame}" \
        --capture "${output_dir}/${name}.png" --capture-frame "${frame}"
}

image_stats() {
    "${metrics}" stats "$1"
}

field() {
    local line="$1"
    local key="$2"
    tr ' ' '\n' <<<"${line}" | awk -F= -v key="${key}" '$1 == key { print $2; exit }'
}

require_le() {
    local actual="$1"
    local limit="$2"
    local label="$3"
    awk -v actual="${actual}" -v limit="${limit}" 'BEGIN { exit !(actual <= limit) }' || {
        echo "Post-processing validation failed: ${label}: ${actual} > ${limit}" >&2
        exit 1
    }
}

require_ge() {
    local actual="$1"
    local limit="$2"
    local label="$3"
    awk -v actual="${actual}" -v limit="${limit}" 'BEGIN { exit !(actual >= limit) }' || {
        echo "Post-processing validation failed: ${label}: ${actual} < ${limit}" >&2
        exit 1
    }
}

"${build_dir}/post_processing_test"
"${build_dir}/shader_reflection_test"

# Exact-zero black and dark fields exercise every live bloom stage. The TAA-on
# dark case is held for 16 frames to expose temporal or stale-resource leakage.
capture black 8 --resolution 1279 719 --post-test black --post-value 65504 \
    --taa off --bloom on --debug-view bloom
black_stats="$(image_stats "${output_dir}/black.png")"
require_le "$(field "${black_stats}" max)" 0 "black-scene bloom maximum"

capture dark_taa 16 --resolution 1279 719 --post-test constant --post-value 0.5 \
    --taa on --bloom on --debug-view bloom
dark_stats="$(image_stats "${output_dir}/dark_taa.png")"
require_le "$(field "${dark_stats}" max)" 0 "dark-scene temporal bloom maximum"

# The shader uses threshold 1.2 with a 0.6 soft knee. Values below 0.6 are
# exactly rejected; values around 1.2 must follow a non-decreasing response.
declare -a threshold_values=(0.599 1.199 1.2 1.201)
previous_threshold_energy=-1
for value in "${threshold_values[@]}"; do
    tag="${value//./_}"
    capture "threshold_${tag}" 4 --resolution 320 180 --post-test constant \
        --post-value "${value}" --taa off --bloom on --debug-view bloom
    stats="$(image_stats "${output_dir}/threshold_${tag}.png")"
    energy="$(field "${stats}" total)"
    require_ge "${energy}" "${previous_threshold_energy}" "soft-knee response at ${value}"
    previous_threshold_energy="${energy}"
done
threshold_low_stats="$(image_stats "${output_dir}/threshold_0_599.png")"
require_le "$(field "${threshold_low_stats}" max)" 0 "below-knee bloom maximum"
threshold_at_stats="$(image_stats "${output_dir}/threshold_1_2.png")"
require_ge "$(field "${threshold_at_stats}" max)" 0.01 "soft-knee response at threshold"

# A square, odd-resolution capture gives a unique center texel and therefore
# exact flip/flop/axis symmetry checks without half-pixel ambiguity.
capture impulse 8 --resolution 719 719 --post-test impulse --post-value 128 \
    --post-position 0.5 0.5 --taa off --bloom on --debug-view bloom
impulse_stats="$(image_stats "${output_dir}/impulse.png")"
impulse_symmetry="$("${metrics}" symmetry "${output_dir}/impulse.png")"
require_le "$(field "${impulse_symmetry}" horizontal_rmse)" 0.001 "impulse horizontal symmetry"
require_le "$(field "${impulse_symmetry}" vertical_rmse)" 0.001 "impulse vertical symmetry"
require_le "$(field "${impulse_symmetry}" axis_rmse)" 0.001 "impulse axis symmetry"
require_le "$(awk -v x="$(field "${impulse_stats}" centroid_x)" 'BEGIN { print (x > 359 ? x-359 : 359-x) }')" \
    0.5 "impulse X centroid"
require_le "$(awk -v y="$(field "${impulse_stats}" centroid_y)" 'BEGIN { print (y > 359 ? y-359 : 359-y) }')" \
    0.5 "impulse Y centroid"

# Live HDR energy must be monotonic after extraction/downsample and both blur
# stages. Values are intentionally powers of two.
previous_energy=-1
for value in 1 2 4 8 16; do
    capture "energy_${value}" 4 --resolution 320 180 --post-test constant \
        --post-value "${value}" --taa off --bloom on --debug-view bloom
    stats="$(image_stats "${output_dir}/energy_${value}.png")"
    require_ge "$(field "${stats}" finite)" 1 "finite bloom at ${value}x"
    energy="$(field "${stats}" total)"
    require_ge "${energy}" "${previous_energy}" "bloom energy at ${value}x"
    previous_energy="${energy}"
done

# Exercise every procedural downsample pattern on odd dimensions. Detailed
# footprint and normalization checks live in post_processing_test.
for pattern in small-square horizontal-line vertical-line multiple checkerboard; do
    capture "pattern_${pattern}" 4 --resolution 321 239 --post-test "${pattern}" \
        --post-value 32 --taa off --bloom on --debug-view bloom
    stats="$(image_stats "${output_dir}/pattern_${pattern}.png")"
    require_ge "$(field "${stats}" finite)" 1 "finite ${pattern} result"
done

# Center, edges, and corners run through Vulkan core, synchronization, and best-
# practices validation. Clamp-to-edge behavior is independently reference-tested.
for entry in center:0.5:0.5 left:0:0.5 right:1:0.5 top:0.5:0 bottom:0.5:1 \
             top_left:0:0 top_right:1:0 bottom_left:0:1 bottom_right:1:1; do
    IFS=: read -r name x y <<<"${entry}"
    run_renderer "edge_${name}" --frames 4 --resolution 321 239 --post-test impulse \
        --post-value 128 --post-position "${x}" "${y}" --taa off --bloom on
done

# Bloom-off creates no bloom graph resources or passes. Repeated captures must
# be exact, while enabling bloom for the same source must change the final image.
for run in a b; do
    capture "bloom_off_${run}" 8 --resolution 640 360 --post-test small-square \
        --post-value 32 --taa off --bloom off --debug-view final
done
bloom_off_compare="$("${metrics}" compare "${output_dir}/bloom_off_a.png" \
    "${output_dir}/bloom_off_b.png")"
require_le "$(field "${bloom_off_compare}" rmse)" 0 "bloom-off deterministic RMSE"
capture bloom_on_final 8 --resolution 640 360 --post-test small-square --post-value 32 \
    --taa off --bloom on --debug-view final
bloom_delta="$("${metrics}" compare "${output_dir}/bloom_off_a.png" \
    "${output_dir}/bloom_on_final.png")"
require_ge "$(field "${bloom_delta}" rmse)" 0.001 "bloom enabled/disabled image delta"

# GPU isolation points supplement the complete CPU reference curves for ACES
# fit and exp2(EV). Both series must be finite and monotonic.
previous_tone=-1
for value in 0 0.000001 0.01 0.18 1 4 65504; do
    tag="${value//./_}"
    capture "tone_${tag}" 4 --resolution 320 180 --post-test constant \
        --post-value "${value}" --taa off --bloom off --exposure-ev 0
    stats="$(image_stats "${output_dir}/tone_${tag}.png")"
    require_ge "$(field "${stats}" finite)" 1 "finite tone map at ${value}"
    mean="$(field "${stats}" mean)"
    require_ge "${mean}" "${previous_tone}" "tone-map monotonicity at ${value}"
    previous_tone="${mean}"
done
tone_black="$(image_stats "${output_dir}/tone_0.png")"
require_le "$(field "${tone_black}" max)" 0 "tone-map black offset"

previous_exposure=-1
for ev in -4 -2 0 2 4; do
    tag="${ev/-/minus_}"
    capture "exposure_${tag}" 4 --resolution 320 180 --post-test constant \
        --post-value 0.18 --taa off --bloom off --exposure-ev "${ev}"
    stats="$(image_stats "${output_dir}/exposure_${tag}.png")"
    mean="$(field "${stats}" mean)"
    require_ge "${mean}" "${previous_exposure}" "exposure monotonicity at EV ${ev}"
    previous_exposure="${mean}"
done

# Static convergence is repeatable; the moving source is compared against its
# TAA-off baseline; a teleport reset must match a no-history frame.
for run in a b; do
    capture "static_taa_${run}" 16 --resolution 719 719 --post-test small-square \
        --post-value 32 --taa on --bloom on --debug-view bloom
done
static_compare="$("${metrics}" compare "${output_dir}/static_taa_a.png" \
    "${output_dir}/static_taa_b.png")"
require_le "$(field "${static_compare}" rmse)" 0 "static TAA repeatability"

for taa in off on; do
    capture "moving_taa_${taa}" 8 --resolution 719 719 --post-test moving-square \
        --post-value 32 --taa "${taa}" --bloom on --debug-view bloom
    stats="$(image_stats "${output_dir}/moving_taa_${taa}.png")"
    require_le "$(awk -v x="$(field "${stats}" centroid_x)" 'BEGIN { print (x > 359 ? x-359 : 359-x) }')" \
        1.0 "moving-source centroid with TAA ${taa}"
done
moving_compare="$("${metrics}" compare "${output_dir}/moving_taa_off.png" \
    "${output_dir}/moving_taa_on.png")"
require_le "$(field "${moving_compare}" rmse)" 0.02 "moving-source TAA deviation"

for taa in off on; do
    capture "reset_taa_${taa}" 3 --resolution 719 719 --post-test moving-square \
        --post-value 32 --camera-script teleport --taa "${taa}" --bloom on --debug-view bloom
done
reset_compare="$("${metrics}" compare "${output_dir}/reset_taa_off.png" \
    "${output_dir}/reset_taa_on.png")"
require_le "$(field "${reset_compare}" rmse)" 0.02 "post-reset bloom history"

# A raster/PBR camera move verifies the integrated path, while the scripted
# resize sequence covers 1920x1080, 1280x720, 1279x719, 2560x1440, restoration,
# and additional repeated odd/even reconstruction.
capture camera_motion 12 --resolution 1280 720 --camera-script translate \
    --taa on --bloom on --debug-view bloom
camera_stats="$(image_stats "${output_dir}/camera_motion.png")"
require_ge "$(field "${camera_stats}" finite)" 1 "camera-motion bloom output"

capture gltf_emissive 8 --resolution 1280 720 \
    --scene "${workspace_root}/glTF-Sample-Assets/Models/BoomBox/glTF/BoomBox.gltf" \
    --taa off --bloom on --debug-view bloom
emissive_stats="$(image_stats "${output_dir}/gltf_emissive.png")"
require_ge "$(field "${emissive_stats}" finite)" 1 "glTF emissive bloom output"

run_renderer resize_odd --frames 12 --resolution 1280 720 --post-test impulse \
    --post-value 128 --taa on --bloom on --resize-script odd
run_renderer resize_stress --frames 18 --resolution 1920 1080 --post-test impulse \
    --post-value 128 --taa on --bloom on --resize-script stress

# Record a primary-resolution timestamp baseline in the selected build.
run_renderer performance --frames 32 --resolution 1280 720 --post-test multiple \
    --post-value 32 --taa off --bloom on --timing-output "${output_dir}/gpu_timings.txt"

{
    echo "Post-processing certification image metrics"
    echo "Black: ${black_stats}"
    echo "Dark + TAA: ${dark_stats}"
    echo "Threshold at 1.2: ${threshold_at_stats}"
    echo "Impulse: ${impulse_stats}"
    echo "Impulse symmetry: ${impulse_symmetry}"
    echo "Bloom-off repeat: ${bloom_off_compare}"
    echo "Bloom enabled delta: ${bloom_delta}"
    echo "Static TAA repeat: ${static_compare}"
    echo "Moving TAA delta: ${moving_compare}"
    echo "History reset delta: ${reset_compare}"
    echo
    cat "${output_dir}/gpu_timings.txt"
} >>"${report}"

echo "Post-processing certification passed; report: ${report}"
