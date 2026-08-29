#!/usr/bin/env bash
set -euo pipefail

renderer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${renderer_root}/build/release}"
output_dir="${2:-${renderer_root}/captures/current/path_tracing}"
maximum_path_ms="${PATH_TRACING_MAX_MS:-5.0}"
maximum_textured_path_ms="${PATH_TRACING_TEXTURED_MAX_MS:-5.0}"
maximum_importance_rmse_ratio="${PATH_TRACING_IMPORTANCE_MAX_RMSE_RATIO:-0.90}"
maximum_emissive_rmse_ratio="${PATH_TRACING_EMISSIVE_MAX_RMSE_RATIO:-0.90}"
maximum_emissive_path_ms="${PATH_TRACING_EMISSIVE_MAX_MS:-1.0}"
maximum_history_ms="${PATH_HISTORY_MAX_MS:-0.25}"
maximum_reconstruction_ms="${PATH_RECONSTRUCTION_MAX_MS:-0.25}"
maximum_reconstructed_motion_ratio="${PATH_RECONSTRUCTION_MAX_MOTION_RATIO:-0.90}"
maximum_denoising_ms="${PATH_DENOISING_MAX_MS:-0.50}"
maximum_spatial_reference_ratio="${PATH_DENOISING_MAX_REFERENCE_RATIO:-0.85}"
maximum_spatial_motion_ratio="${PATH_DENOISING_MAX_MOTION_RATIO:-0.95}"
genesis_render="${build_dir}/bin/genesis_render"
metrics="${build_dir}/post_image_metrics"
textured_scene="${PATH_TRACING_TEXTURED_SCENE:-${renderer_root}/reference/glTF-Sample-Assets/Models/DamagedHelmet/glTF/DamagedHelmet.gltf}"

if [[ ! -f "${textured_scene}" ]]; then
    echo "Textured path-tracing validation scene is missing: ${textured_scene}" >&2
    exit 1
fi

cmake --build "${build_dir}" -j4
mkdir -p "${output_dir}"

capture() {
    local name="$1"
    local frames="$2"
    shift 2
    env SDL_VIDEODRIVER=wayland "${genesis_render}" \
        --validation required \
        --render-mode path-traced \
        --ray-tracing required \
        --demo directional-shadow \
        --taa off \
        --bloom off \
        --occlusion-culling off \
        --camera-input off \
        --resolution 1280 720 \
        --frames "${frames}" \
        --capture "${output_dir}/${name}.png" \
        --capture-frame "${frames}" \
        "$@" 2>&1 | tee "${output_dir}/${name}.log"
}

field() {
    local line="$1"
    local key="$2"
    sed -n "s/.*${key}=\([^ ]*\).*/\1/p" <<<"${line}"
}

capture one_sample 1
capture converged 32 --timing-output "${output_dir}/timing.txt"
capture converged_repeat 32

quality_capture() {
    local name="$1"
    local frames="$2"
    local sampling="$3"
    shift 3
    env SDL_VIDEODRIVER=wayland "${genesis_render}" \
        --validation required \
        --render-mode path-traced \
        --ray-tracing required \
        --demo directional-shadow \
        --taa off \
        --bloom off \
        --occlusion-culling off \
        --camera-input off \
        --path-environment-sampling "${sampling}" \
        --resolution 640 360 \
        --frames "${frames}" \
        --capture "${output_dir}/${name}.png" \
        --capture-frame "${frames}" \
        "$@" \
        2>&1 | tee "${output_dir}/${name}.log"
}

quality_capture uniform_budget 4 uniform --path-max-bounces 2
quality_capture importance_budget 4 importance --path-max-bounces 2
quality_capture importance_reference 64 importance --path-max-bounces 2
quality_capture depth_one 32 importance --path-max-bounces 1
quality_capture depth_four 32 importance --path-max-bounces 4
quality_capture depth_four_repeat 32 importance --path-max-bounces 4

emissive_capture() {
    local name="$1"
    local frames="$2"
    local sampling="$3"
    shift 3
    env SDL_VIDEODRIVER=wayland "${genesis_render}" \
        --validation required \
        --render-mode path-traced \
        --ray-tracing required \
        --demo emissive-sphere \
        --shadows off \
        --taa off \
        --bloom off \
        --occlusion-culling off \
        --camera-input off \
        --path-emissive-sampling "${sampling}" \
        --path-max-bounces 2 \
        --resolution 640 360 \
        --frames "${frames}" \
        --capture "${output_dir}/${name}.png" \
        --capture-frame "${frames}" \
        "$@" \
        2>&1 | tee "${output_dir}/${name}.log"
}

emissive_capture emissive_budget 4 on
emissive_capture emissive_bsdf_budget 4 off
emissive_capture emissive_reference 64 on \
    --timing-output "${output_dir}/emissive_timing.txt"
emissive_capture emissive_reference_repeat 64 on

aov_capture() {
    local name="$1"
    local frames="$2"
    local view="$3"
    shift 3
    env SDL_VIDEODRIVER=wayland "${genesis_render}" \
        --validation required \
        --render-mode path-traced \
        --ray-tracing required \
        --demo directional-shadow \
        --taa off \
        --bloom off \
        --occlusion-culling off \
        --camera-input off \
        --path-debug "${view}" \
        --resolution 640 360 \
        --frames "${frames}" \
        --capture "${output_dir}/${name}.png" \
        --capture-frame "${frames}" \
        "$@" \
        2>&1 | tee "${output_dir}/${name}.log"
}

aov_capture aov_direct 4 direct
aov_capture aov_indirect 4 indirect
aov_capture aov_albedo 1 albedo
aov_capture aov_albedo_repeat 1 albedo
aov_capture aov_roughness 1 roughness
aov_capture aov_geometric_normal 1 geometric-normal
aov_capture aov_shading_normal 1 shading-normal
aov_capture aov_motion_static 2 motion
aov_capture aov_motion 2 motion --camera-script translate
aov_capture aov_depth 1 depth
aov_capture aov_hit_distance 1 hit-distance
aov_capture aov_lighting_residual 4 lighting-residual
aov_capture aov_history_invalid 1 history-validity
aov_capture aov_history_static 2 history-validity
aov_capture aov_history_motion 4 history-validity --camera-script translate
aov_capture aov_history_teleport 3 history-validity --camera-script teleport
aov_capture aov_raw_first 1 raw-radiance
aov_capture aov_temporal_first 1 temporal-radiance
aov_capture aov_spatial_first 1 final
aov_capture aov_temporal_reference 64 temporal-radiance
aov_capture aov_raw_motion_previous 3 raw-radiance --camera-script translate
aov_capture aov_raw_motion 4 raw-radiance --camera-script translate
aov_capture aov_temporal_motion_previous 3 temporal-radiance --camera-script translate
aov_capture aov_temporal_motion 4 temporal-radiance --camera-script translate
aov_capture aov_spatial_motion_previous 3 final --camera-script translate
aov_capture aov_spatial_motion 4 final --camera-script translate
aov_capture aov_reconstruction_weight 4 reconstruction-weight --camera-script translate
aov_capture aov_variance 4 variance

env SDL_VIDEODRIVER=wayland "${genesis_render}" \
    --validation required \
    --render-mode path-traced \
    --ray-tracing required \
    --scene "${textured_scene}" \
    --taa off \
    --bloom off \
    --occlusion-culling off \
    --camera-input off \
    --resolution 1280 720 \
    --frames 32 \
    --capture "${output_dir}/textured.png" \
    --capture-frame 32 \
    --timing-output "${output_dir}/textured_timing.txt" \
    2>&1 | tee "${output_dir}/textured.log"

env SDL_VIDEODRIVER=wayland "${genesis_render}" \
    --validation required \
    --render-mode path-traced \
    --ray-tracing required \
    --demo directional-shadow \
    --taa off \
    --bloom off \
    --occlusion-culling off \
    --camera-input off \
    --resolution 1280 720 \
    --resize-script standard \
    --frames 20 2>&1 | tee "${output_dir}/resize.log"

env SDL_VIDEODRIVER=wayland "${genesis_render}" \
    --validation required \
    --render-mode path-traced \
    --ray-tracing required \
    --demo directional-shadow \
    --taa off \
    --bloom off \
    --occlusion-culling off \
    --camera-input off \
    --camera-script translate \
    --resolution 1280 720 \
    --frames 8 2>&1 | tee "${output_dir}/camera_reset.log"

for log in one_sample converged converged_repeat; do
    grep -F "Render mode: progressive path tracing" "${output_dir}/${log}.log" >/dev/null
    grep -F "Path-tracing ray pipeline:" "${output_dir}/${log}.log" >/dev/null
    grep -F "Shadow traversal: ray-pipeline visibility rays" \
        "${output_dir}/${log}.log" >/dev/null
done
grep -F "Recorded 32 synchronized in-place TLAS update(s)" \
    "${output_dir}/converged.log" >/dev/null
grep -F "Path tracing accumulated 32 sample(s)" \
    "${output_dir}/converged.log" >/dev/null
grep -F "Path tracing accumulated 32 sample(s)" \
    "${output_dir}/textured.log" >/dev/null
grep -F "GPU scene: 1 instances, 1 materials, 6/64 bindless texture descriptors resident" \
    "${output_dir}/textured.log" >/dev/null
grep -F "Path-tracing material textures: 1 textured material(s), 5 imported texture(s); ray-cone mip LOD enabled" \
    "${output_dir}/textured.log" >/dev/null
grep -F "uniform-sphere sampling and power-heuristic MIS enabled" \
    "${output_dir}/uniform_budget.log" >/dev/null
grep -F "solid-angle luminance sampling and power-heuristic MIS enabled" \
    "${output_dir}/importance_budget.log" >/dev/null
grep -F "Path tracing accumulated 64 sample(s)" \
    "${output_dir}/importance_reference.log" >/dev/null
grep -F "2 maximum bounce(s), recursion depth 4" \
    "${output_dir}/converged.log" >/dev/null
grep -F "1 maximum bounce(s), recursion depth 3" \
    "${output_dir}/depth_one.log" >/dev/null
grep -F "4 maximum bounce(s), recursion depth 6" \
    "${output_dir}/depth_four.log" >/dev/null
grep -F "Path tracing accumulated 32 sample(s)" \
    "${output_dir}/depth_four.log" >/dev/null
grep -F "Path-tracing emissive sampler: 2304 resident triangle(s), world-area/luminance CDF and power-heuristic MIS enabled" \
    "${output_dir}/emissive_budget.log" >/dev/null
grep -F "Path-tracing emissive sampler: 2304 resident triangle(s), world-area/luminance CDF and power-heuristic MIS disabled" \
    "${output_dir}/emissive_bsdf_budget.log" >/dev/null
grep -F "Path tracing accumulated 64 sample(s)" \
    "${output_dir}/emissive_reference.log" >/dev/null
grep -F "Path-tracing radiance and reconstruction at 640x360:" \
    "${output_dir}/aov_direct.log" >/dev/null
grep -F "Path reconstruction view: direct radiance" \
    "${output_dir}/aov_direct.log" >/dev/null
grep -F "Path reconstruction view: indirect radiance" \
    "${output_dir}/aov_indirect.log" >/dev/null
grep -F "Path reconstruction view: first-hit albedo" \
    "${output_dir}/aov_albedo.log" >/dev/null
grep -F "Path reconstruction view: first-hit roughness" \
    "${output_dir}/aov_roughness.log" >/dev/null
grep -F "Path reconstruction view: first-hit geometric normal" \
    "${output_dir}/aov_geometric_normal.log" >/dev/null
grep -F "Path reconstruction view: first-hit shading normal" \
    "${output_dir}/aov_shading_normal.log" >/dev/null
grep -F "Path reconstruction view: first-hit motion" \
    "${output_dir}/aov_motion.log" >/dev/null
grep -F "Path reconstruction view: first-hit device depth" \
    "${output_dir}/aov_depth.log" >/dev/null
grep -F "Path reconstruction view: first-hit world distance" \
    "${output_dir}/aov_hit_distance.log" >/dev/null
grep -F "Path reconstruction view: beauty minus direct and indirect residual" \
    "${output_dir}/aov_lighting_residual.log" >/dev/null
grep -F "Path reconstruction view: motion-aware history validity" \
    "${output_dir}/aov_history_static.log" >/dev/null
grep -F "Path reconstruction view: raw progressive radiance" \
    "${output_dir}/aov_raw_first.log" >/dev/null
grep -F "Path reconstruction view: temporal reconstruction weight" \
    "${output_dir}/aov_reconstruction_weight.log" >/dev/null
grep -F "Path reconstruction view: temporal reconstructed radiance" \
    "${output_dir}/aov_temporal_first.log" >/dev/null
grep -F "Path reconstruction view: temporal luminance variance" \
    "${output_dir}/aov_variance.log" >/dev/null
grep -F "Temporal history reset for deterministic camera teleport" \
    "${output_dir}/aov_history_teleport.log" >/dev/null
grep -F "Path-tracing accumulation reset after swapchain recreation" \
    "${output_dir}/resize.log" >/dev/null
grep -F "across 2 reset(s)" "${output_dir}/resize.log" >/dev/null
grep -F "Path tracing accumulated 1 sample(s) since the last reset across 7 reset(s)" \
    "${output_dir}/camera_reset.log" >/dev/null
grep -F "Render mode: path traced" "${output_dir}/timing.txt" >/dev/null
grep -F "Directional shadow backend: ray pipeline" "${output_dir}/timing.txt" >/dev/null
grep -F "Ray-query shadow traversal: disabled" "${output_dir}/timing.txt" >/dev/null
grep -F "Path-tracing accumulated samples: 32" "${output_dir}/timing.txt" >/dev/null
grep -F "Path maximum bounces: 2" "${output_dir}/timing.txt" >/dev/null
grep -F "Path emissive-triangle sampling: enabled" \
    "${output_dir}/emissive_timing.txt" >/dev/null

if grep -E '\[(WARN|ERROR)\]|VUID-|Validation Error' \
    "${output_dir}/one_sample.log" \
    "${output_dir}/converged.log" \
    "${output_dir}/converged_repeat.log" \
    "${output_dir}/uniform_budget.log" \
    "${output_dir}/importance_budget.log" \
    "${output_dir}/importance_reference.log" \
    "${output_dir}/depth_one.log" \
    "${output_dir}/depth_four.log" \
    "${output_dir}/depth_four_repeat.log" \
    "${output_dir}/emissive_budget.log" \
    "${output_dir}/emissive_bsdf_budget.log" \
    "${output_dir}/emissive_reference.log" \
    "${output_dir}/emissive_reference_repeat.log" \
    "${output_dir}/aov_direct.log" \
    "${output_dir}/aov_indirect.log" \
    "${output_dir}/aov_albedo.log" \
    "${output_dir}/aov_albedo_repeat.log" \
    "${output_dir}/aov_roughness.log" \
    "${output_dir}/aov_geometric_normal.log" \
    "${output_dir}/aov_shading_normal.log" \
    "${output_dir}/aov_motion_static.log" \
    "${output_dir}/aov_motion.log" \
    "${output_dir}/aov_depth.log" \
    "${output_dir}/aov_hit_distance.log" \
    "${output_dir}/aov_lighting_residual.log" \
    "${output_dir}/aov_history_invalid.log" \
    "${output_dir}/aov_history_static.log" \
    "${output_dir}/aov_history_motion.log" \
    "${output_dir}/aov_history_teleport.log" \
    "${output_dir}/aov_raw_first.log" \
    "${output_dir}/aov_temporal_first.log" \
    "${output_dir}/aov_spatial_first.log" \
    "${output_dir}/aov_temporal_reference.log" \
    "${output_dir}/aov_raw_motion_previous.log" \
    "${output_dir}/aov_raw_motion.log" \
    "${output_dir}/aov_temporal_motion_previous.log" \
    "${output_dir}/aov_temporal_motion.log" \
    "${output_dir}/aov_spatial_motion_previous.log" \
    "${output_dir}/aov_spatial_motion.log" \
    "${output_dir}/aov_reconstruction_weight.log" \
    "${output_dir}/aov_variance.log" \
    "${output_dir}/textured.log" \
    "${output_dir}/resize.log" \
    "${output_dir}/camera_reset.log"; then
    echo "Path-tracing validation logs contain a warning or error" >&2
    exit 1
fi

one_stats="$("${metrics}" stats "${output_dir}/one_sample.png")"
converged_stats="$("${metrics}" stats "${output_dir}/converged.png")"
convergence_compare="$("${metrics}" compare "${output_dir}/one_sample.png" \
    "${output_dir}/converged.png")"
repeat_compare="$("${metrics}" compare "${output_dir}/converged.png" \
    "${output_dir}/converged_repeat.png")"
textured_stats="$("${metrics}" stats "${output_dir}/textured.png")"
uniform_quality_compare="$("${metrics}" compare "${output_dir}/uniform_budget.png" \
    "${output_dir}/importance_reference.png")"
importance_quality_compare="$("${metrics}" compare "${output_dir}/importance_budget.png" \
    "${output_dir}/importance_reference.png")"
depth_compare="$("${metrics}" compare "${output_dir}/depth_one.png" \
    "${output_dir}/depth_four.png")"
depth_repeat_compare="$("${metrics}" compare "${output_dir}/depth_four.png" \
    "${output_dir}/depth_four_repeat.png")"
depth_four_stats="$("${metrics}" stats "${output_dir}/depth_four.png")"
emissive_stats="$("${metrics}" stats "${output_dir}/emissive_reference.png")"
emissive_quality_compare="$("${metrics}" compare "${output_dir}/emissive_budget.png" \
    "${output_dir}/emissive_reference.png")"
emissive_bsdf_quality_compare="$("${metrics}" compare \
    "${output_dir}/emissive_bsdf_budget.png" \
    "${output_dir}/emissive_reference.png")"
emissive_repeat_compare="$("${metrics}" compare \
    "${output_dir}/emissive_reference.png" \
    "${output_dir}/emissive_reference_repeat.png")"
aov_direct_stats="$("${metrics}" stats "${output_dir}/aov_direct.png")"
aov_indirect_stats="$("${metrics}" stats "${output_dir}/aov_indirect.png")"
aov_albedo_stats="$("${metrics}" stats "${output_dir}/aov_albedo.png")"
aov_roughness_stats="$("${metrics}" stats "${output_dir}/aov_roughness.png")"
aov_motion_stats="$("${metrics}" stats "${output_dir}/aov_motion.png")"
aov_depth_stats="$("${metrics}" stats "${output_dir}/aov_depth.png")"
aov_hit_distance_stats="$("${metrics}" stats "${output_dir}/aov_hit_distance.png")"
aov_residual_stats="$("${metrics}" stats "${output_dir}/aov_lighting_residual.png")"
history_invalid_stats="$("${metrics}" stats "${output_dir}/aov_history_invalid.png")"
history_static_stats="$("${metrics}" stats "${output_dir}/aov_history_static.png")"
history_motion_stats="$("${metrics}" stats "${output_dir}/aov_history_motion.png")"
aov_albedo_repeat_compare="$("${metrics}" compare \
    "${output_dir}/aov_albedo.png" "${output_dir}/aov_albedo_repeat.png")"
aov_normal_compare="$("${metrics}" compare \
    "${output_dir}/aov_geometric_normal.png" \
    "${output_dir}/aov_shading_normal.png")"
aov_lighting_compare="$("${metrics}" compare \
    "${output_dir}/aov_direct.png" "${output_dir}/aov_indirect.png")"
aov_motion_compare="$("${metrics}" compare \
    "${output_dir}/aov_motion_static.png" "${output_dir}/aov_motion.png")"
history_static_compare="$("${metrics}" compare \
    "${output_dir}/aov_history_invalid.png" "${output_dir}/aov_history_static.png")"
history_teleport_compare="$("${metrics}" compare \
    "${output_dir}/aov_history_invalid.png" "${output_dir}/aov_history_teleport.png")"
reconstruction_first_compare="$("${metrics}" compare \
    "${output_dir}/aov_raw_first.png" "${output_dir}/aov_temporal_first.png")"
spatial_first_compare="$("${metrics}" compare \
    "${output_dir}/aov_temporal_first.png" "${output_dir}/aov_spatial_first.png")"
temporal_reference_compare="$("${metrics}" compare \
    "${output_dir}/aov_temporal_first.png" "${output_dir}/aov_temporal_reference.png")"
spatial_reference_compare="$("${metrics}" compare \
    "${output_dir}/aov_spatial_first.png" "${output_dir}/aov_temporal_reference.png")"
raw_motion_compare="$("${metrics}" compare \
    "${output_dir}/aov_raw_motion_previous.png" "${output_dir}/aov_raw_motion.png")"
reconstructed_motion_compare="$("${metrics}" compare \
    "${output_dir}/aov_temporal_motion_previous.png" \
    "${output_dir}/aov_temporal_motion.png")"
spatial_motion_compare="$("${metrics}" compare \
    "${output_dir}/aov_spatial_motion_previous.png" \
    "${output_dir}/aov_spatial_motion.png")"
reconstruction_effect_compare="$("${metrics}" compare \
    "${output_dir}/aov_raw_motion.png" "${output_dir}/aov_temporal_motion.png")"
reconstruction_weight_stats="$("${metrics}" stats \
    "${output_dir}/aov_reconstruction_weight.png")"
variance_stats="$("${metrics}" stats "${output_dir}/aov_variance.png")"
one_mean="$(field "${one_stats}" mean)"
converged_mean="$(field "${converged_stats}" mean)"
convergence_rmse="$(field "${convergence_compare}" rmse)"
repeat_rmse="$(field "${repeat_compare}" rmse)"
path_ms="$(sed -n 's/^path tracing: \([^ ]*\) ms.*/\1/p' "${output_dir}/timing.txt")"
history_ms="$(sed -n 's/^path history validation: \([^ ]*\) ms.*/\1/p' \
    "${output_dir}/timing.txt")"
reconstruction_ms="$(sed -n 's/^path temporal reconstruction: \([^ ]*\) ms.*/\1/p' \
    "${output_dir}/timing.txt")"
denoising_ms="$(sed -n 's/^path spatial denoising: \([^ ]*\) ms.*/\1/p' \
    "${output_dir}/timing.txt")"
textured_path_ms="$(sed -n 's/^path tracing: \([^ ]*\) ms.*/\1/p' \
    "${output_dir}/textured_timing.txt")"
textured_mean="$(field "${textured_stats}" mean)"
uniform_quality_rmse="$(field "${uniform_quality_compare}" rmse)"
importance_quality_rmse="$(field "${importance_quality_compare}" rmse)"
depth_rmse="$(field "${depth_compare}" rmse)"
depth_repeat_rmse="$(field "${depth_repeat_compare}" rmse)"
importance_rmse_ratio="$(awk -v importance="${importance_quality_rmse}" \
    -v uniform="${uniform_quality_rmse}" 'BEGIN { print importance / uniform }')"
emissive_quality_rmse="$(field "${emissive_quality_compare}" rmse)"
emissive_bsdf_quality_rmse="$(field "${emissive_bsdf_quality_compare}" rmse)"
emissive_repeat_rmse="$(field "${emissive_repeat_compare}" rmse)"
emissive_rmse_ratio="$(awk -v sampled="${emissive_quality_rmse}" \
    -v bsdf="${emissive_bsdf_quality_rmse}" 'BEGIN { print sampled / bsdf }')"
emissive_path_ms="$(sed -n 's/^path tracing: \([^ ]*\) ms.*/\1/p' \
    "${output_dir}/emissive_timing.txt")"
aov_albedo_repeat_rmse="$(field "${aov_albedo_repeat_compare}" rmse)"
aov_normal_rmse="$(field "${aov_normal_compare}" rmse)"
aov_lighting_rmse="$(field "${aov_lighting_compare}" rmse)"
aov_motion_rmse="$(field "${aov_motion_compare}" rmse)"
aov_residual_max="$(field "${aov_residual_stats}" max)"
aov_direct_mean="$(field "${aov_direct_stats}" mean)"
aov_indirect_mean="$(field "${aov_indirect_stats}" mean)"
aov_albedo_mean="$(field "${aov_albedo_stats}" mean)"
aov_roughness_mean="$(field "${aov_roughness_stats}" mean)"
aov_depth_mean="$(field "${aov_depth_stats}" mean)"
aov_hit_distance_mean="$(field "${aov_hit_distance_stats}" mean)"
history_invalid_mean="$(field "${history_invalid_stats}" mean)"
history_static_mean="$(field "${history_static_stats}" mean)"
history_motion_mean="$(field "${history_motion_stats}" mean)"
history_static_rmse="$(field "${history_static_compare}" rmse)"
history_teleport_rmse="$(field "${history_teleport_compare}" rmse)"
reconstruction_first_rmse="$(field "${reconstruction_first_compare}" rmse)"
spatial_first_rmse="$(field "${spatial_first_compare}" rmse)"
temporal_reference_rmse="$(field "${temporal_reference_compare}" rmse)"
spatial_reference_rmse="$(field "${spatial_reference_compare}" rmse)"
raw_motion_rmse="$(field "${raw_motion_compare}" rmse)"
reconstructed_motion_rmse="$(field "${reconstructed_motion_compare}" rmse)"
spatial_motion_rmse="$(field "${spatial_motion_compare}" rmse)"
reconstruction_effect_rmse="$(field "${reconstruction_effect_compare}" rmse)"
reconstructed_motion_ratio="$(awk -v reconstructed="${reconstructed_motion_rmse}" \
    -v raw="${raw_motion_rmse}" 'BEGIN { print reconstructed / raw }')"
spatial_reference_ratio="$(awk -v spatial="${spatial_reference_rmse}" \
    -v temporal="${temporal_reference_rmse}" 'BEGIN { print spatial / temporal }')"
spatial_motion_ratio="$(awk -v spatial="${spatial_motion_rmse}" \
    -v temporal="${reconstructed_motion_rmse}" 'BEGIN { print spatial / temporal }')"
reconstruction_weight_mean="$(field "${reconstruction_weight_stats}" mean)"
variance_mean="$(field "${variance_stats}" mean)"

[[ "${one_stats}" == *"finite=1"* && "${converged_stats}" == *"finite=1"* &&
   "${textured_stats}" == *"finite=1"* && "${depth_four_stats}" == *"finite=1"* &&
   "${emissive_stats}" == *"finite=1"* && "${aov_direct_stats}" == *"finite=1"* &&
   "${aov_indirect_stats}" == *"finite=1"* && "${aov_albedo_stats}" == *"finite=1"* &&
   "${aov_roughness_stats}" == *"finite=1"* && "${aov_motion_stats}" == *"finite=1"* &&
   "${aov_depth_stats}" == *"finite=1"* &&
   "${aov_hit_distance_stats}" == *"finite=1"* &&
   "${aov_residual_stats}" == *"finite=1"* &&
   "${history_invalid_stats}" == *"finite=1"* &&
   "${history_static_stats}" == *"finite=1"* &&
   "${history_motion_stats}" == *"finite=1"* &&
   "${reconstruction_weight_stats}" == *"finite=1"* &&
   "${variance_stats}" == *"finite=1"* ]]
awk -v value="${one_mean}" 'BEGIN { if (value <= 0.02 || value >= 0.95) exit 1 }'
awk -v value="${converged_mean}" 'BEGIN { if (value <= 0.02 || value >= 0.95) exit 1 }'
awk -v value="${textured_mean}" 'BEGIN { if (value <= 0.02 || value >= 0.95) exit 1 }'
awk -v value="${convergence_rmse}" 'BEGIN { if (value < 0.01) exit 1 }'
awk -v value="${repeat_rmse}" 'BEGIN { if (value > 0.000001) exit 1 }'
awk -v value="${depth_rmse}" 'BEGIN { if (value < 0.001) exit 1 }'
awk -v value="${depth_repeat_rmse}" 'BEGIN { if (value > 0.000001) exit 1 }'
awk -v value="${importance_rmse_ratio}" -v maximum="${maximum_importance_rmse_ratio}" \
    'BEGIN { if (value != value || value <= 0.0 || value > maximum) exit 1 }'
awk -v value="${emissive_repeat_rmse}" \
    'BEGIN { if (value > 0.000001) exit 1 }'
awk -v value="${emissive_rmse_ratio}" -v maximum="${maximum_emissive_rmse_ratio}" \
    'BEGIN { if (value != value || value <= 0.0 || value > maximum) exit 1 }'
awk -v value="${aov_albedo_repeat_rmse}" 'BEGIN { if (value > 0.000001) exit 1 }'
awk -v value="${aov_normal_rmse}" 'BEGIN { if (value < 0.001) exit 1 }'
awk -v value="${aov_lighting_rmse}" 'BEGIN { if (value < 0.001) exit 1 }'
awk -v value="${aov_motion_rmse}" 'BEGIN { if (value < 0.0001) exit 1 }'
awk -v value="${aov_residual_max}" 'BEGIN { if (value > 0.01) exit 1 }'
awk -v value="${history_invalid_mean}" \
    'BEGIN { if (value < 0.21 || value > 0.22) exit 1 }'
awk -v value="${history_static_mean}" 'BEGIN { if (value < 0.65) exit 1 }'
awk -v value="${history_motion_mean}" 'BEGIN { if (value < 0.60) exit 1 }'
awk -v value="${history_static_rmse}" 'BEGIN { if (value < 0.50) exit 1 }'
awk -v value="${history_teleport_rmse}" 'BEGIN { if (value > 0.000001) exit 1 }'
awk -v value="${reconstruction_first_rmse}" 'BEGIN { if (value > 0.001) exit 1 }'
awk -v value="${spatial_first_rmse}" 'BEGIN { if (value < 0.01) exit 1 }'
awk -v value="${spatial_reference_ratio}" -v maximum="${maximum_spatial_reference_ratio}" \
    'BEGIN { if (value != value || value <= 0.0 || value > maximum) exit 1 }'
awk -v value="${reconstruction_effect_rmse}" 'BEGIN { if (value < 0.01) exit 1 }'
awk -v value="${reconstructed_motion_ratio}" \
    -v maximum="${maximum_reconstructed_motion_ratio}" \
    'BEGIN { if (value != value || value <= 0.0 || value > maximum) exit 1 }'
awk -v value="${spatial_motion_ratio}" -v maximum="${maximum_spatial_motion_ratio}" \
    'BEGIN { if (value != value || value <= 0.0 || value > maximum) exit 1 }'
awk -v value="${reconstruction_weight_mean}" \
    'BEGIN { if (value < 0.65 || value > 0.80) exit 1 }'
awk -v value="${variance_mean}" 'BEGIN { if (value <= 0.001 || value >= 0.50) exit 1 }'
for value in "${aov_direct_mean}" "${aov_indirect_mean}" "${aov_albedo_mean}" \
    "${aov_roughness_mean}" "${aov_depth_mean}" "${aov_hit_distance_mean}"; do
    awk -v mean="${value}" 'BEGIN { if (mean != mean || mean <= 0.0001 || mean >= 0.95) exit 1 }'
done
if [[ -z "${path_ms}" ]] ||
   ! awk -v actual="${path_ms}" -v maximum="${maximum_path_ms}" \
       'BEGIN { if (actual != actual || actual <= 0.0 || actual > maximum) exit 1 }'; then
    echo "Path-tracing GPU time ${path_ms:-missing} ms exceeds ${maximum_path_ms} ms" >&2
    exit 1
fi
if [[ -z "${history_ms}" ]] ||
   ! awk -v actual="${history_ms}" -v maximum="${maximum_history_ms}" \
       'BEGIN { if (actual != actual || actual <= 0.0 || actual > maximum) exit 1 }'; then
    echo "Path-history GPU time ${history_ms:-missing} ms exceeds ${maximum_history_ms} ms" >&2
    exit 1
fi
if [[ -z "${reconstruction_ms}" ]] ||
   ! awk -v actual="${reconstruction_ms}" -v maximum="${maximum_reconstruction_ms}" \
       'BEGIN { if (actual != actual || actual <= 0.0 || actual > maximum) exit 1 }'; then
    echo "Path-reconstruction GPU time ${reconstruction_ms:-missing} ms exceeds ${maximum_reconstruction_ms} ms" >&2
    exit 1
fi
if [[ -z "${denoising_ms}" ]] ||
   ! awk -v actual="${denoising_ms}" -v maximum="${maximum_denoising_ms}" \
       'BEGIN { if (actual != actual || actual <= 0.0 || actual > maximum) exit 1 }'; then
    echo "Path-denoising GPU time ${denoising_ms:-missing} ms exceeds ${maximum_denoising_ms} ms" >&2
    exit 1
fi
if [[ -z "${emissive_path_ms}" ]] ||
   ! awk -v actual="${emissive_path_ms}" -v maximum="${maximum_emissive_path_ms}" \
       'BEGIN { if (actual != actual || actual <= 0.0 || actual > maximum) exit 1 }'; then
    echo "Emissive path-tracing GPU time ${emissive_path_ms:-missing} ms exceeds ${maximum_emissive_path_ms} ms" >&2
    exit 1
fi
if [[ -z "${textured_path_ms}" ]] ||
   ! awk -v actual="${textured_path_ms}" -v maximum="${maximum_textured_path_ms}" \
       'BEGIN { if (actual != actual || actual <= 0.0 || actual > maximum) exit 1 }'; then
    echo "Textured path-tracing GPU time ${textured_path_ms:-missing} ms exceeds ${maximum_textured_path_ms} ms" >&2
    exit 1
fi

{
    echo "Path-tracing validation passed:"
    echo "  one-sample/converged RMSE: ${convergence_rmse} (minimum 0.01)"
    echo "  deterministic repeat RMSE: ${repeat_rmse} (maximum 0.000001)"
    echo "  depth-1/depth-4 RMSE: ${depth_rmse} (minimum 0.001)"
    echo "  depth-4 deterministic repeat RMSE: ${depth_repeat_rmse} (maximum 0.000001)"
    echo "  fixed 4-sample environment RMSE: uniform ${uniform_quality_rmse}, importance ${importance_quality_rmse}"
    echo "  importance/uniform RMSE ratio: ${importance_rmse_ratio} (maximum ${maximum_importance_rmse_ratio})"
    echo "  emissive 4-sample RMSE: sampled ${emissive_quality_rmse}, BSDF-only ${emissive_bsdf_quality_rmse}"
    echo "  emissive sampled/BSDF-only RMSE ratio: ${emissive_rmse_ratio} (maximum ${maximum_emissive_rmse_ratio})"
    echo "  emissive deterministic repeat RMSE: ${emissive_repeat_rmse} (maximum 0.000001)"
    echo "  emissive RTX path dispatch: ${emissive_path_ms} ms (maximum ${maximum_emissive_path_ms} ms)"
    echo "  AOV albedo deterministic repeat RMSE: ${aov_albedo_repeat_rmse} (maximum 0.000001)"
    echo "  geometric/shading normal RMSE: ${aov_normal_rmse} (minimum 0.001)"
    echo "  direct/indirect display RMSE: ${aov_lighting_rmse} (minimum 0.001)"
    echo "  moving/static path-motion RMSE: ${aov_motion_rmse} (minimum 0.0001)"
    echo "  beauty minus direct/indirect residual display maximum: ${aov_residual_max} (maximum 0.01)"
    echo "  history validity luminance: invalid ${history_invalid_mean}, static ${history_static_mean}, moving ${history_motion_mean}"
    echo "  invalid/static history RMSE: ${history_static_rmse} (minimum 0.50)"
    echo "  invalid/teleport history RMSE: ${history_teleport_rmse} (maximum 0.000001)"
    echo "  RTX 2060 SUPER path-history dispatch: ${history_ms} ms (maximum ${maximum_history_ms} ms)"
    echo "  first-frame raw/temporal RMSE: ${reconstruction_first_rmse} (maximum 0.001)"
    echo "  first-frame temporal/spatial RMSE: ${spatial_first_rmse} (minimum 0.01)"
    echo "  one-sample/reference RMSE: temporal ${temporal_reference_rmse}, spatial ${spatial_reference_rmse}"
    echo "  spatial/temporal reference RMSE ratio: ${spatial_reference_ratio} (maximum ${maximum_spatial_reference_ratio})"
    echo "  moving raw/temporal RMSE: ${reconstruction_effect_rmse} (minimum 0.01)"
    echo "  adjacent moving-frame RMSE: raw ${raw_motion_rmse}, temporal ${reconstructed_motion_rmse}, spatial ${spatial_motion_rmse}"
    echo "  reconstructed/raw moving-frame RMSE ratio: ${reconstructed_motion_ratio} (maximum ${maximum_reconstructed_motion_ratio})"
    echo "  spatial/temporal moving-frame RMSE ratio: ${spatial_motion_ratio} (maximum ${maximum_spatial_motion_ratio})"
    echo "  reconstruction-weight debug luminance: ${reconstruction_weight_mean}"
    echo "  variance debug luminance: ${variance_mean}"
    echo "  RTX 2060 SUPER path-reconstruction dispatch: ${reconstruction_ms} ms (maximum ${maximum_reconstruction_ms} ms)"
    echo "  RTX 2060 SUPER path-denoising dispatch: ${denoising_ms} ms (maximum ${maximum_denoising_ms} ms)"
    echo "  mean luminance: one sample ${one_mean}, 32 samples ${converged_mean}"
    echo "  RTX 2060 SUPER path dispatch: ${path_ms} ms (maximum ${maximum_path_ms} ms)"
    echo "  textured glTF mean luminance: ${textured_mean}"
    echo "  textured RTX path dispatch: ${textured_path_ms} ms (maximum ${maximum_textured_path_ms} ms)"
    echo "  resize and moving-camera scripts produced deterministic accumulation resets"
} | tee "${output_dir}/summary.txt"
