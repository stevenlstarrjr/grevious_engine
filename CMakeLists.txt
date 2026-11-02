cmake_minimum_required(VERSION 3.25)

project(GenesisRender VERSION 0.1.0 LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT WIN32)
    message(FATAL_ERROR "Genesis Render supports Linux and Windows only")
endif()

option(RENDERER_VALIDATION_DEFAULT "Request Vulkan validation by default" ON)
option(GENESIS_BUILD_SDL_APP "Build the SDL Genesis Render executable" ON)
option(GENESIS_BUILD_TESTS "Build Genesis unit and integration tests" ${PROJECT_IS_TOP_LEVEL})
option(RENDERER_ENABLE_NRD "Build the NVIDIA Real-time Denoiser backend" ON)
if(WIN32)
    set(RENDERER_STREAMLINE_DEFAULT ON)
else()
    set(RENDERER_STREAMLINE_DEFAULT OFF)
endif()
option(RENDERER_ENABLE_STREAMLINE
    "Build the NVIDIA Streamline DLSS integration"
    ${RENDERER_STREAMLINE_DEFAULT})

if(RENDERER_ENABLE_STREAMLINE AND NOT WIN32)
    message(FATAL_ERROR "NVIDIA Streamline is currently supported by Genesis on Windows only")
endif()
if(RENDERER_ENABLE_STREAMLINE)
    set(RENDERER_STREAMLINE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/Streamline")
    if(NOT EXISTS "${RENDERER_STREAMLINE_ROOT}/include/sl.h" OR
       NOT EXISTS "${RENDERER_STREAMLINE_ROOT}/bin/x64/sl.interposer.dll")
        message(FATAL_ERROR
            "Missing NVIDIA Streamline 2.12.0; run tools/setup_streamline.ps1")
    endif()
endif()

find_package(Vulkan 1.3 REQUIRED)
find_program(GLSLC_EXECUTABLE glslc REQUIRED)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND GENESIS_BUILD_SDL_APP)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(RENDERER_LIBDECOR REQUIRED libdecor-0)
endif()

set(RENDERER_REQUIRED_SUBMODULES volk VMA glm fastgltf simdjson)
if(GENESIS_BUILD_SDL_APP)
    list(APPEND RENDERER_REQUIRED_SUBMODULES SDL)
endif()
foreach(required_submodule IN LISTS RENDERER_REQUIRED_SUBMODULES)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/${required_submodule}/CMakeLists.txt")
        message(FATAL_ERROR
            "Missing thirdparty/${required_submodule}; run git submodule update --init --recursive")
    endif()
endforeach()
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/MikkTSpace/mikktspace.c" OR
   NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/MikkTSpace/mikktspace.h")
    message(FATAL_ERROR
        "Missing thirdparty/MikkTSpace; run git submodule update --init --recursive")
endif()
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/stb/stb_image.h")
    message(FATAL_ERROR
        "Missing thirdparty/stb; run git submodule update --init --recursive")
endif()

if(GENESIS_BUILD_SDL_APP)
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
    set(SDL_AUDIO OFF CACHE BOOL "" FORCE)
    set(SDL_CAMERA OFF CACHE BOOL "" FORCE)
    set(SDL_DIALOG OFF CACHE BOOL "" FORCE)
    set(SDL_DUMMYVIDEO OFF CACHE BOOL "" FORCE)
    set(SDL_GPU OFF CACHE BOOL "" FORCE)
    set(SDL_HAPTIC OFF CACHE BOOL "" FORCE)
    set(SDL_HIDAPI OFF CACHE BOOL "" FORCE)
    set(SDL_JOYSTICK OFF CACHE BOOL "" FORCE)
    set(SDL_OPENGL OFF CACHE BOOL "" FORCE)
    set(SDL_OPENGLES OFF CACHE BOOL "" FORCE)
    set(SDL_RENDER OFF CACHE BOOL "" FORCE)
    set(SDL_SENSOR OFF CACHE BOOL "" FORCE)
    set(SDL_POWER OFF CACHE BOOL "" FORCE)
    set(SDL_TRAY OFF CACHE BOOL "" FORCE)
    set(SDL_NOTIFICATION OFF CACHE BOOL "" FORCE)
    set(SDL_OFFSCREEN OFF CACHE BOOL "" FORCE)
    set(SDL_VULKAN ON CACHE BOOL "" FORCE)
    set(SDL_KMSDRM OFF CACHE BOOL "" FORCE)
    set(SDL_LIBTHAI OFF CACHE BOOL "" FORCE)
    set(SDL_FRIBIDI OFF CACHE BOOL "" FORCE)
    set(SDL_X11 OFF CACHE BOOL "" FORCE)
    set(SDL_X11_XSCRNSAVER OFF CACHE BOOL "" FORCE)
    set(SDL_X11_XTEST OFF CACHE BOOL "" FORCE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(SDL_WAYLAND_LIBDECOR ON CACHE BOOL "" FORCE)
        set(SDL_WAYLAND_LIBDECOR_SHARED ON CACHE BOOL "" FORCE)
        set(SDL_WAYLAND ON CACHE BOOL "" FORCE)
    elseif(WIN32)
        set(SDL_WAYLAND_LIBDECOR OFF CACHE BOOL "" FORCE)
        set(SDL_WAYLAND_LIBDECOR_SHARED OFF CACHE BOOL "" FORCE)
        set(SDL_WAYLAND OFF CACHE BOOL "" FORCE)
        set(SDL_DIRECTX ON CACHE BOOL "" FORCE)
    endif()
endif()

if(RENDERER_ENABLE_NRD)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/NRD/CMakeLists.txt")
        message(FATAL_ERROR
            "Missing thirdparty/NRD; run git submodule update --init --recursive")
    endif()

    # Genesis is Vulkan-only. Embed just the SPIR-V NRD shaders and keep the
    # first Turing-compatible integration independent of NRI and quad
    # intrinsics. Quad intrinsics can be enabled later together with the
    # VK_KHR_compute_shader_derivatives device feature.
    set(NRD_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
    set(NRD_NRI OFF CACHE BOOL "" FORCE)
    set(NRD_EMBEDS_SPIRV_SHADERS ON CACHE BOOL "" FORCE)
    set(NRD_EMBEDS_DXIL_SHADERS OFF CACHE BOOL "" FORCE)
    set(NRD_EMBEDS_DXBC_SHADERS OFF CACHE BOOL "" FORCE)
    set(NRD_SUPPORTS_QUAD_INTRINSICS OFF CACHE BOOL "" FORCE)
    set(NRD_NORMAL_ENCODING "2" CACHE STRING "" FORCE)
    set(NRD_ROUGHNESS_ENCODING "1" CACHE STRING "" FORCE)
    find_program(DXC_SPIRV_PATH dxc
        HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/Bin32"
        REQUIRED)
    add_subdirectory(thirdparty/NRD EXCLUDE_FROM_ALL SYSTEM)
endif()
if(GENESIS_BUILD_SDL_APP)
    add_subdirectory(thirdparty/SDL EXCLUDE_FROM_ALL SYSTEM)
endif()

set(VOLK_INSTALL OFF CACHE BOOL "" FORCE)
set(VOLK_PULL_IN_VULKAN ON CACHE BOOL "" FORCE)
if(WIN32)
    set(VOLK_STATIC_DEFINES "VK_USE_PLATFORM_WIN32_KHR" CACHE STRING "" FORCE)
endif()
add_subdirectory(thirdparty/volk EXCLUDE_FROM_ALL SYSTEM)

set(VMA_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/VMA EXCLUDE_FROM_ALL SYSTEM)

set(GLM_BUILD_LIBRARY OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_INSTALL OFF CACHE BOOL "" FORCE)
set(GLM_ENABLE_CXX_20 ON CACHE BOOL "" FORCE)
set(GLM_QUIET ON CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/glm EXCLUDE_FROM_ALL SYSTEM)

# Configure fastgltf against the pinned simdjson submodule before adding it.
# This keeps configuration and builds offline and avoids fastgltf's download
# fallback when a simdjson target is not already present.
set(SIMDJSON_INSTALL OFF CACHE BOOL "" FORCE)
set(SIMDJSON_DEVELOPER_MODE OFF CACHE BOOL "" FORCE)
set(SIMDJSON_BUILD_STATIC_LIB OFF CACHE BOOL "" FORCE)
set(SIMDJSON_ENABLE_THREADS OFF CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/simdjson EXCLUDE_FROM_ALL SYSTEM)

set(FASTGLTF_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_DOCS OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(FASTGLTF_COMPILE_AS_CPP20 ON CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_CPP_MODULES OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_KHR_IMPLICIT_SHAPES OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_KHR_PHYSICS_RIGID_BODIES OFF CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/fastgltf EXCLUDE_FROM_ALL SYSTEM)

add_library(renderer_scene STATIC src/scene/Scene.cpp)
target_compile_features(renderer_scene PUBLIC cxx_std_20)
target_include_directories(renderer_scene PUBLIC include)
target_link_libraries(renderer_scene PUBLIC glm::glm)
set_target_properties(renderer_scene PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(renderer_scene PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
    )
endif()

add_library(renderer_mikktspace STATIC thirdparty/MikkTSpace/mikktspace.c)
target_include_directories(renderer_mikktspace PUBLIC thirdparty/MikkTSpace)
set_target_properties(renderer_mikktspace PROPERTIES C_EXTENSIONS OFF)

if(RENDERER_ENABLE_NRD)
    add_library(renderer_nrd STATIC
        src/nvidia/NrdContext.cpp
        src/nvidia/NrdVulkan.cpp
        src/nvidia/NrdVulkanDenoiser.cpp
    )
    target_compile_features(renderer_nrd PUBLIC cxx_std_20)
    target_include_directories(renderer_nrd PUBLIC include)
    target_link_libraries(renderer_nrd PUBLIC NRD volk::volk)
    set_target_properties(renderer_nrd PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(renderer_nrd PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
endif()

add_library(renderer_gltf STATIC
    src/assets/GltfImporter.cpp
    src/assets/TangentGenerator.cpp
)
target_compile_features(renderer_gltf PUBLIC cxx_std_20)
target_include_directories(renderer_gltf PUBLIC include)
target_link_libraries(renderer_gltf
    PUBLIC renderer_scene
    PRIVATE fastgltf::fastgltf renderer_mikktspace
)
set_target_properties(renderer_gltf PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(renderer_gltf PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
    )
endif()

add_library(renderer_pbr STATIC src/materials/PbrMath.cpp)
target_compile_features(renderer_pbr PUBLIC cxx_std_20)
target_include_directories(renderer_pbr PUBLIC include)
target_link_libraries(renderer_pbr PUBLIC glm::glm)
set_target_properties(renderer_pbr PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
target_compile_options(renderer_pbr PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
    )
endif()

add_library(renderer_procedural STATIC src/procedural/NoiseVolume.cpp)
target_compile_features(renderer_procedural PUBLIC cxx_std_20)
target_include_directories(renderer_procedural PUBLIC include)
set_target_properties(renderer_procedural PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(renderer_procedural PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
    )
endif()

add_library(renderer_images STATIC
    src/assets/ImageDecoder.cpp
    src/assets/ImageWriter.cpp
)
target_compile_features(renderer_images PUBLIC cxx_std_20)
target_include_directories(renderer_images PUBLIC include)
target_include_directories(renderer_images SYSTEM PRIVATE thirdparty/stb)
set_target_properties(renderer_images PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(renderer_images PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
    )
endif()

add_library(renderer_output STATIC src/output/AovExport.cpp)
target_compile_features(renderer_output PUBLIC cxx_std_20)
target_include_directories(renderer_output PUBLIC include)
target_link_libraries(renderer_output PRIVATE renderer_images)
set_target_properties(renderer_output PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(renderer_output PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
    )
endif()

add_library(renderer_upscaling STATIC src/upscaling/Upscaler.cpp)
target_compile_features(renderer_upscaling PUBLIC cxx_std_20)
target_include_directories(renderer_upscaling PUBLIC include)
set_target_properties(renderer_upscaling PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(renderer_upscaling PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
    )
endif()

add_library(renderer_ibl STATIC
    src/lighting/Atmosphere.cpp
    src/lighting/IblBaker.cpp
)
target_compile_features(renderer_ibl PUBLIC cxx_std_20)
target_include_directories(renderer_ibl PUBLIC include)
target_link_libraries(renderer_ibl PUBLIC renderer_images glm::glm)
target_compile_definitions(renderer_ibl PRIVATE
    GENESIS_BRUNETON_ATMOSPHERE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/BrunetonAtmosphere/assets"
)
set_target_properties(renderer_ibl PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(renderer_ibl PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
    )
endif()

add_library(renderer_gpu_scene STATIC src/gpu_scene/GpuScene.cpp)
target_compile_features(renderer_gpu_scene PUBLIC cxx_std_20)
target_include_directories(renderer_gpu_scene PUBLIC include)
target_link_libraries(renderer_gpu_scene PUBLIC renderer_scene glm::glm)
set_target_properties(renderer_gpu_scene PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(renderer_gpu_scene PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
    )
endif()

add_library(renderer_temporal STATIC
    src/temporal/TemporalSampling.cpp
    src/temporal/TemporalState.cpp
    src/temporal/MotionVectors.cpp
    src/temporal/TemporalResolve.cpp
)
target_compile_features(renderer_temporal PUBLIC cxx_std_20)
target_include_directories(renderer_temporal PUBLIC include)
target_link_libraries(renderer_temporal PUBLIC glm::glm)
set_target_properties(renderer_temporal PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(renderer_temporal PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
    )
endif()

add_library(renderer_post STATIC src/post/PostProcessing.cpp)
target_compile_features(renderer_post PUBLIC cxx_std_20)
target_include_directories(renderer_post PUBLIC include)
target_link_libraries(renderer_post PUBLIC glm::glm)
set_target_properties(renderer_post PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(renderer_post PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
    )
endif()

add_library(renderer_ray_tracing STATIC
    src/ray_tracing/AccelerationStructurePlan.cpp
    src/ray_tracing/RayTracingPolicy.cpp
)
target_compile_features(renderer_ray_tracing PUBLIC cxx_std_20)
target_include_directories(renderer_ray_tracing PUBLIC include)
target_link_libraries(renderer_ray_tracing PUBLIC renderer_gpu_scene)
set_target_properties(renderer_ray_tracing PROPERTIES CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(renderer_ray_tracing PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
    )
endif()

add_library(renderer_vma STATIC
    src/vulkan/VmaImplementation.cpp
    src/vulkan/MemoryAllocator.cpp
)
target_compile_features(renderer_vma PUBLIC cxx_std_20)
target_include_directories(renderer_vma PUBLIC include)
target_link_libraries(renderer_vma
    PUBLIC volk::volk
    PRIVATE GPUOpen::VulkanMemoryAllocator
)
set_target_properties(renderer_vma PROPERTIES CXX_EXTENSIONS OFF)

set(RENDERER_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")
set(RENDERER_SHADER_SOURCES
    shaders/mesh.vert
    shaders/mesh.frag
    shaders/shadow.vert
    shaders/shadow.frag
    shaders/shadow_cull.comp
    shaders/frustum_cull.comp
    shaders/occlusion_cull.comp
    shaders/hiz_copy.comp
    shaders/hiz_reduce.comp
    shaders/sky.frag
    shaders/post_test.frag
    shaders/bloom_downsample.frag
    shaders/bloom_blur_h.frag
    shaders/bloom_blur_v.frag
    shaders/editor_overlay.frag
    shaders/editor_overlay.vert
    shaders/taa.frag
    shaders/upscale_spatial.frag
    shaders/upscale_reactive.frag
    shaders/upscale_temporal.frag
    shaders/tonemap.vert
    shaders/tonemap.frag
    shaders/rain.frag
    shaders/atmosphere_environment.comp
    shaders/path_trace.rgen
    shaders/path_trace.rmiss
    shaders/path_trace_shadow.rmiss
    shaders/path_trace.rchit
    shaders/path_trace.rahit
    shaders/path_history.comp
    shaders/path_reconstruct.comp
    shaders/path_denoise.comp
    shaders/path_nrd_prepare.comp
    shaders/path_nrd_resolve.comp
)
set(RENDERER_SHADER_HEADERS
    shaders/include/look_frame.glsl
    shaders/include/first_hit_ids.glsl
    shaders/include/bruneton_atmosphere.glsl
    shaders/include/genesis_atmosphere.glsl
    shaders/include/genesis_clouds.glsl
    shaders/include/genesis_noise.glsl
    shaders/include/pbr.glsl
    shaders/include/ibl.glsl
    shaders/include/post.glsl
    shaders/include/shadows.glsl
    shaders/include/path_trace.glsl
    shaders/include/path_trace_bsdf.glsl
    shaders/include/path_trace_environment.glsl
    shaders/include/virtual_texture.glsl
    shaders/include/weather_wetness.glsl
)
set(RENDERER_SHADER_OUTPUTS)
foreach(shader_source IN LISTS RENDERER_SHADER_SOURCES)
    get_filename_component(shader_name "${shader_source}" NAME)
    set(shader_output "${RENDERER_SHADER_OUTPUT_DIR}/${shader_name}.spv")
    add_custom_command(
        OUTPUT "${shader_output}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${RENDERER_SHADER_OUTPUT_DIR}"
        COMMAND "${GLSLC_EXECUTABLE}"
            --target-env=vulkan1.3
            -Werror
            -O
            -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
            "${CMAKE_CURRENT_SOURCE_DIR}/${shader_source}"
            -o "${shader_output}"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/${shader_source}"
            ${RENDERER_SHADER_HEADERS}
        VERBATIM
        COMMENT "Compiling ${shader_source}"
    )
    list(APPEND RENDERER_SHADER_OUTPUTS "${shader_output}")
endforeach()

set(RENDERER_RAY_QUERY_SHADER_OUTPUT "${RENDERER_SHADER_OUTPUT_DIR}/mesh_ray_query.frag.spv")
add_custom_command(
    OUTPUT "${RENDERER_RAY_QUERY_SHADER_OUTPUT}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${RENDERER_SHADER_OUTPUT_DIR}"
    COMMAND "${GLSLC_EXECUTABLE}"
        --target-env=vulkan1.3
        -Werror
        -O
        -DRENDERER_RAY_QUERY_SHADOWS=1
        -I "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh.frag"
        -o "${RENDERER_RAY_QUERY_SHADER_OUTPUT}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/mesh.frag"
        ${RENDERER_SHADER_HEADERS}
    VERBATIM
    COMMENT "Compiling ray-query PBR fragment shader"
)
list(APPEND RENDERER_SHADER_OUTPUTS "${RENDERER_RAY_QUERY_SHADER_OUTPUT}")
add_custom_target(renderer_shaders DEPENDS ${RENDERER_SHADER_OUTPUTS})

set(GENESIS_RENDERER_SOURCES
    src/core/Log.cpp
    src/vulkan/Instance.cpp
    src/vulkan/Device.cpp
    src/vulkan/AccelerationStructure.cpp
    src/vulkan/Buffer.cpp
    src/vulkan/Image.cpp
    src/vulkan/Sampler.cpp
    src/vulkan/UploadContext.cpp
    src/vulkan/Shader.cpp
    src/vulkan/Descriptors.cpp
    src/vulkan/Pipeline.cpp
    src/vulkan/Swapchain.cpp
    src/render_graph/RenderGraph.cpp
    src/renderer/Renderer.cpp
    src/renderer/RendererEditorOverlay.cpp
    src/renderer/RendererPathDescriptors.cpp
    src/renderer/VirtualTexture.cpp
    src/renderer/RendererPathPasses.cpp
)
if(WIN32)
    list(APPEND GENESIS_RENDERER_SOURCES src/platform/Win32RenderSurface.cpp)
endif()

add_library(genesis_renderer STATIC ${GENESIS_RENDERER_SOURCES})
add_library(Genesis::Renderer ALIAS genesis_renderer)
if(RENDERER_ENABLE_STREAMLINE)
    target_sources(genesis_renderer PRIVATE src/nvidia/StreamlineRuntime.cpp)
endif()
add_dependencies(genesis_renderer renderer_shaders)

target_compile_features(genesis_renderer PUBLIC cxx_std_20)
target_include_directories(genesis_renderer PUBLIC include)
target_include_directories(genesis_renderer SYSTEM PRIVATE thirdparty/stb)
if(RENDERER_ENABLE_STREAMLINE)
    target_include_directories(genesis_renderer PRIVATE "${RENDERER_STREAMLINE_ROOT}/include")
endif()
target_link_libraries(genesis_renderer
    PUBLIC
        volk::volk
        glm::glm
    PRIVATE
    renderer_gltf
    renderer_ibl
    renderer_output
    renderer_gpu_scene
    renderer_post
    renderer_procedural
    renderer_ray_tracing
    renderer_temporal
    renderer_upscaling
    renderer_vma
)
if(RENDERER_ENABLE_NRD)
    target_link_libraries(genesis_renderer PRIVATE renderer_nrd)
    target_compile_definitions(genesis_renderer PUBLIC RENDERER_ENABLE_NRD=1)
else()
    target_compile_definitions(genesis_renderer PUBLIC RENDERER_ENABLE_NRD=0)
endif()
if(RENDERER_ENABLE_STREAMLINE)
    target_compile_definitions(genesis_renderer PUBLIC RENDERER_ENABLE_STREAMLINE=1)
    set_property(TARGET genesis_renderer PROPERTY GENESIS_STREAMLINE_RUNTIME_DIR
        "${RENDERER_STREAMLINE_ROOT}/bin/x64")
else()
    target_compile_definitions(genesis_renderer PUBLIC RENDERER_ENABLE_STREAMLINE=0)
endif()
target_compile_definitions(genesis_renderer PRIVATE
    RENDERER_SHADER_DIR="${RENDERER_SHADER_OUTPUT_DIR}"
    RENDERER_PIPELINE_CACHE_PATH="${CMAKE_BINARY_DIR}/cache/$<CONFIG>/pipeline.bin"
    RENDERER_MATERIAL_REFERENCE_DIR="${CMAKE_SOURCE_DIR}/reference/materials"
    RENDERER_BUILD_CONFIGURATION="$<CONFIG>"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(genesis_renderer PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
        # Vulkan structs are value-initialized; named members are then selected
        # explicitly. GCC otherwise warns for every intentionally-zero field.
        -Wno-missing-field-initializers
    )
endif()

set_target_properties(genesis_renderer PROPERTIES
    CXX_EXTENSIONS OFF
)

# Call this for any Windows executable that creates StreamlineRuntime. It keeps
# the SDL demo and external Qt hosts on the same signed NVIDIA runtime files.
function(genesis_copy_streamline_runtime target)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "genesis_copy_streamline_runtime: unknown target ${target}")
    endif()
    if(RENDERER_ENABLE_STREAMLINE)
        set(streamline_runtime_files
            sl.interposer.dll
            sl.common.dll
            sl.dlss.dll
            sl.dlss_d.dll
            nvngx_dlss.dll
            nvngx_dlssd.dll
            nvngx_dlss.license.txt
        )
        get_target_property(streamline_runtime_dir genesis_renderer
            GENESIS_STREAMLINE_RUNTIME_DIR)
        foreach(streamline_runtime_file IN LISTS streamline_runtime_files)
            add_custom_command(TARGET "${target}" POST_BUILD
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${streamline_runtime_dir}/${streamline_runtime_file}"
                    "$<TARGET_FILE_DIR:${target}>/${streamline_runtime_file}"
                VERBATIM
            )
        endforeach()
    endif()
endfunction()

if(GENESIS_BUILD_SDL_APP)
    add_executable(genesis_render
        src/main.cpp
        src/platform/Window.cpp
    )
    target_compile_features(genesis_render PRIVATE cxx_std_20)
    target_link_libraries(genesis_render PRIVATE Genesis::Renderer SDL3::SDL3-static)
    if(RENDERER_VALIDATION_DEFAULT)
        target_compile_definitions(genesis_render PRIVATE RENDERER_VALIDATION_DEFAULT=1)
    else()
        target_compile_definitions(genesis_render PRIVATE RENDERER_VALIDATION_DEFAULT=0)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(genesis_render PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
            -Wno-missing-field-initializers
        )
    endif()
    set_target_properties(genesis_render PROPERTIES
        CXX_EXTENSIONS OFF
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    genesis_copy_streamline_runtime(genesis_render)
endif()

include(CTest)
if(BUILD_TESTING AND GENESIS_BUILD_TESTS)
    if(GENESIS_BUILD_SDL_APP)
        add_test(NAME genesis_render_cli_help COMMAND genesis_render --help)
    endif()

    add_executable(aov_export_test tests/AovExportTest.cpp)
    target_compile_features(aov_export_test PRIVATE cxx_std_20)
    target_link_libraries(aov_export_test PRIVATE renderer_output)
    set_target_properties(aov_export_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(aov_export_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
    add_test(NAME aov_export COMMAND aov_export_test)

    add_executable(aov_export_metrics tests/AovExportMetrics.cpp)
    target_compile_features(aov_export_metrics PRIVATE cxx_std_20)
    set_target_properties(aov_export_metrics PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(aov_export_metrics PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()

    add_executable(shader_reflection_test
        tests/ShaderReflectionTest.cpp
        src/vulkan/Shader.cpp
    )
    add_dependencies(shader_reflection_test renderer_shaders)
    target_compile_features(shader_reflection_test PRIVATE cxx_std_20)
    target_include_directories(shader_reflection_test PRIVATE include)
    target_link_libraries(shader_reflection_test PRIVATE volk::volk)
    target_compile_definitions(shader_reflection_test PRIVATE
        RENDERER_SHADER_DIR="${RENDERER_SHADER_OUTPUT_DIR}"
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(shader_reflection_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
            -Wno-missing-field-initializers
        )
    endif()
    add_test(NAME shader_reflection COMMAND shader_reflection_test)

    add_executable(render_graph_test
        tests/RenderGraphTest.cpp
        src/core/Log.cpp
        src/vulkan/Buffer.cpp
        src/vulkan/Image.cpp
        src/render_graph/RenderGraph.cpp
    )
    target_compile_features(render_graph_test PRIVATE cxx_std_20)
    target_include_directories(render_graph_test PRIVATE include)
    target_link_libraries(render_graph_test PRIVATE renderer_vma volk::volk)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(render_graph_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
            -Wno-missing-field-initializers
        )
    endif()
    add_test(NAME render_graph COMMAND render_graph_test)

    add_executable(scene_ir_test tests/SceneTest.cpp)
    target_compile_features(scene_ir_test PRIVATE cxx_std_20)
    target_link_libraries(scene_ir_test PRIVATE renderer_scene)
    set_target_properties(scene_ir_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(scene_ir_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
            -Wno-missing-field-initializers
        )
    endif()
    add_test(NAME scene_ir COMMAND scene_ir_test)

    add_executable(gltf_importer_test tests/GltfImporterTest.cpp)
    target_compile_features(gltf_importer_test PRIVATE cxx_std_20)
    target_link_libraries(gltf_importer_test PRIVATE renderer_gltf)
    target_compile_definitions(gltf_importer_test PRIVATE
        RENDERER_GLTF_SAMPLE_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/reference/glTF-Sample-Assets"
    )
    set_target_properties(gltf_importer_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(gltf_importer_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
    add_test(NAME gltf_importer COMMAND gltf_importer_test)

    add_executable(pbr_math_test tests/PbrMathTest.cpp)
    target_compile_features(pbr_math_test PRIVATE cxx_std_20)
    target_link_libraries(pbr_math_test PRIVATE renderer_pbr)
    set_target_properties(pbr_math_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(pbr_math_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
    add_test(NAME pbr_math COMMAND pbr_math_test)

    add_executable(noise_volume_test tests/NoiseVolumeTest.cpp)
    target_compile_features(noise_volume_test PRIVATE cxx_std_20)
    target_link_libraries(noise_volume_test PRIVATE renderer_procedural)
    set_target_properties(noise_volume_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(noise_volume_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
    add_test(NAME noise_volume COMMAND noise_volume_test)

    add_executable(gltf_pbr_integration_test tests/GltfPbrIntegrationTest.cpp)
    target_compile_features(gltf_pbr_integration_test PRIVATE cxx_std_20)
    target_link_libraries(gltf_pbr_integration_test PRIVATE
        renderer_gltf
        renderer_gpu_scene
        renderer_pbr
    )
    target_compile_definitions(gltf_pbr_integration_test PRIVATE
        RENDERER_GLTF_SAMPLE_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/reference/glTF-Sample-Assets"
    )
    set_target_properties(gltf_pbr_integration_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(gltf_pbr_integration_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
    add_test(NAME gltf_pbr_integration COMMAND gltf_pbr_integration_test)

    add_executable(image_decoder_test tests/ImageDecoderTest.cpp)
    target_compile_features(image_decoder_test PRIVATE cxx_std_20)
    target_link_libraries(image_decoder_test PRIVATE renderer_images)
    target_compile_definitions(image_decoder_test PRIVATE
        RENDERER_GLTF_SAMPLE_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/reference/glTF-Sample-Assets"
    )
    set_target_properties(image_decoder_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(image_decoder_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
    add_test(NAME image_decoder COMMAND image_decoder_test)

    add_executable(ibl_baker_test tests/IblBakerTest.cpp)
    target_compile_features(ibl_baker_test PRIVATE cxx_std_20)
    target_link_libraries(ibl_baker_test PRIVATE renderer_ibl)
    set_target_properties(ibl_baker_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(ibl_baker_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
    add_test(NAME ibl_baker COMMAND ibl_baker_test)

    add_executable(gpu_scene_test tests/GpuSceneTest.cpp)
    target_compile_features(gpu_scene_test PRIVATE cxx_std_20)
    target_link_libraries(gpu_scene_test PRIVATE renderer_gpu_scene)
    set_target_properties(gpu_scene_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(gpu_scene_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
    add_test(NAME gpu_scene COMMAND gpu_scene_test)

    add_executable(gpu_scene_scale_test tests/GpuSceneScaleTest.cpp)
    target_compile_features(gpu_scene_scale_test PRIVATE cxx_std_20)
    target_link_libraries(gpu_scene_scale_test PRIVATE renderer_gpu_scene)
    set_target_properties(gpu_scene_scale_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(gpu_scene_scale_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
    add_test(NAME gpu_scene_scale COMMAND gpu_scene_scale_test)

    add_executable(visibility_test tests/VisibilityTest.cpp)
    target_compile_features(visibility_test PRIVATE cxx_std_20)
    target_link_libraries(visibility_test PRIVATE renderer_gpu_scene)
    set_target_properties(visibility_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(visibility_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
    add_test(NAME visibility_layers COMMAND visibility_test)

    add_executable(temporal_sampling_test tests/TemporalSamplingTest.cpp)
    target_compile_features(temporal_sampling_test PRIVATE cxx_std_20)
    target_link_libraries(temporal_sampling_test PRIVATE renderer_temporal)
    set_target_properties(temporal_sampling_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(temporal_sampling_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
    add_test(NAME temporal_sampling COMMAND temporal_sampling_test)

    add_executable(upscaling_test tests/UpscalingTest.cpp)
    target_compile_features(upscaling_test PRIVATE cxx_std_20)
    target_link_libraries(upscaling_test PRIVATE renderer_upscaling)
    set_target_properties(upscaling_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(upscaling_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
    add_test(NAME upscaling COMMAND upscaling_test)

    add_executable(post_processing_test tests/PostProcessingTest.cpp)
    target_compile_features(post_processing_test PRIVATE cxx_std_20)
    target_link_libraries(post_processing_test PRIVATE renderer_post)
    set_target_properties(post_processing_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(post_processing_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
    add_test(NAME post_processing COMMAND post_processing_test)

    add_executable(ray_tracing_policy_test tests/RayTracingPolicyTest.cpp)
    target_compile_features(ray_tracing_policy_test PRIVATE cxx_std_20)
    target_link_libraries(ray_tracing_policy_test PRIVATE renderer_ray_tracing)
    set_target_properties(ray_tracing_policy_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(ray_tracing_policy_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
    add_test(NAME ray_tracing_policy COMMAND ray_tracing_policy_test)

    add_executable(hardware_policy_test tests/HardwarePolicyTest.cpp)
    target_compile_features(hardware_policy_test PRIVATE cxx_std_20)
    target_include_directories(hardware_policy_test PRIVATE include)
    set_target_properties(hardware_policy_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(hardware_policy_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
    add_test(NAME hardware_policy COMMAND hardware_policy_test)

    if(RENDERER_ENABLE_NRD)
        add_executable(nrd_context_test tests/NrdContextTest.cpp)
        target_compile_features(nrd_context_test PRIVATE cxx_std_20)
        target_link_libraries(nrd_context_test PRIVATE renderer_nrd)
        set_target_properties(nrd_context_test PROPERTIES CXX_EXTENSIONS OFF)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(nrd_context_test PRIVATE
                -Wall
                -Wextra
                -Wpedantic
                -Wconversion
                -Wshadow
                -Wnon-virtual-dtor
            )
        endif()
        add_test(NAME nrd_context COMMAND nrd_context_test)
    endif()

    add_executable(acceleration_structure_plan_test tests/AccelerationStructurePlanTest.cpp)
    target_compile_features(acceleration_structure_plan_test PRIVATE cxx_std_20)
    target_link_libraries(acceleration_structure_plan_test PRIVATE renderer_ray_tracing)
    set_target_properties(acceleration_structure_plan_test PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(acceleration_structure_plan_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
    add_test(NAME acceleration_structure_plan COMMAND acceleration_structure_plan_test)

    add_executable(post_image_metrics tests/PostImageMetrics.cpp)
    target_compile_features(post_image_metrics PRIVATE cxx_std_20)
    target_link_libraries(post_image_metrics PRIVATE renderer_images)
    set_target_properties(post_image_metrics PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(post_image_metrics PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wshadow
            -Wnon-virtual-dtor
        )
    endif()
endif()
