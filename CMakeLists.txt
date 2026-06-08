cmake_minimum_required(VERSION 3.20)
project(WFE LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─── GPU targets ──────────────────────────────────────────────────────────────
# sm_75  = Turing  (RTX 2060)
# sm_86  = Ampere  (RTX 3070/3080)
# sm_89  = Ada     (RTX 4090)
# sm_120 = Blackwell (RTX 5070 Ti — primary target)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # sm_120 (Blackwell/RTX5070Ti) requires CUDA 12.8+; add it on the main PC
    set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
endif()

# ─── Build type ───────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE  "/O2 /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -march=native -DNDEBUG")
endif()
set(CMAKE_CUDA_FLAGS_RELEASE "--use_fast_math -O3 -DNDEBUG")

# ─── Sources ──────────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/io/output.cpp
    src/solver/swe1d.cpp
    src/solver/cuda/swe1d_gpu.cu
)

add_executable(wfe ${SOURCES})

target_include_directories(wfe PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

# CUDA separable compilation for device-side function calls across .cu files
set_target_properties(wfe PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(wfe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
    >
)

# ─── 2D Density Current ───────────────────────────────────────────────────────
add_executable(wfe2d
    src/main2d.cpp
    src/solver/cuda/euler2d_gpu.cu
)

target_include_directories(wfe2d PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(wfe2d PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(wfe2d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
    >
)

# ─── 2D Mountain Wave ─────────────────────────────────────────────────────────
add_executable(wfe_mw
    src/main_mw.cpp
    src/solver/cuda/euler2d_gpu.cu
)

target_include_directories(wfe_mw PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(wfe_mw PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(wfe_mw PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
    >
)

# ─── 3D Warm Bubble ───────────────────────────────────────────────────────────
add_executable(wfe3d
    src/main3d.cpp
    src/solver/cuda/euler3d_gpu.cu
)

target_include_directories(wfe3d PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(wfe3d PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(wfe3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
    >
)

