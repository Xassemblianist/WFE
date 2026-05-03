cmake_minimum_required(VERSION 3.20)
project(CPPWRF LANGUAGES CXX CUDA)

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

set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -march=native -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "--use_fast_math -O3 -DNDEBUG")

# ─── Sources ──────────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/io/output.cpp
    src/solver/swe1d.cpp
    src/solver/cuda/swe1d_gpu.cu
)

add_executable(cppwrf ${SOURCES})

target_include_directories(cppwrf PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

# CUDA separable compilation for device-side function calls across .cu files
set_target_properties(cppwrf PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(cppwrf PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
    >
)
