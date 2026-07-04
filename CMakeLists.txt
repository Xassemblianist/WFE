cmake_minimum_required(VERSION 3.24)
project(wfe LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

option(WFE_DOUBLE "Build with double precision (real = double)" OFF)

execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                OUTPUT_VARIABLE WFE_GIT_HASH
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT WFE_GIT_HASH)
  set(WFE_GIT_HASH "unknown")
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(wfe
  src/main.cpp
  src/core/config.cpp
  src/core/base_state.cpp
  src/core/metric.cpp
  src/dynamics/boundary.cpp
  src/dynamics/integrator.cpp
  src/dynamics/kernels.cu
  src/physics/kessler.cu
  src/physics/surface.cu
  src/io/input.cpp
  src/io/writer.cpp
)

target_include_directories(wfe PRIVATE src)
target_link_libraries(wfe PRIVATE CUDA::cudart)

if(WFE_DOUBLE)
  target_compile_definitions(wfe PRIVATE WFE_DOUBLE)
endif()
target_compile_definitions(wfe PRIVATE WFE_GIT_HASH="${WFE_GIT_HASH}")

if(MSVC)
  target_compile_options(wfe PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3>)
  target_compile_definitions(wfe PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
target_compile_options(wfe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
