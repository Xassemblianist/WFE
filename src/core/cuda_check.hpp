#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define WFE_CUDA_CHECK(call)                                                 \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                   \
                   cudaGetErrorName(err__), __FILE__, __LINE__,              \
                   cudaGetErrorString(err__));                               \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

namespace wfe {

// Kernel launch sonrasi hata kontrolu.
inline void check_kernel(const char* name) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA kernel error in %s: %s\n", name,
                 cudaGetErrorString(err));
    std::exit(1);
  }
}

} // namespace wfe
