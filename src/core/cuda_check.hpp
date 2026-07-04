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

// Kernel launch sonrasi hata kontrolu. WFE_SYNC=1 ortam degiskeniyle her
// kontrolde cudaDeviceSynchronize yapilir: asenkron hatalar (illegal access
// vb.) tam olarak hatali kernelde yakalanir (yavas; yalniz hata ayiklama).
inline void check_kernel(const char* name) {
  static const bool sync = [] {
    const char* e = std::getenv("WFE_SYNC");
    return e && e[0] == '1';
  }();
  if (sync) {
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(cudaStreamPerThread, &cap);
    if (cap == cudaStreamCaptureStatusNone) cudaDeviceSynchronize();
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA kernel error in %s: %s\n", name,
                 cudaGetErrorString(err));
    std::exit(1);
  }
}

} // namespace wfe
