#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

#include "core/cuda_check.hpp"
#include "core/precision.hpp"

namespace wfe {

// GPU uzerinde tek bir 3B alan. Sahiplik tekil; kopya yasak, swap ucuz.
struct Field3D {
  real* d = nullptr;
  size_t n = 0;

  Field3D() = default;
  Field3D(const Field3D&) = delete;
  Field3D& operator=(const Field3D&) = delete;
  ~Field3D() { release(); }

  void alloc(size_t count) {
    release();
    n = count;
    WFE_CUDA_CHECK(cudaMalloc(&d, n * sizeof(real)));
    zero();
  }
  void zero() {
    if (d) WFE_CUDA_CHECK(cudaMemset(d, 0, n * sizeof(real)));
  }
  void release() {
    if (d) {
      cudaFree(d);
      d = nullptr;
      n = 0;
    }
  }
  void copy_from(const Field3D& o) {
    WFE_CUDA_CHECK(cudaMemcpy(d, o.d, n * sizeof(real), cudaMemcpyDeviceToDevice));
  }
  void upload(const real* h) {
    WFE_CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(real), cudaMemcpyHostToDevice));
  }
  void download(real* h) const {
    WFE_CUDA_CHECK(cudaMemcpy(h, d, n * sizeof(real), cudaMemcpyDeviceToHost));
  }
  void swap(Field3D& o) {
    std::swap(d, o.d);
    std::swap(n, o.n);
  }
};

} // namespace wfe
