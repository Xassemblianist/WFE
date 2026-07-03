#include "core/base_state.hpp"

#include <cuda_runtime.h>

#include <cmath>

#include "core/constants.hpp"
#include "core/cuda_check.hpp"

namespace wfe {

void BaseState::build(const GDims& g, real theta0) {
  release();
  nz_ = (size_t)g.NZ;
  h_thb.assign(nz_, 0);
  h_pib.assign(nz_, 0);
  h_rhob.assign(nz_, 0);
  h_dthbdz.assign(nz_, 0);
  h_thbw.assign(nz_, 0);
  h_rhobw.assign(nz_, 0);

  // Izentropik atmosfer: pi(z) = 1 - g z / (cp theta0), analitik hidrostatik cozum.
  auto exner = [&](real z) { return (real)1 - phys::grav * z / (phys::cp * theta0); };
  auto rho = [&](real pi) {
    return phys::p00 * std::pow(pi, phys::cv / phys::Rd) / (phys::Rd * theta0);
  };

  for (int kk = 0; kk < (int)nz_; ++kk) {
    int k = kk - g.ng;                    // hucre indeksi (ghost'lar negatif/asiri olabilir)
    real zc = ((real)k + (real)0.5) * g.dz;  // hucre merkezi yuksekligi
    real zw = (real)k * g.dz;                // w seviyesi yuksekligi
    real pic = exner(zc);
    real piw = exner(zw);
    h_thb[kk] = theta0;
    h_pib[kk] = pic;
    h_rhob[kk] = rho(pic);
    h_dthbdz[kk] = 0;
    h_thbw[kk] = theta0;
    h_rhobw[kk] = rho(piw);
  }

  WFE_CUDA_CHECK(cudaMalloc(&d_all_, 6 * nz_ * sizeof(real)));
  const std::vector<real>* profs[6] = {&h_thb, &h_pib, &h_rhob, &h_dthbdz, &h_thbw, &h_rhobw};
  for (int p = 0; p < 6; ++p) {
    WFE_CUDA_CHECK(cudaMemcpy(d_all_ + p * nz_, profs[p]->data(), nz_ * sizeof(real),
                              cudaMemcpyHostToDevice));
  }
}

void BaseState::release() {
  if (d_all_) {
    cudaFree(d_all_);
    d_all_ = nullptr;
    nz_ = 0;
  }
}

DevProf BaseState::dev() const {
  return DevProf{d_all_,           d_all_ + nz_,     d_all_ + 2 * nz_,
                 d_all_ + 3 * nz_, d_all_ + 4 * nz_, d_all_ + 5 * nz_};
}

} // namespace wfe
