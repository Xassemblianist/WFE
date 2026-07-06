#include "dynamics/boundary.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace wfe {

void BdyManager::init(const GDims& g, const DynParams& dp, const InputData* input,
                      const std::vector<real>& thb3, const std::vector<real>& pib3,
                      int bdy_width, real bdy_tau, real nudge_tau) {
  g_ = g;
  input_ = input;
  thb3_ = &thb3;
  pib3_ = &pib3;
  size_t n = g.npts();
  for (int v = 0; v < 5; ++v) {
    lo_[v].alloc(n);
    hi_[v].alloc(n);
  }
  wgt_.alloc((size_t)g.NX * g.NY);
  hbuf_.assign(n, 0);

  // 2D relaksasyon katsayisi: kenarda Davies (cos^2/bdy_tau), ic bolgede
  // zayif analiz-nudging tabani (1/nudge_tau; 0 kapali). GFS-guducu LAM'de
  // ic bolgenin buyuk-olcek surklenmesini kisitlar (WRF analiz nudging).
  real inner = (nudge_tau > (real)0) ? (real)1 / nudge_tau : (real)0;
  std::vector<real> w((size_t)g.NX * g.NY, 0);
  const real pi_half = (real)1.5707963267948966;
  for (int j = 0; j < g.ny; ++j)
    for (int i = 0; i < g.nx; ++i) {
      real wij = inner;
      int dist = 1 << 20;
      if (dp.bc_x_open) dist = std::min({dist, i, g.nx - 1 - i});
      if (dp.bc_y_open) dist = std::min({dist, j, g.ny - 1 - j});
      if (dist < bdy_width) {
        real s = std::cos(pi_half * (real)dist / (real)bdy_width);
        wij = std::max(wij, s * s / bdy_tau);
      }
      w[(size_t)(j + g.ng) * g.NX + (i + g.ng)] = wij;
    }
  wgt_.upload(w.data());

  load_into(0, lo_);
  load_into(std::min(1, input->n_bdy - 1), hi_);
  t_lo_ = 0;
  t_hi_ = input->bdy_interval;
  idx_hi_ = 1;
}

void BdyManager::load_into(int idx, Field3D* dst) {
  std::vector<real> f[5];
  if (!input_->load_bdy(g_, idx, f)) std::exit(1);
  auto src = [&](const std::vector<real>& a, int i, int j, int k) {
    return a[((size_t)k * g_.ny + j) * g_.nx + i];
  };
  for (int v = 0; v < 5; ++v) {
    std::fill(hbuf_.begin(), hbuf_.end(), (real)0);
    for (int k = 0; k < g_.nz; ++k)
      for (int j = 0; j < g_.ny; ++j)
        for (int i = 0; i < g_.nx + (v == 0 ? 1 : 0); ++i) {
          size_t c = g_.idx(i, j, k);
          real val;
          if (v == 0) {  // u: merkezlerden yuzlere
            int il = i > 0 ? i - 1 : 0, ir = i < g_.nx ? i : g_.nx - 1;
            val = (real)0.5 * (src(f[0], il, j, k) + src(f[0], ir, j, k));
          } else if (v == 1) {  // v: merkezlerden yuzlere (j araligi ny'de kalir)
            int jl = j > 0 ? j - 1 : 0;
            val = (real)0.5 * (src(f[1], i, jl, k) + src(f[1], i, j, k));
          } else if (v == 2) {  // theta -> theta'
            val = src(f[2], i, j, k) - (*thb3_)[c];
          } else if (v == 3) {  // pi -> pi'
            val = src(f[3], i, j, k) - (*pib3_)[c];
          } else {
            val = src(f[4], i, j, k);
          }
          hbuf_[c] = val;
        }
    dst[v].upload(hbuf_.data());
  }
}

void BdyManager::update(real t) {
  while (t >= t_hi_ && idx_hi_ + 1 < input_->n_bdy) {
    for (int v = 0; v < 5; ++v) lo_[v].swap(hi_[v]);
    ++idx_hi_;
    load_into(idx_hi_, hi_);
    t_lo_ = t_hi_;
    t_hi_ += input_->bdy_interval;
  }
}

real BdyManager::tfrac(real t) const {
  if (t_hi_ <= t_lo_) return 0;
  real f = (t - t_lo_) / (t_hi_ - t_lo_);
  return f < 0 ? 0 : (f > 1 ? (real)1 : f);
}

} // namespace wfe
