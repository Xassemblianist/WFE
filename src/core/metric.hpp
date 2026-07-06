#pragma once

#include <vector>

#include "core/config.hpp"
#include "core/grid.hpp"
#include "core/precision.hpp"

namespace wfe {

// Gal-Chen (BTF) arazi-takip eden koordinat metrikleri.
//   zeta in [0, zt],  z(i,j,zeta) = h(i,j) + zeta * (zt - h(i,j)) / zt
//   Jacobian J = dz/dzeta = (zt - h)/zt   (Gal-Chen'de kolon icinde sabit)
//   dz/dx|_zeta = dh/dx * (1 - zeta/zt)
// Kernel'lere deger olarak gecirilen isaretciler:
struct DevMetric {
  const real* zeta_w;   // [NZ] w seviyelerinin zeta degeri (ham k indeksi)
  const real* zeta_c;   // [NZ] hucre merkezi zeta
  const real* dzeta_c;  // [NZ] zeta_w(k+1)-zeta_w(k): hucre kalinligi
  const real* dzeta_w;  // [NZ] zeta_c(k)-zeta_c(k-1): w seviyesi araligi
  const real* h;        // [NX*NY] arazi yuksekligi (merkez kolonlar, 2D)
  const real* hx;       // [NX*NY] dh/dx merkezlerde (2D)
  const real* hy;       // [NX*NY] dh/dy merkezlerde
  const real* hx_u;     // [NX*NY] dh/dx u noktalarinda
  const real* hy_v;     // [NX*NY] dh/dy v noktalarinda
  const real* jac;      // [NX*NY] J = (zt-h)/zt merkez kolonlarda
  const real* fcor;     // [NX*NY] Coriolis parametresi f(lat) [1/s]
  const real* mapf;     // [NX*NY] harita olcek faktoru m (izotropik konformal; idealize=1)
  real zt;              // model tepesi [m]
};

// 2D (yatay) padded indeks.
inline size_t idx2(const GDims& g, int i, int j) {
  return (size_t)(j + g.ng) * g.NX + (i + g.ng);
}

// terrain = none | schaer | gauss | file (h_file: interior ny*nx, prep'ten)
// stretch = none | geometric (dz0'dan dz_max'a buyuyen hucre kalinligi)
// fcor: sabit (cfg coriolis_f) veya dosyadan f(lat) (fcor_file, interior)
class Metric {
 public:
  void build(const GDims& g, const Config& cfg,
             const std::vector<real>* h_file = nullptr,
             const std::vector<real>* fcor_file = nullptr,
             const std::vector<real>* mapf_file = nullptr);
  void release();
  DevMetric dev() const;

  real zt() const { return zt_; }
  // Ana bilgisayar kopyalari (baslangic kosullari icin).
  std::vector<real> h_zeta_w, h_zeta_c, h_h, h_jac;
  real z_at(const GDims& g, int i, int j, real zeta) const {
    real hh = h_h[idx2(g, i, j)];
    return hh + zeta * (zt_ - hh) / zt_;
  }

 private:
  real* d_all_ = nullptr;
  size_t n1_ = 0, n2_ = 0;
  real zt_ = 0;
};

} // namespace wfe
