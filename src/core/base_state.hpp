#pragma once

#include <vector>

#include "core/grid.hpp"
#include "core/precision.hpp"

namespace wfe {

// Kernel'lere deger olarak gecirilen taban durumu profil isaretcileri.
// Tum diziler NZ boyutlu, ham (ghost dahil) k indeksiyle erisilir: [k + ng].
struct DevProf {
  const real* thb;     // taban potansiyel sicaklik, hucre merkezi [K]
  const real* pib;     // taban Exner fonksiyonu, hucre merkezi [-]
  const real* rhob;    // taban yogunluk, hucre merkezi [kg m-3]
  const real* dthbdz;  // d(thb)/dz, hucre merkezi [K m-1]
  const real* thbw;    // taban potansiyel sicaklik, w seviyeleri
  const real* rhobw;   // taban yogunluk, w seviyeleri
};

// Hidrostatik dengede yatay-homojen taban durumu. Simdilik izentropik
// (sabit theta0); ileride sounding dosyasindan genel profil okunacak.
class BaseState {
 public:
  void build(const GDims& g, real theta0);
  void release();
  DevProf dev() const;

  // Ana bilgisayar kopyalari (baslangic kosullari ve tanilar icin).
  std::vector<real> h_thb, h_pib, h_rhob, h_dthbdz, h_thbw, h_rhobw;

 private:
  real* d_all_ = nullptr;  // tek cudaMalloc, 6 profil arka arkaya
  size_t nz_ = 0;
};

} // namespace wfe
