#pragma once

#include "core/precision.hpp"

namespace wfe {

// Dinamik cekirdegin config'den gelen parametreleri (bkz. cases/*.ini).
struct DynParams {
  real diff_K = 0;          // sabit yayilim katsayisi [m2 s-1] (0 = kapali)
  real coriolis_f = 0;      // f-plane parametresi [s-1] (0 = kapali; dosyada f(lat))
  bool coriolis_use_ub = true;  // idealize: f(u-ub); gercek veri: tam ruzgar
  real rayleigh_zd = -1;    // Rayleigh katmani taban yuksekligi [m] (<0 = kapali)
  real rayleigh_alpha = 0;  // maksimum sonumleme katsayisi [s-1]
  int acoustic_ns = 6;      // buyuk adim basina akustik alt-adim sayisi
  real acoustic_beta = (real)0.2;    // dikey implicit off-centering
  real acoustic_smdiv = (real)0.1;   // diverjans sonumleme (pi' ileri agirliklama)
  bool bc_x_open = false;   // x siniri: false=periyodik, true=acik/radyasyon
  bool bc_y_open = false;
  real cstar = 30;          // radyasyon BC faz hizi [m s-1] (Klemp-Wilhelmson)
  bool moisture = false;    // qv/qc/qr prognostikleri + Kessler mikrofizigi
};

} // namespace wfe
