#pragma once

#include "core/base_state.hpp"
#include "core/field3d.hpp"
#include "core/grid.hpp"
#include "core/metric.hpp"
#include "dynamics/state.hpp"
#include "io/input.hpp"

namespace wfe {

// Fizik v1 (Faz 4): her buyuk zaman adiminda operator bolmesiyle uygulanir.
//  - Yuzey katmani: Louis (1979) stabilite fonksiyonlari ile toplu (bulk)
//    aerodinamik surtunme + duyulur/gizli isi akilari.
//  - PBL/dusey karisim: Ri-bagimli yerel-K profili, kolon-implicit dusey
//    difuzyon (u, v, theta, qv) — kosulsuz kararli.
//  - Levha toprak (force-restore): T_sfc prognostik (karada); denizde SST sabit.
//  - Radyasyon zorlamasi: gunes geometrisi + bulut-zayiflatmali kisa dalga,
//    Brunt ampirik uzun dalga, troposferde sabit LW sogumasi (-2 K/gun).
class SfcPBL {
 public:
  void init(const GDims& g, const InputData& in, real start_hour_utc, int doy,
            bool nonlocal);
  void release();
  // t: adim SONUNDAKI simulasyon zamani [s] (gunes acisi icin)
  void step(const GDims& g, const DevProf& p, const DevMetric& m, real dt, real t,
            State& s);

  const Field3D& tsk() const { return tsk_; }
  const Field3D& pblh() const { return pblh_; }  // 2D PBL yuksekligi [m]

 private:
  Field3D km_;                              // momentum K, w seviyeleri [m2/s]
  Field3D tsk_, tdeep_, land_, lat_, lon_;  // 2D yuzey alanlari
  Field3D cdv_;                             // 2D Cd*|V1| (momentum icin)
  Field3D pblh_;                            // 2D PBL yuksekligi (nonlocal) [m]
  real start_hour_ = 0;
  int doy_ = 1;
  bool nonlocal_ = false;
};

} // namespace wfe
