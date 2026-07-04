#pragma once

#include <vector>

#include "core/config.hpp"
#include "core/field3d.hpp"
#include "core/grid.hpp"
#include "core/metric.hpp"
#include "core/precision.hpp"

namespace wfe {

// Kernel'lere deger olarak gecirilen taban durumu isaretcileri.
// 3B alanlar GDims::idx ile indekslenir (arazi-takip eden gridde taban durumu
// yatayda da degisir); ub 1B'dir ([k+ng]).
struct DevProf {
  const real* thb;     // taban potansiyel sicaklik (kuru), hucre merkezi [K]
  const real* pib;     // taban Exner fonksiyonu, hucre merkezi [-]
  const real* rhob;    // taban yogunluk, hucre merkezi [kg m-3]
  const real* dthbdz;  // d(thb)/dz (fiziksel z), hucre merkezi [K m-1]
  const real* thbw;    // taban potansiyel sicaklik, w seviyeleri
  const real* rhobw;   // taban yogunluk, w seviyeleri
  const real* thvb;    // taban SANAL pot. sicaklik thb(1+0.61 qvb), merkez
  const real* thvbw;   // sanal pot. sicaklik, w seviyeleri
  const real* qvb;     // taban su buhari karisim orani [kg/kg], merkez
  const real* ub;      // taban ruzgari u bileseni [m s-1], 1B profil
};

// Hidrostatik dengede yatay-homojen (fiziksel z'de) taban durumu; grid
// noktalarina z(i,j,k) uzerinden degerlendirilir. Nemli profillerde denge
// sanal potansiyel sicaklikla kurulur (iteratif). (Sounding dosyasi Faz 3'te.)
// profile = isentropic | constant_N | wk82 | file (prep tablolarindan)
// file modunda tablolar (z, theta, qv, u alan-ortalama profilleri) verilir.
struct ProfileTables {
  const std::vector<real>* z = nullptr;
  const std::vector<real>* th = nullptr;
  const std::vector<real>* qv = nullptr;
  const std::vector<real>* u = nullptr;
};

class BaseState {
 public:
  void build(const GDims& g, const Config& cfg, const Metric& metric,
             const ProfileTables* tables = nullptr);
  void release();
  DevProf dev() const;

  std::vector<real> h_ub;   // 1B taban ruzgari (baslangic kosulu icin)
  std::vector<real> h_thb3, h_pib3;  // 3B host kopyalar (girdi/sinir donusumu)
  bool has_moisture() const { return has_moisture_; }
  // q̄v'yi grid noktasina degerlendir (baslangic kosulu icin)
  real qvb_at(const GDims& g, int i, int j, int k) const {
    return h_qvb_.empty() ? (real)0 : h_qvb_[g.idx(i, j, k)];
  }

 private:
  Field3D thb_, pib_, rhob_, dthbdz_, thbw_, rhobw_, thvb_, thvbw_, qvb_;
  real* d_ub_ = nullptr;
  std::vector<real> h_qvb_;
  bool has_moisture_ = false;
};

} // namespace wfe
