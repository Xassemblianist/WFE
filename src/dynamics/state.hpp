#pragma once

#include "core/field3d.hpp"

namespace wfe {

// Prognostik degiskenler (C-grid):
//   u, v, w : ruzgar bilesenleri [m s-1] (yuzlerde; taban sapmasi degil, tam)
//   thp     : potansiyel sicaklik sapmasi theta' [K] (merkezlerde)
//   pip     : Exner fonksiyonu sapmasi pi' [-] (merkezlerde)
//   qv, qc, qr : su buhari / bulut suyu / yagmur karisim orani [kg/kg]
//                (tam alanlar; nem kapaliyken 0 kalirlar)
struct State {
  Field3D u, v, w, thp, pip, qv, qc, qr;

  void alloc(size_t n) {
    u.alloc(n);
    v.alloc(n);
    w.alloc(n);
    thp.alloc(n);
    pip.alloc(n);
    qv.alloc(n);
    qc.alloc(n);
    qr.alloc(n);
  }
  void copy_from(const State& o) {
    u.copy_from(o.u);
    v.copy_from(o.v);
    w.copy_from(o.w);
    thp.copy_from(o.thp);
    pip.copy_from(o.pip);
    qv.copy_from(o.qv);
    qc.copy_from(o.qc);
    qr.copy_from(o.qr);
  }
  void swap(State& o) {
    u.swap(o.u);
    v.swap(o.v);
    w.swap(o.w);
    thp.swap(o.thp);
    pip.swap(o.pip);
    qv.swap(o.qv);
    qc.swap(o.qc);
    qr.swap(o.qr);
  }
};

} // namespace wfe
