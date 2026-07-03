#pragma once

#include "core/field3d.hpp"

namespace wfe {

// Prognostik degiskenler (taban durumundan sapmalar, C-grid):
//   u, v, w : ruzgar bilesenleri [m s-1] (yuzlerde)
//   thp     : potansiyel sicaklik sapmasi theta' [K] (merkezlerde)
//   pip     : Exner fonksiyonu sapmasi pi' [-] (merkezlerde)
struct State {
  Field3D u, v, w, thp, pip;

  void alloc(size_t n) {
    u.alloc(n);
    v.alloc(n);
    w.alloc(n);
    thp.alloc(n);
    pip.alloc(n);
  }
  void copy_from(const State& o) {
    u.copy_from(o.u);
    v.copy_from(o.v);
    w.copy_from(o.w);
    thp.copy_from(o.thp);
    pip.copy_from(o.pip);
  }
  void swap(State& o) {
    u.swap(o.u);
    v.swap(o.v);
    w.swap(o.w);
    thp.swap(o.thp);
    pip.swap(o.pip);
  }
};

} // namespace wfe
