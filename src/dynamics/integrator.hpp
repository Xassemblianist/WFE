#pragma once

#include "core/base_state.hpp"
#include "core/field3d.hpp"
#include "core/grid.hpp"
#include "dynamics/state.hpp"

namespace wfe {

// Wicker-Skamarock (2002) RK3 zaman entegratoru. Su an tamamen explicit
// (akustik modlar dahil) — dt ses hizi CFL'iyle sinirli. Faz 1'de
// split-explicit akustik alt-adimlama eklenecek (docs/ROADMAP.md).
class Integrator {
 public:
  void init(const GDims& g, const DevProf& prof);
  void step(real dt);

  State& state() { return s_n_; }

 private:
  GDims g_{};
  DevProf prof_{};
  State s_n_, s_stage_, tend_;
  Field3D div_;
};

} // namespace wfe
