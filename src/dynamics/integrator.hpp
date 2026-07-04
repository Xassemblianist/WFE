#pragma once

#include "core/base_state.hpp"
#include "core/field3d.hpp"
#include "core/grid.hpp"
#include "core/metric.hpp"
#include "dynamics/params.hpp"
#include "dynamics/state.hpp"

namespace wfe {

// Wicker-Skamarock (2002) RK3 + Klemp-Wilhelmson split-explicit akustik
// alt-adimlama, Gal-Chen arazi-takip eden koordinatta. Her RK asamasi:
// yavas egilimler son tahminden, hizli sistem zaman-n'den akustik
// alt-adimlarla (dikey implicit). Alt-adim sayilari {1, ns/2, ns}.
class Integrator {
 public:
  void init(const GDims& g, const DevProf& prof, const DevMetric& m, const DynParams& dp);
  void step(real dt);

  State& state() { return s_n_; }
  const Field3D& rain() const { return rain_; }  // 2D birikmis yagis [mm]

 private:
  GDims g_{};
  DevProf prof_{};
  DevMetric m_{};
  DynParams dp_{};
  State s_n_, s_stage_, s_work_, tend_;
  Field3D div_, piprev_, mfx_, mfy_, mfz_, rain_;
};

} // namespace wfe
