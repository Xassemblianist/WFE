#include "dynamics/integrator.hpp"

#include "dynamics/kernels.hpp"

namespace wfe {

void Integrator::init(const GDims& g, const DevProf& prof) {
  g_ = g;
  prof_ = prof;
  size_t n = g.npts();
  s_n_.alloc(n);
  s_stage_.alloc(n);
  tend_.alloc(n);
  div_.alloc(n);
}

void Integrator::step(real dt) {
  const real dts[3] = {dt / 3, dt / 2, dt};
  State* cur = &s_n_;
  for (int stage = 0; stage < 3; ++stage) {
    apply_bcs(g_, *cur);
    compute_divergence(g_, prof_, *cur, div_);
    compute_tendencies(g_, prof_, *cur, div_, tend_);
    update_state(s_n_, tend_, dts[stage], s_stage_);
    cur = &s_stage_;
  }
  s_n_.swap(s_stage_);
}

} // namespace wfe
