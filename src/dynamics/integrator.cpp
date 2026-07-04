#include "dynamics/integrator.hpp"

#include <algorithm>
#include <cstdio>

#include "dynamics/kernels.hpp"

namespace wfe {

void Integrator::init(const GDims& g, const DevProf& prof, const DevMetric& m,
                      const DynParams& dp) {
  g_ = g;
  prof_ = prof;
  m_ = m;
  dp_ = dp;
  if (g.nz + 1 > 320) {  // k_acou_wpi kolon cozucusunun yerel dizi siniri
    std::fprintf(stderr, "nz=%d cok buyuk (kolon cozucu siniri 319)\n", g.nz);
    std::exit(1);
  }
  size_t n = g.npts();
  s_n_.alloc(n);
  s_stage_.alloc(n);
  s_work_.alloc(n);
  tend_.alloc(n);
  div_.alloc(n);
  piprev_.alloc(n);
  mfx_.alloc(n);
  mfy_.alloc(n);
  mfz_.alloc(n);
}

void Integrator::step(real dt) {
  const real stage_dt[3] = {dt / 3, dt / 2, dt};
  const int ns = std::max(1, dp_.acoustic_ns);
  const int ns_stage[3] = {1, std::max(1, ns / 2), ns};

  for (int m = 0; m < 3; ++m) {
    State& est = (m == 0) ? s_n_ : s_stage_;  // yavas egilimlerin kaynagi
    apply_bcs(g_, dp_, est);
    compute_mass_fluxes(g_, prof_, m_, est, mfx_, mfy_, mfz_);
    compute_divergence(g_, m_, mfx_, mfy_, mfz_, div_);
    compute_tendencies(g_, prof_, m_, dp_, est, mfx_, mfy_, mfz_, div_, tend_);

    s_work_.copy_from(s_n_);  // hizli sistem her asamada zaman-n'den baslar
    piprev_.copy_from(s_n_.pip);
    real dtau = stage_dt[m] / ns_stage[m];
    for (int t = 0; t < ns_stage[m]; ++t)
      acoustic_substep(g_, prof_, m_, dp_, dtau, s_work_, tend_, piprev_);
    s_stage_.swap(s_work_);
  }
  s_n_.swap(s_stage_);
}

} // namespace wfe
