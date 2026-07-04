#include "dynamics/integrator.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "core/cuda_check.hpp"
#include "dynamics/kernels.hpp"
#include "physics/kessler.hpp"
#include "physics/surface.hpp"

namespace wfe {

namespace {
// WFE_PROF=1: bolum bazli GPU zamanlamasi (senkronizasyonlu, sadece profil modunda)
enum Section { P_BC, P_MF, P_TEND, P_BDY, P_ACOU, P_MOIST, P_KESSLER, P_PHYS };
const char* kSectionName[8] = {"sinir kosullari", "kutle akisi+div", "yavas tendency",
                               "sinir relaks",    "akustik dongu",   "nem asama",
                               "kessler",         "yuzey/PBL"};

struct Tic {
  bool on;
  std::chrono::steady_clock::time_point t0;
  explicit Tic(bool enabled) : on(enabled) {
    if (on) {
      cudaDeviceSynchronize();
      t0 = std::chrono::steady_clock::now();
    }
  }
  void toc(double* acc) {
    if (on) {
      cudaDeviceSynchronize();
      *acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      t0 = std::chrono::steady_clock::now();
    }
  }
};
} // namespace

void Integrator::init(const GDims& g, const DevProf& prof, const DevMetric& m,
                      const DynParams& dp) {
  g_ = g;
  prof_ = prof;
  m_ = m;
  dp_ = dp;
  if (g.nz + 1 > MAX_COLUMN_LEVELS) {  // kolon cozuculerin yerel dizi siniri
    std::fprintf(stderr, "nz=%d cok buyuk (kolon cozucu siniri %d)\n", g.nz,
                 MAX_COLUMN_LEVELS - 1);
    std::exit(1);
  }
  const char* e = std::getenv("WFE_PROF");
  prof_on_ = e && e[0] == '1';
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
  rain_.alloc((size_t)g.NX * g.NY);
  precompute_acoustic_coef(g_, prof_, m_, acoef_, div_);  // div_ gecici tampon
}

void Integrator::step(real dt, real t) {
  const real stage_dt[3] = {dt / 3, dt / 2, dt};
  const int ns = std::max(1, dp_.acoustic_ns);
  const int ns_stage[3] = {1, std::max(1, ns / 2), ns};

  if (bdy_ && bdy_->active()) bdy_->update(t);

  for (int m = 0; m < 3; ++m) {
    State& est = (m == 0) ? s_n_ : s_stage_;  // yavas egilimlerin kaynagi
    Tic tic(prof_on_);
    apply_bcs(g_, dp_, est);
    tic.toc(&ptimes_[P_BC]);
    compute_mass_fluxes(g_, prof_, m_, est, mfx_, mfy_, mfz_);
    compute_divergence(g_, m_, mfx_, mfy_, mfz_, div_);
    tic.toc(&ptimes_[P_MF]);
    compute_tendencies(g_, prof_, m_, dp_, dt, est, mfx_, mfy_, mfz_, div_, tend_);
    tic.toc(&ptimes_[P_TEND]);
    if (bdy_ && bdy_->active())
      bdy_relax(g_, est, bdy_->lo_, bdy_->hi_, bdy_->tfrac(t), bdy_->wgt_, tend_);
    tic.toc(&ptimes_[P_BDY]);

    s_work_.copy_from(s_n_);  // hizli sistem her asamada zaman-n'den baslar
    piprev_.copy_from(s_n_.pip);
    real dtau = stage_dt[m] / ns_stage[m];
    // NOT: CUDA Graph denendi ve geri alindi — State::swap tampon rotasyonuyla
    // uyumsuz (sabit pointer yakalar) ve olcumde kazanc sifirdi (bkz. ROADMAP).
    for (int t = 0; t < ns_stage[m]; ++t)
      acoustic_substep(g_, prof_, m_, dp_, dtau, s_work_, tend_, piprev_, acoef_);
    tic.toc(&ptimes_[P_ACOU]);
    if (dp_.moisture) update_moisture_stage(s_n_, tend_, stage_dt[m], s_work_);
    tic.toc(&ptimes_[P_MOIST]);
    s_stage_.swap(s_work_);
  }
  s_n_.swap(s_stage_);
  Tic tic(prof_on_);
  if (dp_.moisture) kessler_step(g_, prof_, m_, dt, s_n_, rain_);
  tic.toc(&ptimes_[P_KESSLER]);
  if (phys_) phys_->step(g_, prof_, m_, dt, t + dt, s_n_);
  tic.toc(&ptimes_[P_PHYS]);
}

void Integrator::print_profile() const {
  if (!prof_on_) return;
  double tot = 0;
  for (double v : ptimes_) tot += v;
  std::printf("\nprofil (WFE_PROF=1, toplam %.2fs GPU-senkron):\n", tot);
  for (int i = 0; i < 8; ++i)
    if (ptimes_[i] > 0)
      std::printf("  %-16s %7.2fs  %5.1f%%\n", kSectionName[i], ptimes_[i],
                  100.0 * ptimes_[i] / tot);
}

} // namespace wfe
