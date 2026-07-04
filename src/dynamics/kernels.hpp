#pragma once

#include "core/base_state.hpp"
#include "core/field3d.hpp"
#include "core/grid.hpp"
#include "core/metric.hpp"
#include "dynamics/params.hpp"
#include "dynamics/state.hpp"

namespace wfe {

// Sinir kosullari: yanal periyodik veya acik (dp'ye gore), altta Omega=0
// (arazi boyunca akis; w(0) diagnostik), ustte rijit w=0.
void apply_bcs(const GDims& g, const DynParams& dp, State& s);

// Kutle akilari (hesaplama uzayinda, rho_b*J agirlikli):
//   mfx = (rho_b J)_u * u,  mfy = (rho_b J)_v * v,
//   mfz = rho_b^w * (w - u dz/dx - v dz/dy)   [= rho_b J Omega]
// ve bunlarin diverjansi D (adveksiyonun advektif-form duzeltmesi icin).
void compute_mass_fluxes(const GDims& g, const DevProf& p, const DevMetric& m,
                         const State& s, Field3D& mfx, Field3D& mfy, Field3D& mfz);
void compute_divergence(const GDims& g, const DevMetric& m, const Field3D& mfx,
                        const Field3D& mfy, const Field3D& mfz, Field3D& div);

// YAVAS egilimler: adveksiyon (kutle akilariyla) + difuzyon + Coriolis +
// Rayleigh. Hizli terimler akustik alt-adimlarda.
void compute_tendencies(const GDims& g, const DevProf& p, const DevMetric& m,
                        const DynParams& dp, real dt, const State& s,
                        const Field3D& mfx, const Field3D& mfy, const Field3D& mfz,
                        const Field3D& div, State& tend);

// Akustik cozucunun kosu boyunca sabit katsayi alanlari (Faz 6 performans):
// bir kez hesaplanir, her alt-adimda yeniden turetim yerine okunur.
struct AcousticCoef {
  Field3D rtjx, rtjy;  // rho_b thv_b J yuz ortalamalari (u/v yuzleri)
  Field3D kt3;         // Rd pib / (cv rho_b thv_b J)
  Field3D aw;          // rho_b^w thv_b^w (w seviyeleri)
};
void precompute_acoustic_coef(const GDims& g, const DevProf& p, const DevMetric& m,
                              AcousticCoef& ac, Field3D& scratch);

// Bir akustik alt-adim: u,v yatay explicit (capraz-terimli PGF); w,pi'
// dikey implicit; theta' stratifikasyon; yuzey w'si diagnostik.
// Nemli kaldirma acoustic dongude asama-sabit qv/qc/qr ile hesaplanir.
void acoustic_substep(const GDims& g, const DevProf& p, const DevMetric& m,
                      const DynParams& dp, real dtau, State& s, const State& tend,
                      Field3D& piprev, const AcousticCoef& ac);

// Nem skalarlarinin asama guncellemesi: out.q = s0.q + dt * tend.q
// (hizli terimleri olmadigindan akustik donguye girmezler).
void update_moisture_stage(const State& s0, const State& tend, real dt, State& out);

// Davies sinir relaksasyonu (u,v,thp,pip,qv): yavas egilimlere nudge ekler.
void bdy_relax(const GDims& g, const State& s, const Field3D lo[5], const Field3D hi[5],
               real tf, const Field3D& wgt, State& tend);

// Alan uzerinde max|f| (GPU reduce). NaN'lar cok buyuk deger olarak doner —
// patlama/NaN bekcisi icin ucuz kontrol.
float field_absmax(const Field3D& f);

} // namespace wfe
