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
                        const DynParams& dp, const State& s, const Field3D& mfx,
                        const Field3D& mfy, const Field3D& mfz, const Field3D& div,
                        State& tend);

// Bir akustik alt-adim: u,v yatay explicit (capraz-terimli PGF); w,pi'
// dikey implicit; theta' stratifikasyon; yuzey w'si diagnostik.
void acoustic_substep(const GDims& g, const DevProf& p, const DevMetric& m,
                      const DynParams& dp, real dtau, State& s, const State& tend,
                      Field3D& piprev);

} // namespace wfe
