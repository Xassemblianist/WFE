#pragma once

#include "core/base_state.hpp"
#include "core/field3d.hpp"
#include "core/grid.hpp"
#include "dynamics/state.hpp"

namespace wfe {

// Sinir kosullari: x/y periyodik, altta/ustte rijit serbest-kayma sinir
// (w=0, diger alanlar sifir-gradyan ghost).
void apply_bcs(const GDims& g, State& s);

// rho_b * V ruzgar alaninin diverjansi, hucre merkezlerinde (adveksiyonun
// advektif-form duzeltmesi icin).
void compute_divergence(const GDims& g, const DevProf& p, const State& s, Field3D& div);

// Tum prognostik degiskenlerin toplam egilimleri (adveksiyon + basinc
// gradyani + kaldirma + surekllilik).
void compute_tendencies(const GDims& g, const DevProf& p, const State& s,
                        const Field3D& div, State& tend);

// out = s0 + dt * tend (tum tampon uzerinde; tend ghost'lari hep sifir).
void update_state(const State& s0, const State& tend, real dt, State& out);

} // namespace wfe
