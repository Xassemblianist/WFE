#pragma once

#include "core/base_state.hpp"
#include "core/field3d.hpp"
#include "core/grid.hpp"
#include "core/metric.hpp"
#include "dynamics/state.hpp"

namespace wfe {

// Kessler (1969) warm-rain mikrofizigi, zaman adimi sonunda operator
// bolmesiyle uygulanir (WRF ile ayni yaklasim):
//   1) yagmur sedimentasyonu (kolon upwind, CFL alt-adimli) + yuzey birikimi
//   2) otokonversiyon (qc->qr), akresyon (qc->qr), yagmur buharlasmasi (qr->qv)
//   3) doygunluk ayarlamasi (qv<->qc, gizli isi -> theta')
// rain: [NX*NY] 2D birikmis yagis [kg m-2 = mm].
void kessler_step(const GDims& g, const DevProf& p, const DevMetric& m, real dt,
                  State& s, Field3D& rain);

} // namespace wfe
