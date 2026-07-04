#include "physics/kessler.hpp"

#include "core/constants.hpp"
#include "core/cuda_check.hpp"

namespace wfe {
namespace {

__device__ __forceinline__ size_t gidx(const GDims& g, int i, int j, int k) {
  return ((size_t)(k + g.ng) * g.NY + (j + g.ng)) * g.NX + (i + g.ng);
}
__device__ __forceinline__ size_t g2(const GDims& g, int i, int j) {
  return (size_t)(j + g.ng) * g.NX + (i + g.ng);
}

// Tetens doygunluk karisim orani (p [Pa], T [K]).
__device__ __forceinline__ real qsat_of(real p, real T) {
  real es = (real)610.78 * exp((real)17.269 * (T - (real)273.16) / (T - (real)35.86));
  return (real)0.622 * es / (p - es);
}

constexpr int KES_MAX_NZ = 320;

// Kolon basina bir thread: sedimentasyon + mikrofizik + doygunluk ayarlamasi.
__global__ void k_kessler(GDims g, DevProf p, DevMetric m, real dt, real* thp,
                          const real* pip, real* qv, real* qc, real* qr, real* rain) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;

  const real J = m.jac[g2(g, i, j)];
  const real k1 = (real)0.001;      // otokonversiyon hizi [1/s]
  const real qc0 = (real)0.001;     // otokonversiyon esigi [kg/kg]

  real qrl[KES_MAX_NZ], rhol[KES_MAX_NZ], dzl[KES_MAX_NZ], vr[KES_MAX_NZ];

  // --- 1) sedimentasyon (birinci mertebe upwind, CFL alt-adimli) ---
  real rho_sfc = p.rhob[gidx(g, i, j, 0)];
  real vrmax = 0, dzmin = (real)1e30;
  for (int k = 0; k < g.nz; ++k) {
    int kk = k + g.ng;
    size_t c = gidx(g, i, j, k);
    qrl[k] = qr[c] > (real)0 ? qr[c] : (real)0;
    rhol[k] = p.rhob[c];
    dzl[k] = m.dzeta_c[kk] * J;
    real rq = rhol[k] * qrl[k];
    vr[k] = rq > (real)1e-9
                ? (real)36.34 * pow(rq, (real)0.1364) * sqrt(rho_sfc / rhol[k])
                : (real)0;
    if (vr[k] > vrmax) vrmax = vr[k];
    if (dzl[k] < dzmin) dzmin = dzl[k];
  }
  int nsub = 1 + (int)(vrmax * dt / ((real)0.8 * dzmin));
  real dts = dt / (real)nsub;
  real acc = 0;
  for (int s = 0; s < nsub; ++s) {
    real fb = rhol[0] * qrl[0] * vr[0];  // yuzeyden cikan aki [kg m-2 s-1]
    acc += dts * fb;
    for (int k = 0; k < g.nz; ++k) {
      real ftop = (k + 1 < g.nz) ? rhol[k + 1] * qrl[k + 1] * vr[k + 1] : (real)0;
      real fbot = rhol[k] * qrl[k] * vr[k];
      qrl[k] += dts * (ftop - fbot) / (rhol[k] * dzl[k]);
    }
    if (s + 1 < nsub) {
      for (int k = 0; k < g.nz; ++k) {
        real rq = rhol[k] * (qrl[k] > (real)0 ? qrl[k] : (real)0);
        vr[k] = rq > (real)1e-9
                    ? (real)36.34 * pow(rq, (real)0.1364) * sqrt(rho_sfc / rhol[k])
                    : (real)0;
      }
    }
  }
  rain[g2(g, i, j)] += acc;

  // --- 2+3) mikrofizik donusumleri + doygunluk ayarlamasi ---
  for (int k = 0; k < g.nz; ++k) {
    size_t c = gidx(g, i, j, k);
    real qvk = qv[c] > (real)0 ? qv[c] : (real)0;
    real qck = qc[c] > (real)0 ? qc[c] : (real)0;
    real qrk = qrl[k] > (real)0 ? qrl[k] : (real)0;
    real pi = p.pib[c] + pip[c];
    real th = p.thb[c] + thp[c];
    real T = th * pi;
    real pres = phys::p00 * pow(pi, phys::cp / phys::Rd);
    real qs = qsat_of(pres, T);

    // otokonversiyon + akresyon (qc -> qr)
    real dqauto = k1 * (qck > qc0 ? qck - qc0 : (real)0);
    real dqacc = (real)2.2 * qck * pow(qrk > (real)1e-9 ? qrk : (real)1e-9, (real)0.875);
    real dqr = dt * (dqauto + dqacc);
    if (dqr > qck) dqr = qck;
    qck -= dqr;
    qrk += dqr;

    // yagmur buharlasmasi (doymamis havada, qr -> qv; sogutma)
    if (qvk < qs && qrk > (real)0) {
      real rq = rhol[k] * qrk;
      real vent = (real)1.6 + (real)124.9 * pow(rq, (real)0.2046);
      real ern = vent * pow(rq, (real)0.525) /
                 ((real)2.55e8 / (pres * qs) + (real)5.4e5) *
                 (qs - qvk) / (rhol[k] * qs);
      real dqe = dt * ern;
      if (dqe > qrk) dqe = qrk;
      if (dqe > qs - qvk) dqe = qs - qvk;
      qrk -= dqe;
      qvk += dqe;
      thp[c] -= phys::Lv * dqe / (phys::cp * pi);
      th = p.thb[c] + thp[c];
      T = th * pi;
      qs = qsat_of(pres, T);
    }

    // doygunluk ayarlamasi (tek Newton adimi; qv <-> qc, gizli isi)
    real gam = phys::Lv * phys::Lv * qs / (phys::cp * phys::Rv * T * T);
    real dq = (qvk - qs) / ((real)1 + gam);
    if (dq < (real)0) {
      real evap = -dq;
      if (evap > qck) evap = qck;
      dq = -evap;
    }
    qvk -= dq;
    qck += dq;
    thp[c] += phys::Lv * dq / (phys::cp * pi);

    qv[c] = qvk > (real)0 ? qvk : (real)0;
    qc[c] = qck > (real)0 ? qck : (real)0;
    qr[c] = qrk > (real)0 ? qrk : (real)0;
  }
}

} // namespace

void kessler_step(const GDims& g, const DevProf& p, const DevMetric& m, real dt,
                  State& s, Field3D& rain) {
  dim3 b(32, 8);
  dim3 gr((g.nx + 31) / 32, (g.ny + 7) / 8);
  k_kessler<<<gr, b>>>(g, p, m, dt, s.thp.d, s.pip.d, s.qv.d, s.qc.d, s.qr.d, rain.d);
  check_kernel("kessler_step");
}

} // namespace wfe
