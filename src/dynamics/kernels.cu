#include "dynamics/kernels.hpp"

#include <cstring>

#include "core/constants.hpp"
#include "core/cuda_check.hpp"
#include "dynamics/params.hpp"

// Ayriklastirma notlari (ayrinti: docs/EQUATIONS.md):
//  - Gal-Chen arazi-takip eden zeta koordinati; adveksiyon hesaplama uzayinda
//    kutle akilariyla (mfx, mfy, mfz) akı formunda, advektif-tutarli:
//    tend = -(1/rho~)[ div(MF q) - q div(MF) ],  rho~ = rho_b J.
//  - 5. mertebe upwind arayuz degerleri (WS2002), RK3 ile eslesir.
//  - Akustik alt-adimlar: yatayda forward-backward (pi* diverjans sonumlemeli,
//    capraz metrik terimi explicit), dikeyde off-centered implicit w-pi'
//    tridiagonal cozucu (kolon basina bir thread).
//  - pi' adveksiyonu ihmal (Klemp-Wilhelmson).

namespace wfe {
namespace {

struct DevState {
  const real* u;
  const real* v;
  const real* w;
  const real* thp;
  const real* pip;
  const real* qv;
  const real* qc;
  const real* qr;
};

__device__ __forceinline__ size_t gidx(const GDims& g, int i, int j, int k) {
  return ((size_t)(k + g.ng) * g.NY + (j + g.ng)) * g.NX + (i + g.ng);
}
__device__ __forceinline__ size_t g2(const GDims& g, int i, int j) {
  return (size_t)(j + g.ng) * g.NX + (i + g.ng);
}

// Ust Rayleigh sonumleme katsayisi alpha(zeta) (kapaliysa 0). Sonumleme
// katmani duz model tepesine yakin oldugu icin zeta ~ z kabul edilir.
__device__ __forceinline__ real ray_alpha(real zeta, real zt, const DynParams& dp) {
  if (dp.rayleigh_zd < (real)0) return 0;
  if (zeta <= dp.rayleigh_zd || zt <= dp.rayleigh_zd) return 0;
  const real halfpi = (real)1.5707963267948966;
  real s = sin(halfpi * (zeta - dp.rayleigh_zd) / (zt - dp.rayleigh_zd));
  return dp.rayleigh_alpha * s * s;
}

// q0 (alt) ile q1 (ust) arasindaki yuzeyde 5. mertebe upwind arayuz degeri.
__device__ __forceinline__ real iface5(real qm2, real qm1, real q0, real q1,
                                       real q2, real q3, real vel) {
  const real c = (real)(1.0 / 60.0);
  real cen = c * ((real)37 * (q0 + q1) - (real)8 * (qm1 + q2) + (qm2 + q3));
  real upw = c * ((q3 - qm2) - (real)5 * (q2 - qm1) + (real)10 * (q1 - q0));
  return (vel >= (real)0) ? (cen - upw) : (cen + upw);
}

#define WFE_IJK_GUARD(imax, jmax, kmin, kmax)                        \
  int i = blockIdx.x * blockDim.x + threadIdx.x;                     \
  int j = blockIdx.y * blockDim.y + threadIdx.y;                     \
  int k = (int)(blockIdx.z * blockDim.z + threadIdx.z) + (kmin);     \
  if (i >= (imax) || j >= (jmax) || k >= (kmax)) return;

// ------------------------------------------------------------ kutle akilari

// Genisletilmis halka (+-2) uzerinde hesaplanir; tuketiciler ghost'suz okur.
__global__ void k_massflux_xy(GDims g, DevProf p, DevMetric m, DevState s, real* mfx,
                              real* mfy) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x) - 2;
  int j = (int)(blockIdx.y * blockDim.y + threadIdx.y) - 2;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i > g.nx + 2 || j > g.ny + 2 || k >= g.nz) return;
  size_t c = gidx(g, i, j, k);
  real rjx = (real)0.5 * (p.rhob[gidx(g, i - 1, j, k)] * m.jac[g2(g, i - 1, j)] +
                          p.rhob[c] * m.jac[g2(g, i, j)]);
  mfx[c] = rjx * s.u[c];
  real rjy = (real)0.5 * (p.rhob[gidx(g, i, j - 1, k)] * m.jac[g2(g, i, j - 1)] +
                          p.rhob[c] * m.jac[g2(g, i, j)]);
  mfy[c] = rjy * s.v[c];
}

__global__ void k_massflux_z(GDims g, DevProf p, DevMetric m, DevState s, real* mfz) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x) - 2;
  int j = (int)(blockIdx.y * blockDim.y + threadIdx.y) - 2;
  int k = (int)(blockIdx.z * blockDim.z + threadIdx.z) + 1;
  if (i > g.nx + 1 || j > g.ny + 1 || k >= g.nz) return;  // k=0, nz: Omega=0
  int kk = k + g.ng;
  size_t c = gidx(g, i, j, k);
  real fac = (real)1 - m.zeta_w[kk] / m.zt;
  real uw = (real)0.25 * (s.u[gidx(g, i, j, k - 1)] + s.u[gidx(g, i + 1, j, k - 1)] +
                          s.u[gidx(g, i, j, k)] + s.u[gidx(g, i + 1, j, k)]);
  real vw = (real)0.25 * (s.v[gidx(g, i, j, k - 1)] + s.v[gidx(g, i, j + 1, k - 1)] +
                          s.v[gidx(g, i, j, k)] + s.v[gidx(g, i, j + 1, k)]);
  real cross = (uw * m.hx[g2(g, i, j)] + vw * m.hy[g2(g, i, j)]) * fac;
  mfz[c] = p.rhobw[c] * (s.w[c] - cross);
}

__global__ void k_divergence(GDims g, DevMetric m, const real* mfx, const real* mfy,
                             const real* mfz, real* div) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x) - 1;
  int j = (int)(blockIdx.y * blockDim.y + threadIdx.y) - 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i > g.nx || j > g.ny || k >= g.nz) return;
  int kk = k + g.ng;
  div[gidx(g, i, j, k)] =
      (mfx[gidx(g, i + 1, j, k)] - mfx[gidx(g, i, j, k)]) / g.dx +
      (mfy[gidx(g, i, j + 1, k)] - mfy[gidx(g, i, j, k)]) / g.dy +
      (mfz[gidx(g, i, j, k + 1)] - mfz[gidx(g, i, j, k)]) / m.dzeta_c[kk];
}

// ------------------------------------------------------------- skaler theta'

__global__ void k_tend_thp(GDims g, DevProf p, DevMetric m, DynParams dp, DevState s,
                           const real* mfx, const real* mfy, const real* mfz,
                           const real* div, real* tend) {
  WFE_IJK_GUARD(g.nx, g.ny, 0, g.nz)
  int kk = k + g.ng;
  const real* q = s.thp;
  auto Q = [&](int ii, int jj, int kz) { return q[gidx(g, ii, jj, kz)]; };

  real fxL = mfx[gidx(g, i, j, k)];
  real fxR = mfx[gidx(g, i + 1, j, k)];
  real fyL = mfy[gidx(g, i, j, k)];
  real fyR = mfy[gidx(g, i, j + 1, k)];
  real fzB = mfz[gidx(g, i, j, k)];
  real fzT = mfz[gidx(g, i, j, k + 1)];

  real FxL = fxL * iface5(Q(i - 3, j, k), Q(i - 2, j, k), Q(i - 1, j, k), Q(i, j, k),
                          Q(i + 1, j, k), Q(i + 2, j, k), fxL);
  real FxR = fxR * iface5(Q(i - 2, j, k), Q(i - 1, j, k), Q(i, j, k), Q(i + 1, j, k),
                          Q(i + 2, j, k), Q(i + 3, j, k), fxR);
  real FyL = fyL * iface5(Q(i, j - 3, k), Q(i, j - 2, k), Q(i, j - 1, k), Q(i, j, k),
                          Q(i, j + 1, k), Q(i, j + 2, k), fyL);
  real FyR = fyR * iface5(Q(i, j - 2, k), Q(i, j - 1, k), Q(i, j, k), Q(i, j + 1, k),
                          Q(i, j + 2, k), Q(i, j + 3, k), fyR);
  real FzB = fzB * iface5(Q(i, j, k - 3), Q(i, j, k - 2), Q(i, j, k - 1), Q(i, j, k),
                          Q(i, j, k + 1), Q(i, j, k + 2), fzB);
  real FzT = fzT * iface5(Q(i, j, k - 2), Q(i, j, k - 1), Q(i, j, k), Q(i, j, k + 1),
                          Q(i, j, k + 2), Q(i, j, k + 3), fzT);

  real fdiv = (FxR - FxL) / g.dx + (FyR - FyL) / g.dy + (FzT - FzB) / m.dzeta_c[kk];
  real rt = p.rhob[gidx(g, i, j, k)] * m.jac[g2(g, i, j)];
  tend[gidx(g, i, j, k)] = (-fdiv + Q(i, j, k) * div[gidx(g, i, j, k)]) / rt -
                           ray_alpha(m.zeta_c[kk], m.zt, dp) * Q(i, j, k);
}

// Genel skaler adveksiyon (nem turleri): thp ile ayni sema, Rayleigh'siz.
__global__ void k_tend_q(GDims g, DevProf p, DevMetric m, const real* q,
                         const real* mfx, const real* mfy, const real* mfz,
                         const real* div, real* tend) {
  WFE_IJK_GUARD(g.nx, g.ny, 0, g.nz)
  int kk = k + g.ng;
  auto Q = [&](int ii, int jj, int kz) { return q[gidx(g, ii, jj, kz)]; };

  real fxL = mfx[gidx(g, i, j, k)];
  real fxR = mfx[gidx(g, i + 1, j, k)];
  real fyL = mfy[gidx(g, i, j, k)];
  real fyR = mfy[gidx(g, i, j + 1, k)];
  real fzB = mfz[gidx(g, i, j, k)];
  real fzT = mfz[gidx(g, i, j, k + 1)];

  real FxL = fxL * iface5(Q(i - 3, j, k), Q(i - 2, j, k), Q(i - 1, j, k), Q(i, j, k),
                          Q(i + 1, j, k), Q(i + 2, j, k), fxL);
  real FxR = fxR * iface5(Q(i - 2, j, k), Q(i - 1, j, k), Q(i, j, k), Q(i + 1, j, k),
                          Q(i + 2, j, k), Q(i + 3, j, k), fxR);
  real FyL = fyL * iface5(Q(i, j - 3, k), Q(i, j - 2, k), Q(i, j - 1, k), Q(i, j, k),
                          Q(i, j + 1, k), Q(i, j + 2, k), fyL);
  real FyR = fyR * iface5(Q(i, j - 2, k), Q(i, j - 1, k), Q(i, j, k), Q(i, j + 1, k),
                          Q(i, j + 2, k), Q(i, j + 3, k), fyR);
  real FzB = fzB * iface5(Q(i, j, k - 3), Q(i, j, k - 2), Q(i, j, k - 1), Q(i, j, k),
                          Q(i, j, k + 1), Q(i, j, k + 2), fzB);
  real FzT = fzT * iface5(Q(i, j, k - 2), Q(i, j, k - 1), Q(i, j, k), Q(i, j, k + 1),
                          Q(i, j, k + 2), Q(i, j, k + 3), fzT);

  real fdiv = (FxR - FxL) / g.dx + (FyR - FyL) / g.dy + (FzT - FzB) / m.dzeta_c[kk];
  real rt = p.rhob[gidx(g, i, j, k)] * m.jac[g2(g, i, j)];
  tend[gidx(g, i, j, k)] = (-fdiv + Q(i, j, k) * div[gidx(g, i, j, k)]) / rt;
}

// Asama sonunda nem skalarlarinin guncellenmesi: out = s0 + dt * tend
// (nem turlerinin hizli terimi yok; akustik donguye girmezler).
__global__ void k_stage_q(const real* s0, const real* tend, real dt, real* out,
                          size_t n) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = s0[idx] + dt * tend[idx];
}

// Alan uzerinde max|f| (patlama/NaN dedektoru icin). Pozitif float'larda
// int karsilastirmasi IEEE siralamasiyla uyumludur; NaN bitleri her sonlu
// degerden buyuk oldugundan NaN da "buyuk deger" olarak yakalanir.
__global__ void k_absmax(const real* f, size_t n, unsigned int* result) {
  __shared__ unsigned int smax;
  if (threadIdx.x == 0) smax = 0;
  __syncthreads();
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float v = fabsf((float)f[idx]);
    atomicMax(&smax, __float_as_uint(v));
  }
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(result, smax);
}

// Davies sinir relaksasyonu: tend += w(i,j) [ (1-tf) lo + tf hi - f ].
__global__ void k_bdy_relax(GDims g, int imax, int jmax, const real* f, const real* lo,
                            const real* hi, real tf, const real* wgt, real* tend) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i >= imax || j >= jmax || k >= g.nz) return;
  int iw = i < g.nx ? i : g.nx - 1;
  int jw = j < g.ny ? j : g.ny - 1;
  real w = wgt[g2(g, iw, jw)];
  if (w <= (real)0) return;
  size_t c = gidx(g, i, j, k);
  tend[c] += w * (((real)1 - tf) * lo[c] + tf * hi[c] - f[c]);
}

// ----------------------------------------------------------------- momentum

__global__ void k_tend_u(GDims g, DevProf p, DevMetric m, DynParams dp, DevState s,
                         const real* mfx, const real* mfy, const real* mfz,
                         const real* div, real* tend) {
  WFE_IJK_GUARD(g.nx, g.ny, 0, g.nz)
  int kk = k + g.ng;
  auto U = [&](int ii, int jj, int kz) { return s.u[gidx(g, ii, jj, kz)]; };

  // x akilari: hucre merkezleri (i-1) ve (i)
  real fcL = (real)0.5 * (mfx[gidx(g, i - 1, j, k)] + mfx[gidx(g, i, j, k)]);
  real fcR = (real)0.5 * (mfx[gidx(g, i, j, k)] + mfx[gidx(g, i + 1, j, k)]);
  real FxL = fcL * iface5(U(i - 3, j, k), U(i - 2, j, k), U(i - 1, j, k), U(i, j, k),
                          U(i + 1, j, k), U(i + 2, j, k), fcL);
  real FxR = fcR * iface5(U(i - 2, j, k), U(i - 1, j, k), U(i, j, k), U(i + 1, j, k),
                          U(i + 2, j, k), U(i + 3, j, k), fcR);
  // y akilari: koseler
  real fyB = (real)0.5 * (mfy[gidx(g, i - 1, j, k)] + mfy[gidx(g, i, j, k)]);
  real fyT = (real)0.5 * (mfy[gidx(g, i - 1, j + 1, k)] + mfy[gidx(g, i, j + 1, k)]);
  real FyB = fyB * iface5(U(i, j - 3, k), U(i, j - 2, k), U(i, j - 1, k), U(i, j, k),
                          U(i, j + 1, k), U(i, j + 2, k), fyB);
  real FyT = fyT * iface5(U(i, j - 2, k), U(i, j - 1, k), U(i, j, k), U(i, j + 1, k),
                          U(i, j + 2, k), U(i, j + 3, k), fyT);
  // zeta akilari
  real fzB = (real)0.5 * (mfz[gidx(g, i - 1, j, k)] + mfz[gidx(g, i, j, k)]);
  real fzT = (real)0.5 * (mfz[gidx(g, i - 1, j, k + 1)] + mfz[gidx(g, i, j, k + 1)]);
  real FzB = fzB * iface5(U(i, j, k - 3), U(i, j, k - 2), U(i, j, k - 1), U(i, j, k),
                          U(i, j, k + 1), U(i, j, k + 2), fzB);
  real FzT = fzT * iface5(U(i, j, k - 2), U(i, j, k - 1), U(i, j, k), U(i, j, k + 1),
                          U(i, j, k + 2), U(i, j, k + 3), fzT);

  real fdiv = (FxR - FxL) / g.dx + (FyT - FyB) / g.dy + (FzT - FzB) / m.dzeta_c[kk];
  real dv = (real)0.5 * (div[gidx(g, i - 1, j, k)] + div[gidx(g, i, j, k)]);
  real ru = (real)0.5 * (p.rhob[gidx(g, i - 1, j, k)] * m.jac[g2(g, i - 1, j)] +
                         p.rhob[gidx(g, i, j, k)] * m.jac[g2(g, i, j)]);
  real vavg = (real)0.25 * (s.v[gidx(g, i - 1, j, k)] + s.v[gidx(g, i - 1, j + 1, k)] +
                            s.v[gidx(g, i, j, k)] + s.v[gidx(g, i, j + 1, k)]);
  real fc = (real)0.5 * (m.fcor[g2(g, i - 1, j)] + m.fcor[g2(g, i, j)]);
  real relax = -ray_alpha(m.zeta_c[kk], m.zt, dp) * (U(i, j, k) - p.ub[kk]);
  tend[gidx(g, i, j, k)] = (-fdiv + U(i, j, k) * dv) / ru + fc * vavg + relax;
}

__global__ void k_tend_v(GDims g, DevProf p, DevMetric m, DynParams dp, DevState s,
                         const real* mfx, const real* mfy, const real* mfz,
                         const real* div, real* tend) {
  WFE_IJK_GUARD(g.nx, g.ny, 0, g.nz)
  int kk = k + g.ng;
  auto V = [&](int ii, int jj, int kz) { return s.v[gidx(g, ii, jj, kz)]; };

  real fcB = (real)0.5 * (mfy[gidx(g, i, j - 1, k)] + mfy[gidx(g, i, j, k)]);
  real fcT = (real)0.5 * (mfy[gidx(g, i, j, k)] + mfy[gidx(g, i, j + 1, k)]);
  real FyB = fcB * iface5(V(i, j - 3, k), V(i, j - 2, k), V(i, j - 1, k), V(i, j, k),
                          V(i, j + 1, k), V(i, j + 2, k), fcB);
  real FyT = fcT * iface5(V(i, j - 2, k), V(i, j - 1, k), V(i, j, k), V(i, j + 1, k),
                          V(i, j + 2, k), V(i, j + 3, k), fcT);
  real fxL = (real)0.5 * (mfx[gidx(g, i, j - 1, k)] + mfx[gidx(g, i, j, k)]);
  real fxR = (real)0.5 * (mfx[gidx(g, i + 1, j - 1, k)] + mfx[gidx(g, i + 1, j, k)]);
  real FxL = fxL * iface5(V(i - 3, j, k), V(i - 2, j, k), V(i - 1, j, k), V(i, j, k),
                          V(i + 1, j, k), V(i + 2, j, k), fxL);
  real FxR = fxR * iface5(V(i - 2, j, k), V(i - 1, j, k), V(i, j, k), V(i + 1, j, k),
                          V(i + 2, j, k), V(i + 3, j, k), fxR);
  real fzB = (real)0.5 * (mfz[gidx(g, i, j - 1, k)] + mfz[gidx(g, i, j, k)]);
  real fzT = (real)0.5 * (mfz[gidx(g, i, j - 1, k + 1)] + mfz[gidx(g, i, j, k + 1)]);
  real FzB = fzB * iface5(V(i, j, k - 3), V(i, j, k - 2), V(i, j, k - 1), V(i, j, k),
                          V(i, j, k + 1), V(i, j, k + 2), fzB);
  real FzT = fzT * iface5(V(i, j, k - 2), V(i, j, k - 1), V(i, j, k), V(i, j, k + 1),
                          V(i, j, k + 2), V(i, j, k + 3), fzT);

  real fdiv = (FxR - FxL) / g.dx + (FyT - FyB) / g.dy + (FzT - FzB) / m.dzeta_c[kk];
  real dv = (real)0.5 * (div[gidx(g, i, j - 1, k)] + div[gidx(g, i, j, k)]);
  real rv = (real)0.5 * (p.rhob[gidx(g, i, j - 1, k)] * m.jac[g2(g, i, j - 1)] +
                         p.rhob[gidx(g, i, j, k)] * m.jac[g2(g, i, j)]);
  real uavg = (real)0.25 * (s.u[gidx(g, i, j - 1, k)] + s.u[gidx(g, i + 1, j - 1, k)] +
                            s.u[gidx(g, i, j, k)] + s.u[gidx(g, i + 1, j, k)]);
  real fc = (real)0.5 * (m.fcor[g2(g, i, j - 1)] + m.fcor[g2(g, i, j)]);
  real uref = dp.coriolis_use_ub ? p.ub[kk] : (real)0;
  real relax = -ray_alpha(m.zeta_c[kk], m.zt, dp) * V(i, j, k);
  tend[gidx(g, i, j, k)] =
      (-fdiv + V(i, j, k) * dv) / rv - fc * (uavg - uref) + relax;
}

__global__ void k_tend_w(GDims g, DevProf p, DevMetric m, DynParams dp, real dt,
                         DevState s, const real* mfx, const real* mfy,
                         const real* mfz, const real* div, real* tend) {
  WFE_IJK_GUARD(g.nx, g.ny, 1, g.nz)  // w(0) diagnostik, w(nz)=0
  int kk = k + g.ng;
  auto W = [&](int ii, int jj, int kz) { return s.w[gidx(g, ii, jj, kz)]; };

  real fxL = (real)0.5 * (mfx[gidx(g, i, j, k - 1)] + mfx[gidx(g, i, j, k)]);
  real fxR = (real)0.5 * (mfx[gidx(g, i + 1, j, k - 1)] + mfx[gidx(g, i + 1, j, k)]);
  real FxL = fxL * iface5(W(i - 3, j, k), W(i - 2, j, k), W(i - 1, j, k), W(i, j, k),
                          W(i + 1, j, k), W(i + 2, j, k), fxL);
  real FxR = fxR * iface5(W(i - 2, j, k), W(i - 1, j, k), W(i, j, k), W(i + 1, j, k),
                          W(i + 2, j, k), W(i + 3, j, k), fxR);
  real fyB = (real)0.5 * (mfy[gidx(g, i, j, k - 1)] + mfy[gidx(g, i, j, k)]);
  real fyT = (real)0.5 * (mfy[gidx(g, i, j + 1, k - 1)] + mfy[gidx(g, i, j + 1, k)]);
  real FyB = fyB * iface5(W(i, j - 3, k), W(i, j - 2, k), W(i, j - 1, k), W(i, j, k),
                          W(i, j + 1, k), W(i, j + 2, k), fyB);
  real FyT = fyT * iface5(W(i, j - 2, k), W(i, j - 1, k), W(i, j, k), W(i, j + 1, k),
                          W(i, j + 2, k), W(i, j + 3, k), fyT);
  // zeta akilari: hucre merkezleri (k-1) ve (k)
  real fzB = (real)0.5 * (mfz[gidx(g, i, j, k - 1)] + mfz[gidx(g, i, j, k)]);
  real fzT = (real)0.5 * (mfz[gidx(g, i, j, k)] + mfz[gidx(g, i, j, k + 1)]);
  real FzB = fzB * iface5(W(i, j, k - 3), W(i, j, k - 2), W(i, j, k - 1), W(i, j, k),
                          W(i, j, k + 1), W(i, j, k + 2), fzB);
  real FzT = fzT * iface5(W(i, j, k - 2), W(i, j, k - 1), W(i, j, k), W(i, j, k + 1),
                          W(i, j, k + 2), W(i, j, k + 3), fzT);

  real fdiv = (FxR - FxL) / g.dx + (FyT - FyB) / g.dy + (FzT - FzB) / m.dzeta_w[kk];
  real dv = (real)0.5 * (div[gidx(g, i, j, k - 1)] + div[gidx(g, i, j, k)]);
  real rw = p.rhobw[gidx(g, i, j, k)] * m.jac[g2(g, i, j)];
  real relax = -ray_alpha(m.zeta_w[kk], m.zt, dp) * W(i, j, k);
  real wk = W(i, j, k);
  if (dp.w_damping) {
    // dikey Courant > 1: kosuyu kurtaran yerel sonumleme (WRF w_damping)
    real cr = fabs(wk) * dt / (m.dzeta_w[kk] * m.jac[g2(g, i, j)]);
    if (cr > (real)1) relax -= (real)0.3 * (cr - (real)1) * wk / dt;
  }
  tend[gidx(g, i, j, k)] = (-fdiv + wk * dv) / rw + relax;
}

// ---------------------------------------------------------------- difuzyon

// Sabit-K 2. mertebe Laplasyen (idealize testler; dikeyde yerel fiziksel aralik).
__global__ void k_diffuse(GDims g, DevMetric m, real K, const real* f, real* tend,
                          int kmin, int knum, int on_w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = (int)(blockIdx.z * blockDim.z + threadIdx.z) + kmin;
  if (i >= g.nx || j >= g.ny || k >= kmin + knum) return;
  int kk = k + g.ng;
  real dzl = (on_w ? m.dzeta_w[kk] : m.dzeta_c[kk]) * m.jac[g2(g, i, j)];
  real c = f[gidx(g, i, j, k)];
  real lap = (f[gidx(g, i + 1, j, k)] - 2 * c + f[gidx(g, i - 1, j, k)]) / (g.dx * g.dx) +
             (f[gidx(g, i, j + 1, k)] - 2 * c + f[gidx(g, i, j - 1, k)]) / (g.dy * g.dy) +
             (f[gidx(g, i, j, k + 1)] - 2 * c + f[gidx(g, i, j, k - 1)]) / (dzl * dzl);
  tend[gidx(g, i, j, k)] += K * lap;
}

// ------------------------------------------------- akustik alt-adimlama

constexpr int WFE_MAX_NZ = 320;  // kolon cozucunun yerel dizi siniri (nz+1 <= bu)

// u,v guncellemesi: pi* = pi' + smdiv (pi' - pi'_onceki); arazi capraz terimi
// (dz/dx * dpi/dzeta) explicit. PGF katsayisi TAM sanal pot. sicaklik
// (theta_b + theta')(1 + 0.61 qv): sinoptik olcekli buyuk pertubasyonlarda
// dogrulugu korur (Faz 3).
__global__ void k_acou_uv(GDims g, DevProf p, DevMetric m, DynParams dp, real dtau,
                          real* u, real* v, const real* pip, const real* piprev,
                          const real* tu, const real* tv, const real* thp,
                          const real* qv) {
  WFE_IJK_GUARD(g.nx, g.ny, 0, g.nz)
  const int kk = k + g.ng;
  auto ps = [&](int ii, int jj, int kz) {
    size_t c = gidx(g, ii, jj, kz);
    return pip[c] + dp.acoustic_smdiv * (pip[c] - piprev[c]);
  };
  auto thv = [&](int ii, int jj) {
    size_t c = gidx(g, ii, jj, k);
    real t = p.thb[c] + thp[c];
    if (dp.moisture) t *= (real)1 + phys::eps61 * qv[c];
    return t;
  };
  real ps0 = ps(i, j, k);
  real dzc2 = m.zeta_c[kk + 1] - m.zeta_c[kk - 1];

  if (!(dp.bc_x_open && i == 0)) {
    real thbu = (real)0.5 * (thv(i - 1, j) + thv(i, j));
    real grad = (ps0 - ps(i - 1, j, k)) / g.dx;
    real zxu = m.hx_u[g2(g, i, j)] * ((real)1 - m.zeta_c[kk] / m.zt);
    if (zxu != (real)0) {
      real ju = (real)0.5 * (m.jac[g2(g, i - 1, j)] + m.jac[g2(g, i, j)]);
      real dpdz = (real)0.5 *
                  ((ps(i - 1, j, k + 1) - ps(i - 1, j, k - 1)) +
                   (ps(i, j, k + 1) - ps(i, j, k - 1))) / dzc2;
      grad -= zxu / ju * dpdz;
    }
    u[gidx(g, i, j, k)] += dtau * (-phys::cp * thbu * grad + tu[gidx(g, i, j, k)]);
  }
  if (!(dp.bc_y_open && j == 0)) {
    real thbv = (real)0.5 * (thv(i, j - 1) + thv(i, j));
    real grad = (ps0 - ps(i, j - 1, k)) / g.dy;
    real zyv = m.hy_v[g2(g, i, j)] * ((real)1 - m.zeta_c[kk] / m.zt);
    if (zyv != (real)0) {
      real jv = (real)0.5 * (m.jac[g2(g, i, j - 1)] + m.jac[g2(g, i, j)]);
      real dpdz = (real)0.5 *
                  ((ps(i, j - 1, k + 1) - ps(i, j - 1, k - 1)) +
                   (ps(i, j, k + 1) - ps(i, j, k - 1))) / dzc2;
      grad -= zyv / jv * dpdz;
    }
    v[gidx(g, i, j, k)] += dtau * (-phys::cp * thbv * grad + tv[gidx(g, i, j, k)]);
  }
}

// w-pi' dikey implicit cozucu + theta' + diagnostik yuzey w'si.
// Kaldirma: g[ theta'/thb + 0.61(qv - qvb) - qc - qr ] (KW78 nemli form).
__global__ void k_acou_wpi(GDims g, DevProf p, DevMetric m, DynParams dp, real dtau,
                           const real* u, const real* v, real* w, real* pip,
                           real* piprev, real* thp, const real* tw, const real* tth,
                           const real* tpi, const real* qv, const real* qc,
                           const real* qr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;

  const real c1 = ((real)1 + dp.acoustic_beta) * (real)0.5;
  const real c2 = ((real)1 - dp.acoustic_beta) * (real)0.5;
  const real J = m.jac[g2(g, i, j)];
  const real hxc = m.hx[g2(g, i, j)];
  const real hyc = m.hy[g2(g, i, j)];

  real P[WFE_MAX_NZ], E[WFE_MAX_NZ], cs[WFE_MAX_NZ], ds[WFE_MAX_NZ], Wn[WFE_MAX_NZ];
  real Av[WFE_MAX_NZ], Sv[WFE_MAX_NZ];

  // Av: rho_b^w theta_b^w (yuzey ve tepede 0: Omega=0 kosulu — akiya girmez)
  // Sv: capraz dikey aki (explicit, guncel u,v ile)
  Av[0] = 0;
  Av[g.nz] = 0;
  Sv[0] = 0;
  Sv[g.nz] = 0;
  for (int k = 1; k < g.nz; ++k) {
    int kk = k + g.ng;
    size_t c = gidx(g, i, j, k);
    Av[k] = p.rhobw[c] * p.thvbw[c];
    real fac = (real)1 - m.zeta_w[kk] / m.zt;
    real uw = (real)0.25 * (u[gidx(g, i, j, k - 1)] + u[gidx(g, i + 1, j, k - 1)] +
                            u[gidx(g, i, j, k)] + u[gidx(g, i + 1, j, k)]);
    real vw = (real)0.25 * (v[gidx(g, i, j, k - 1)] + v[gidx(g, i, j + 1, k - 1)] +
                            v[gidx(g, i, j, k)] + v[gidx(g, i, j + 1, k)]);
    Sv[k] = Av[k] * (uw * hxc + vw * hyc) * fac;
  }

  for (int k = 0; k < g.nz; ++k) {
    int kk = k + g.ng;
    size_t c = gidx(g, i, j, k);
    real Kt = phys::Rd * p.pib[c] / (phys::cv * p.rhob[c] * p.thvb[c] * J);
    auto rtj = [&](int ii, int jj) {
      size_t cc = gidx(g, ii, jj, k);
      return p.rhob[cc] * p.thvb[cc] * m.jac[g2(g, ii, jj)];
    };
    real rc = rtj(i, j);
    real Dh = ((real)0.5 * (rtj(i, j) + rtj(i + 1, j)) * u[gidx(g, i + 1, j, k)] -
               (real)0.5 * (rtj(i - 1, j) + rc) * u[gidx(g, i, j, k)]) / g.dx +
              ((real)0.5 * (rtj(i, j) + rtj(i, j + 1)) * v[gidx(g, i, j + 1, k)] -
               (real)0.5 * (rtj(i, j - 1) + rc) * v[gidx(g, i, j, k)]) / g.dy;
    real dzc = m.dzeta_c[kk];
    real wk = w[gidx(g, i, j, k)];
    real wkp = w[gidx(g, i, j, k + 1)];
    P[k] = pip[c] + dtau * tpi[c] -
           dtau * Kt *
               (Dh - (Sv[k + 1] - Sv[k]) / dzc +
                c2 * (Av[k + 1] * wkp - Av[k] * wk) / dzc);
    E[k] = dtau * Kt * c1 / dzc;
  }

  for (int k = 1; k < g.nz; ++k) {
    int kk = k + g.ng;
    size_t c = gidx(g, i, j, k);
    size_t cm = gidx(g, i, j, k - 1);
    // w-PGF katsayisi: tam sanal pot. sicaklik w seviyesinde
    real thvw = p.thbw[c] + (real)0.5 * (thp[cm] + thp[c]);
    if (dp.moisture) thvw *= (real)1 + phys::eps61 * (real)0.5 * (qv[cm] + qv[c]);
    real G = dtau * phys::cp * thvw * c1 / (J * m.dzeta_w[kk]);
    real A = -G * E[k - 1] * Av[k - 1];
    real B = (real)1 + G * Av[k] * (E[k] + E[k - 1]);
    real C = -G * E[k] * Av[k + 1];
    real buoy = phys::grav * (real)0.5 * (thp[cm] + thp[c]) / p.thbw[c];
    if (dp.moisture) {
      real dqv = (real)0.5 * (qv[cm] + qv[c]) -
                 (real)0.5 * (p.qvb[cm] + p.qvb[c]);
      real qcr = (real)0.5 * (qc[cm] + qc[c]) + (real)0.5 * (qr[cm] + qr[c]);
      buoy += phys::grav * (phys::eps61 * dqv - qcr);
    }
    real RHS = w[c] + dtau * (buoy + tw[c]) -
               dtau * phys::cp * thvw * c2 *
                   (pip[gidx(g, i, j, k)] - pip[gidx(g, i, j, k - 1)]) /
                   (J * m.dzeta_w[kk]);
    real D = RHS - G * (P[k] - P[k - 1]);
    if (k == 1) {
      cs[k] = C / B;
      ds[k] = D / B;
    } else {
      real mm = B - A * cs[k - 1];
      cs[k] = C / mm;
      ds[k] = (D - A * ds[k - 1]) / mm;
    }
  }

  Wn[g.nz] = 0;
  real wabove = 0;
  for (int k = g.nz - 1; k >= 1; --k) {
    real wk = ds[k] - cs[k] * wabove;
    Wn[k] = wk;
    wabove = wk;
  }
  // yuzey: Omega=0 => w = u dz/dx + v dz/dy (arazi boyunca akis)
  {
    real us = (real)0.5 * (u[gidx(g, i, j, 0)] + u[gidx(g, i + 1, j, 0)]);
    real vs = (real)0.5 * (v[gidx(g, i, j, 0)] + v[gidx(g, i, j + 1, 0)]);
    Wn[0] = us * hxc + vs * hyc;
  }

  for (int k = 0; k < g.nz; ++k) {
    size_t c = gidx(g, i, j, k);
    real oldpi = pip[c];
    pip[c] = P[k] - E[k] * (Av[k + 1] * Wn[k + 1] - Av[k] * Wn[k]);
    piprev[c] = oldpi;
    real wc = (real)0.5 * (Wn[k] + Wn[k + 1]);
    thp[c] += dtau * (tth[c] - wc * p.dthbdz[c]);
  }
  for (int k = 0; k < g.nz; ++k) w[gidx(g, i, j, k)] = Wn[k];
}

// Acik x sinirinda normal hiz (u) yuzleri: giris taban durumuna sabit,
// cikis Klemp-Wilhelmson radyasyonu: du/dt = -(u + c*) du/dx.
__global__ void k_bc_rad_u_x(GDims g, DevProf p, real dtau, real cstar, real* u) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (j >= g.ny || k >= g.nz) return;
  int kk = k + g.ng;
  real u0 = u[gidx(g, 0, j, k)];
  if (u0 >= (real)0) {
    u[gidx(g, 0, j, k)] = p.ub[kk];
  } else {
    real cb = u0 - cstar;
    u[gidx(g, 0, j, k)] = u0 - dtau * cb * (u[gidx(g, 1, j, k)] - u0) / g.dx;
  }
  real un = u[gidx(g, g.nx, j, k)];
  if (un <= (real)0) {
    u[gidx(g, g.nx, j, k)] = p.ub[kk];
  } else {
    real cb = un + cstar;
    u[gidx(g, g.nx, j, k)] = un - dtau * cb * (un - u[gidx(g, g.nx - 1, j, k)]) / g.dx;
  }
}

__global__ void k_bc_rad_v_y(GDims g, real dtau, real cstar, real* v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || k >= g.nz) return;
  real v0 = v[gidx(g, i, 0, k)];
  if (v0 >= (real)0) {
    v[gidx(g, i, 0, k)] = 0;
  } else {
    real cb = v0 - cstar;
    v[gidx(g, i, 0, k)] = v0 - dtau * cb * (v[gidx(g, i, 1, k)] - v0) / g.dy;
  }
  real vn = v[gidx(g, i, g.ny, k)];
  if (vn <= (real)0) {
    v[gidx(g, i, g.ny, k)] = 0;
  } else {
    real cb = vn + cstar;
    v[gidx(g, i, g.ny, k)] = vn - dtau * cb * (vn - v[gidx(g, i, g.ny - 1, k)]) / g.dy;
  }
}

// --------------------------------------------------------- sinir kosullari

// Ghost doldurma sirasi onemli: once dusey, sonra x (tum j,k satirlari),
// sonra y (x ghost'lari dahil tum i) — koseler boylece tutarli dolar.

__global__ void k_bc_z_zerograd(GDims g, real* f) {
  int ir = blockIdx.x * blockDim.x + threadIdx.x;
  int jr = blockIdx.y * blockDim.y + threadIdx.y;
  if (ir >= g.NX || jr >= g.NY) return;
  size_t col = (size_t)jr * g.NX + ir;
  size_t stride = (size_t)g.NY * g.NX;
  real bot = f[col + stride * (size_t)g.ng];               // k = 0
  real top = f[col + stride * (size_t)(g.nz - 1 + g.ng)];  // k = nz-1
  for (int m = 1; m <= g.ng; ++m)
    f[col + stride * (size_t)(g.ng - m)] = bot;
  for (int m = 1; m <= g.ng + 1; ++m)
    f[col + stride * (size_t)(g.nz - 1 + g.ng + m)] = top;
}

// w: tepede 0; yuzeyde diagnostik deger korunur, ghost'lar w(0) etrafinda tek.
__global__ void k_bc_z_w(GDims g, real* f) {
  int ir = blockIdx.x * blockDim.x + threadIdx.x;
  int jr = blockIdx.y * blockDim.y + threadIdx.y;
  if (ir >= g.NX || jr >= g.NY) return;
  size_t col = (size_t)jr * g.NX + ir;
  size_t stride = (size_t)g.NY * g.NX;
  auto at = [&](int k) -> real& { return f[col + stride * (size_t)(k + g.ng)]; };
  at(g.nz) = 0;
  real w0 = at(0);
  for (int m = 1; m <= g.ng; ++m) {
    at(-m) = 2 * w0 - at(m);
    at(g.nz + m) = -at(g.nz - m);
  }
}

__global__ void k_bc_periodic_x(GDims g, real* f) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  int jr = blockIdx.y * blockDim.y + threadIdx.y;
  int kr = blockIdx.z * blockDim.z + threadIdx.z;
  if (m >= g.ng || jr >= g.NY || kr >= g.NZ) return;
  size_t row = ((size_t)kr * g.NY + jr) * g.NX;
  f[row + (size_t)(g.ng + g.nx + m)] = f[row + (size_t)(g.ng + m)];
  f[row + (size_t)(g.ng - 1 - m)] = f[row + (size_t)(g.ng + g.nx - 1 - m)];
}

__global__ void k_bc_periodic_y(GDims g, real* f) {
  int ir = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int kr = blockIdx.z * blockDim.z + threadIdx.z;
  if (ir >= g.NX || m >= g.ng || kr >= g.NZ) return;
  size_t plane = (size_t)kr * g.NY * g.NX;
  f[plane + (size_t)(g.ng + g.ny + m) * g.NX + ir] =
      f[plane + (size_t)(g.ng + m) * g.NX + ir];
  f[plane + (size_t)(g.ng - 1 - m) * g.NX + ir] =
      f[plane + (size_t)(g.ng + g.ny - 1 - m) * g.NX + ir];
}

// Acik sinir: sifir-gradyan ghost. ilast = son gecerli indeks.
__global__ void k_bc_zg_x(GDims g, real* f, int ilast) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  int jr = blockIdx.y * blockDim.y + threadIdx.y;
  int kr = blockIdx.z * blockDim.z + threadIdx.z;
  if (m >= g.ng || jr >= g.NY || kr >= g.NZ) return;
  size_t row = ((size_t)kr * g.NY + jr) * g.NX;
  real left = f[row + (size_t)g.ng];
  real right = f[row + (size_t)(g.ng + ilast)];
  f[row + (size_t)(g.ng - 1 - m)] = left;
  f[row + (size_t)(g.ng + ilast + 1 + m)] = right;
}

__global__ void k_bc_zg_y(GDims g, real* f, int jlast) {
  int ir = blockIdx.x * blockDim.x + threadIdx.x;
  int m = blockIdx.y * blockDim.y + threadIdx.y;
  int kr = blockIdx.z * blockDim.z + threadIdx.z;
  if (ir >= g.NX || m >= g.ng || kr >= g.NZ) return;
  size_t plane = (size_t)kr * g.NY * g.NX;
  real lo = f[plane + (size_t)g.ng * g.NX + ir];
  real hi = f[plane + (size_t)(g.ng + jlast) * g.NX + ir];
  f[plane + (size_t)(g.ng - 1 - m) * g.NX + ir] = lo;
  f[plane + (size_t)(g.ng + jlast + 1 + m) * g.NX + ir] = hi;
}

// ------------------------------------------------------------- yardimcilar

dim3 tile_block() { return dim3(32, 4, 2); }

dim3 tile_grid(int ni, int nj, int nk) {
  dim3 b = tile_block();
  return dim3((ni + b.x - 1) / b.x, (nj + b.y - 1) / b.y, (nk + b.z - 1) / b.z);
}

DevState dev_state(const State& s) {
  return DevState{s.u.d, s.v.d, s.w.d, s.thp.d, s.pip.d, s.qv.d, s.qc.d, s.qr.d};
}

void bc_lateral_x(const GDims& g, const DynParams& dp, real* f, int ilast) {
  dim3 bx(4, 8, 8);
  dim3 gx((g.ng + 3) / 4, (g.NY + 7) / 8, (g.NZ + 7) / 8);
  if (dp.bc_x_open)
    k_bc_zg_x<<<gx, bx>>>(g, f, ilast);
  else
    k_bc_periodic_x<<<gx, bx>>>(g, f);
}

void bc_lateral_y(const GDims& g, const DynParams& dp, real* f, int jlast) {
  dim3 by(32, 2, 8);
  dim3 gy((g.NX + 31) / 32, (g.ng + 1) / 2, (g.NZ + 7) / 8);
  if (dp.bc_y_open)
    k_bc_zg_y<<<gy, by>>>(g, f, jlast);
  else
    k_bc_periodic_y<<<gy, by>>>(g, f);
}

} // namespace

void apply_bcs(const GDims& g, const DynParams& dp, State& s) {
  dim3 b2(32, 8);
  dim3 g2d((g.NX + 31) / 32, (g.NY + 7) / 8);
  k_bc_z_zerograd<<<g2d, b2>>>(g, s.u.d);
  k_bc_z_zerograd<<<g2d, b2>>>(g, s.v.d);
  k_bc_z_zerograd<<<g2d, b2>>>(g, s.thp.d);
  k_bc_z_zerograd<<<g2d, b2>>>(g, s.pip.d);
  k_bc_z_w<<<g2d, b2>>>(g, s.w.d);

  bc_lateral_x(g, dp, s.u.d, g.nx);
  bc_lateral_x(g, dp, s.v.d, g.nx - 1);
  bc_lateral_x(g, dp, s.w.d, g.nx - 1);
  bc_lateral_x(g, dp, s.thp.d, g.nx - 1);
  bc_lateral_x(g, dp, s.pip.d, g.nx - 1);

  bc_lateral_y(g, dp, s.u.d, g.ny - 1);
  bc_lateral_y(g, dp, s.v.d, g.ny);
  bc_lateral_y(g, dp, s.w.d, g.ny - 1);
  bc_lateral_y(g, dp, s.thp.d, g.ny - 1);
  bc_lateral_y(g, dp, s.pip.d, g.ny - 1);

  if (dp.moisture) {
    real* qs[3] = {s.qv.d, s.qc.d, s.qr.d};
    for (real* f : qs) {
      k_bc_z_zerograd<<<g2d, b2>>>(g, f);
      bc_lateral_x(g, dp, f, g.nx - 1);
      bc_lateral_y(g, dp, f, g.ny - 1);
    }
  }
  check_kernel("apply_bcs");
}

void compute_mass_fluxes(const GDims& g, const DevProf& p, const DevMetric& m,
                         const State& s, Field3D& mfx, Field3D& mfy, Field3D& mfz) {
  DevState ds = dev_state(s);
  dim3 blk = tile_block();
  k_massflux_xy<<<tile_grid(g.nx + 5, g.ny + 5, g.nz), blk>>>(g, p, m, ds, mfx.d, mfy.d);
  k_massflux_z<<<tile_grid(g.nx + 4, g.ny + 4, g.nz - 1), blk>>>(g, p, m, ds, mfz.d);
  check_kernel("compute_mass_fluxes");
}

void compute_divergence(const GDims& g, const DevMetric& m, const Field3D& mfx,
                        const Field3D& mfy, const Field3D& mfz, Field3D& div) {
  k_divergence<<<tile_grid(g.nx + 2, g.ny + 2, g.nz), tile_block()>>>(g, m, mfx.d, mfy.d,
                                                                      mfz.d, div.d);
  check_kernel("compute_divergence");
}

void compute_tendencies(const GDims& g, const DevProf& p, const DevMetric& m,
                        const DynParams& dp, real dt, const State& s,
                        const Field3D& mfx, const Field3D& mfy, const Field3D& mfz,
                        const Field3D& div, State& tend) {
  DevState ds = dev_state(s);
  dim3 blk = tile_block();
  k_tend_u<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(g, p, m, dp, ds, mfx.d, mfy.d, mfz.d,
                                                 div.d, tend.u.d);
  k_tend_v<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(g, p, m, dp, ds, mfx.d, mfy.d, mfz.d,
                                                 div.d, tend.v.d);
  k_tend_w<<<tile_grid(g.nx, g.ny, g.nz - 1), blk>>>(g, p, m, dp, dt, ds, mfx.d, mfy.d,
                                                     mfz.d, div.d, tend.w.d);
  k_tend_thp<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(g, p, m, dp, ds, mfx.d, mfy.d, mfz.d,
                                                   div.d, tend.thp.d);
  if (dp.moisture) {
    k_tend_q<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(g, p, m, s.qv.d, mfx.d, mfy.d,
                                                   mfz.d, div.d, tend.qv.d);
    k_tend_q<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(g, p, m, s.qc.d, mfx.d, mfy.d,
                                                   mfz.d, div.d, tend.qc.d);
    k_tend_q<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(g, p, m, s.qr.d, mfx.d, mfy.d,
                                                   mfz.d, div.d, tend.qr.d);
  }
  // pi' yavas egilimi: adveksiyon yok; sinir relaksasyonu sonradan ekleyebilir
  tend.pip.zero();
  if (dp.diff_K > (real)0) {
    k_diffuse<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(g, m, dp.diff_K, s.u.d, tend.u.d, 0,
                                                    g.nz, 0);
    k_diffuse<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(g, m, dp.diff_K, s.v.d, tend.v.d, 0,
                                                    g.nz, 0);
    k_diffuse<<<tile_grid(g.nx, g.ny, g.nz - 1), blk>>>(g, m, dp.diff_K, s.w.d,
                                                        tend.w.d, 1, g.nz - 1, 1);
    k_diffuse<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(g, m, dp.diff_K, s.thp.d,
                                                    tend.thp.d, 0, g.nz, 0);
  }
  check_kernel("compute_tendencies");
}

void acoustic_substep(const GDims& g, const DevProf& p, const DevMetric& m,
                      const DynParams& dp, real dtau, State& s, const State& tend,
                      Field3D& piprev) {
  dim3 blk = tile_block();
  k_acou_uv<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(g, p, m, dp, dtau, s.u.d, s.v.d,
                                                  s.pip.d, piprev.d, tend.u.d,
                                                  tend.v.d, s.thp.d, s.qv.d);
  if (dp.bc_x_open) {
    dim3 b2(32, 8);
    dim3 g2d((g.ny + 31) / 32, (g.nz + 7) / 8);
    k_bc_rad_u_x<<<g2d, b2>>>(g, p, dtau, dp.cstar, s.u.d);
  }
  if (dp.bc_y_open) {
    dim3 b2(32, 8);
    dim3 g2d((g.nx + 31) / 32, (g.nz + 7) / 8);
    k_bc_rad_v_y<<<g2d, b2>>>(g, dtau, dp.cstar, s.v.d);
  }
  // Dh icin u(nx), v(ny) ve komsu kolonlar taze olmali
  bc_lateral_x(g, dp, s.u.d, g.nx);
  bc_lateral_y(g, dp, s.u.d, g.ny - 1);
  bc_lateral_x(g, dp, s.v.d, g.nx - 1);
  bc_lateral_y(g, dp, s.v.d, g.ny);

  dim3 bcol(32, 8);
  dim3 gcol((g.nx + 31) / 32, (g.ny + 7) / 8);
  k_acou_wpi<<<gcol, bcol>>>(g, p, m, dp, dtau, s.u.d, s.v.d, s.w.d, s.pip.d, piprev.d,
                             s.thp.d, tend.w.d, tend.thp.d, tend.pip.d, s.qv.d,
                             s.qc.d, s.qr.d);

  // sonraki alt-adim pi' yanal ghost'lari + capraz terim icin dusey ghost'lari okur
  bc_lateral_x(g, dp, s.pip.d, g.nx - 1);
  bc_lateral_y(g, dp, s.pip.d, g.ny - 1);
  dim3 b2(32, 8);
  dim3 g2d((g.NX + 31) / 32, (g.NY + 7) / 8);
  k_bc_z_zerograd<<<g2d, b2>>>(g, s.pip.d);
  check_kernel("acoustic_substep");
}

float field_absmax(const Field3D& f) {
  static unsigned int* d_res = nullptr;
  if (!d_res) WFE_CUDA_CHECK(cudaMalloc(&d_res, sizeof(unsigned int)));
  WFE_CUDA_CHECK(cudaMemset(d_res, 0, sizeof(unsigned int)));
  int blk = 256;
  int grd = (int)((f.n + blk - 1) / blk);
  k_absmax<<<grd, blk>>>(f.d, f.n, d_res);
  unsigned int bits = 0;
  WFE_CUDA_CHECK(cudaMemcpy(&bits, d_res, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  float v;
  std::memcpy(&v, &bits, sizeof(float));
  return v;
}

void bdy_relax(const GDims& g, const State& s, const Field3D lo[5], const Field3D hi[5],
               real tf, const Field3D& wgt, State& tend) {
  dim3 blk = tile_block();
  k_bdy_relax<<<tile_grid(g.nx + 1, g.ny, g.nz), blk>>>(
      g, g.nx + 1, g.ny, s.u.d, lo[0].d, hi[0].d, tf, wgt.d, tend.u.d);
  k_bdy_relax<<<tile_grid(g.nx, g.ny + 1, g.nz), blk>>>(
      g, g.nx, g.ny + 1, s.v.d, lo[1].d, hi[1].d, tf, wgt.d, tend.v.d);
  k_bdy_relax<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(
      g, g.nx, g.ny, s.thp.d, lo[2].d, hi[2].d, tf, wgt.d, tend.thp.d);
  k_bdy_relax<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(
      g, g.nx, g.ny, s.pip.d, lo[3].d, hi[3].d, tf, wgt.d, tend.pip.d);
  k_bdy_relax<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(
      g, g.nx, g.ny, s.qv.d, lo[4].d, hi[4].d, tf, wgt.d, tend.qv.d);
  check_kernel("bdy_relax");
}

void update_moisture_stage(const State& s0, const State& tend, real dt, State& out) {
  size_t n = s0.qv.n;
  int blk = 256;
  int grd = (int)((n + blk - 1) / blk);
  k_stage_q<<<grd, blk>>>(s0.qv.d, tend.qv.d, dt, out.qv.d, n);
  k_stage_q<<<grd, blk>>>(s0.qc.d, tend.qc.d, dt, out.qc.d, n);
  k_stage_q<<<grd, blk>>>(s0.qr.d, tend.qr.d, dt, out.qr.d, n);
  check_kernel("update_moisture_stage");
}

} // namespace wfe
