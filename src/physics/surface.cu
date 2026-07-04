#include "physics/surface.hpp"

#include <cmath>
#include <vector>

#include "core/constants.hpp"
#include "core/cuda_check.hpp"
#include "core/thermo.hpp"

namespace wfe {
namespace {

constexpr int SFC_MAX_NZ = MAX_COLUMN_LEVELS;

__device__ __forceinline__ size_t gidx(const GDims& g, int i, int j, int k) {
  return ((size_t)(k + g.ng) * g.NY + (j + g.ng)) * g.NX + (i + g.ng);
}
__device__ __forceinline__ size_t g2(const GDims& g, int i, int j) {
  return (size_t)(j + g.ng) * g.NX + (i + g.ng);
}
using thermo::qsat_tetens;

constexpr real KAPPA = (real)0.4;
constexpr real SIGMA = (real)5.67e-8;
constexpr real S0 = (real)1361;
constexpr real CSOIL = (real)1.4e5;   // levha toprak isi kapasitesi [J m-2 K-1]
constexpr real KM_MAX = (real)1000, KM_MIN = (real)0.1;

// Kolon-implicit dusey difuzyon (Thomas). phi merkez degerleri, Kw w-seviyeleri.
// src0: k=0'a eklenen yuzey aki kaynagi [birim*kg m-2 s-1] (rho agirlikli).
__device__ void diffuse_column(int nz, real dt, const real* rho, const real* rhow,
                               const real* dzc, const real* dzw, const real* Kw,
                               real src0, real* phi) {
  real A[SFC_MAX_NZ], B[SFC_MAX_NZ], C[SFC_MAX_NZ], D[SFC_MAX_NZ];
  for (int k = 0; k < nz; ++k) {
    real a = (k > 0) ? rhow[k] * Kw[k] / dzw[k] : (real)0;
    real c = (k < nz - 1) ? rhow[k + 1] * Kw[k + 1] / dzw[k + 1] : (real)0;
    real m = rho[k] * dzc[k] / dt;
    A[k] = -a;
    B[k] = m + a + c;
    C[k] = -c;
    D[k] = m * phi[k];
  }
  D[0] += src0;
  // Thomas
  for (int k = 1; k < nz; ++k) {
    real w = A[k] / B[k - 1];
    B[k] -= w * C[k - 1];
    D[k] -= w * D[k - 1];
  }
  phi[nz - 1] = D[nz - 1] / B[nz - 1];
  for (int k = nz - 2; k >= 0; --k) phi[k] = (D[k] - C[k] * phi[k + 1]) / B[k];
}

// Kutle kolonu: yuzey katmani + radyasyon + toprak + K profili + skalar karisim.
__global__ void k_sfc_scalar(GDims g, DevProf p, DevMetric m, real dt, real utc,
                             int doy, real* thp, const real* pip, real* qv,
                             const real* qc, const real* u, const real* v, real* tsk,
                             const real* tdeep, const real* land, const real* lat,
                             const real* lon, real* km, real* cdv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  size_t c2 = g2(g, i, j);
  const real J = m.jac[c2];

  real rho[SFC_MAX_NZ], rhow[SFC_MAX_NZ], dzc[SFC_MAX_NZ], dzw[SFC_MAX_NZ];
  real uc[SFC_MAX_NZ], vc[SFC_MAX_NZ], thv[SFC_MAX_NZ], Kw[SFC_MAX_NZ];
  real thcol[SFC_MAX_NZ], qvcol[SFC_MAX_NZ];

  real lwp = 0;  // bulut su yolu [g m-2]
  for (int k = 0; k < g.nz; ++k) {
    int kk = k + g.ng;
    size_t c = gidx(g, i, j, k);
    rho[k] = p.rhob[c];
    rhow[k] = p.rhobw[c];
    dzc[k] = m.dzeta_c[kk] * J;
    dzw[k] = m.dzeta_w[kk] * J;
    uc[k] = (real)0.5 * (u[gidx(g, i, j, k)] + u[gidx(g, i + 1, j, k)]);
    vc[k] = (real)0.5 * (v[gidx(g, i, j, k)] + v[gidx(g, i, j + 1, k)]);
    thcol[k] = p.thb[c] + thp[c];
    qvcol[k] = qv[c];
    thv[k] = thcol[k] * ((real)1 + phys::eps61 * qvcol[k]);
    lwp += rho[k] * (qc[c] > (real)0 ? qc[c] : (real)0) * dzc[k] * (real)1000;
  }

  // --- yuzey katmani (Louis 1979) ---
  bool onland = land[c2] > (real)0.5;
  real z0 = onland ? (real)0.1 : (real)2e-4;
  real alb = onland ? (real)0.2 : (real)0.08;
  real beta = onland ? (real)0.3 : (real)1.0;
  real z1 = (real)0.5 * dzc[0];
  real V1 = sqrt(uc[0] * uc[0] + vc[0] * vc[0]);
  if (V1 < (real)0.5) V1 = (real)0.5;
  size_t c0 = gidx(g, i, j, 0);
  real pi1 = p.pib[c0] + pip[c0];
  real T1 = thcol[0] * pi1;
  real p1 = phys::p00 * pow(pi1, phys::cp / phys::Rd);
  real Ts = tsk[c2];
  real pis = pi1 + phys::grav * z1 / (phys::cp * thv[0]);
  real ths = Ts / pis;
  real qs_s = qsat_tetens(p1, Ts);
  real thvs = ths * ((real)1 + phys::eps61 * (beta * qs_s + ((real)1 - beta) * qvcol[0]));
  real Rib = phys::grav * z1 * (thv[0] - thvs) / (thvs * V1 * V1);
  real a2 = KAPPA / log(z1 / z0);
  a2 *= a2;
  real Fm, Fh;
  if (Rib < (real)0) {
    real sq = sqrt(-Rib * z1 / z0);
    Fm = (real)1 - (real)9.4 * Rib / ((real)1 + (real)7.4 * (real)9.4 * a2 * sq);
    Fh = (real)1 - (real)9.4 * Rib / ((real)1 + (real)5.3 * (real)9.4 * a2 * sq);
  } else {
    real d = (real)1 + (real)4.7 * Rib;
    Fm = Fh = (real)1 / (d * d);
  }
  real Cd = a2 * Fm;
  real Ch = a2 * Fh;
  cdv[c2] = Cd * V1;

  real shf = Ch * V1 * (ths - thcol[0]);                       // kinematik [K m/s]
  real qflx = Ch * V1 * beta * (qs_s - qvcol[0]);              // [kg/kg m/s]
  if (!onland && qflx < (real)0) qflx = 0;                     // deniz: cig yok

  // --- radyasyon + levha toprak ---
  real decl = (real)0.40928 * sin((real)6.2832 * ((real)284 + doy) / (real)365);
  real latr = lat[c2] * (real)0.0174533;
  real hsol = utc + lon[c2] / (real)15;
  real H = (real)0.2618 * (hsol - (real)12);
  real cosz = sin(latr) * sin(decl) + cos(latr) * cos(decl) * cos(H);
  if (cosz < (real)0) cosz = 0;
  real trc = (real)1 / ((real)1 + (real)0.02 * lwp);           // bulut gecirgenligi
  real swn = S0 * cosz * (real)0.75 * trc * ((real)1 - alb);
  real e_hpa = qvcol[0] * p1 / ((real)0.622 * (real)100);
  real cc = lwp / (real)50;
  if (cc > (real)1) cc = 1;
  real lwd = SIGMA * T1 * T1 * T1 * T1 *
             ((real)0.60 + (real)0.042 * sqrt(e_hpa > (real)0 ? e_hpa : (real)0)) *
             ((real)1 + (real)0.22 * cc * cc);
  real lwu = (real)0.97 * SIGMA * Ts * Ts * Ts * Ts;
  real G = swn + (real)0.97 * lwd - lwu - rho[0] * phys::cp * shf * pi1 -
           rho[0] * phys::Lv * qflx;
  if (onland) {
    Ts += dt * (G / CSOIL - (real)7.27e-5 * (Ts - tdeep[c2]));
    if (Ts < (real)200) Ts = 200;
    if (Ts > (real)340) Ts = 340;
    tsk[c2] = Ts;
  }

  // --- yerel-K profili (Ri bagimli, Louis benzeri) ---
  Kw[0] = 0;
  for (int k = 1; k < g.nz; ++k) {
    int kk = k + g.ng;
    real dz = dzw[k];
    real du = (uc[k] - uc[k - 1]) / dz;
    real dv = (vc[k] - vc[k - 1]) / dz;
    real S2 = du * du + dv * dv + (real)1e-8;
    real thvw = (real)0.5 * (thv[k - 1] + thv[k]);
    real N2 = phys::grav / thvw * (thv[k] - thv[k - 1]) / dz;
    real Ri = N2 / S2;
    real zw = m.zeta_w[kk] * J;  // yuzeyden yukseklik ~ zeta*J
    real l = KAPPA * zw / ((real)1 + KAPPA * zw / (real)150);
    real Kv;
    if (Ri < (real)0)
      Kv = l * l * sqrt(S2) * sqrt((real)1 - (real)16 * Ri);
    else {
      real d = (real)1 + (real)5 * Ri;
      Kv = l * l * sqrt(S2) / (d * d);
    }
    if (Kv < KM_MIN) Kv = KM_MIN;
    if (Kv > KM_MAX) Kv = KM_MAX;
    Kw[k] = Kv;
    km[gidx(g, i, j, k)] = Kv;
  }

  // --- skalar dusey karisim (implicit) + yuzey akilari ---
  diffuse_column(g.nz, dt, rho, rhow, dzc, dzw, Kw, rho[0] * shf, thcol);
  diffuse_column(g.nz, dt, rho, rhow, dzc, dzw, Kw, rho[0] * qflx, qvcol);

  // troposferik LW sogumasi: -2 K/gun, 11 km ustunde sifira iner
  const real coolrate = (real)-2.31e-5;  // -2 K / 86400 s
  for (int k = 0; k < g.nz; ++k) {
    size_t c = gidx(g, i, j, k);
    real zc = m.zeta_c[k + g.ng] * J;
    real ramp = zc < (real)11000 ? (real)1
                                 : ((real)13000 - zc) / (real)2000;
    if (ramp < (real)0) ramp = 0;
    thp[c] = thcol[k] - p.thb[c] + dt * coolrate * ramp;
    real q = qvcol[k];
    qv[c] = q > (real)0 ? q : (real)0;
  }
}

// u yuzu kolonu: yuzey surtunmesi (explicit) + implicit dusey karisim.
__global__ void k_sfc_u(GDims g, DevProf p, DevMetric m, real dt, real* u,
                        const real* km, const real* cdv) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x) + 1;  // sinir yuzleri haric
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  real J = (real)0.5 * (m.jac[g2(g, i - 1, j)] + m.jac[g2(g, i, j)]);
  real rho[SFC_MAX_NZ], rhow[SFC_MAX_NZ], dzc[SFC_MAX_NZ], dzw[SFC_MAX_NZ];
  real Kw[SFC_MAX_NZ], phi[SFC_MAX_NZ];
  for (int k = 0; k < g.nz; ++k) {
    int kk = k + g.ng;
    size_t c = gidx(g, i, j, k);
    rho[k] = (real)0.5 * (p.rhob[gidx(g, i - 1, j, k)] + p.rhob[c]);
    rhow[k] = (real)0.5 * (p.rhobw[gidx(g, i - 1, j, k)] + p.rhobw[c]);
    dzc[k] = m.dzeta_c[kk] * J;
    dzw[k] = m.dzeta_w[kk] * J;
    Kw[k] = (k > 0) ? (real)0.5 * (km[gidx(g, i - 1, j, k)] + km[c]) : (real)0;
    phi[k] = u[c];
  }
  real cdva = (real)0.5 * (cdv[g2(g, i - 1, j)] + cdv[g2(g, i, j)]);
  real src0 = -rho[0] * cdva * phi[0];  // yuzey momentum akisi (surukleme)
  diffuse_column(g.nz, dt, rho, rhow, dzc, dzw, Kw, src0, phi);
  for (int k = 0; k < g.nz; ++k) u[gidx(g, i, j, k)] = phi[k];
}

__global__ void k_sfc_v(GDims g, DevProf p, DevMetric m, real dt, real* v,
                        const real* km, const real* cdv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = (int)(blockIdx.y * blockDim.y + threadIdx.y) + 1;
  if (i >= g.nx || j >= g.ny) return;
  real J = (real)0.5 * (m.jac[g2(g, i, j - 1)] + m.jac[g2(g, i, j)]);
  real rho[SFC_MAX_NZ], rhow[SFC_MAX_NZ], dzc[SFC_MAX_NZ], dzw[SFC_MAX_NZ];
  real Kw[SFC_MAX_NZ], phi[SFC_MAX_NZ];
  for (int k = 0; k < g.nz; ++k) {
    int kk = k + g.ng;
    size_t c = gidx(g, i, j, k);
    rho[k] = (real)0.5 * (p.rhob[gidx(g, i, j - 1, k)] + p.rhob[c]);
    rhow[k] = (real)0.5 * (p.rhobw[gidx(g, i, j - 1, k)] + p.rhobw[c]);
    dzc[k] = m.dzeta_c[kk] * J;
    dzw[k] = m.dzeta_w[kk] * J;
    Kw[k] = (k > 0) ? (real)0.5 * (km[gidx(g, i, j - 1, k)] + km[c]) : (real)0;
    phi[k] = v[c];
  }
  real cdva = (real)0.5 * (cdv[g2(g, i, j - 1)] + cdv[g2(g, i, j)]);
  real src0 = -rho[0] * cdva * phi[0];
  diffuse_column(g.nz, dt, rho, rhow, dzc, dzw, Kw, src0, phi);
  for (int k = 0; k < g.nz; ++k) v[gidx(g, i, j, k)] = phi[k];
}

} // namespace

void SfcPBL::init(const GDims& g, const InputData& in, real start_hour_utc, int doy) {
  start_hour_ = start_hour_utc;
  doy_ = doy;
  size_t n = g.npts();
  size_t n2 = (size_t)g.NX * g.NY;
  km_.alloc(n);
  tsk_.alloc(n2);
  tdeep_.alloc(n2);
  land_.alloc(n2);
  lat_.alloc(n2);
  lon_.alloc(n2);
  cdv_.alloc(n2);

  auto up2 = [&](Field3D& f, const std::vector<real>& src) {
    std::vector<real> h(n2, 0);
    for (int j = 0; j < g.ny; ++j)
      for (int i = 0; i < g.nx; ++i)
        h[(size_t)(j + g.ng) * g.NX + (i + g.ng)] = src[(size_t)j * g.nx + i];
    f.upload(h.data());
  };
  up2(tsk_, in.tsk);
  up2(tdeep_, in.tsk);  // derin sicaklik ~ baslangic yuzey sicakligi
  up2(land_, in.land);
  up2(lat_, in.lat);
  up2(lon_, in.lon);
}

void SfcPBL::release() {
  km_.release();
  tsk_.release();
  tdeep_.release();
  land_.release();
  lat_.release();
  lon_.release();
  cdv_.release();
}

void SfcPBL::step(const GDims& g, const DevProf& p, const DevMetric& m, real dt, real t,
                  State& s) {
  real utc = start_hour_ + t / (real)3600;
  utc = utc - (real)24 * floor(utc / (real)24);
  dim3 b(32, 8);
  dim3 gr((g.nx + 31) / 32, (g.ny + 7) / 8);
  k_sfc_scalar<<<gr, b>>>(g, p, m, dt, utc, doy_, s.thp.d, s.pip.d, s.qv.d, s.qc.d,
                          s.u.d, s.v.d, tsk_.d, tdeep_.d, land_.d, lat_.d, lon_.d,
                          km_.d, cdv_.d);
  dim3 gu((g.nx - 1 + 31) / 32, (g.ny + 7) / 8);
  k_sfc_u<<<gu, b>>>(g, p, m, dt, s.u.d, km_.d, cdv_.d);
  dim3 gv((g.nx + 31) / 32, (g.ny - 1 + 7) / 8);
  k_sfc_v<<<gv, b>>>(g, p, m, dt, s.v.d, km_.d, cdv_.d);
  check_kernel("sfc_pbl");
}

} // namespace wfe

