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
constexpr real KM_MAX = (real)1000, KM_MIN = (real)0.1;

// Businger-Dyer stabilite fonksiyonlari (2m/10m tanilari icin).
// zeta = z/L; kararsizda x=(1-16 zeta)^(1/4).
__device__ __forceinline__ real psi_m(real zeta) {
  if (zeta >= (real)0) return (real)-5 * fmin(zeta, (real)2);   // kararli (kelepceli)
  real x = pow((real)1 - (real)16 * zeta, (real)0.25);
  const real halfpi = (real)1.5707963267948966;
  return (real)2 * log(((real)1 + x) * (real)0.5) +
         log(((real)1 + x * x) * (real)0.5) - (real)2 * atan(x) + halfpi;
}
__device__ __forceinline__ real psi_h(real zeta) {
  if (zeta >= (real)0) return (real)-5 * fmin(zeta, (real)2);
  real x = pow((real)1 - (real)16 * zeta, (real)0.25);
  return (real)2 * log(((real)1 + x * x) * (real)0.5);
}

// Cok katmanli toprak (Noah-benzeri): 4 katman, hacimsel isi kap. + iletkenlik.
// Termal ozellikler TOPRAK NEMI ile degisir (Johansen-tipi basitlestirme):
// kuru bozkir (w~0.1): C~1.5e6, K~0.55 (termal atalet ~900) — sabit "nemli kil"
// degerleri (C=2.2e6, K=1.5, atalet ~1800) yaz gunduz isinmasini yariya
// bastiriyordu -> METAR'da -4C gunduz soguk sapmasinin ana bileseni.
constexpr int NSOIL = 4;
__constant__ real SOIL_DZ[NSOIL] = {(real)0.1, (real)0.3, (real)0.6, (real)1.0};  // [m]
__device__ __forceinline__ real soil_C(real w) {   // [J m-3 K-1]
  return (real)1.2e6 + (real)3.2e6 * fmin(fmax(w, (real)0), (real)0.4);
}
__device__ __forceinline__ real soil_K(real w) {   // [W m-1 K-1]
  return (real)0.25 + (real)3.0 * fmin(fmax(w, (real)0), (real)0.4);
}

// Kolon-implicit dusey difuzyon (Thomas). phi merkez degerleri, Kw w-seviyeleri.
// src0: k=0'a eklenen yuzey aki kaynagi [birim*kg m-2 s-1] (rho agirlikli).
// fcg (nullable): w seviyelerinde YUKARI karsi-gradyan akisi (nonlocal PBL);
// hucre k'ye (fcg[k]-fcg[k+1]) explicit kaynak olarak eklenir.
__device__ void diffuse_column(int nz, real dt, const real* rho, const real* rhow,
                               const real* dzc, const real* dzw, const real* Kw,
                               real src0, real* phi, const real* fcg = nullptr) {
  real A[SFC_MAX_NZ], B[SFC_MAX_NZ], C[SFC_MAX_NZ], D[SFC_MAX_NZ];
  for (int k = 0; k < nz; ++k) {
    real a = (k > 0) ? rhow[k] * Kw[k] / dzw[k] : (real)0;
    real c = (k < nz - 1) ? rhow[k + 1] * Kw[k + 1] / dzw[k + 1] : (real)0;
    real m = rho[k] * dzc[k] / dt;
    A[k] = -a;
    B[k] = m + a + c;
    C[k] = -c;
    D[k] = m * phi[k];
    if (fcg) D[k] += fcg[k] - fcg[k + 1];
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
                             int doy, int nonlocal, real* thp, const real* pip,
                             real* qv, const real* qc, const real* u, const real* v,
                             real* tsk, const real* tdeep, const real* land,
                             const real* lat, const real* lon, const real* soilw,
                             real* km, real* cdv, real* pblh, real* t2m, real* u10,
                             real* soilt, size_t n2plane) {
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
  // evaporasyon verimi: karesel nem-stresi (solma 0.08, tarla kap. 0.32).
  // Lineer oran kuru bozkirda (w~0.2 -> beta~0.57) asiri buharlatiyordu:
  // 100-200 W/m2 gizli isiya kacip gunduz soguk sapmasi buyutuyordu.
  real beta;
  if (onland) {
    real f = (soilw[c2] - (real)0.08) / ((real)0.32 - (real)0.08);
    f = fmin((real)1, fmax((real)0, f));
    beta = fmax((real)0.05, f * f);
  } else {
    beta = (real)1;
  }
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

  // 2m sicaklik + 10m ruzgar tanilari: psi-duzeltmeli flux-profil
  // (Businger-Dyer). Duz log-orani kararli gecelerde 2m'yi yuzeye fazla
  // yapistiriyordu (gece soguk sapmasinin bir kaynagi). z/L, Rib'den
  // Launiainen-tipi yaklasimla kestirilir; z/L yukseklikle lineer olceklenir.
  real z0h = z0 * (real)0.1;
  real lnz1z0 = log(z1 / z0);
  real zeta1 = (Rib >= (real)0)
                   ? Rib * lnz1z0 / ((real)1 - (real)5 * fmin(Rib, (real)0.19))
                   : Rib * lnz1z0;
  zeta1 = fmax((real)-8, fmin((real)2, zeta1));
  real zol = zeta1 / z1;                              // 1/L
  real den_m = lnz1z0 - psi_m(zeta1) + psi_m(z0 * zol);
  real num_m = log((real)10 / z0) - psi_m((real)10 * zol) + psi_m(z0 * zol);
  real r10 = fmax((real)0, fmin((real)1.2, num_m / fmax(den_m, (real)0.1)));
  u10[c2] = V1 * r10;
  real den_h = log(z1 / z0h) - psi_h(zeta1) + psi_h(z0h * zol);
  real num_h = log((real)2 / z0h) - psi_h((real)2 * zol) + psi_h(z0h * zol);
  real r2 = fmax((real)0, fmin((real)1.2, num_h / fmax(den_h, (real)0.1)));
  real th2 = ths + (thcol[0] - ths) * r2;
  t2m[c2] = th2 * pi1;

  real shf = Ch * V1 * (ths - thcol[0]);                       // kinematik [K m/s]
  real qflx = Ch * V1 * beta * (qs_s - qvcol[0]);              // [kg/kg m/s]
  if (!onland && qflx < (real)0) qflx = 0;                     // deniz: cig yok

  // --- kolon radyasyonu (iki-aki broadband) ---
  // Gunes geometrisi
  real decl = (real)0.40928 * sin((real)6.2832 * ((real)284 + doy) / (real)365);
  real latr = lat[c2] * (real)0.0174533;
  real hsol = utc + lon[c2] / (real)15;
  real H = (real)0.2618 * (hsol - (real)12);
  real cosz = sin(latr) * sin(decl) + cos(latr) * cos(decl) * cos(H);
  if (cosz < (real)0) cosz = 0;

  real Tlev[SFC_MAX_NZ], epslw[SFC_MAX_NZ], Fup[SFC_MAX_NZ], Fdn[SFC_MAX_NZ];
  real radht[SFC_MAX_NZ];  // radyatif isitma hizi [K/s] (theta)
  for (int k = 0; k < g.nz; ++k) {
    size_t c = gidx(g, i, j, k);
    real pik = p.pib[c] + pip[c];
    Tlev[k] = thcol[k] * pik;
    real du = rho[k] * qvcol[k] * dzc[k];                    // su buhari yolu
    real dl = rho[k] * (qc[c] > (real)0 ? qc[c] : (real)0) * dzc[k] * (real)1000;
    // nemli alt atmosfer LW'de opak (yuzeye asagi LW; dusuk katsayi -> asiri sogur)
    epslw[k] = (real)1 - exp(-((real)0.22 * du + (real)0.30 * dl));
  }
  // uzun dalga: asagi (tepeden) sonra yukari (yuzeyden)
  Fdn[g.nz] = (real)0;                                        // model tepesinde ~0
  for (int k = g.nz - 1; k >= 0; --k)
    Fdn[k] = Fdn[k + 1] * ((real)1 - epslw[k]) +
             epslw[k] * SIGMA * Tlev[k] * Tlev[k] * Tlev[k] * Tlev[k];
  real lwd = Fdn[0];
  Fup[0] = (real)0.97 * SIGMA * Ts * Ts * Ts * Ts + (real)0.03 * Fdn[0];
  for (int k = 0; k < g.nz; ++k)
    Fup[k + 1] = Fup[k] * ((real)1 - epslw[k]) +
                 epslw[k] * SIGMA * Tlev[k] * Tlev[k] * Tlev[k] * Tlev[k];
  // kisa dalga: tek isin, su buhari + bulut soguurma.
  // Bulut ALBEDOSU: kolon LWP'sine bagli yansitma (Stephens-tipi);
  // onceden bulut yalniz sogururdu -> bulutlu gunde yuzey fazla isiniyordu.
  real mu = fmax(cosz, (real)0.05);
  real cldalb = lwp / (lwp + (real)60);                       // ~60 g/m2 -> A=0.5
  real Sdn = S0 * cosz * (real)0.95 * ((real)1 - cldalb);     // tepe (ozon/rayleigh + bulut yansimasi)
  real swabs[SFC_MAX_NZ];
  for (int k = g.nz - 1; k >= 0; --k) {
    size_t c = gidx(g, i, j, k);
    real du = rho[k] * qvcol[k] * dzc[k];
    real dl = rho[k] * (qc[c] > (real)0 ? qc[c] : (real)0) * dzc[k] * (real)1000;
    // broadband SW su buhari soguurma katsayisi (~toplam OD 0.2/25kg -> 0.008)
    real tau = exp(-((real)0.008 * du + (real)0.15 * dl) / mu);
    real ab = Sdn * ((real)1 - tau);
    swabs[k] = ab;
    Sdn *= tau;
  }
  real swn = Sdn * ((real)1 - alb);                           // yuzeyde net SW
  // katman radyatif isitma: LW aki yakinsamasi + SW soguurma
  for (int k = 0; k < g.nz; ++k) {
    size_t c = gidx(g, i, j, k);
    real pik = p.pib[c] + pip[c];
    real lwconv = (Fdn[k + 1] - Fdn[k]) + (Fup[k] - Fup[k + 1]);  // [W/m2 katmana]
    real net = lwconv + swabs[k];
    radht[k] = net / (rho[k] * phys::cp * dzc[k]) / pik;      // theta hizi [K/s]
  }

  // --- cok katmanli toprak (Noah-benzeri isi iletimi) ---
  real lwu = (real)0.97 * SIGMA * Ts * Ts * Ts * Ts;
  real G = swn + (real)0.97 * lwd - lwu - rho[0] * phys::cp * shf * pi1 -
           rho[0] * phys::Lv * qflx;                       // yuzeye net aki [W/m2]
  if (onland) {
    // 4 katman implicit 1B isi difuzyonu (Thomas). Ust: G akisi; alt: sabit Tdeep.
    // C ve K toprak nemine bagli (kuru yaz topragi hizli isinir/sogur).
    real Cso = soil_C(soilw[c2]);
    real Kso = soil_K(soilw[c2]);
    real Tso[NSOIL], A[NSOIL], B[NSOIL], C[NSOIL], D[NSOIL];
    for (int L = 0; L < NSOIL; ++L) Tso[L] = soilt[(size_t)L * n2plane + c2];
    for (int L = 0; L < NSOIL; ++L) {
      real cap = Cso * SOIL_DZ[L] / dt;
      // katman arayuz iletim katsayilari (kat L ile L+1 arasi)
      real ku = (L > 0) ? Kso / ((real)0.5 * (SOIL_DZ[L - 1] + SOIL_DZ[L])) : (real)0;
      real kl = (L < NSOIL - 1) ? Kso / ((real)0.5 * (SOIL_DZ[L] + SOIL_DZ[L + 1]))
                                : Kso / SOIL_DZ[L];  // alt: Tdeep'e
      A[L] = -ku;
      C[L] = (L < NSOIL - 1) ? -kl : (real)0;
      B[L] = cap + ku + kl;
      D[L] = cap * Tso[L];
    }
    D[0] += G;                                    // ust sinir: yuzey isi akisi
    D[NSOIL - 1] += (Kso / SOIL_DZ[NSOIL - 1]) * tdeep[c2];  // alt: Tdeep
    for (int L = 1; L < NSOIL; ++L) {
      real w = A[L] / B[L - 1];
      B[L] -= w * C[L - 1];
      D[L] -= w * D[L - 1];
    }
    Tso[NSOIL - 1] = D[NSOIL - 1] / B[NSOIL - 1];
    for (int L = NSOIL - 2; L >= 0; --L) Tso[L] = (D[L] - C[L] * Tso[L + 1]) / B[L];
    for (int L = 0; L < NSOIL; ++L) {
      if (Tso[L] < (real)200) Tso[L] = 200;
      if (Tso[L] > (real)350) Tso[L] = 350;
      soilt[(size_t)L * n2plane + c2] = Tso[L];
    }
    tsk[c2] = Tso[0];                             // yuzey = ust toprak katmani
  }

  // yerel-Ri difuzyon katsayisi (serbest atmosfer / kararli kolon fallback)
  auto local_K = [&](int k) -> real {
    real dz = dzw[k];
    real du = (uc[k] - uc[k - 1]) / dz;
    real dv = (vc[k] - vc[k - 1]) / dz;
    real S2 = du * du + dv * dv + (real)1e-8;
    real thvw = (real)0.5 * (thv[k - 1] + thv[k]);
    real N2 = phys::grav / thvw * (thv[k] - thv[k - 1]) / dz;
    real Ri = N2 / S2;
    real zw = m.zeta_w[k + g.ng] * J;
    real l = KAPPA * zw / ((real)1 + KAPPA * zw / (real)150);
    if (Ri < (real)0) return l * l * sqrt(S2) * sqrt((real)1 - (real)16 * Ri);
    real d = (real)1 + (real)5 * Ri;
    return l * l * sqrt(S2) / (d * d);
  };

  real fcg_th[SFC_MAX_NZ], fcg_qv[SFC_MAX_NZ];
  Kw[0] = 0;
  fcg_th[0] = fcg_qv[0] = fcg_th[g.nz] = fcg_qv[g.nz] = 0;

  if (nonlocal) {
    // --- nonlocal PBL (Troen-Mahrt / Hong-Pan K-profili + karsi-gradyan) ---
    real ust = sqrt(Cd) * V1;                       // surtunme hizi u*
    real thv1 = thv[0];
    // sanal kinematik yuzey isi akisi (kaldirma uretimi)
    real wthv0 = shf * ((real)1 + phys::eps61 * qvcol[0]) + phys::eps61 * thcol[0] * qflx;

    const real Ric = (real)0.25;
    real hpbl = m.zeta_c[(g.nz - 1) + g.ng] * J;
    real ws = ust > (real)0.1 ? ust : (real)0.1;
    for (int it = 0; it < 3; ++it) {                // h <-> w* iterasyonu
      real exc = (wthv0 > (real)0) ? (real)6.8 * wthv0 / ws : (real)0;
      real thv_s = thv1 + exc;
      real prevRb = 0;
      hpbl = m.zeta_c[(g.nz - 1) + g.ng] * J;
      for (int k = 1; k < g.nz; ++k) {
        real zc = m.zeta_c[k + g.ng] * J;
        real shear = uc[k] * uc[k] + vc[k] * vc[k] + (real)0.1;
        real Rib = phys::grav * (thv[k] - thv_s) * zc / (thv1 * shear);
        if (Rib >= Ric) {
          real zcm = m.zeta_c[(k - 1) + g.ng] * J;
          real f = (Ric - prevRb) / (Rib - prevRb + (real)1e-9);
          f = fmin((real)1, fmax((real)0, f));
          hpbl = zcm + f * (zc - zcm);
          break;
        }
        prevRb = Rib;
      }
      real zc0 = m.zeta_c[g.ng] * J;
      if (hpbl < zc0) hpbl = zc0;
      real wstar = (wthv0 > (real)0) ? cbrt(phys::grav / thv1 * wthv0 * hpbl) : (real)0;
      ws = cbrt(ust * ust * ust + (real)0.6 * wstar * wstar * wstar);
      if (ws < (real)0.1) ws = (real)0.1;
    }
    pblh[c2] = hpbl;

    real gth = (wthv0 > (real)0) ? (real)6.8 * shf / (ws * hpbl) : (real)0;
    real gqv = (wthv0 > (real)0) ? (real)6.8 * qflx / (ws * hpbl) : (real)0;

    for (int k = 1; k < g.nz; ++k) {
      real zw = m.zeta_w[k + g.ng] * J;
      real Kv;
      if (wthv0 > (real)0 && zw < hpbl) {           // konvektif BL: nonlocal profil
        real zf = zw / hpbl;
        Kv = KAPPA * ws * zw * (1 - zf) * (1 - zf);
        fcg_th[k] = rhow[k] * Kv * gth;             // yukari karsi-gradyan akisi
        fcg_qv[k] = rhow[k] * Kv * gqv;
      } else {                                      // kararli/serbest atmosfer
        Kv = local_K(k);
        fcg_th[k] = fcg_qv[k] = 0;
      }
      if (Kv < KM_MIN) Kv = KM_MIN;
      if (Kv > KM_MAX) Kv = KM_MAX;
      Kw[k] = Kv;
      km[gidx(g, i, j, k)] = Kv;
    }
  } else {
    // --- yerel-K profili (Faz 4 v1) ---
    pblh[c2] = 0;
    for (int k = 1; k < g.nz; ++k) {
      real Kv = local_K(k);
      if (Kv < KM_MIN) Kv = KM_MIN;
      if (Kv > KM_MAX) Kv = KM_MAX;
      Kw[k] = Kv;
      km[gidx(g, i, j, k)] = Kv;
      fcg_th[k] = fcg_qv[k] = 0;
    }
  }

  // --- skalar dusey karisim (implicit) + yuzey akilari + karsi-gradyan ---
  diffuse_column(g.nz, dt, rho, rhow, dzc, dzw, Kw, rho[0] * shf, thcol, fcg_th);
  diffuse_column(g.nz, dt, rho, rhow, dzc, dzw, Kw, rho[0] * qflx, qvcol, fcg_qv);

  // kolon radyatif isitma (iki-aki hesabindan) — sabit sogumanin yerine
  for (int k = 0; k < g.nz; ++k) {
    size_t c = gidx(g, i, j, k);
    real dth = dt * radht[k];
    if (dth > (real)0.02) dth = (real)0.02;    // adim basi guvenlik siniri [K]
    if (dth < (real)-0.02) dth = (real)-0.02;
    thp[c] = thcol[k] - p.thb[c] + dth;
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

void SfcPBL::init(const GDims& g, const InputData& in, real start_hour_utc, int doy,
                  bool nonlocal) {
  start_hour_ = start_hour_utc;
  doy_ = doy;
  nonlocal_ = nonlocal;
  size_t n = g.npts();
  size_t n2 = (size_t)g.NX * g.NY;
  km_.alloc(n);
  tsk_.alloc(n2);
  tdeep_.alloc(n2);
  land_.alloc(n2);
  lat_.alloc(n2);
  lon_.alloc(n2);
  soilw_.alloc(n2);
  cdv_.alloc(n2);
  pblh_.alloc(n2);
  t2m_.alloc(n2);
  u10_.alloc(n2);
  soilt_.alloc(4 * n2);  // 4 toprak katmani

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
  up2(soilw_, in.soilw);

  // toprak katmanlari: baslangicta hepsi GFS yuzey sicakligi (ilk saatlerde dengelenir)
  std::vector<real> hs(4 * n2, 0);
  for (int L = 0; L < 4; ++L)
    for (int j = 0; j < g.ny; ++j)
      for (int i = 0; i < g.nx; ++i)
        hs[(size_t)L * n2 + (size_t)(j + g.ng) * g.NX + (i + g.ng)] =
            in.tsk[(size_t)j * g.nx + i];
  soilt_.upload(hs.data());
}

void SfcPBL::release() {
  km_.release();
  tsk_.release();
  tdeep_.release();
  land_.release();
  lat_.release();
  lon_.release();
  soilw_.release();
  cdv_.release();
  pblh_.release();
  t2m_.release();
  u10_.release();
  soilt_.release();
}

void SfcPBL::step(const GDims& g, const DevProf& p, const DevMetric& m, real dt, real t,
                  State& s) {
  real utc = start_hour_ + t / (real)3600;
  utc = utc - (real)24 * floor(utc / (real)24);
  dim3 b(32, 8);
  dim3 gr((g.nx + 31) / 32, (g.ny + 7) / 8);
  k_sfc_scalar<<<gr, b>>>(g, p, m, dt, utc, doy_, nonlocal_ ? 1 : 0, s.thp.d, s.pip.d,
                          s.qv.d, s.qc.d, s.u.d, s.v.d, tsk_.d, tdeep_.d, land_.d,
                          lat_.d, lon_.d, soilw_.d, km_.d, cdv_.d, pblh_.d,
                          t2m_.d, u10_.d, soilt_.d, (size_t)g.NX * g.NY);
  dim3 gu((g.nx - 1 + 31) / 32, (g.ny + 7) / 8);
  k_sfc_u<<<gu, b>>>(g, p, m, dt, s.u.d, km_.d, cdv_.d);
  dim3 gv((g.nx + 31) / 32, (g.ny - 1 + 7) / 8);
  k_sfc_v<<<gv, b>>>(g, p, m, dt, s.v.d, km_.d, cdv_.d);
  check_kernel("sfc_pbl");
}

} // namespace wfe

