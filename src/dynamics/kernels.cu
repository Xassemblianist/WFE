#include "dynamics/kernels.hpp"

#include "core/constants.hpp"
#include "core/cuda_check.hpp"

// Ayriklastirma notlari (ayrinti: docs/EQUATIONS.md):
//  - 5. mertebe upwind-egilimli akilar (Wicker & Skamarock 2002), RK3 ile eslesir.
//  - Adveksiyon "advektif-tutarli" akı formunda: momentum ve skalarlar icin
//    tend = -(1/rho_b)[ div(rho_b V q) - q div(rho_b V) ], boylece sabit alan
//    sabit kalir ve taban yogunluk agirlikli korunum yaklasik saglanir.
//  - pi' denklemi Klemp-Wilhelmson: pi' adveksiyonu ihmal edilir.

namespace wfe {
namespace {

struct DevState {
  const real* u;
  const real* v;
  const real* w;
  const real* thp;
  const real* pip;
};

__device__ __forceinline__ size_t gidx(const GDims& g, int i, int j, int k) {
  return ((size_t)(k + g.ng) * g.NY + (j + g.ng)) * g.NX + (i + g.ng);
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

// ---------------------------------------------------------------- divergans

__global__ void k_divergence(GDims g, DevProf p, DevState s, real* div) {
  WFE_IJK_GUARD(g.nx, g.ny, 0, g.nz)
  int kk = k + g.ng;
  real hd = p.rhob[kk] * ((s.u[gidx(g, i + 1, j, k)] - s.u[gidx(g, i, j, k)]) / g.dx +
                          (s.v[gidx(g, i, j + 1, k)] - s.v[gidx(g, i, j, k)]) / g.dy);
  real vd = (p.rhobw[kk + 1] * s.w[gidx(g, i, j, k + 1)] -
             p.rhobw[kk] * s.w[gidx(g, i, j, k)]) / g.dz;
  div[gidx(g, i, j, k)] = hd + vd;
}

// ------------------------------------------------------------- skaler theta'

__global__ void k_tend_thp(GDims g, DevProf p, DevState s, const real* div, real* tend) {
  WFE_IJK_GUARD(g.nx, g.ny, 0, g.nz)
  int kk = k + g.ng;
  const real* q = s.thp;
  auto Q = [&](int ii, int jj, int kz) { return q[gidx(g, ii, jj, kz)]; };

  real uL = s.u[gidx(g, i, j, k)];
  real uR = s.u[gidx(g, i + 1, j, k)];
  real vL = s.v[gidx(g, i, j, k)];
  real vR = s.v[gidx(g, i, j + 1, k)];
  real wB = s.w[gidx(g, i, j, k)];
  real wT = s.w[gidx(g, i, j, k + 1)];

  real rk = p.rhob[kk];
  real FxL = rk * uL * iface5(Q(i - 3, j, k), Q(i - 2, j, k), Q(i - 1, j, k),
                              Q(i, j, k), Q(i + 1, j, k), Q(i + 2, j, k), uL);
  real FxR = rk * uR * iface5(Q(i - 2, j, k), Q(i - 1, j, k), Q(i, j, k),
                              Q(i + 1, j, k), Q(i + 2, j, k), Q(i + 3, j, k), uR);
  real FyL = rk * vL * iface5(Q(i, j - 3, k), Q(i, j - 2, k), Q(i, j - 1, k),
                              Q(i, j, k), Q(i, j + 1, k), Q(i, j + 2, k), vL);
  real FyR = rk * vR * iface5(Q(i, j - 2, k), Q(i, j - 1, k), Q(i, j, k),
                              Q(i, j + 1, k), Q(i, j + 2, k), Q(i, j + 3, k), vR);
  real FzB = p.rhobw[kk] * wB * iface5(Q(i, j, k - 3), Q(i, j, k - 2), Q(i, j, k - 1),
                                       Q(i, j, k), Q(i, j, k + 1), Q(i, j, k + 2), wB);
  real FzT = p.rhobw[kk + 1] * wT * iface5(Q(i, j, k - 2), Q(i, j, k - 1), Q(i, j, k),
                                           Q(i, j, k + 1), Q(i, j, k + 2), Q(i, j, k + 3), wT);

  real fdiv = (FxR - FxL) / g.dx + (FyR - FyL) / g.dy + (FzT - FzB) / g.dz;
  real wc = (real)0.5 * (wB + wT);
  tend[gidx(g, i, j, k)] =
      (-fdiv + Q(i, j, k) * div[gidx(g, i, j, k)]) / rk - wc * p.dthbdz[kk];
}

// ------------------------------------------------------------------ pi' (KW)

__global__ void k_tend_pip(GDims g, DevProf p, DevState s, real* tend) {
  WFE_IJK_GUARD(g.nx, g.ny, 0, g.nz)
  int kk = k + g.ng;
  real uL = s.u[gidx(g, i, j, k)];
  real uR = s.u[gidx(g, i + 1, j, k)];
  real vL = s.v[gidx(g, i, j, k)];
  real vR = s.v[gidx(g, i, j + 1, k)];
  real wB = s.w[gidx(g, i, j, k)];
  real wT = s.w[gidx(g, i, j, k + 1)];

  real rt = p.rhob[kk] * p.thb[kk];
  real hdiv = rt * ((uR - uL) / g.dx + (vR - vL) / g.dy);
  real vdiv = (p.rhobw[kk + 1] * p.thbw[kk + 1] * wT -
               p.rhobw[kk] * p.thbw[kk] * wB) / g.dz;
  real coef = phys::Rd * p.pib[kk] / (phys::cv * rt);
  tend[gidx(g, i, j, k)] = -coef * (hdiv + vdiv);
}

// ----------------------------------------------------------------- momentum

__global__ void k_tend_u(GDims g, DevProf p, DevState s, const real* div, real* tend) {
  WFE_IJK_GUARD(g.nx, g.ny, 0, g.nz)
  int kk = k + g.ng;
  auto U = [&](int ii, int jj, int kz) { return s.u[gidx(g, ii, jj, kz)]; };

  real rk = p.rhob[kk];
  // x akilari: hucre merkezleri (i-1) ve (i)
  real vcL = (real)0.5 * (U(i - 1, j, k) + U(i, j, k));
  real vcR = (real)0.5 * (U(i, j, k) + U(i + 1, j, k));
  real FxL = rk * vcL * iface5(U(i - 3, j, k), U(i - 2, j, k), U(i - 1, j, k),
                               U(i, j, k), U(i + 1, j, k), U(i + 2, j, k), vcL);
  real FxR = rk * vcR * iface5(U(i - 2, j, k), U(i - 1, j, k), U(i, j, k),
                               U(i + 1, j, k), U(i + 2, j, k), U(i + 3, j, k), vcR);
  // y akilari: koseler (i-1/2, j-1/2) ve (i-1/2, j+1/2)
  real vvB = (real)0.5 * (s.v[gidx(g, i - 1, j, k)] + s.v[gidx(g, i, j, k)]);
  real vvT = (real)0.5 * (s.v[gidx(g, i - 1, j + 1, k)] + s.v[gidx(g, i, j + 1, k)]);
  real FyB = rk * vvB * iface5(U(i, j - 3, k), U(i, j - 2, k), U(i, j - 1, k),
                               U(i, j, k), U(i, j + 1, k), U(i, j + 2, k), vvB);
  real FyT = rk * vvT * iface5(U(i, j - 2, k), U(i, j - 1, k), U(i, j, k),
                               U(i, j + 1, k), U(i, j + 2, k), U(i, j + 3, k), vvT);
  // z akilari: (i-1/2, k-1/2) ve (i-1/2, k+1/2)
  real wwB = (real)0.5 * (s.w[gidx(g, i - 1, j, k)] + s.w[gidx(g, i, j, k)]);
  real wwT = (real)0.5 * (s.w[gidx(g, i - 1, j, k + 1)] + s.w[gidx(g, i, j, k + 1)]);
  real FzB = p.rhobw[kk] * wwB * iface5(U(i, j, k - 3), U(i, j, k - 2), U(i, j, k - 1),
                                        U(i, j, k), U(i, j, k + 1), U(i, j, k + 2), wwB);
  real FzT = p.rhobw[kk + 1] * wwT * iface5(U(i, j, k - 2), U(i, j, k - 1), U(i, j, k),
                                            U(i, j, k + 1), U(i, j, k + 2), U(i, j, k + 3), wwT);

  real fdiv = (FxR - FxL) / g.dx + (FyT - FyB) / g.dy + (FzT - FzB) / g.dz;
  real dv = (real)0.5 * (div[gidx(g, i - 1, j, k)] + div[gidx(g, i, j, k)]);
  real pgf = phys::cp * p.thb[kk] *
             (s.pip[gidx(g, i, j, k)] - s.pip[gidx(g, i - 1, j, k)]) / g.dx;
  tend[gidx(g, i, j, k)] = (-fdiv + U(i, j, k) * dv) / rk - pgf;
}

__global__ void k_tend_v(GDims g, DevProf p, DevState s, const real* div, real* tend) {
  WFE_IJK_GUARD(g.nx, g.ny, 0, g.nz)
  int kk = k + g.ng;
  auto V = [&](int ii, int jj, int kz) { return s.v[gidx(g, ii, jj, kz)]; };

  real rk = p.rhob[kk];
  // y akilari: hucre merkezleri (j-1) ve (j)
  real vcB = (real)0.5 * (V(i, j - 1, k) + V(i, j, k));
  real vcT = (real)0.5 * (V(i, j, k) + V(i, j + 1, k));
  real FyB = rk * vcB * iface5(V(i, j - 3, k), V(i, j - 2, k), V(i, j - 1, k),
                               V(i, j, k), V(i, j + 1, k), V(i, j + 2, k), vcB);
  real FyT = rk * vcT * iface5(V(i, j - 2, k), V(i, j - 1, k), V(i, j, k),
                               V(i, j + 1, k), V(i, j + 2, k), V(i, j + 3, k), vcT);
  // x akilari: koseler (i-1/2, j-1/2) ve (i+1/2, j-1/2)
  real uuL = (real)0.5 * (s.u[gidx(g, i, j - 1, k)] + s.u[gidx(g, i, j, k)]);
  real uuR = (real)0.5 * (s.u[gidx(g, i + 1, j - 1, k)] + s.u[gidx(g, i + 1, j, k)]);
  real FxL = rk * uuL * iface5(V(i - 3, j, k), V(i - 2, j, k), V(i - 1, j, k),
                               V(i, j, k), V(i + 1, j, k), V(i + 2, j, k), uuL);
  real FxR = rk * uuR * iface5(V(i - 2, j, k), V(i - 1, j, k), V(i, j, k),
                               V(i + 1, j, k), V(i + 2, j, k), V(i + 3, j, k), uuR);
  // z akilari
  real wwB = (real)0.5 * (s.w[gidx(g, i, j - 1, k)] + s.w[gidx(g, i, j, k)]);
  real wwT = (real)0.5 * (s.w[gidx(g, i, j - 1, k + 1)] + s.w[gidx(g, i, j, k + 1)]);
  real FzB = p.rhobw[kk] * wwB * iface5(V(i, j, k - 3), V(i, j, k - 2), V(i, j, k - 1),
                                        V(i, j, k), V(i, j, k + 1), V(i, j, k + 2), wwB);
  real FzT = p.rhobw[kk + 1] * wwT * iface5(V(i, j, k - 2), V(i, j, k - 1), V(i, j, k),
                                            V(i, j, k + 1), V(i, j, k + 2), V(i, j, k + 3), wwT);

  real fdiv = (FxR - FxL) / g.dx + (FyT - FyB) / g.dy + (FzT - FzB) / g.dz;
  real dv = (real)0.5 * (div[gidx(g, i, j - 1, k)] + div[gidx(g, i, j, k)]);
  real pgf = phys::cp * p.thb[kk] *
             (s.pip[gidx(g, i, j, k)] - s.pip[gidx(g, i, j - 1, k)]) / g.dy;
  tend[gidx(g, i, j, k)] = (-fdiv + V(i, j, k) * dv) / rk - pgf;
}

__global__ void k_tend_w(GDims g, DevProf p, DevState s, const real* div, real* tend) {
  WFE_IJK_GUARD(g.nx, g.ny, 1, g.nz)  // k = 1 .. nz-1 (sinirlarda w = 0 sabit)
  int kk = k + g.ng;
  auto W = [&](int ii, int jj, int kz) { return s.w[gidx(g, ii, jj, kz)]; };

  real rw = p.rhobw[kk];
  // x akilari: (i-1/2, k-1/2) ve (i+1/2, k-1/2)
  real uuL = (real)0.5 * (s.u[gidx(g, i, j, k - 1)] + s.u[gidx(g, i, j, k)]);
  real uuR = (real)0.5 * (s.u[gidx(g, i + 1, j, k - 1)] + s.u[gidx(g, i + 1, j, k)]);
  real FxL = rw * uuL * iface5(W(i - 3, j, k), W(i - 2, j, k), W(i - 1, j, k),
                               W(i, j, k), W(i + 1, j, k), W(i + 2, j, k), uuL);
  real FxR = rw * uuR * iface5(W(i - 2, j, k), W(i - 1, j, k), W(i, j, k),
                               W(i + 1, j, k), W(i + 2, j, k), W(i + 3, j, k), uuR);
  // y akilari
  real vvB = (real)0.5 * (s.v[gidx(g, i, j, k - 1)] + s.v[gidx(g, i, j, k)]);
  real vvT = (real)0.5 * (s.v[gidx(g, i, j + 1, k - 1)] + s.v[gidx(g, i, j + 1, k)]);
  real FyB = rw * vvB * iface5(W(i, j - 3, k), W(i, j - 2, k), W(i, j - 1, k),
                               W(i, j, k), W(i, j + 1, k), W(i, j + 2, k), vvB);
  real FyT = rw * vvT * iface5(W(i, j - 2, k), W(i, j - 1, k), W(i, j, k),
                               W(i, j + 1, k), W(i, j + 2, k), W(i, j + 3, k), vvT);
  // z akilari: hucre merkezleri (k-1) ve (k)
  real wcB = (real)0.5 * (W(i, j, k - 1) + W(i, j, k));
  real wcT = (real)0.5 * (W(i, j, k) + W(i, j, k + 1));
  real FzB = p.rhob[kk - 1] * wcB * iface5(W(i, j, k - 3), W(i, j, k - 2), W(i, j, k - 1),
                                           W(i, j, k), W(i, j, k + 1), W(i, j, k + 2), wcB);
  real FzT = p.rhob[kk] * wcT * iface5(W(i, j, k - 2), W(i, j, k - 1), W(i, j, k),
                                       W(i, j, k + 1), W(i, j, k + 2), W(i, j, k + 3), wcT);

  real fdiv = (FxR - FxL) / g.dx + (FyT - FyB) / g.dy + (FzT - FzB) / g.dz;
  real dv = (real)0.5 * (div[gidx(g, i, j, k - 1)] + div[gidx(g, i, j, k)]);
  real pgf = phys::cp * p.thbw[kk] *
             (s.pip[gidx(g, i, j, k)] - s.pip[gidx(g, i, j, k - 1)]) / g.dz;
  real buoy = phys::grav * (real)0.5 *
              (s.thp[gidx(g, i, j, k - 1)] + s.thp[gidx(g, i, j, k)]) / p.thbw[kk];
  tend[gidx(g, i, j, k)] = (-fdiv + W(i, j, k) * dv) / rw - pgf + buoy;
}

// ------------------------------------------------------------------- update

__global__ void k_update(const real* s0, const real* tend, real dt, real* out, size_t n) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = s0[idx] + dt * tend[idx];
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

__global__ void k_bc_z_w(GDims g, real* f) {
  int ir = blockIdx.x * blockDim.x + threadIdx.x;
  int jr = blockIdx.y * blockDim.y + threadIdx.y;
  if (ir >= g.NX || jr >= g.NY) return;
  size_t col = (size_t)jr * g.NX + ir;
  size_t stride = (size_t)g.NY * g.NX;
  auto at = [&](int k) -> real& { return f[col + stride * (size_t)(k + g.ng)]; };
  at(0) = 0;
  at(g.nz) = 0;
  for (int m = 1; m <= g.ng; ++m) {
    at(-m) = -at(m);
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

// ------------------------------------------------------------- yardimcilar

dim3 tile_block() { return dim3(32, 4, 2); }

dim3 tile_grid(int ni, int nj, int nk) {
  dim3 b = tile_block();
  return dim3((ni + b.x - 1) / b.x, (nj + b.y - 1) / b.y, (nk + b.z - 1) / b.z);
}

DevState dev_state(const State& s) {
  return DevState{s.u.d, s.v.d, s.w.d, s.thp.d, s.pip.d};
}

} // namespace

void apply_bcs(const GDims& g, State& s) {
  dim3 b2(32, 8);
  dim3 g2((g.NX + 31) / 32, (g.NY + 7) / 8);
  k_bc_z_zerograd<<<g2, b2>>>(g, s.u.d);
  k_bc_z_zerograd<<<g2, b2>>>(g, s.v.d);
  k_bc_z_zerograd<<<g2, b2>>>(g, s.thp.d);
  k_bc_z_zerograd<<<g2, b2>>>(g, s.pip.d);
  k_bc_z_w<<<g2, b2>>>(g, s.w.d);

  dim3 bx(4, 8, 8);
  dim3 gx((g.ng + 3) / 4, (g.NY + 7) / 8, (g.NZ + 7) / 8);
  real* fields[5] = {s.u.d, s.v.d, s.w.d, s.thp.d, s.pip.d};
  for (real* f : fields) k_bc_periodic_x<<<gx, bx>>>(g, f);

  dim3 by(32, 2, 8);
  dim3 gy((g.NX + 31) / 32, (g.ng + 1) / 2, (g.NZ + 7) / 8);
  for (real* f : fields) k_bc_periodic_y<<<gy, by>>>(g, f);

  check_kernel("apply_bcs");
}

void compute_divergence(const GDims& g, const DevProf& p, const State& s, Field3D& div) {
  k_divergence<<<tile_grid(g.nx, g.ny, g.nz), tile_block()>>>(g, p, dev_state(s), div.d);
  check_kernel("k_divergence");
}

void compute_tendencies(const GDims& g, const DevProf& p, const State& s,
                        const Field3D& div, State& tend) {
  DevState ds = dev_state(s);
  dim3 blk = tile_block();
  k_tend_u<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(g, p, ds, div.d, tend.u.d);
  k_tend_v<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(g, p, ds, div.d, tend.v.d);
  k_tend_w<<<tile_grid(g.nx, g.ny, g.nz - 1), blk>>>(g, p, ds, div.d, tend.w.d);
  k_tend_thp<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(g, p, ds, div.d, tend.thp.d);
  k_tend_pip<<<tile_grid(g.nx, g.ny, g.nz), blk>>>(g, p, ds, tend.pip.d);
  check_kernel("compute_tendencies");
}

void update_state(const State& s0, const State& tend, real dt, State& out) {
  size_t n = s0.u.n;
  int blk = 256;
  int grd = (int)((n + blk - 1) / blk);
  k_update<<<grd, blk>>>(s0.u.d, tend.u.d, dt, out.u.d, n);
  k_update<<<grd, blk>>>(s0.v.d, tend.v.d, dt, out.v.d, n);
  k_update<<<grd, blk>>>(s0.w.d, tend.w.d, dt, out.w.d, n);
  k_update<<<grd, blk>>>(s0.thp.d, tend.thp.d, dt, out.thp.d, n);
  k_update<<<grd, blk>>>(s0.pip.d, tend.pip.d, dt, out.pip.d, n);
  check_kernel("update_state");
}

} // namespace wfe
