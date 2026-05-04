/*
 * CPPWRF — Phase 2
 * GPU solver: 2D non-hydrostatic compressible Euler equations
 * Scheme: WENO5 reconstruction · Rusanov flux · Wicker-Skamarock SSP-RK3
 * Memory: SoA, layout [nz+2h][nx+2h], threadIdx.x→ix (coalesced in both sweeps)
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * PEER REVIEW & TECHNICAL NOTES FOR IMPLEMENTATION:
 * ─────────────────────────────────────────────────────────────────────────────
 * 1. WELL-BALANCED FORMULATION: 
 *    - Current flux splitting (p - pb) in k_x_fluxes/k_z_fluxes correctly preserves
 *      hydrostatic equilibrium by only advecting pressure perturbations.
 *    - Gravity source in k_rk_update (-rho' * g) is consistent with this.
 *
 * 2. ROBERT (1993) DENSITY CURRENT VALIDATION (Target Case):
 *    - Domain: 51.2km x 6.4km. Initial Theta_base = 300K.
 *    - Cold Bubble Perturbation (to be implemented in IC setup):
 *      R = sqrt( ((x-xc)/rx)^2 + ((z-zc)/rz)^2 )
 *      If R <= 1: Theta' = -15.0 * cos^2(pi * R / 2)
 *      Coords: xc=0 (or 25.6km), zc=3000m, rx=4000m, rz=2000m.
 *
 * 3. POTENTIAL OPTIMIZATIONS:
 *    - k_z_fluxes: The arithmetic mean of pi_b (line 381) is a good first-order 
 *      approximation, but for higher-order hydrostatic consistency, consider
 *      sampling pi_b exactly at interface points if the profile is analytical.
 *    - Boundary Conditions: Reflective 'w' at z-boundaries (line 242) is correct
 *      for rigid lids.
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "euler2d_gpu.cuh"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

// ─── Error macro ──────────────────────────────────────────────────────────────
#define CK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA ") + cudaGetErrorString(_e) \
            + " at " __FILE__ ":" + std::to_string(__LINE__)); \
} while(0)

// ─── WENO5 device functions ───────────────────────────────────────────────────
// Left-biased WENO5: reconstructed value at right edge of cell i  (interface i+½)
// Inputs: qm2=q_{i-2}, qm1=q_{i-1}, q0=q_i, qp1=q_{i+1}, qp2=q_{i+2}

__device__ __forceinline__ Real weno5L(Real qm2, Real qm1, Real q0, Real qp1, Real qp2)
{
    // Three candidate stencils (JS smoothness indicators)
    const Real b0 = (13.0/12.0)*(qm2-2*qm1+q0)*(qm2-2*qm1+q0)
                  + 0.25*(qm2-4*qm1+3*q0)*(qm2-4*qm1+3*q0);
    const Real b1 = (13.0/12.0)*(qm1-2*q0+qp1)*(qm1-2*q0+qp1)
                  + 0.25*(qm1-qp1)*(qm1-qp1);
    const Real b2 = (13.0/12.0)*(q0-2*qp1+qp2)*(q0-2*qp1+qp2)
                  + 0.25*(3*q0-4*qp1+qp2)*(3*q0-4*qp1+qp2);

    // Optimal weights: d = {1/10, 3/5, 3/10}
    constexpr Real eps = 1.0e-6;
    const Real a0 = 0.1  / ((b0+eps)*(b0+eps));
    const Real a1 = 0.6  / ((b1+eps)*(b1+eps));
    const Real a2 = 0.3  / ((b2+eps)*(b2+eps));
    const Real w  = 1.0  / (a0+a1+a2);

    // Candidate polynomials
    const Real q0s = ( 2*qm2 - 7*qm1 + 11*q0)  / 6.0;
    const Real q1s = (-  qm1 + 5*q0  +  2*qp1) / 6.0;
    const Real q2s = ( 2*q0  + 5*qp1 -    qp2) / 6.0;
    const Real res = (a0*q0s + a1*q1s + a2*q2s) * w;
    if (isnan(res)) {
        printf("weno5L NaN! eps=%g b0=%g a0=%g w=%g qm2=%g qm1=%g q0=%g qp1=%g qp2=%g\n",
               eps, b0, a0, w, qm2, qm1, q0, qp1, qp2);
    }
    return res;
}

// Right-biased WENO5: reconstructed value at left edge of cell i+1 (interface i+½)
// Mirror of weno5L: swap stencil order, optimal weights d = {3/10, 3/5, 1/10}
__device__ __forceinline__ Real weno5R(Real qm2, Real qm1, Real q0, Real qp1, Real qp2)
{
    const Real b0 = (13.0/12.0)*(qm2-2*qm1+q0)*(qm2-2*qm1+q0)
                  + 0.25*(qm2-4*qm1+3*q0)*(qm2-4*qm1+3*q0);
    const Real b1 = (13.0/12.0)*(qm1-2*q0+qp1)*(qm1-2*q0+qp1)
                  + 0.25*(qm1-qp1)*(qm1-qp1);
    const Real b2 = (13.0/12.0)*(q0-2*qp1+qp2)*(q0-2*qp1+qp2)
                  + 0.25*(3*q0-4*qp1+qp2)*(3*q0-4*qp1+qp2);

    constexpr Real eps = 1.0e-6;
    const Real a0 = 0.3  / ((b0+eps)*(b0+eps));
    const Real a1 = 0.6  / ((b1+eps)*(b1+eps));
    const Real a2 = 0.1  / ((b2+eps)*(b2+eps));
    const Real w  = 1.0  / (a0+a1+a2);

    // Right-biased polynomials (mirror of left)
    const Real q0s = (-   qm2 + 5*qm1 +  2*q0)  / 6.0;
    const Real q1s = ( 2* qm1 + 5*q0  -    qp1) / 6.0;
    const Real q2s = (11* q0  - 7*qp1 +  2*qp2) / 6.0;

    return (a0*q0s + a1*q1s + a2*q2s) * w;
}

__device__ __forceinline__ void hllc_x(
    Real& F_rho, Real& F_rhou, Real& F_rhow, Real& F_rhoTh,
    Real rL, Real ruL, Real rwL, Real rTL, Real pL,
    Real rR, Real ruR, Real rwR, Real rTR, Real pR,
    Real pb)
{
    const Real uL = (rL > 1.0e-10) ? ruL / rL : 0.0;
    const Real uR = (rR > 1.0e-10) ? ruR / rR : 0.0;
    const Real wL = (rL > 1.0e-10) ? rwL / rL : 0.0;
    const Real wR = (rR > 1.0e-10) ? rwR / rR : 0.0;
    const Real aL = sqrt(atm::gamma * pL / max(rL, 1.0e-10));
    const Real aR = sqrt(atm::gamma * pR / max(rR, 1.0e-10));

    const Real SL = min(uL - aL, uR - aR);
    const Real SR = max(uL + aL, uR + aR);

    const Real FL_rho = ruL;
    const Real FL_rhou = ruL * uL + (pL - pb);
    const Real FL_rhow = ruL * wL;
    const Real FL_rhoTh = rTL * uL;

    const Real FR_rho = ruR;
    const Real FR_rhou = ruR * uR + (pR - pb);
    const Real FR_rhow = ruR * wR;
    const Real FR_rhoTh = rTR * uR;

    if (0.0 <= SL) {
        F_rho = FL_rho; F_rhou = FL_rhou; F_rhow = FL_rhow; F_rhoTh = FL_rhoTh;
        return;
    }
    if (0.0 >= SR) {
        F_rho = FR_rho; F_rhou = FR_rhou; F_rhow = FR_rhow; F_rhoTh = FR_rhoTh;
        return;
    }

    const Real S_star = (pR - pL + ruL*(SL - uL) - ruR*(SR - uR)) / (rL*(SL - uL) - rR*(SR - uR) - 1e-15);

    if (0.0 <= S_star) {
        const Real inv_SL_Sstar = 1.0 / (SL - S_star - 1e-15);
        const Real r_star = rL * (SL - uL) * inv_SL_Sstar;
        const Real ru_star = r_star * S_star;
        const Real rw_star = r_star * wL;
        const Real rT_star = r_star * (rTL / rL);
        F_rho   = FL_rho   + SL * (r_star - rL);
        F_rhou  = FL_rhou  + SL * (ru_star - ruL);
        F_rhow  = FL_rhow  + SL * (rw_star - rwL);
        F_rhoTh = FL_rhoTh + SL * (rT_star - rTL);
    } else {
        const Real inv_SR_Sstar = 1.0 / (SR - S_star + 1e-15);
        const Real r_star = rR * (SR - uR) * inv_SR_Sstar;
        const Real ru_star = r_star * S_star;
        const Real rw_star = r_star * wR;
        const Real rT_star = r_star * (rTR / rR);
        F_rho   = FR_rho   + SR * (r_star - rR);
        F_rhou  = FR_rhou  + SR * (ru_star - ruR);
        F_rhow  = FR_rhow  + SR * (rw_star - rwR);
        F_rhoTh = FR_rhoTh + SR * (rT_star - rTR);
    }

    if (isnan(F_rhow) || isnan(F_rho)) {
        printf("HLLC_X NaN! SL=%g SR=%g S_star=%g rL=%g rR=%g ruL=%g ruR=%g rwL=%g rwR=%g rTL=%g rTR=%g pL=%g pR=%g\n",
               SL, SR, S_star, rL, rR, ruL, ruR, rwL, rwR, rTL, rTR, pL, pR);
    }
}

__device__ __forceinline__ void hllc_z(
    Real& F_rho, Real& F_rhou, Real& F_rhow, Real& F_rhoTh,
    Real rL, Real ruL, Real rwL, Real rTL, Real pL,
    Real rR, Real ruR, Real rwR, Real rTR, Real pR,
    Real pb)
{
    const Real uL = (rL > 1.0e-10) ? ruL / rL : 0.0;
    const Real uR = (rR > 1.0e-10) ? ruR / rR : 0.0;
    const Real wL = (rL > 1.0e-10) ? rwL / rL : 0.0;
    const Real wR = (rR > 1.0e-10) ? rwR / rR : 0.0;
    const Real aL = sqrt(atm::gamma * pL / max(rL, 1.0e-10));
    const Real aR = sqrt(atm::gamma * pR / max(rR, 1.0e-10));

    const Real SL = min(wL - aL, wR - aR);
    const Real SR = max(wL + aL, wR + aR);

    const Real FL_rho = rwL;
    const Real FL_rhou = rwL * uL;
    const Real FL_rhow = rwL * wL + (pL - pb);
    const Real FL_rhoTh = rTL * wL;

    const Real FR_rho = rwR;
    const Real FR_rhou = rwR * uR;
    const Real FR_rhow = rwR * wR + (pR - pb);
    const Real FR_rhoTh = rTR * wR;

    if (0.0 <= SL) {
        F_rho = FL_rho; F_rhou = FL_rhou; F_rhow = FL_rhow; F_rhoTh = FL_rhoTh;
        return;
    }
    if (0.0 >= SR) {
        F_rho = FR_rho; F_rhou = FR_rhou; F_rhow = FR_rhow; F_rhoTh = FR_rhoTh;
        return;
    }

    const Real S_star = (pR - pL + rwL*(SL - wL) - rwR*(SR - wR)) / (rL*(SL - wL) - rR*(SR - wR) - 1e-15);

    if (0.0 <= S_star) {
        const Real inv_SL_Sstar = 1.0 / (SL - S_star - 1e-15);
        const Real r_star = rL * (SL - wL) * inv_SL_Sstar;
        const Real ru_star = r_star * uL;
        const Real rw_star = r_star * S_star;
        const Real rT_star = r_star * (rTL / rL);
        F_rho   = FL_rho   + SL * (r_star - rL);
        F_rhou  = FL_rhou  + SL * (ru_star - ruL);
        F_rhow  = FL_rhow  + SL * (rw_star - rwL);
        F_rhoTh = FL_rhoTh + SL * (rT_star - rTL);
    } else {
        const Real inv_SR_Sstar = 1.0 / (SR - S_star + 1e-15);
        const Real r_star = rR * (SR - wR) * inv_SR_Sstar;
        const Real ru_star = r_star * uR;
        const Real rw_star = r_star * S_star;
        const Real rT_star = r_star * (rTR / rR);
        F_rho   = FR_rho   + SR * (r_star - rR);
        F_rhou  = FR_rhou  + SR * (ru_star - ruR);
        F_rhow  = FR_rhow  + SR * (rw_star - rwR);
        F_rhoTh = FR_rhoTh + SR * (rT_star - rTR);
    }
}

// ─── Equation of state helpers ────────────────────────────────────────────────
// p = p0 * (Rd * ρθ / p0)^(cp/cv)
__device__ __forceinline__ Real pressure(Real rhoTh) {
    return atm::p0 * pow(atm::Rd * rhoTh / atm::p0, atm::gamma);
}
// sound speed: a = sqrt(γ p / ρ)
__device__ __forceinline__ Real sound_speed(Real rho, Real rhoTh) {
    return sqrt(atm::gamma * pressure(rhoTh) / max(rho, 1.0e-10));
}

__global__ void k_fill_halos(Real* r, Real* ru, Real* rw, Real* rT,
                             const Real* rho_b, const Real* rT_b,
                             int stride, int nz_full, int nx, int nz, int halo)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    // x halos: transmissive for perturbations
    if (k < halo) {
        for (int iz = 0; iz < nz_full; ++iz) {
            int iL = iz*stride + halo;
            int iR = iz*stride + halo + nx - 1;
            
            // left
            r [iz*stride + halo - 1 - k] = r[iL];
            ru[iz*stride + halo - 1 - k] = ru[iL];
            rw[iz*stride + halo - 1 - k] = rw[iL];
            rT[iz*stride + halo - 1 - k] = rT[iL];
            
            // right
            r [iz*stride + halo + nx + k] = r[iR];
            ru[iz*stride + halo + nx + k] = ru[iR];
            rw[iz*stride + halo + nx + k] = rw[iR];
            rT[iz*stride + halo + nx + k] = rT[iR];
        }
    }
    // z halos: zero-gradient for perturbations, reflective for w
    if (k < halo) {
        for (int ix = 0; ix < stride; ++ix) {
            int izB_i = halo + k;
            int izT_i = halo + nz - 1 - k;
            
            int iB = izB_i*stride + ix;
            int iT = izT_i*stride + ix;
            
            int izB_g = halo - 1 - k;
            int izT_g = halo + nz + k;
            
            // bottom
            r [izB_g*stride + ix] = r[iB] - rho_b[izB_i] + rho_b[izB_g];
            ru[izB_g*stride + ix] = ru[iB];
            rw[izB_g*stride + ix] = -rw[iB];
            rT[izB_g*stride + ix] = rT[iB] - rT_b[izB_i] + rT_b[izB_g];
            
            // top
            r [izT_g*stride + ix] = r[iT] - rho_b[izT_i] + rho_b[izT_g];
            ru[izT_g*stride + ix] = ru[iT];
            rw[izT_g*stride + ix] = -rw[iT];
            rT[izT_g*stride + ix] = rT[iT] - rT_b[izT_i] + rT_b[izT_g];
        }
    }
}

// ─── Kernel: x-direction Rusanov fluxes ───────────────────────────────────────
// Thread (ix, iz) computes flux at interface  ix+½  (between cells ix and ix+1).
// Interfaces: 0..nx  → (nx+1) interfaces, stored at d_Fx[iz*nx1 + ix]
// where nx1 = nx+1.

__global__ void k_x_fluxes(const Real* __restrict__ rho,
                             const Real* __restrict__ rhou,
                             const Real* __restrict__ rhow,
                             const Real* __restrict__ rhoTh,
                             Real* __restrict__ Fx_rho,
                             Real* __restrict__ Fx_rhou,
                             Real* __restrict__ Fx_rhow,
                             Real* __restrict__ Fx_rhoTh,
                             const Real* __restrict__ p_b,
                             const Real* __restrict__ rho_b,
                             const Real* __restrict__ rT_b,
                             int stride, int nz, int nx, int halo)
{
    // Thread computes interface f (0..nx)
    const int f  = blockIdx.x * blockDim.x + threadIdx.x;  // interface index
    const int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (f > nx || iz >= nz) return;

    // Left cell index in full array: il = halo + (f-1) in x,  halo + iz in z
    // f=0 → left cell is ghost halo-1 (il = halo-1), safe because WENO5 needs 2 more
    const int il  = halo + f - 1;  // left cell x-index in full array
    const int iz_ = halo + iz;     // z-index in full array

    // Load 5-point stencil in x
    auto load = [&](int dx_off, auto& q_arr, Real* out) {
        *out = q_arr[iz_ * stride + (il + dx_off)];
    };

    Real r[6], ru[6], rw[6], rT[6];
    for (int k = -2; k <= 3; ++k) {
        const int ix_k = il + k;
        r [k+2] = rho  [iz_*stride + ix_k] - rho_b[iz_];
        ru[k+2] = rhou [iz_*stride + ix_k];
        rw[k+2] = rhow [iz_*stride + ix_k];
        rT[k+2] = rhoTh[iz_*stride + ix_k] - rT_b[iz_];
    }

    // WENO5 reconstruction of perturbations at interface f+½
    const Real rL_p  = weno5L(r[0],r[1],r[2],r[3],r[4]);
    const Real rR_p  = weno5R(r[1],r[2],r[3],r[4],r[5]);
    const Real ruL = weno5L(ru[0],ru[1],ru[2],ru[3],ru[4]);
    const Real ruR = weno5R(ru[1],ru[2],ru[3],ru[4],ru[5]);
    const Real rwL = weno5L(rw[0],rw[1],rw[2],rw[3],rw[4]);
    const Real rwR = weno5R(rw[1],rw[2],rw[3],rw[4],rw[5]);
    const Real rTL_p = weno5L(rT[0],rT[1],rT[2],rT[3],rT[4]);
    const Real rTR_p = weno5R(rT[1],rT[2],rT[3],rT[4],rT[5]);

    // Add base state back (since x-derivatives of base state are zero, we just add the iz_ base state)
    const Real rL = rL_p + rho_b[iz_];
    const Real rR = rR_p + rho_b[iz_];
    const Real rTL = rTL_p + rT_b[iz_];
    const Real rTR = rTR_p + rT_b[iz_];

    // Primitive reconstruction
    const Real uL = (rL > 1.0e-10) ? ruL / rL : 0.0;
    const Real uR = (rR > 1.0e-10) ? ruR / rR : 0.0;
    const Real wL = (rL > 1.0e-10) ? rwL / rL : 0.0;
    const Real wR = (rR > 1.0e-10) ? rwR / rR : 0.0;
    const Real pL = pressure(rTL);
    const Real pR = pressure(rTR);
    const Real pb = p_b[iz_];

    const int fidx = iz * (nx+1) + f;
    Real F_rho, F_rhou, F_rhow, F_rhoTh;
    hllc_x(F_rho, F_rhou, F_rhow, F_rhoTh,
           rL, ruL, rwL, rTL, pL,
           rR, ruR, rwR, rTR, pR, pb);

    Fx_rho  [fidx] = F_rho;
    Fx_rhou [fidx] = F_rhou;
    Fx_rhow [fidx] = F_rhow;
    Fx_rhoTh[fidx] = F_rhoTh;
}

// ─── Kernel: z-direction Rusanov fluxes ───────────────────────────────────────
// Thread (ix, iz) computes flux at interface iz+½.
// Interfaces: 0..nz → (nz+1) interfaces, stored at d_Fz[iz * stride + (halo+ix)]

__global__ void k_z_fluxes(const Real* __restrict__ rho,
                             const Real* __restrict__ rhou,
                             const Real* __restrict__ rhow,
                             const Real* __restrict__ rhoTh,
                             Real* __restrict__ Fz_rho,
                             Real* __restrict__ Fz_rhou,
                             Real* __restrict__ Fz_rhow,
                             Real* __restrict__ Fz_rhoTh,
                             const Real* __restrict__ p_b,
                             const Real* __restrict__ rho_b,
                             const Real* __restrict__ rT_b,
                             const Real* __restrict__ pi_b,
                             int stride, int nz, int nx, int halo, Real dz)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;  // x index (interior)
    const int f  = blockIdx.y * blockDim.y + threadIdx.y;  // interface 0..nz
    if (ix >= nx || f > nz) return;

    const int ix_ = halo + ix;     // x-index in full array

    // Bypass WENO5/HLLC at wall faces: exact hydrostatic pressure extrapolation.
    // This is the well-balanced fix — WENO5 produces a zero pressure gradient at
    // the wall ghost cells, leaving gravity unbalanced and causing exponential blowup.
    if (f == 0) {
        const int fidx  = f * stride + ix_;
        const int icell = halo * stride + ix_;          // bottom interior cell center
        const Real p_cell        = pressure(rhoTh[icell]);
        const Real p_prime       = p_cell - p_b[halo];
        const Real r_prime       = rho[icell] - rho_b[halo];
        const Real p_prime_wall  = p_prime + 0.5 * dz * r_prime * atm::g;
        Fz_rho  [fidx] = 0.0;
        Fz_rhou [fidx] = 0.0;
        Fz_rhow [fidx] = p_prime_wall;
        Fz_rhoTh[fidx] = 0.0;
        return;
    }
    if (f == nz) {
        const int fidx  = f * stride + ix_;
        const int icell = (halo + nz - 1) * stride + ix_;  // top interior cell center
        const Real p_cell        = pressure(rhoTh[icell]);
        const Real p_prime       = p_cell - p_b[halo + nz - 1];
        const Real r_prime       = rho[icell] - rho_b[halo + nz - 1];
        const Real p_prime_wall  = p_prime - 0.5 * dz * r_prime * atm::g;
        Fz_rho  [fidx] = 0.0;
        Fz_rhou [fidx] = 0.0;
        Fz_rhow [fidx] = p_prime_wall;
        Fz_rhoTh[fidx] = 0.0;
        return;
    }

    // Bottom cell z-index in full array for this interface
    const int il  = halo + f - 1;  // z-index of bottom cell

    // Load 5-point stencil in z (stride = nx+2h per row)
    // Warp: threadIdx.x varies → all 32 threads read same-z rows → coalesced ✓
    Real r[6], ru[6], rw[6], rT[6];
    for (int k = -2; k <= 3; ++k) {
        const int iz_k = il + k;
        r [k+2] = rho  [iz_k * stride + ix_] - rho_b[iz_k];
        ru[k+2] = rhou [iz_k * stride + ix_];
        rw[k+2] = rhow [iz_k * stride + ix_];
        rT[k+2] = rhoTh[iz_k * stride + ix_] - rT_b[iz_k];
    }

    const Real rL_p  = weno5L(r[0],r[1],r[2],r[3],r[4]);
    const Real rR_p  = weno5R(r[1],r[2],r[3],r[4],r[5]);
    const Real ruL = weno5L(ru[0],ru[1],ru[2],ru[3],ru[4]);
    const Real ruR = weno5R(ru[1],ru[2],ru[3],ru[4],ru[5]);
    const Real rwL = weno5L(rw[0],rw[1],rw[2],rw[3],rw[4]);
    const Real rwR = weno5R(rw[1],rw[2],rw[3],rw[4],rw[5]);
    const Real rTL_p = weno5L(rT[0],rT[1],rT[2],rT[3],rT[4]);
    const Real rTR_p = weno5R(rT[1],rT[2],rT[3],rT[4],rT[5]);

    // Add base state back at the interface iz+1/2 using EXACT hydrostatic formula
    // pi_b is linear in z, so its arithmetic mean is exact at the interface
    const Real pi_b_f = 0.5 * (pi_b[il] + pi_b[il+1]);
    
    // Extract constant theta_bar from cell center il
    const Real theta_bar = atm::p0 / (atm::Rd * rho_b[il]) * pow(pi_b[il], atm::cv / atm::Rd);
    
    const Real rho_b_f = (atm::p0 / (atm::Rd * theta_bar)) * pow(pi_b_f, atm::cv / atm::Rd);
    const Real rT_b_f  = rho_b_f * theta_bar;
    const Real pb      = atm::p0 * pow(pi_b_f, atm::cp / atm::Rd);

    const Real rL = rL_p + rho_b_f;
    const Real rR = rR_p + rho_b_f;
    const Real rTL = rTL_p + rT_b_f;
    const Real rTR = rTR_p + rT_b_f;

    const Real uL = (rL > 1.0e-10) ? ruL / rL : 0.0;
    const Real uR = (rR > 1.0e-10) ? ruR / rR : 0.0;
    const Real wL = (rL > 1.0e-10) ? rwL / rL : 0.0;
    const Real wR = (rR > 1.0e-10) ? rwR / rR : 0.0;
    const Real pL = pressure(rTL);
    const Real pR = pressure(rTR);

    const int fidx = f * stride + ix_;
    Real F_rho, F_rhou, F_rhow, F_rhoTh;
    hllc_z(F_rho, F_rhou, F_rhow, F_rhoTh,
           rL, ruL, rwL, rTL, pL,
           rR, ruR, rwR, rTR, pR, pb);

    Fz_rho  [fidx] = F_rho;
    Fz_rhou [fidx] = F_rhou;
    Fz_rhow [fidx] = F_rhow;
    Fz_rhoTh[fidx] = F_rhoTh;
}

// ─── Kernel: SSP-RK3 stage update + gravity source ───────────────────────────
// dst = coeff_old * q_old  +  coeff_stage * (q_stage + dt * L(q_stage))
// Gravity source: only ρw equation gets -ρg term.

__global__ void k_rk_update(Real* dst_rho,  Real* dst_rhou,
                              Real* dst_rhow, Real* dst_rhoTh,
                              const Real* old_rho,  const Real* old_rhou,
                              const Real* old_rhow, const Real* old_rhoTh,
                              const Real* stg_rho,  const Real* stg_rhou,
                              const Real* stg_rhow, const Real* stg_rhoTh,
                              const Real* Fx_rho,   const Real* Fx_rhou,
                              const Real* Fx_rhow,  const Real* Fx_rhoTh,
                              const Real* Fz_rho,   const Real* Fz_rhou,
                              const Real* Fz_rhow,  const Real* Fz_rhoTh,
                              const Real* rho_b,
                              Real dt, Real inv_dx, Real inv_dz,
                              Real coeff_old, Real coeff_stage,
                              int stride, int nz, int nx, int halo)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iz >= nz) return;

    const int ix_ = halo + ix;
    const int iz_ = halo + iz;
    const int fix = iz * (nx + 1) + ix;
    const int fiz = iz * stride + ix_;
    const int gi  = iz_ * stride + ix_;

    // L(q) = -(dFx/dx + dFz/dz)
    const Real Lrho  = -(Fx_rho  [fix+1] - Fx_rho  [fix])  * inv_dx
                      -(Fz_rho  [fiz+stride] - Fz_rho  [fiz]) * inv_dz;
    const Real Lrhou = -(Fx_rhou [fix+1] - Fx_rhou [fix])  * inv_dx
                      -(Fz_rhou [fiz+stride] - Fz_rhou [fiz]) * inv_dz;
    const Real Lrhow = -(Fx_rhow [fix+1] - Fx_rhow [fix])  * inv_dx
                      -(Fz_rhow [fiz+stride] - Fz_rhow [fiz]) * inv_dz
                      - (stg_rho[gi] - rho_b[iz+halo]) * atm::g;  // gravity source: -ρ'g
    const Real LrhoTh= -(Fx_rhoTh[fix+1] - Fx_rhoTh[fix]) * inv_dx
                      -(Fz_rhoTh[fiz+stride] - Fz_rhoTh[fiz]) * inv_dz;

    dst_rho  [gi] = coeff_old * old_rho  [gi] + coeff_stage*(stg_rho  [gi] + dt*Lrho  );
    dst_rhou [gi] = coeff_old * old_rhou [gi] + coeff_stage*(stg_rhou [gi] + dt*Lrhou );
    dst_rhow [gi] = coeff_old * old_rhow [gi] + coeff_stage*(stg_rhow [gi] + dt*Lrhow );
    dst_rhoTh[gi] = coeff_old * old_rhoTh[gi] + coeff_stage*(stg_rhoTh[gi] + dt*LrhoTh);

    // Positivity floor
    if (dst_rho[gi] < 1.0e-10) dst_rho[gi] = 1.0e-10;
}

// ─── Kernel: max wave speed reduction ────────────────────────────────────────
__global__ void k_wave_speed(const Real* __restrict__ rho,
                              const Real* __restrict__ rhou,
                              const Real* __restrict__ rhow,
                              const Real* __restrict__ rhoTh,
                              Real* __restrict__ smax_out,
                              int stride, int nx, int nz, int halo)
{
    extern __shared__ Real sdata[];
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iz = blockIdx.y * blockDim.y + threadIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    Real s = 0.0;
    if (ix < nx && iz < nz) {
        const int gi = (iz+halo)*stride + (ix+halo);
        const Real r = rho[gi];
        const Real u = (r > 1.0e-10) ? rhou[gi]/r : 0.0;
        const Real w = (r > 1.0e-10) ? rhow[gi]/r : 0.0;
        const Real a = sound_speed(r, rhoTh[gi]);
        s = max(fabs(u)+a, fabs(w)+a);
    }
    sdata[tid] = s;
    __syncthreads();

    // Block reduction
    for (int stride2 = blockDim.x*blockDim.y/2; stride2 > 0; stride2 >>= 1) {
        if (tid < stride2) sdata[tid] = max(sdata[tid], sdata[tid+stride2]);
        __syncthreads();
    }
    if (tid == 0) atomicMax((unsigned long long*)smax_out,
                             __double_as_longlong(sdata[0]));
}

// ─── Constructor / Destructor ─────────────────────────────────────────────────

Euler2D_GPU::Euler2D_GPU(int nx, int nz, Real dx, Real dz, Real cfl)
    : cfl_(cfl), t_(0.0)
{
    g_.nx   = nx; g_.nz   = nz;
    g_.dx   = dx; g_.dz   = dz;
    g_.halo = atm::HALO2;
    alloc_arrays();
    CK(cudaEventCreate(&ev_smax_ready_));
}

Euler2D_GPU::~Euler2D_GPU() {
    free_arrays();
    cudaEventDestroy(ev_smax_ready_);
}

void Euler2D_GPU::alloc_arrays() {
    const size_t N  = (size_t)g_.size() * sizeof(Real);
    const size_t Nx = (size_t)(g_.nx+1) * g_.nz * sizeof(Real);
    const size_t Nz = (size_t)g_.size() * sizeof(Real);  // same stride, nz+1 rows

    auto alloc = [&](Real*& p, size_t bytes) {
        CK(cudaMalloc(&p, bytes));
        CK(cudaMemset(p, 0, bytes));
    };

    alloc(d_rho_,    N); alloc(d_rhou_,    N);
    alloc(d_rhow_,   N); alloc(d_rhoTh_,   N);
    alloc(d_rho1_,   N); alloc(d_rhou1_,   N);
    alloc(d_rhow1_,  N); alloc(d_rhoTh1_,  N);
    alloc(d_rho2_,   N); alloc(d_rhou2_,   N);
    alloc(d_rhow2_,  N); alloc(d_rhoTh2_,  N);

    alloc(d_Fx_rho_,   Nx); alloc(d_Fx_rhou_,  Nx);
    alloc(d_Fx_rhow_,  Nx); alloc(d_Fx_rhoTh_, Nx);
    alloc(d_Fz_rho_,   Nz); alloc(d_Fz_rhou_,  Nz);
    alloc(d_Fz_rhow_,  Nz); alloc(d_Fz_rhoTh_, Nz);

    CK(cudaMalloc(&d_smax_, sizeof(Real)));
    CK(cudaHostAlloc(&h_smax_, sizeof(Real), cudaHostAllocDefault));
}

void Euler2D_GPU::free_arrays() {
    auto fr = [](Real* p){ if(p) cudaFree(p); };
    fr(d_rho_);   fr(d_rhou_);   fr(d_rhow_);   fr(d_rhoTh_);
    fr(d_rho1_);  fr(d_rhou1_);  fr(d_rhow1_);  fr(d_rhoTh1_);
    fr(d_rho2_);  fr(d_rhou2_);  fr(d_rhow2_);  fr(d_rhoTh2_);
    fr(d_Fx_rho_); fr(d_Fx_rhou_); fr(d_Fx_rhow_); fr(d_Fx_rhoTh_);
    fr(d_Fz_rho_); fr(d_Fz_rhou_); fr(d_Fz_rhow_); fr(d_Fz_rhoTh_);
    fr(d_smax_);
    fr(base_.rho_b); fr(base_.pi_b); fr(base_.p_b); fr(base_.theta_b);
    if (h_smax_) cudaFreeHost(h_smax_);
}

// ─── set_state ────────────────────────────────────────────────────────────────

void Euler2D_GPU::set_state(const std::vector<Real>& rho,
                              const std::vector<Real>& rhou,
                              const std::vector<Real>& rhow,
                              const std::vector<Real>& rhoTh)
{
    const int h = g_.halo;
    const int s = g_.stride();
    for (int iz = 0; iz < g_.nz; ++iz) {
        const int row_dev = (iz + h) * s + h;
        CK(cudaMemcpy(d_rho_   + row_dev, rho   .data() + iz*g_.nx, g_.nx*sizeof(Real), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_rhou_  + row_dev, rhou  .data() + iz*g_.nx, g_.nx*sizeof(Real), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_rhow_  + row_dev, rhow  .data() + iz*g_.nx, g_.nx*sizeof(Real), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_rhoTh_ + row_dev, rhoTh .data() + iz*g_.nx, g_.nx*sizeof(Real), cudaMemcpyHostToDevice));
    }
    launch_fill_halos(d_rho_, d_rhou_, d_rhow_, d_rhoTh_);
    CK(cudaDeviceSynchronize());
    t_ = 0.0;
}

void Euler2D_GPU::set_base_state(const std::vector<Real>& rho_b,
                                 const std::vector<Real>& pi_b)
{
    const int nz_full = g_.nz + 2 * g_.halo;
    CK(cudaMalloc(&base_.rho_b, nz_full * sizeof(Real)));
    CK(cudaMalloc(&base_.pi_b,  nz_full * sizeof(Real)));
    CK(cudaMalloc(&base_.p_b,   nz_full * sizeof(Real)));
    CK(cudaMalloc(&base_.theta_b, nz_full * sizeof(Real)));
    
    std::vector<Real> p_b(nz_full);
    std::vector<Real> rT_b(nz_full);
    for (int iz = 0; iz < nz_full; ++iz) {
        p_b[iz] = atm::pressure_from_exner(pi_b[iz]);
        rT_b[iz] = p_b[iz] / (atm::Rd * pi_b[iz]);
    }
    
    CK(cudaMemcpy(base_.rho_b, rho_b.data(), nz_full * sizeof(Real), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(base_.pi_b,  pi_b.data(),  nz_full * sizeof(Real), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(base_.p_b,   p_b.data(),   nz_full * sizeof(Real), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(base_.theta_b, rT_b.data(), nz_full * sizeof(Real), cudaMemcpyHostToDevice));
}

// ─── get_state ────────────────────────────────────────────────────────────────

static std::vector<Real> download_interior(const Real* d_arr, const Grid2D& g) {
    CK(cudaDeviceSynchronize());
    const int h = g.halo, s = g.stride();
    std::vector<Real> out(g.nx * g.nz);
    for (int iz = 0; iz < g.nz; ++iz)
        CK(cudaMemcpy(out.data() + iz*g.nx,
                      d_arr + (iz+h)*s + h, g.nx*sizeof(Real), cudaMemcpyDeviceToHost));
    return out;
}
std::vector<Real> Euler2D_GPU::get_rho()   const { return download_interior(d_rho_,   g_); }
std::vector<Real> Euler2D_GPU::get_rhou()  const { return download_interior(d_rhou_,  g_); }
std::vector<Real> Euler2D_GPU::get_rhow()  const { return download_interior(d_rhow_,  g_); }
std::vector<Real> Euler2D_GPU::get_rhoTh() const { return download_interior(d_rhoTh_, g_); }

// ─── Launch helpers ───────────────────────────────────────────────────────────

void Euler2D_GPU::launch_fill_halos(Real* r, Real* ru, Real* rw, Real* rT) {
    const int h = g_.halo;
    dim3 thr(64); dim3 blk((h + 63)/64);
    // Since base_.theta_b contains rT_b (we will allocate it next), we pass it here.
    k_fill_halos<<<blk,thr>>>(r, ru, rw, rT, base_.rho_b, base_.theta_b, g_.stride(), g_.nz+2*h, g_.nx, g_.nz, h);
}

void Euler2D_GPU::launch_x_fluxes(const Real* r, const Real* ru,
                                    const Real* rw, const Real* rT,
                                    const Real* p_b, const Real* rho_b, const Real* rT_b)
{
    dim3 thr(32, 8);
    dim3 blk((g_.nx+1+31)/32, (g_.nz+7)/8);
    k_x_fluxes<<<blk,thr>>>(r,ru,rw,rT,
        d_Fx_rho_,d_Fx_rhou_,d_Fx_rhow_,d_Fx_rhoTh_, p_b, rho_b, rT_b,
        g_.stride(), g_.nz, g_.nx, g_.halo);
}

void Euler2D_GPU::launch_z_fluxes(const Real* r, const Real* ru,
                                    const Real* rw, const Real* rT,
                                    const Real* p_b, const Real* rho_b, const Real* rT_b, const Real* pi_b)
{
    dim3 thr(32, 8);
    dim3 blk((g_.nx+31)/32, (g_.nz+1+7)/8);
    k_z_fluxes<<<blk,thr>>>(r,ru,rw,rT,
        d_Fz_rho_,d_Fz_rhou_,d_Fz_rhow_,d_Fz_rhoTh_, p_b, rho_b, rT_b, pi_b,
        g_.stride(), g_.nz, g_.nx, g_.halo, g_.dz);
}

void Euler2D_GPU::launch_rk_update(
    Real* dr,  Real* dru,  Real* drw,  Real* drT,
    const Real* or_, const Real* oru, const Real* orw, const Real* orT,
    const Real* sr,  const Real* sru, const Real* srw, const Real* srT,
    const Real* rho_b,
    Real dt, Real coeff_old, Real coeff_stage)
{
    dim3 thr(32, 8);
    dim3 blk((g_.nx+31)/32, (g_.nz+7)/8);
    k_rk_update<<<blk,thr>>>(dr,dru,drw,drT, or_,oru,orw,orT, sr,sru,srw,srT,
        d_Fx_rho_,d_Fx_rhou_,d_Fx_rhow_,d_Fx_rhoTh_,
        d_Fz_rho_,d_Fz_rhou_,d_Fz_rhow_,d_Fz_rhoTh_,
        rho_b, dt, 1.0/g_.dx, 1.0/g_.dz,
        coeff_old, coeff_stage,
        g_.stride(), g_.nz, g_.nx, g_.halo);
}

// ─── Async dt ─────────────────────────────────────────────────────────────────

void Euler2D_GPU::async_compute_dt(const Real* d_rho, const Real* d_rhoTh) {
    CK(cudaMemset(d_smax_, 0, sizeof(Real)));
    dim3 thr(32,8); dim3 blk((g_.nx+31)/32, (g_.nz+7)/8);
    const int smem = 32*8*sizeof(Real);
    k_wave_speed<<<blk,thr,smem>>>(d_rho, d_rhou_, d_rhow_, d_rhoTh, d_smax_,
                                    g_.stride(), g_.nx, g_.nz, g_.halo);
    CK(cudaMemcpyAsync(h_smax_, d_smax_, sizeof(Real), cudaMemcpyDeviceToHost));
    CK(cudaEventRecord(ev_smax_ready_));
    smax_pending_ = true;
}

void Euler2D_GPU::collect_dt() {
    if (!smax_pending_) return;
    CK(cudaEventSynchronize(ev_smax_ready_));
    smax_pending_ = false;
    const Real smax = *h_smax_;
    if (smax > 0.0) {
        const Real dt_x = cfl_ * g_.dx / smax;
        const Real dt_z = cfl_ * g_.dz / smax;
        dt_ = min(dt_x, dt_z);
    } else {
        dt_ = 1.0e30;
    }
}

// ─── step ─────────────────────────────────────────────────────────────────────
// Wicker-Skamarock SSP-RK3 (Shu-Osher coefficients):
//   q1 = q  + dt * L(q)
//   q2 = 3/4 q + 1/4 (q1 + dt * L(q1))
//   q  = 1/3 q + 2/3 (q2 + dt * L(q2))

void Euler2D_GPU::step() {
    if (step_n_ % DT_UPDATE_FREQ == 0) {
        collect_dt();
        async_compute_dt(d_rho_, d_rhoTh_);
        if (dt_ == 0.0) { collect_dt(); }  // first step: block
    }

    const Real dt = dt_;

    // Stage 1
    launch_x_fluxes(d_rho_, d_rhou_, d_rhow_, d_rhoTh_, base_.p_b, base_.rho_b, base_.theta_b);
    launch_z_fluxes(d_rho_, d_rhou_, d_rhow_, d_rhoTh_, base_.p_b, base_.rho_b, base_.theta_b, base_.pi_b);
    launch_rk_update(d_rho1_,d_rhou1_,d_rhow1_,d_rhoTh1_,
                     d_rho_, d_rhou_, d_rhow_, d_rhoTh_,
                     d_rho_, d_rhou_, d_rhow_, d_rhoTh_,
                     base_.rho_b, dt, 0.0, 1.0);
    launch_fill_halos(d_rho1_, d_rhou1_, d_rhow1_, d_rhoTh1_);

    // Stage 2
    launch_x_fluxes(d_rho1_, d_rhou1_, d_rhow1_, d_rhoTh1_, base_.p_b, base_.rho_b, base_.theta_b);
    launch_z_fluxes(d_rho1_, d_rhou1_, d_rhow1_, d_rhoTh1_, base_.p_b, base_.rho_b, base_.theta_b, base_.pi_b);
    launch_rk_update(d_rho2_,d_rhou2_,d_rhow2_,d_rhoTh2_,
                     d_rho_, d_rhou_, d_rhow_, d_rhoTh_,
                     d_rho1_,d_rhou1_,d_rhow1_,d_rhoTh1_,
                     base_.rho_b, dt, 0.75, 0.25);
    launch_fill_halos(d_rho2_, d_rhou2_, d_rhow2_, d_rhoTh2_);

    // Stage 3
    launch_x_fluxes(d_rho2_, d_rhou2_, d_rhow2_, d_rhoTh2_, base_.p_b, base_.rho_b, base_.theta_b);
    launch_z_fluxes(d_rho2_, d_rhou2_, d_rhow2_, d_rhoTh2_, base_.p_b, base_.rho_b, base_.theta_b, base_.pi_b);
    launch_rk_update(d_rho_, d_rhou_, d_rhow_, d_rhoTh_,
                     d_rho_, d_rhou_, d_rhow_, d_rhoTh_,
                     d_rho2_,d_rhou2_,d_rhow2_,d_rhoTh2_,
                     base_.rho_b, dt, 1.0/3.0, 2.0/3.0);
    launch_fill_halos(d_rho_, d_rhou_, d_rhow_, d_rhoTh_);

    t_ += dt;
    ++step_n_;
}
