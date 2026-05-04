#pragma once
#include "types.hpp"
#include <cmath>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

// ─── Atmospheric constants ────────────────────────────────────────────────────
namespace atm {

constexpr Real Rd    = 287.0;       // dry air gas constant [J/(kg·K)]
constexpr Real cp    = 1004.0;      // specific heat at const pressure [J/(kg·K)]
constexpr Real cv    = 717.0;       // specific heat at const volume [J/(kg·K)]
constexpr Real gamma = cp / cv;     // ratio of specific heats
constexpr Real p0    = 1.0e5;       // reference pressure [Pa]
constexpr Real g     = 9.81;        // gravity [m/s²]

// WENO5 needs 3-cell halo each side
constexpr int HALO2 = 3;

// Exner pressure from density and potential temperature
__host__ __device__ inline Real exner_from_rhoTheta(Real rhoTheta) {
    // π = (Rd * ρθ / p0)^(Rd/cv)
    // Physical derivation check: 
    // p = rho*Rd*T = rho*Rd*theta*pi. Since pi = (p/p0)^(Rd/cp),
    // substituting p gives pi = (rho*Rd*theta/p0)^(Rd/cv). Correct.
    return std::pow(Rd * rhoTheta / p0, Rd / cv);
}

// Pressure from Exner
__host__ __device__ inline Real pressure_from_exner(Real pi) {
    return p0 * std::pow(pi, cp / Rd);
}

} // namespace atm

// ─── 2D grid descriptor (passed by value to kernels) ─────────────────────────
struct Grid2D {
    int  nx, nz;          // interior cells
    Real dx, dz;          // cell sizes [m]
    int  halo;            // ghost cells (= atm::HALO2)

    // Layout: [nz+2h][nx+2h] → x is the fast dimension (unit stride)
    // This is critical for memory coalescing in CUDA where threadIdx.x 
    // maps to the x-direction.
    [[nodiscard]] __host__ __device__ int stride() const { return nx + 2 * halo; }
    // Total elements per variable
    [[nodiscard]] __host__ __device__ int size() const {
        return (nx + 2 * halo) * (nz + 2 * halo);
    }
    // Flat index: layout [nz+2h][nx+2h], coalesced in x
    [[nodiscard]] __host__ __device__ int idx(int ix, int iz) const {
        return (iz + halo) * stride() + (ix + halo);
    }
};

// ─── SoA state arrays (host or device) ───────────────────────────────────────
// Each pointer has length grid.size().  ix in [-halo, nx+halo), iz likewise.
// Layout: [nz+2h][nx+2h] → index = (iz+halo)*(nx+2h) + (ix+halo)
struct State2D {
    Real* rho   = nullptr;   // density              [kg/m³]
    Real* rhou  = nullptr;   // x-momentum density   [kg/(m²·s)]
    Real* rhow  = nullptr;   // z-momentum density   [kg/(m²·s)]
    Real* rhoTh = nullptr;   // ρθ                   [kg·K/m³]
    // π (Exner pressure perturbation) is diagnosed each stage from rhoTh
};

// ─── Base-state profiles (1D in z, device-accessible) ────────────────────────
struct BaseState {
    Real* rho_b   = nullptr;   // [nz + 2*halo]
    Real* theta_b = nullptr;
    Real* pi_b    = nullptr;
    Real* p_b     = nullptr;
};
