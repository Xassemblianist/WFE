#pragma once
#include "../src/types2d.hpp"
#include <vector>
#include <cmath>
#include <string>

namespace cases {

// ─── Schär et al. (2002) Mountain Wave test case ─────────────────────────────
// Reference: Schär, C. et al. (2002) J. Atmos. Sci. 59, 2078–2092
//
// Domain:   x ∈ [-25, 25] km,  z ∈ [0, 21 km]
// Background: isothermal atmosphere, T̄ = 250 K, Brunt-Väisälä N = 0.01 s⁻¹
// Uniform inflow: ū = 10 m/s
// Bell-shaped mountain: h(x) = h0 · exp(-(x/a)²) · cos²(πx/λ)
//   h0 = 250 m,  a = 5000 m,  λ = 4000 m
// Expected: steady-state orographic gravity wave pattern (no breaking)

struct MountainWaveParams {
    int  nx             = 500;          // interior cells (x: 25.6 km each side → 200m dx)
    int  nz             = 105;          // interior cells (z: 21 km, 200m dz)
    Real dx             = 100.0;        // [m]  (100m resolution)
    Real dz             = 200.0;        // [m]
    Real cfl            = 0.4;
    Real tend           = 10000.0;      // [s] — run to approximate steady state
    Real output_interval = 1000.0;      // [s]
    Real x0             = -25000.0;     // domain left [m]
    Real T_bar          = 250.0;        // isothermal background temperature [K]
    Real N_bv           = 0.01;         // Brunt-Väisälä frequency [1/s]
    Real u_bar          = 10.0;        // background horizontal wind [m/s]
    // Terrain parameters
    Real h0             = 250.0;        // mountain peak height [m]
    Real a              = 5000.0;       // mountain half-width [m]
    Real lambda_m       = 4000.0;       // cosine modulation wavelength [m]
    // Rayleigh damping (sponge) layer
    Real sponge_top     = 15000.0;      // bottom of sponge layer [m]
    Real tau_sponge     = 100.0;        // damping time scale [s]
    std::string output_dir = "";
};

// ─── Terrain height h(x) ────────────────────────────────────────────────────
inline Real mountain_height(Real x, const MountainWaveParams& p) {
    const Real bell = std::exp(-(x / p.a) * (x / p.a));
    const Real cos2 = std::cos(M_PI * x / p.lambda_m);
    return p.h0 * bell * cos2 * cos2;
}

// ─── Constant-N base state (Schär 2002) ──────────────────────────────────────
// For constant Brunt-Väisälä frequency N:
//   θ̄(z) = θ₀ · exp(N²z/g)                 [monotonically increasing]
//   dπ/dz = -g/(cp·θ̄(z))
//   π(z)  = 1 + g²/(cp·θ₀·N²) · [exp(-N²z/g) - 1]   [analytic integral]
//   ρ̄(z)  = p0/(Rd·θ̄(z)) · π(z)^(cv/Rd)
// With θ₀=280K, N=0.01/s this matches Schär et al. (2002).
inline void make_base_state_mw(const MountainWaveParams& p,
                                std::vector<Real>& rho_b,
                                std::vector<Real>& pi_b,
                                int halo)
{
    const int nz_full = p.nz + 2 * halo;
    rho_b.resize(nz_full);
    pi_b .resize(nz_full);

    const Real theta0 = p.T_bar;          // surface θ₀ [K]
    const Real N2     = p.N_bv * p.N_bv;  // N² [s⁻²]
    // Coefficient for analytic π(z) integration
    const Real coeff  = atm::g * atm::g / (atm::cp * theta0 * N2);

    for (int iz = -halo; iz < p.nz + halo; ++iz) {
        const Real z     = (iz + 0.5) * p.dz;
        const Real theta = theta0 * std::exp(N2 * z / atm::g);
        const Real pi    = 1.0 + coeff * (std::exp(-N2 * z / atm::g) - 1.0);
        const Real rho   = (atm::p0 / (atm::Rd * theta)) * std::pow(pi, atm::cv / atm::Rd);
        rho_b[iz + halo] = rho;
        pi_b [iz + halo] = pi;
    }
}

// ─── Build initial condition ──────────────────────────────────────────────────
// Uniform horizontal flow ū over the constant-N base state, no perturbations.
inline void make_ic_mw(const MountainWaveParams& p,
                       std::vector<Real>& rho,
                       std::vector<Real>& rhou,
                       std::vector<Real>& rhow,
                       std::vector<Real>& rhoTh)
{
    const int N_cells = p.nx * p.nz;
    rho  .assign(N_cells, 0.0);
    rhou .assign(N_cells, 0.0);
    rhow .assign(N_cells, 0.0);
    rhoTh.assign(N_cells, 0.0);

    const Real N2    = p.N_bv * p.N_bv;
    const Real coeff = atm::g * atm::g / (atm::cp * p.T_bar * N2);

    for (int iz = 0; iz < p.nz; ++iz) {
        const Real z     = (iz + 0.5) * p.dz;
        const Real theta = p.T_bar * std::exp(N2 * z / atm::g);
        const Real pi    = 1.0 + coeff * (std::exp(-N2 * z / atm::g) - 1.0);
        const Real rho_b = (atm::p0 / (atm::Rd * theta)) * std::pow(pi, atm::cv / atm::Rd);

        for (int ix = 0; ix < p.nx; ++ix) {
            const int idx = iz * p.nx + ix;
            rho  [idx] = rho_b;
            rhou [idx] = rho_b * p.u_bar;
            rhow [idx] = 0.0;
            rhoTh[idx] = rho_b * theta;
        }
    }
}

} // namespace cases
