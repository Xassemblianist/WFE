#pragma once
#include "../src/types2d.hpp"
#include <vector>
#include <cmath>
#include <string>

namespace cases {

// ─── Robert (1993) cold-bubble density current ───────────────────────────────
// Domain: x ∈ [-12.8, 12.8] km,  z ∈ [0, 6.4] km
// Background: θ̄ = 300 K, ρ̄ from hydrostatic balance
// Perturbation: θ' = -15 · cos²(πr/2)  for r ≤ 1,  else 0
//   r = sqrt( ((x-xc)/4000)² + ((z-zc)/2000)² )
//   centre (xc, zc) = (0, 3000) m

struct DensityCurrentParams {
    int  nx             = 256;         // interior cells
    int  nz             = 64;
    Real dx             = 100.0;       // [m]
    Real dz             = 100.0;       // [m]
    Real cfl            = 0.4;
    Real tend           = 900.0;       // [s]  (~15 min)
    Real output_interval = 60.0;       // [s]
    Real x0             = -12800.0;    // domain left [m]
    Real theta_bar      = 300.0;       // background potential temperature [K]
    Real dtheta_cold    = -15.0;       // cold bubble θ perturbation [K]
    Real xc             = 0.0;         // bubble centre x [m]
    Real zc             = 3000.0;      // bubble centre z [m]
    Real rx             = 4000.0;      // bubble half-width x [m]
    Real rz             = 2000.0;      // bubble half-width z [m]
    std::string output_dir = "";
};

// ─── Hydrostatic base state ───────────────────────────────────────────────────
// Dry atmosphere, θ̄ = const → exponential ρ̄ profile.
// Integrate downward from p(ztop) = p_ref_top analytically.
// With θ = const: T = θ·π, π decreases linearly with z.
//   dπ/dz = -g/(cp·θ)  → π(z) = π_sfc - (g/(cp·θ))·z
//   ρ(z) = p0/(Rd·θ) · π^(cv/Rd)

inline void make_base_state(const DensityCurrentParams& p,
                             std::vector<Real>& rho_b,
                             std::vector<Real>& pi_b,
                             int halo)
{
    const int nz_full = p.nz + 2 * halo;
    rho_b.resize(nz_full);
    pi_b .resize(nz_full);

    // π at z=0 from surface pressure ≈ p0
    const Real pi_sfc = 1.0;  // Exner pressure at surface (π(p=p0) = 1)
    const Real dpi_dz = -atm::g / (atm::cp * p.theta_bar);

    for (int iz = -halo; iz < p.nz + halo; ++iz) {
        const Real z  = (iz + 0.5) * p.dz;
        const Real pi = pi_sfc + dpi_dz * z;
        // ρ̄ = p0/(Rd·θ̄) · π^(cv/Rd)
        const Real rho = (atm::p0 / (atm::Rd * p.theta_bar))
                       * std::pow(pi, atm::cv / atm::Rd);
        rho_b[iz + halo] = rho;
        pi_b [iz + halo] = pi;
    }
}

// ─── Build initial condition ──────────────────────────────────────────────────
// Returns flat arrays length nx*nz, x-fastest (row-major ix)
inline void make_ic(const DensityCurrentParams& p,
                    std::vector<Real>& rho,
                    std::vector<Real>& rhou,
                    std::vector<Real>& rhow,
                    std::vector<Real>& rhoTh)
{
    const int N = p.nx * p.nz;
    rho  .assign(N, 0.0);
    rhou .assign(N, 0.0);
    rhow .assign(N, 0.0);
    rhoTh.assign(N, 0.0);

    const Real pi_sfc = 1.0;
    const Real dpi_dz = -atm::g / (atm::cp * p.theta_bar);

    for (int iz = 0; iz < p.nz; ++iz) {
        const Real zc = (iz + 0.5) * p.dz;
        const Real pi = pi_sfc + dpi_dz * zc;
        const Real rho_b = (atm::p0 / (atm::Rd * p.theta_bar))
                         * std::pow(pi, atm::cv / atm::Rd);

        for (int ix = 0; ix < p.nx; ++ix) {
            const Real xc = p.x0 + (ix + 0.5) * p.dx;
            const Real r  = std::sqrt( ((xc - p.xc) / p.rx) * ((xc - p.xc) / p.rx)
                                     + ((zc - p.zc) / p.rz) * ((zc - p.zc) / p.rz) );
            const Real dtheta = (r <= 1.0)
                ? p.dtheta_cold * std::cos(0.5 * M_PI * r) * std::cos(0.5 * M_PI * r)
                : 0.0;
            const Real theta = p.theta_bar + dtheta;
            // ρ adjusted to keep p (and thus π) unchanged:
            // p = ρ Rd θ / p0)^(γ) * p0  → same π means ρ = ρ_b * θ_b/θ
            const Real rho_cell = rho_b * p.theta_bar / theta;

            const int idx = iz * p.nx + ix;
            rho  [idx] = rho_cell;
            rhou [idx] = 0.0;
            rhow [idx] = 0.0;
            rhoTh[idx] = rho_cell * theta;
        }
    }
}

} // namespace cases
