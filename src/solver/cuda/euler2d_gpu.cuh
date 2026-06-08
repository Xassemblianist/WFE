#pragma once
#include "../../types2d.hpp"
#include <cuda_runtime.h>
#include <array>
#include <vector>

// ─── GPU solver: 2D non-hydrostatic compressible Euler ────────────────────────
// Scheme: WENO5 reconstruction · HLLC flux · RK3 + Acoustic Sub-cycling
// Memory: SoA, layout [nz+2h][nx+2h]
class Euler2D_GPU {
public:
    static constexpr int  N_SPLIT       = 10;  // acoustic sub-steps per RK3 stage
    static constexpr int  DT_UPDATE_FREQ = 1;  // recompute CFL every N steps

    Euler2D_GPU(int nx, int nz, Real dx, Real dz, Real cfl = 0.4);
    ~Euler2D_GPU();

    Euler2D_GPU(const Euler2D_GPU&)            = delete;
    Euler2D_GPU& operator=(const Euler2D_GPU&) = delete;

    void set_state(const std::vector<Real>& rho,
                   const std::vector<Real>& rhou,
                   const std::vector<Real>& rhow,
                   const std::vector<Real>& rhoTh);

    void set_base_state(const std::vector<Real>& rho_b,
                        const std::vector<Real>& pi_b);

    void set_sponge_layer(Real z_bot, Real z_top, Real tau, Real u_bar = 0.0);
    void set_terrain(const std::vector<Real>& dhdx_arr);

    // Explicit diffusion: fixed eddy diffusivity (K_m for momentum, K_theta for θ)
    void set_diffusion(Real K_m, Real K_theta = -1.0);  // K_theta<0 → same as K_m
    // Dynamic Smagorinsky turbulence closure (Cs≈0.18, Prt≈1/3 for atmosphere)
    void set_smagorinsky(Real Cs = 0.18, Real Prt = 1.0/3.0);

    void step();

    [[nodiscard]] std::vector<Real> get_rho()   const;
    [[nodiscard]] std::vector<Real> get_rhou()  const;
    [[nodiscard]] std::vector<Real> get_rhow()  const;
    [[nodiscard]] std::vector<Real> get_rhoTh() const;

    // Conservation diagnostics: [total_mass, total_KE, total_PE, total_rhoTheta]
    // All in SI units, integrated over the 2D domain (per unit depth, × dx×dz each cell)
    [[nodiscard]] std::array<Real,4> get_diagnostics() const;

    [[nodiscard]] Real time() const { return t_; }

private:
    Grid2D g_;
    Real   cfl_, t_;
    Real   dt_      = 0.0;
    int    step_n_  = 0;

    // ── Device SoA arrays ─────────────────────────────────────────────────────
    Real *d_rho_   = nullptr, *d_rhou_   = nullptr,
         *d_rhow_  = nullptr, *d_rhoTh_  = nullptr;
    Real *d_rho1_  = nullptr, *d_rhou1_  = nullptr,
         *d_rhow1_ = nullptr, *d_rhoTh1_ = nullptr;
    Real *d_rho2_  = nullptr, *d_rhou2_  = nullptr,
         *d_rhow2_ = nullptr, *d_rhoTh2_ = nullptr;

    // ── Slow Tendency Buffers (Advection L(q)) ───────────────────────────────
    Real *d_Trho_  = nullptr, *d_Trhou_  = nullptr,
         *d_Trhow_ = nullptr, *d_TrhoTh_ = nullptr;

    // ── Flux buffers ──────────────────────────────────────────────────────────
    Real *d_Fx_rho_  = nullptr, *d_Fx_rhou_ = nullptr,
         *d_Fx_rhow_ = nullptr, *d_Fx_rhoTh_= nullptr;
    Real *d_Fz_rho_  = nullptr, *d_Fz_rhou_ = nullptr,
         *d_Fz_rhow_ = nullptr, *d_Fz_rhoTh_= nullptr;

    Real*       d_smax_  = nullptr;
    Real*       h_smax_  = nullptr;
    cudaEvent_t ev_smax_ready_;
    bool        smax_pending_ = false;

    mutable Real* d_diag_ = nullptr;   // 4 * nblocks reduction buffer

    BaseState   base_;

    Real  sponge_z_bot_   = -1.0;
    Real  sponge_z_top_   =  0.0;
    Real  sponge_alpha_   =  0.0;
    Real  sponge_u_bar_   =  0.0;

    // Diffusion parameters (Smagorinsky or fixed K)
    Real  K_m_         = 0.0;   // fixed momentum eddy diffusivity [m²/s]
    Real  K_theta_     = 0.0;   // fixed theta eddy diffusivity [m²/s]
    Real  K_smag_coef_ = 0.0;   // (Cs*Δ)² for Smagorinsky [m²]
    Real  Prt_         = 1.0/3.0; // turbulent Prandtl number

    Real* d_dhdx_         = nullptr;

    // ── Internal helpers ───────────────────────────────────────────────────────
    void alloc_arrays();
    void free_arrays();

    void launch_fill_halos(Real* d_rho, Real* d_rhou, Real* d_rhow, Real* d_rhoTh);
    
    void launch_x_fluxes(const Real* r, const Real* ru, const Real* rw, const Real* rT,
                          const Real* p_b, const Real* rho_b, const Real* rhoTh_b);
    void launch_z_fluxes(const Real* r, const Real* ru, const Real* rw, const Real* rT,
                          const Real* p_b, const Real* rho_b, const Real* rhoTh_b, const Real* pi_b);
    
    // Split-explicit helpers
    void launch_slow_tendencies(const Real* d_rho, const Real* d_rhou, 
                                 const Real* d_rhow, const Real* d_rhoTh);
    
    void launch_acoustic_step(Real* d_rho, Real* d_rhou, Real* d_rhow, Real* d_rhoTh, 
                               const Real* r0, const Real* ru0, const Real* rw0, const Real* rT0,
                               Real dtt);

    void launch_rk_update(Real* dst_rho,  Real* dst_rhou,
                           Real* dst_rhow, Real* dst_rhoTh,
                           const Real* old_rho,  const Real* old_rhou,
                           const Real* old_rhow, const Real* old_rhoTh,
                           const Real* stg_rho,  const Real* stg_rhou,
                           const Real* stg_rhow, const Real* stg_rhoTh,
                           Real dt, Real coeff_old, Real coeff_stage);

    void launch_rayleigh_damping(Real* d_rho, Real* d_rhou, Real* d_rhow, Real* d_rhoTh, Real dt);

    void async_compute_dt(const Real* d_rho, const Real* d_rhoTh);
    void collect_dt();
};
