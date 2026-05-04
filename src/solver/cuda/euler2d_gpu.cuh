#pragma once
#include "../../types2d.hpp"
#include <cuda_runtime.h>
#include <vector>

// ─── GPU solver: 2D non-hydrostatic compressible Euler ────────────────────────
// Scheme: WENO5 reconstruction · Rusanov flux · Wicker-Skamarock RK3
// Memory: SoA, layout [nz+2h][nx+2h], threadIdx.x → ix (coalesced in x & z)
//
// Acoustic sub-cycling (N_split Forward-Backward steps per RK3 stage) is
// implemented inside each stage to satisfy the acoustic CFL without shrinking Δt.
// NOTE FOR IMPLEMENTER: The current step() implementation in .cu file does 
// NOT yet include the sub-cycling loop. N_SPLIT is currently unused.
class Euler2D_GPU {
public:
    static constexpr int  N_SPLIT       = 6;   // acoustic sub-steps per RK3 stage
    static constexpr int  DT_UPDATE_FREQ = 1;  // recompute CFL every N steps

    Euler2D_GPU(int nx, int nz, Real dx, Real dz, Real cfl = 0.4);
    ~Euler2D_GPU();

    Euler2D_GPU(const Euler2D_GPU&)            = delete;
    Euler2D_GPU& operator=(const Euler2D_GPU&) = delete;

    // Upload interior IC (length nx*nz, row-major x-fastest)
    void set_state(const std::vector<Real>& rho,
                   const std::vector<Real>& rhou,
                   const std::vector<Real>& rhow,
                   const std::vector<Real>& rhoTh);

    // Upload base state (length nz)
    void set_base_state(const std::vector<Real>& rho_b,
                        const std::vector<Real>& pi_b);

    void step();

    // Download interior cells (length nx*nz, x-fastest)
    [[nodiscard]] std::vector<Real> get_rho()   const;
    [[nodiscard]] std::vector<Real> get_rhou()  const;
    [[nodiscard]] std::vector<Real> get_rhow()  const;
    [[nodiscard]] std::vector<Real> get_rhoTh() const;

    [[nodiscard]] Real time() const { return t_; }

private:
    Grid2D g_;
    Real   cfl_, t_;
    Real   dt_      = 0.0;
    int    step_n_  = 0;

    // ── Device SoA arrays (each length g_.size()) ─────────────────────────────
    Real *d_rho_   = nullptr, *d_rhou_   = nullptr,
         *d_rhow_  = nullptr, *d_rhoTh_  = nullptr;  // current step q^n
    Real *d_rho1_  = nullptr, *d_rhou1_  = nullptr,
         *d_rhow1_ = nullptr, *d_rhoTh1_ = nullptr;  // stage 1
    Real *d_rho2_  = nullptr, *d_rhou2_  = nullptr,
         *d_rhow2_ = nullptr, *d_rhoTh2_ = nullptr;  // stage 2

    // ── Flux buffers: (nx+1)*(nz+2h) for x-faces, (nx+2h)*(nz+1) for z-faces ─
    // Stored as flat arrays; indexed per variable
    Real *d_Fx_rho_  = nullptr, *d_Fx_rhou_ = nullptr,
         *d_Fx_rhow_ = nullptr, *d_Fx_rhoTh_= nullptr;
    Real *d_Fz_rho_  = nullptr, *d_Fz_rhou_ = nullptr,
         *d_Fz_rhow_ = nullptr, *d_Fz_rhoTh_= nullptr;

    // ── Async dt ──────────────────────────────────────────────────────────────
    Real*       d_smax_  = nullptr;
    Real*       h_smax_  = nullptr;  // pinned
    cudaEvent_t ev_smax_ready_;
    bool        smax_pending_ = false;

    // ── Base state ────────────────────────────────────────────────────────────
    BaseState   base_;

    // ── Internal helpers ───────────────────────────────────────────────────────
    void alloc_arrays();
    void free_arrays();

    void launch_fill_halos(Real* d_rho, Real* d_rhou,
                            Real* d_rhow, Real* d_rhoTh);
    void launch_x_fluxes(const Real* d_rho, const Real* d_rhou,
                          const Real* d_rhow, const Real* d_rhoTh,
                          const Real* p_b, const Real* rho_b, const Real* rT_b);
    void launch_z_fluxes(const Real* d_rho, const Real* d_rhou,
                          const Real* d_rhow, const Real* d_rhoTh,
                          const Real* p_b, const Real* rho_b, const Real* rT_b, const Real* pi_b);
    // RHS update: dst = coeff_old*q_old + coeff_stage*(q_stage + dt*L)
    void launch_rk_update(Real* dst_rho,  Real* dst_rhou,
                           Real* dst_rhow, Real* dst_rhoTh,
                           const Real* old_rho,  const Real* old_rhou,
                           const Real* old_rhow, const Real* old_rhoTh,
                           const Real* stg_rho,  const Real* stg_rhou,
                           const Real* stg_rhow, const Real* stg_rhoTh,
                           const Real* rho_b,
                           Real dt, Real coeff_old, Real coeff_stage);

    void async_compute_dt(const Real* d_rho, const Real* d_rhoTh);
    void collect_dt();
};
