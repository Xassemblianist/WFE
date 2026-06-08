#pragma once
#include "../../types3d.hpp"
#include <cuda_runtime.h>
#include <array>
#include <vector>

// ─── GPU solver: 3D non-hydrostatic compressible Euler ────────────────────────
// Scheme: WENO5 · HLLC · split-explicit SSP-RK3 (same design as 2D solver)
// Layout: SoA [nz+2h][ny+2h][nx+2h], x fastest dimension
// 5 prognostic variables: ρ, ρu, ρv, ρw, ρθ
class Euler3D_GPU {
public:
    static constexpr int N_SPLIT        = 10;
    static constexpr int DT_UPDATE_FREQ = 1;

    Euler3D_GPU(int nx, int ny, int nz, Real dx, Real dy, Real dz, Real cfl = 0.4);
    ~Euler3D_GPU();

    Euler3D_GPU(const Euler3D_GPU&)            = delete;
    Euler3D_GPU& operator=(const Euler3D_GPU&) = delete;

    void set_state(const std::vector<Real>& rho,
                   const std::vector<Real>& rhou,
                   const std::vector<Real>& rhov,
                   const std::vector<Real>& rhow,
                   const std::vector<Real>& rhoTh);

    void set_base_state(const std::vector<Real>& rho_b,
                        const std::vector<Real>& pi_b);

    void set_sponge_layer(Real z_bot, Real z_top, Real tau, Real u_bar = 0.0);

    void set_diffusion(Real K_m, Real K_theta = -1.0);
    void set_smagorinsky(Real Cs = 0.18, Real Prt = 1.0/3.0);

    void step();

    [[nodiscard]] std::array<Real,4> get_diagnostics() const;

    [[nodiscard]] std::vector<Real> get_rho()   const;
    [[nodiscard]] std::vector<Real> get_rhou()  const;
    [[nodiscard]] std::vector<Real> get_rhov()  const;
    [[nodiscard]] std::vector<Real> get_rhow()  const;
    [[nodiscard]] std::vector<Real> get_rhoTh() const;

    [[nodiscard]] Real time()   const { return t_; }
    [[nodiscard]] int  step_n() const { return step_n_; }

private:
    Grid3D g_;
    Real   cfl_, t_;
    Real   dt_           = 0.0;
    int    step_n_       = 0;
    bool   smax_pending_ = false;

    Real sponge_z_bot_ = -1.0;
    Real sponge_z_top_ =  0.0;
    Real sponge_alpha_ =  0.0;
    Real sponge_u_bar_ =  0.0;

    Real K_m_          = 0.0;
    Real K_theta_      = 0.0;
    Real K_smag_coef_  = 0.0;
    Real Prt_          = 1.0/3.0;

    mutable Real* d_diag_ = nullptr;

    // ── Device SoA: current + 2 RK stages + slow tendencies ──────────────────
    Real *d_q_[5]  = {};   // rho, rhou, rhov, rhow, rhoTh
    Real *d_q1_[5] = {};
    Real *d_q2_[5] = {};
    Real *d_T_[5]  = {};   // slow tendencies (advection only)

    // ── Flux buffers (interior, no halos) ─────────────────────────────────────
    Real *d_Fx_[5] = {};   // [nz][ny][nx+1]
    Real *d_Fy_[5] = {};   // [nz][ny+1][nx]
    Real *d_Fz_[5] = {};   // [nz+1][ny][nx]

    // ── Base state (1D in z, length nz+2h) ────────────────────────────────────
    Real *d_rho_b_   = nullptr;
    Real *d_p_b_     = nullptr;
    Real *d_pi_b_    = nullptr;
    Real *d_rhoTh_b_ = nullptr;

    Real        *d_smax_ = nullptr;
    Real        *h_smax_ = nullptr;
    cudaEvent_t  ev_smax_ready_{};

    void alloc_arrays();
    void free_arrays();

    std::vector<Real> download_interior(const Real* d_ptr) const;

    void launch_fill_halos(Real** q);
    void launch_slow_tendencies(Real** q);
    void launch_acoustic_step(Real** q_sub, Real** q_old, Real dtt);
    void launch_rk_combine(Real** dst, Real** q_old, Real** q_new,
                            Real c_old, Real c_new);
    void launch_rayleigh_damping(Real** q, Real dt);
    void async_compute_dt(Real** q);
    void collect_dt();
};
