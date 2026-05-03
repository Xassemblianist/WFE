#pragma once
#include "../swe1d.hpp"
#include <cuda_runtime.h>
#include <vector>

// ─── GPU solver: 1D SWE, WENO3 + HLLC + SSP-RK3 ─────────────────────────────
// Same scheme as the CPU reference but runs entirely on device memory.
// Designed for sm_75 (RTX 2060) through sm_120 (RTX 5070 Ti / Blackwell).
//
// Performance strategy: dt is recomputed asynchronously every DT_UPDATE_FREQ
// steps using pinned host memory + async copy, avoiding a blocking sync every
// step. The GPU pipeline runs freely between dt updates.
class SWE1D_GPU {
public:
    static constexpr int DT_UPDATE_FREQ = 32; // recompute CFL dt every N steps

    SWE1D_GPU(int nx, Real dx, Real cfl = 0.4);
    ~SWE1D_GPU();

    SWE1D_GPU(const SWE1D_GPU&)            = delete;
    SWE1D_GPU& operator=(const SWE1D_GPU&) = delete;

    void set_state(const std::vector<State>& ic);

    // Advance one SSP-RK3 step. Avoids blocking CPU-GPU sync in the hot path.
    void step();

    // Flush the GPU pipeline and copy interior cells to host.
    [[nodiscard]] std::vector<State> get_state() const;

    [[nodiscard]] Real time() const { return t_; }

private:
    int  nx_;
    Real dx_, cfl_, t_;
    Real dt_     = 0.0;  // current time step (updated every DT_UPDATE_FREQ steps)
    int  step_n_ = 0;

    // Device arrays (length = nx + 2*HALO)
    State* d_q_  = nullptr;
    State* d_q1_ = nullptr;
    State* d_q2_ = nullptr;

    // Reconstructed face values and fluxes (length = nx+1)
    State* d_qL_ = nullptr;
    State* d_qR_ = nullptr;
    Flux*  d_F_  = nullptr;

    // Async dt computation: device smax + pinned host mirror + CUDA event
    Real*        d_smax_   = nullptr;  // device reduction result
    Real*        h_smax_   = nullptr;  // pinned host buffer
    cudaEvent_t  ev_smax_ready_;       // signals when async copy is done
    bool         smax_pending_ = false;

    void launch_fill_halos(State* d_q);
    void launch_reconstruct(const State* d_q);
    void launch_fluxes();
    void launch_update(State* d_dst, const State* d_q_old,
                       const State* d_q_stage, Real dt,
                       Real coeff_old, Real coeff_stage);

    // Kick off async dt computation (non-blocking).
    void async_compute_dt();
    // Poll (may block once) for the pending dt result.
    void collect_dt();
};
