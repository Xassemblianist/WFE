#pragma once
#include "../types.hpp"
#include <vector>

// ─── CPU reference solver  (1D SWE, WENO3 + HLLC + SSP-RK3) ─────────────────
class SWE1D_CPU {
public:
    SWE1D_CPU(int nx, Real dx, Real cfl = 0.4);

    void set_state(const std::vector<State>& ic);
    void step();                            // one complete SSP-RK3 step

    [[nodiscard]] const std::vector<State>& state() const { return q_; }
    [[nodiscard]] Real compute_dt() const;
    [[nodiscard]] Real time()       const { return t_; }

private:
    int  nx_;
    Real dx_, cfl_, t_;

    // Interior cells: [HALO .. HALO+nx-1]
    // Total storage : nx + 2*HALO
    std::vector<State> q_;     // current state (with halos)
    std::vector<State> q1_;    // RK stage 1
    std::vector<State> q2_;    // RK stage 2

    // Reconstructed face values (nx+1 interfaces)
    std::vector<State> qL_, qR_;

    // Numerical fluxes (nx+1 interfaces)
    std::vector<Flux>  F_;

    // ─── helpers ─────────────────────────────────────────────────────────────
    void fill_halos(std::vector<State>& q);
    void weno3_reconstruct(const std::vector<State>& q);
    static Flux hllc_flux(State qL, State qR);
    void rhs(const std::vector<State>& q,
             std::vector<State>&       dqdt,
             Real                      dt_dummy = 0.0);
};
