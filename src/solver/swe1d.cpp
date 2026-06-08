/*
 * WFE — Phase 1
 * CPU reference solver: 1D shallow water equations
 * Scheme: WENO3 reconstruction · HLLC flux · SSP-RK3 time integration
 */

#include "swe1d.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace wfe;

// ─── Constructor ──────────────────────────────────────────────────────────────

SWE1D_CPU::SWE1D_CPU(int nx, Real dx, Real cfl)
    : nx_(nx), dx_(dx), cfl_(cfl), t_(0.0)
{
    const int N = nx + 2 * HALO;
    q_ .resize(N);
    q1_.resize(N);
    q2_.resize(N);
    qL_.resize(nx + 1);
    qR_.resize(nx + 1);
    F_ .resize(nx + 1);
}

void SWE1D_CPU::set_state(const std::vector<State>& ic) {
    if ((int)ic.size() != nx_)
        throw std::runtime_error("IC size mismatch");
    for (int i = 0; i < nx_; ++i)
        q_[HALO + i] = ic[i];
    fill_halos(q_);
    t_ = 0.0;
}

// ─── Time step estimate ───────────────────────────────────────────────────────

Real SWE1D_CPU::compute_dt() const {
    Real smax = 0.0;
    for (int i = HALO; i < HALO + nx_; ++i) {
        const Real s = std::abs(q_[i].u()) + q_[i].a();
        smax = std::max(smax, s);
    }
    return (smax > 0.0) ? cfl_ * dx_ / smax : 1.0e30;
}

// ─── Transmissive (outflow) halos ─────────────────────────────────────────────

void SWE1D_CPU::fill_halos(std::vector<State>& q) {
    for (int k = 0; k < HALO; ++k) {
        q[HALO - 1 - k] = q[HALO];
        q[HALO + nx_ + k] = q[HALO + nx_ - 1];
    }
}

// ─── WENO3 reconstruction ─────────────────────────────────────────────────────
// Produces left (qL) and right (qR) states at each of the nx+1 interfaces.
// Interface i+1/2 sits between interior cells i and i+1.
// q array is indexed with HALO offset.

namespace {

// WENO3 for a scalar. Returns reconstructed value at the right edge of cell i
// (i.e. interface i+1/2 from the left).
inline Real weno3_left(Real qm1, Real q0, Real qp1) {
    // Two candidate stencils
    const Real q0s = -0.5 * qm1 + 1.5 * q0;
    const Real q1s =  0.5 * q0  + 0.5 * qp1;

    const Real b0 = (q0 - qm1) * (q0 - qm1);
    const Real b1 = (qp1 - q0) * (qp1 - q0);

    const Real a0 = d0L / ((b0 + eps) * (b0 + eps));
    const Real a1 = d1L / ((b1 + eps) * (b1 + eps));
    const Real asum = a0 + a1;

    return (a0 * q0s + a1 * q1s) / asum;
}

// WENO3 for right state at interface i+1/2.
// Inputs: q0=q_i, qp1=q_{i+1}, qp2=q_{i+2} where cell i is LEFT of the interface.
// Stencil 0 {i+1, i+2}: qs0 = 3/2*q_{i+1} - 1/2*q_{i+2}, optimal weight d=2/3.
// Stencil 1 {i,   i+1}: qs1 = 1/2*q_i + 1/2*q_{i+1},     optimal weight d=1/3.
inline Real weno3_right(Real q0, Real qp1, Real qp2) {
    const Real qs0 =  1.5 * qp1 - 0.5 * qp2;   // stencil {i+1, i+2}
    const Real qs1 =  0.5 * q0  + 0.5 * qp1;   // stencil {i,   i+1}

    const Real b0 = (qp2 - qp1) * (qp2 - qp1); // smoothness of stencil 0
    const Real b1 = (qp1 - q0)  * (qp1 - q0);  // smoothness of stencil 1

    const Real a0 = d0R / ((b0 + eps) * (b0 + eps));  // d0R = 2/3
    const Real a1 = d1R / ((b1 + eps) * (b1 + eps));  // d1R = 1/3

    return (a0 * qs0 + a1 * qs1) / (a0 + a1);
}

} // namespace

void SWE1D_CPU::weno3_reconstruct(const std::vector<State>& q) {
    // Interface 0 is between ghost cell HALO-1 and interior cell HALO.
    // Interface nx is between interior cell HALO+nx-1 and ghost cell HALO+nx.
    for (int f = 0; f <= nx_; ++f) {
        // Left state at interface f (f = i+1/2 relative to interior cell i = f-1)
        // Interior index: left cell is HALO + f - 1
        const int il = HALO + f - 1;  // left cell (may be a halo cell when f=0)
        qL_[f].h  = weno3_left(q[il-1].h,  q[il].h,  q[il+1].h);
        qL_[f].hu = weno3_left(q[il-1].hu, q[il].hu, q[il+1].hu);
        if (qL_[f].h < hMin) { qL_[f].h = hMin; qL_[f].hu = 0.0; }

        // Right state: same center cell as left (il = cell f-1), looking rightward.
        // Uses cells f-1, f, f+1 → indices il, il+1, il+2.
        qR_[f].h  = weno3_right(q[il].h,  q[il+1].h,  q[il+2].h);
        qR_[f].hu = weno3_right(q[il].hu, q[il+1].hu, q[il+2].hu);
        if (qR_[f].h < hMin) { qR_[f].h = hMin; qR_[f].hu = 0.0; }
    }
}

// ─── HLLC flux ────────────────────────────────────────────────────────────────

Flux SWE1D_CPU::hllc_flux(State L, State R) {
    const Real uL = L.u(), uR = R.u();
    const Real aL = L.a(), aR = R.a();

    // Roe-averaged wave speed estimates (simple but stable)
    const Real sL = std::min(uL - aL, uR - aR);
    const Real sR = std::max(uL + aL, uR + aR);

    if (sL >= 0.0) return physical_flux(L);
    if (sR <= 0.0) return physical_flux(R);

    // Contact (middle) wave speed — particle velocity in the star region.
    // Correct SWE HLLC formula: s* = [h_L*u_L*(s_L-u_L) - h_R*u_R*(s_R-u_R) + g*(h_R²-h_L²)/2]
    //                                / [h_L*(s_L-u_L) - h_R*(s_R-u_R)]
    const Real sStar = (L.hu * (sL - uL) - R.hu * (sR - uR)
                       + g * (0.5 * R.h * R.h - 0.5 * L.h * L.h))
                     / (L.h * (sL - uL) - R.h * (sR - uR));

    // HLLC star states
    auto star_state = [&](const State& q, Real s) -> State {
        const Real factor = q.h * (s - q.u()) / (s - sStar);
        return { factor, factor * sStar };
    };

    Flux FL = physical_flux(L);
    Flux FR = physical_flux(R);

    if (sStar >= 0.0) {
        State qStar = star_state(L, sL);
        Flux FStar;
        FStar.mass     = FL.mass     + sL * (qStar.h  - L.h);
        FStar.momentum = FL.momentum + sL * (qStar.hu - L.hu);
        return FStar;
    } else {
        State qStar = star_state(R, sR);
        Flux FStar;
        FStar.mass     = FR.mass     + sR * (qStar.h  - R.h);
        FStar.momentum = FR.momentum + sR * (qStar.hu - R.hu);
        return FStar;
    }
}

// ─── Right-hand side L(q) = -dF/dx ──────────────────────────────────────────

void SWE1D_CPU::rhs(const std::vector<State>& q,
                    std::vector<State>&        dqdt,
                    Real /*dt_dummy*/)
{
    // Reconstruct and compute fluxes
    weno3_reconstruct(q);
    for (int f = 0; f <= nx_; ++f)
        F_[f] = hllc_flux(qL_[f], qR_[f]);

    // Update
    dqdt.resize(q.size());
    for (int i = 0; i < nx_; ++i) {
        const Real inv_dx = 1.0 / dx_;
        dqdt[HALO + i].h  = -(F_[i+1].mass     - F_[i].mass)     * inv_dx;
        dqdt[HALO + i].hu = -(F_[i+1].momentum - F_[i].momentum) * inv_dx;
    }
}

// ─── SSP-RK3 step ─────────────────────────────────────────────────────────────

void SWE1D_CPU::step() {
    const Real dt = compute_dt();

    // Temporaries for RHS
    std::vector<State> L0(q_.size()), L1(q_.size()), L2(q_.size());

    // Stage 1: q1 = q + dt * L(q)
    rhs(q_, L0);
    for (int i = 0; i < nx_; ++i) {
        q1_[HALO + i].h  = q_[HALO + i].h  + dt * L0[HALO + i].h;
        q1_[HALO + i].hu = q_[HALO + i].hu + dt * L0[HALO + i].hu;
        if (q1_[HALO + i].h < hMin) { q1_[HALO + i].h = hMin; q1_[HALO + i].hu = 0.0; }
    }
    fill_halos(q1_);

    // Stage 2: q2 = 3/4 q + 1/4 (q1 + dt * L(q1))
    rhs(q1_, L1);
    for (int i = 0; i < nx_; ++i) {
        q2_[HALO + i].h  = 0.75 * q_[HALO + i].h  + 0.25 * (q1_[HALO + i].h  + dt * L1[HALO + i].h);
        q2_[HALO + i].hu = 0.75 * q_[HALO + i].hu + 0.25 * (q1_[HALO + i].hu + dt * L1[HALO + i].hu);
        if (q2_[HALO + i].h < hMin) { q2_[HALO + i].h = hMin; q2_[HALO + i].hu = 0.0; }
    }
    fill_halos(q2_);

    // Stage 3: q = 1/3 q + 2/3 (q2 + dt * L(q2))
    rhs(q2_, L2);
    for (int i = 0; i < nx_; ++i) {
        q_[HALO + i].h  = (1.0/3.0) * q_[HALO + i].h  + (2.0/3.0) * (q2_[HALO + i].h  + dt * L2[HALO + i].h);
        q_[HALO + i].hu = (1.0/3.0) * q_[HALO + i].hu + (2.0/3.0) * (q2_[HALO + i].hu + dt * L2[HALO + i].hu);
        if (q_[HALO + i].h < hMin) { q_[HALO + i].h = hMin; q_[HALO + i].hu = 0.0; }
    }
    fill_halos(q_);

    t_ += dt;
}
