#pragma once
#include <cmath>
#include <cstdint>

// ─── Scalar type ─────────────────────────────────────────────────────────────
using Real = double;
using Int  = int32_t;

// ─── Physical / numerical constants ──────────────────────────────────────────
namespace cppwrf {

constexpr Real g    = 9.81;      // gravity [m/s²]
constexpr Real eps  = 1.0e-6;   // WENO regularisation
constexpr Real hMin = 1.0e-10;  // wet/dry threshold [m]
constexpr int  HALO = 3;        // ghost cells each side (WENO3 needs 2, use 3)

// WENO3 optimal weights
constexpr Real d0L = 1.0/3.0, d1L = 2.0/3.0;   // left-biased
constexpr Real d0R = 2.0/3.0, d1R = 1.0/3.0;   // right-biased

} // namespace cppwrf

// ─── Conservative state  q = [h, hu] ─────────────────────────────────────────
struct State {
    Real h  = 0.0;   // water depth [m]
    Real hu = 0.0;   // x-momentum  [m²/s]

    // velocity (safe)
    [[nodiscard]] Real u() const {
        return (h > cppwrf::hMin) ? hu / h : 0.0;
    }
    // wave speed
    [[nodiscard]] Real a() const {
        return (h > cppwrf::hMin) ? std::sqrt(cppwrf::g * h) : 0.0;
    }
};

// ─── Physical flux  F(q) = [hu, hu²/h + g h²/2] ─────────────────────────────
struct Flux {
    Real mass     = 0.0;
    Real momentum = 0.0;
};

inline Flux physical_flux(const State& q) {
    const Real u = q.u();
    return { q.hu,
             u * q.hu + 0.5 * cppwrf::g * q.h * q.h };
}
