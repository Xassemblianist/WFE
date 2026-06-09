#pragma once
#include <vector>
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using Real = double;

namespace atm {
    static constexpr Real Rd    = 287.0;
    static constexpr Real Cp    = 1004.0;
    static constexpr Real Cv    = 717.0;
    static constexpr Real p0    = 100000.0;
    static constexpr Real gamma = 1.4;
    static constexpr Real g     = 9.81;
    static constexpr int  HALO3 = 3;  // WENO5 needs 3 cells for stability in 3D

    inline Real pressure_from_exner(Real pi) {
        return p0 * pow(pi, Cp/Rd);
    }
}

struct Grid3D {
    int nx, ny, nz;
    int halo;
    Real dx, dy, dz;

    [[nodiscard]] size_t size() const {
        return (size_t)(nx + 2*halo) * (ny + 2*halo) * (nz + 2*halo);
    }
    [[nodiscard]] int stride_x() const { return 1; }
    [[nodiscard]] int stride_y() const { return (nx + 2*halo); }
    [[nodiscard]] int stride_z() const { return (nx + 2*halo) * (ny + 2*halo); }
};

struct BaseState3D {
    Real *rho_b   = nullptr; // [nz+2h]
    Real *p_b     = nullptr; // [nz+2h]
    Real *pi_b    = nullptr; // [nz+2h]
    Real *rhoTh_b = nullptr; // [nz+2h]
};
