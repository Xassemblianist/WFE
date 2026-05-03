#pragma once
#include "../src/types.hpp"
#include <vector>
#include <string>

namespace cases {

// ─── Dam-break parameters ────────────────────────────────────────────────────
struct DamBreakParams {
    int  nx              = 1000;
    Real dx              = 1.0 / 1000.0;   // recomputed from nx
    Real tend            = 0.5;
    Real cfl             = 0.4;
    Real output_interval = 0.05;
    Real x_break         = 0.5;
    Real h_left          = 1.0;
    Real h_right         = 0.1;
    std::string output_dir = "";  // set at runtime by main
    std::string device   = "gpu";
};

// Build the initial condition vector (length nx)
inline std::vector<State>
make_ic(const DamBreakParams& p) {
    std::vector<State> ic(p.nx);
    for (int i = 0; i < p.nx; ++i) {
        const Real xc = (i + 0.5) * p.dx;   // cell-centre
        ic[i].h  = (xc < p.x_break) ? p.h_left : p.h_right;
        ic[i].hu = 0.0;
    }
    return ic;
}

} // namespace cases
