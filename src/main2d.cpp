/*
 * CPPWRF — Phase 2 entry point
 * Runs the 2D density current test case on GPU, writes CSV snapshots.
 *
 * Usage: ./cppwrf2d
 */

#include "types2d.hpp"
#include "solver/cuda/euler2d_gpu.cuh"
#include "../cases/density_current.hpp"
#include "io/output.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>
#include <libgen.h>
#include <climits>

// Resolve the output directory relative to the binary, not the CWD.
static std::string binary_dir() {
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n < 0) return "./";
    buf[n] = '\0';
    return std::string(dirname(buf)) + "/";
}

int main(int argc, char* argv[]) {
    cases::DensityCurrentParams p;
    p.output_dir = binary_dir() + "../results_2d/";
    
    // Parse args if needed
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--nx") == 0 && i + 1 < argc) {
            p.nx = std::stoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--nz") == 0 && i + 1 < argc) {
            p.nz = std::stoi(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--tend") == 0 && i + 1 < argc) {
            p.tend = std::stod(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--cfl") == 0 && i + 1 < argc) {
            p.cfl = std::stod(argv[++i]);
        }
        else if (std::strcmp(argv[i], "--dtheta") == 0 && i + 1 < argc) {
            p.dtheta_cold = std::stod(argv[++i]);
        }
    }

    std::printf("CPPWRF Phase 2 — 2D Density Current\n");
    std::printf("  device : GPU (CUDA)\n");
    std::printf("  nx x nz: %d x %d\n", p.nx, p.nz);
    std::printf("  dx, dz : %.2f m, %.2f m\n", p.dx, p.dz);
    std::printf("  tend   : %.3f s\n", p.tend);
    std::printf("  dtheta : %.1f K\n", p.dtheta_cold);
    std::printf("  output : %s\n\n", p.output_dir.c_str());

    std::vector<Real> rho, rhou, rhow, rhoTh;
    cases::make_ic(p, rho, rhou, rhow, rhoTh);
    
    std::vector<Real> rho_b, pi_b;
    cases::make_base_state(p, rho_b, pi_b, atm::HALO2);

    auto write_snapshot = [&](const Euler2D_GPU& solver, int step, Real t) {
        auto h_rho = solver.get_rho();
        auto h_rhou = solver.get_rhou();
        auto h_rhow = solver.get_rhow();
        auto h_rhoTh = solver.get_rhoTh();
        
        char filename[1024];
        std::sprintf(filename, "%s/density_current_%04d.csv", p.output_dir.c_str(), step);
        FILE* f = std::fopen(filename, "w");
        if (!f) {
            std::string cmd = "mkdir -p " + p.output_dir;
            system(cmd.c_str());
            f = std::fopen(filename, "w");
            if (!f) return;
        }
        
        std::fprintf(f, "x,z,rho,u,w,theta\n");
        for (int iz = 0; iz < p.nz; ++iz) {
            const Real z = (iz + 0.5) * p.dz;
            for (int ix = 0; ix < p.nx; ++ix) {
                const Real x = p.x0 + (ix + 0.5) * p.dx;
                const int idx = iz * p.nx + ix;
                const Real r = h_rho[idx];
                const Real u = h_rhou[idx] / r;
                const Real w = h_rhow[idx] / r;
                const Real theta = h_rhoTh[idx] / r;
                std::fprintf(f, "%.2f,%.2f,%.6f,%.6f,%.6f,%.6f\n", x, z, r, u, w, theta);
            }
        }
        std::fclose(f);
    };

    Euler2D_GPU solver(p.nx, p.nz, p.dx, p.dz, p.cfl);
    solver.set_base_state(rho_b, pi_b);
    solver.set_state(rho, rhou, rhow, rhoTh);

    auto t0 = std::chrono::high_resolution_clock::now();

    Real next_out = 0.0;
    int out_snap  = 0;
    int step_n    = 0;
    while (solver.time() < p.tend - 1.0e-12) {
        if (solver.time() >= next_out - 1.0e-12) {
            std::printf("  snapshot %d, t = %.4f\n", out_snap, (double)solver.time());
            write_snapshot(solver, out_snap++, solver.time());
            next_out += p.output_interval;
        }
        solver.step();
        ++step_n;

        // Download diagnostics every 10 steps to limit PCIe bandwidth usage
        if (step_n % 10 == 0) {
            auto hr  = solver.get_rho();
            auto hru = solver.get_rhou();
            auto hrw = solver.get_rhow();
            Real max_u = 0, max_w = 0;
            for (size_t i = 0; i < hr.size(); ++i) {
                if (hr[i] > 1e-10) {
                    max_u = std::max(max_u, std::abs(hru[i] / hr[i]));
                    max_w = std::max(max_w, std::abs(hrw[i] / hr[i]));
                }
            }
            std::printf("  step %4d, t = %.4f s,  max|u| = %8.4f m/s,  max|w| = %8.4f m/s\n",
                        step_n, (double)solver.time(), max_u, max_w);
        }
    }
    std::printf("  snapshot %d, t = %.4f (final)\n", out_snap, (double)solver.time());
    write_snapshot(solver, out_snap++, solver.time());

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("Runtime: %.2f ms\n", ms);

    return 0;
}
