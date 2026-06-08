/*
 * WFE — Phase 2 Mountain Wave entry point
 * Runs the Schär et al. (2002) mountain wave test case on GPU.
 * Rayleigh sponge layer at domain top absorbs upward-propagating gravity waves.
 *
 * Usage: ./build/wfe_mw [--tend T] [--cfl C] [--nx N] [--nz N]
 */

#include "types2d.hpp"
#include "solver/cuda/euler2d_gpu.cuh"
#include "../cases/mountain_wave.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <filesystem>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

static std::string binary_dir() {
#ifdef _WIN32
    char buf[MAX_PATH];
    DWORD length = GetModuleFileNameA(NULL, buf, MAX_PATH);
    if (length == 0) return "./";
    return std::filesystem::path(buf).parent_path().string() + "/";
#else
    char buf[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n < 0) return "./";
    buf[n] = '\0';
    return std::filesystem::path(buf).parent_path().string() + "/";
#endif
}

int main(int argc, char* argv[]) {
    cases::MountainWaveParams p;
    p.output_dir = binary_dir() + "../results_mw/";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--nx")   == 0 && i+1 < argc) p.nx   = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--nz")   == 0 && i+1 < argc) p.nz   = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--tend") == 0 && i+1 < argc) p.tend = std::stod(argv[++i]);
        else if (std::strcmp(argv[i], "--cfl")  == 0 && i+1 < argc) p.cfl  = std::stod(argv[++i]);
    }

    std::printf("WFE Phase 2 — Schär (2002) Mountain Wave\n");
    std::printf("  device    : GPU (CUDA)\n");
    std::printf("  nx x nz   : %d x %d\n", p.nx, p.nz);
    std::printf("  dx, dz    : %.1f m, %.1f m\n", p.dx, p.dz);
    std::printf("  domain    : %.1f km x %.1f km\n",
                p.nx * p.dx / 1e3, p.nz * p.dz / 1e3);
    std::printf("  tend      : %.0f s\n", p.tend);
    std::printf("  u_bar     : %.1f m/s\n", p.u_bar);
    std::printf("  sponge    : %.0f – %.0f m  (tau=%.0f s)\n",
                p.sponge_top, p.nz * p.dz, p.tau_sponge);
    std::printf("  output    : %s\n\n", p.output_dir.c_str());

    // ── Build initial condition and base state ────────────────────────────────
    std::vector<Real> rho, rhou, rhow, rhoTh;
    cases::make_ic_mw(p, rho, rhou, rhow, rhoTh);

    std::vector<Real> rho_b, pi_b;
    cases::make_base_state_mw(p, rho_b, pi_b, atm::HALO2);

    // ── Build terrain slope array dh/dx at each interior x-cell ──────────────
    std::vector<Real> dhdx(p.nx);
    for (int ix = 0; ix < p.nx; ++ix) {
        const Real x  = p.x0 + (ix + 0.5) * p.dx;
        const Real dx = p.dx;
        // Centred difference of h(x) to get dh/dx
        const Real hm = cases::mountain_height(x - dx, p);
        const Real hp = cases::mountain_height(x + dx, p);
        dhdx[ix] = (hp - hm) / (2.0 * dx);
    }

    // ── Output helper ─────────────────────────────────────────────────────────
    auto write_snapshot = [&](const Euler2D_GPU& solver, int snap, Real t) {
        auto h_rho  = solver.get_rho();
        auto h_rhou = solver.get_rhou();
        auto h_rhow = solver.get_rhow();
        auto h_rhoTh= solver.get_rhoTh();

        char filename[1024];
        std::sprintf(filename, "%s/mountain_wave_%05d.csv", p.output_dir.c_str(), snap);
        FILE* f = std::fopen(filename, "w");
        if (!f) {
            std::filesystem::create_directories(p.output_dir);
            f = std::fopen(filename, "w");
            if (!f) return;
        }
        std::fprintf(f, "x,z,rho,u,w,theta\n");
        for (int iz = 0; iz < p.nz; ++iz) {
            const Real z = (iz + 0.5) * p.dz;
            for (int ix = 0; ix < p.nx; ++ix) {
                const Real x   = p.x0 + (ix + 0.5) * p.dx;
                const int  idx = iz * p.nx + ix;
                const Real r   = h_rho[idx];
                if (r < 1.0e-10) continue;
                const Real u   = h_rhou[idx] / r;
                const Real w   = h_rhow[idx] / r;
                const Real th  = h_rhoTh[idx] / r;
                std::fprintf(f, "%.1f,%.1f,%.6f,%.4f,%.4f,%.4f\n", x, z, r, u, w, th);
            }
        }
        std::fclose(f);
    };

    // ── Build and configure solver ────────────────────────────────────────────
    Euler2D_GPU solver(p.nx, p.nz, p.dx, p.dz, p.cfl);
    solver.set_base_state(rho_b, pi_b);
    solver.set_state(rho, rhou, rhow, rhoTh);
    solver.set_terrain(dhdx);
    solver.set_sponge_layer(p.sponge_top, p.nz * p.dz, p.tau_sponge, p.u_bar);
    solver.set_smagorinsky(0.18, 1.0/3.0);

    // ── Time loop ─────────────────────────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();

    Real next_out = 0.0;
    int  out_snap = 0;
    int  step_n   = 0;

    std::filesystem::create_directories(p.output_dir);

    while (solver.time() < p.tend - 1.0e-12) {
        if (solver.time() >= next_out - 1.0e-12) {
            std::printf("  snapshot %d, t = %.1f s\n", out_snap, (double)solver.time());
            write_snapshot(solver, out_snap++, solver.time());
            next_out += p.output_interval;
        }
        solver.step();
        ++step_n;

        if (step_n % 50 == 0) {
            auto h_r  = solver.get_rho();
            auto h_rw = solver.get_rhow();
            Real max_w = 0;
            for (size_t i = 0; i < h_r.size(); ++i)
                if (h_r[i] > 1e-10) max_w = std::max(max_w, std::abs(h_rw[i] / h_r[i]));
            std::printf("  step %5d, t = %7.1f s,  max|w| = %6.3f m/s\n",
                        step_n, (double)solver.time(), max_w);
        }
    }

    std::printf("  snapshot %d, t = %.1f s (final)\n", out_snap, (double)solver.time());
    write_snapshot(solver, out_snap++, solver.time());

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("\nRuntime: %.2f s  (%d steps)\n", ms / 1000.0, step_n);

    return 0;
}
