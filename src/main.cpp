/*
 * WFE — Phase 1 entry point
 * Runs the 1D dam-break test case on CPU and GPU, writes CSV snapshots.
 *
 * Usage: ./wfe [--cpu | --gpu] [--nx N] [--tend T]
 */

#include "types.hpp"
#include "solver/swe1d.hpp"
#include "solver/cuda/swe1d_gpu.hpp"
#include "io/output.hpp"
#include "../cases/dam_break.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <filesystem>
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

// Resolve the output directory relative to the binary, not the CWD.
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
    // ── Parse args ────────────────────────────────────────────────────────────
    bool use_gpu = true;
    cases::DamBreakParams p;

    p.output_dir = binary_dir() + "../results/";

    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i], "--cpu") == 0) { use_gpu = false; }
        else if (std::strcmp(argv[i], "--gpu") == 0) { use_gpu = true;  }
        else if (std::strcmp(argv[i], "--nx")  == 0 && i+1 < argc) {
            p.nx = std::stoi(argv[++i]);
            p.dx = 1.0 / p.nx;
        }
        else if (std::strcmp(argv[i], "--tend") == 0 && i+1 < argc) {
            p.tend = std::stod(argv[++i]);
        }
    }

    std::printf("WFE Phase 1 — 1D Dam-Break\n");
    std::printf("  device : %s\n", use_gpu ? "GPU (CUDA)" : "CPU");
    std::printf("  nx     : %d\n", p.nx);
    std::printf("  dx     : %.6f m\n", p.dx);
    std::printf("  tend   : %.3f s\n", p.tend);
    std::printf("  h_L/h_R: %.2f / %.2f m\n", p.h_left, p.h_right);
    std::printf("  output : %s\n\n", p.output_dir.c_str());

    auto ic = cases::make_ic(p);
    io::CSVWriter writer(p.output_dir);

    auto t0 = std::chrono::high_resolution_clock::now();

    if (!use_gpu) {
        // ── CPU path ──────────────────────────────────────────────────────────
        SWE1D_CPU solver(p.nx, p.dx, p.cfl);
        solver.set_state(ic);

        Real next_out = 0.0;
        while (solver.time() < p.tend - 1.0e-12) {
            if (solver.time() >= next_out - 1.0e-12) {
                const auto& q = solver.state();
                std::vector<State> interior(q.begin() + wfe::HALO,
                                            q.begin() + wfe::HALO + p.nx);
                writer.write_snapshot(interior, p.nx, p.dx, solver.time());
                std::printf("  t = %.4f\n", (double)solver.time());
                next_out += p.output_interval;
            }
            solver.step();
        }
        // Final snapshot
        const auto& q = solver.state();
        std::vector<State> interior(q.begin() + wfe::HALO,
                                    q.begin() + wfe::HALO + p.nx);
        writer.write_snapshot(interior, p.nx, p.dx, solver.time());
        std::printf("  t = %.4f (final)\n", (double)solver.time());

    } else {
        // ── GPU path ──────────────────────────────────────────────────────────
        SWE1D_GPU solver(p.nx, p.dx, p.cfl);
        solver.set_state(ic);

        Real next_out = 0.0;
        while (solver.time() < p.tend - 1.0e-12) {
            if (solver.time() >= next_out - 1.0e-12) {
                auto state = solver.get_state();
                writer.write_snapshot(state, p.nx, p.dx, solver.time());
                std::printf("  t = %.4f\n", (double)solver.time());
                next_out += p.output_interval;
            }
            solver.step();
        }
        auto state = solver.get_state();
        writer.write_snapshot(state, p.nx, p.dx, solver.time());
        std::printf("  t = %.4f (final)\n", (double)solver.time());
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    io::CSVWriter::print_perf(ms, p.nx, p.tend);

    return 0;
}
