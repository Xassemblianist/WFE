/*
 * WFE — Phase 3 entry point
 * 3D warm bubble test (Bryan & Fritsch 2002 style):
 *   Background: isentropic θ̄=300K, hydrostatic
 *   Perturbation: θ'=+2K spherical Gaussian centred at (xc, yc, 2km)
 *   Domain: 20km x 20km x 10km  (200m grid → 100x100x50 cells)
 *   Tests: 3D compressible convection, GPU performance, Stage 3 RK fix
 *
 * Usage: ./build/wfe3d [--nx N] [--ny N] [--nz N] [--tend T] [--cfl C]
 */
#include "solver/cuda/euler3d_gpu.cuh"
#include "types3d.hpp"

#include <array>
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

static void make_base_state(int nz, Real dz, int halo,
                            std::vector<Real>& rho_b, std::vector<Real>& pi_b)
{
    const int nz_full = nz + 2 * halo;
    rho_b.resize(nz_full);
    pi_b .resize(nz_full);

    const Real theta_bar = 300.0;
    const Real dpi_dz    = -atm::g / (atm::Cp * theta_bar);

    for (int iz = -halo; iz < nz + halo; ++iz) {
        const Real z   = (iz + 0.5) * dz;
        const Real pi  = 1.0 + dpi_dz * z;
        const Real rho = (atm::p0 / (atm::Rd * theta_bar))
                       * std::pow(pi, atm::Cv / atm::Rd);
        rho_b[iz + halo] = rho;
        pi_b [iz + halo] = pi;
    }
}

int main(int argc, char* argv[])
{
    int  nx   = 100, ny   = 100, nz  = 50;
    Real dx   = 200.0, dy = 200.0, dz = 200.0;
    Real cfl  = 0.4;
    Real tend = 600.0;
    Real output_interval = 60.0;
    std::string outdir = binary_dir() + "../results_3d/";

    for (int i = 1; i < argc; ++i) {
        if      (std::strcmp(argv[i],"--nx"  )==0 && i+1<argc) nx   = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i],"--ny"  )==0 && i+1<argc) ny   = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i],"--nz"  )==0 && i+1<argc) nz   = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i],"--tend")==0 && i+1<argc) tend = std::stod(argv[++i]);
        else if (std::strcmp(argv[i],"--cfl" )==0 && i+1<argc) cfl  = std::stod(argv[++i]);
    }

    std::printf("WFE Phase 3 — 3D Warm Bubble\n");
    std::printf("  device    : GPU (CUDA)\n");
    std::printf("  grid      : %d x %d x %d\n", nx, ny, nz);
    std::printf("  spacing   : %.0f x %.0f x %.0f m\n", dx, dy, dz);
    std::printf("  domain    : %.1f x %.1f x %.1f km\n",
                nx*dx/1e3, ny*dy/1e3, nz*dz/1e3);
    std::printf("  tend      : %.0f s\n", tend);
    std::printf("  output    : %s\n\n", outdir.c_str());

    const int halo = atm::HALO3;
    std::vector<Real> rho_b, pi_b;
    make_base_state(nz, dz, halo, rho_b, pi_b);

    const long N = (long)nx * ny * nz;
    std::vector<Real> rho(N), rhou(N), rhov(N), rhow(N), rhoTh(N);

    const Real theta_bar = 300.0;
    const Real dpi_dz    = -atm::g / (atm::Cp * theta_bar);
    const Real xc = nx * dx * 0.5;
    const Real yc = ny * dy * 0.5;
    const Real zc = 2000.0;
    const Real rb = 2000.0;

    for (int iz = 0; iz < nz; ++iz) {
        const Real z  = (iz + 0.5) * dz;
        const Real pi = 1.0 + dpi_dz * z;
        const Real rho_base = (atm::p0 / (atm::Rd * theta_bar))
                            * std::pow(pi, atm::Cv / atm::Rd);
        for (int iy = 0; iy < ny; ++iy) {
            const Real y = (iy + 0.5) * dy;
            for (int ix = 0; ix < nx; ++ix) {
                const Real x = (ix + 0.5) * dx;
                const long idx = (long)iz * ny * nx + (long)iy * nx + ix;

                const Real r2 = ((x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc))
                              / (rb * rb);
                const Real dtheta = (r2 <= 1.0)
                    ? 2.0 * std::cos(0.5 * M_PI * std::sqrt(r2))
                          * std::cos(0.5 * M_PI * std::sqrt(r2))
                    : 0.0;
                const Real theta   = theta_bar + dtheta;
                const Real rho_val = rho_base * theta_bar / theta;

                rho  [idx] = rho_val;
                rhou [idx] = 0.0;
                rhov [idx] = 0.0;
                rhow [idx] = 0.0;
                rhoTh[idx] = rho_val * theta;
            }
        }
    }

    Euler3D_GPU solver(nx, ny, nz, dx, dy, dz, cfl);
    solver.set_base_state(rho_b, pi_b);
    solver.set_state(rho, rhou, rhov, rhow, rhoTh);
    solver.set_sponge_layer(nz * dz - 1000.0, nz * dz, 10.0, 0.0);
    solver.set_smagorinsky(0.18, 1.0/3.0);

    std::filesystem::create_directories(outdir);

    auto write_snapshot = [&](int snap, Real t) {
        auto h_rho  = solver.get_rho();
        auto h_rhow = solver.get_rhow();
        auto h_rhoTh= solver.get_rhoTh();

        char fname[1024];
        std::sprintf(fname, "%s/bubble3d_%05d.csv", outdir.c_str(), snap);
        FILE* f = std::fopen(fname, "w");
        if (!f) return;
        std::fprintf(f, "x,z,rho,w,theta\n");
        const int iy_c = ny / 2;
        for (int iz = 0; iz < nz; ++iz) {
            const Real z_val = (iz + 0.5) * dz;
            for (int ix = 0; ix < nx; ++ix) {
                const Real x_val = (ix + 0.5) * dx;
                const long idx = (long)iz * ny * nx + (long)iy_c * nx + ix;
                const Real r   = h_rho[idx];
                if (r < 1.0e-10) continue;
                std::fprintf(f, "%.0f,%.0f,%.6f,%.4f,%.4f\n",
                             x_val, z_val, r, h_rhow[idx]/r, h_rhoTh[idx]/r);
            }
        }
        std::fclose(f);
    };

    auto t0 = std::chrono::high_resolution_clock::now();

    std::array<Real,4> diag0 = solver.get_diagnostics();
    std::printf("  Conservation (mass [kg], KE [J], PE [J], ρθ [kg·K]):\n");
    std::printf("    t=0: mass=%.6e  KE=%.6e  PE=%.6e\n\n",
                diag0[0], diag0[1], diag0[2]);

    Real next_out = 0.0;
    int  out_snap = 0, step_n = 0;

    while (solver.time() < tend - 1.0e-12) {
        if (solver.time() >= next_out - 1.0e-12) {
            auto diag = solver.get_diagnostics();
            std::printf("  snapshot %d, t = %.1f s  |  Δmass=%.3e  ΔKE=%.3e  ΔPE=%.3e\n",
                        out_snap, (double)solver.time(),
                        (diag[0]-diag0[0])/diag0[0],
                        diag[1]-diag0[1], diag[2]-diag0[2]);
            write_snapshot(out_snap++, solver.time());
            next_out += output_interval;
        }
        solver.step();
        ++step_n;

        if (step_n % 20 == 0) {
            auto h_r  = solver.get_rho();
            auto h_rw = solver.get_rhow();
            auto h_rT = solver.get_rhoTh();
            Real max_w = 0, min_th = 1e10, max_th = 0;
            for (long i = 0; i < (long)h_r.size(); ++i) {
                if (h_r[i] > 1e-10) {
                    max_w  = std::max(max_w, std::abs(h_rw[i] / h_r[i]));
                    Real th = h_rT[i] / h_r[i];
                    min_th = std::min(min_th, th);
                    max_th = std::max(max_th, th);
                }
            }
            std::printf("  step %4d, t = %6.1f s,  max|w| = %6.3f m/s,"
                        "  θ ∈ [%.3f, %.3f] K\n",
                        step_n, (double)solver.time(), max_w, min_th, max_th);
        }
    }
    std::printf("  snapshot %d, t = %.1f s (final)\n", out_snap, (double)solver.time());
    write_snapshot(out_snap++, (double)solver.time());

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("\nRuntime: %.2f s  (%d steps,  %d M cells)\n",
                ms/1000.0, step_n, nx*ny*nz/1000000);
    return 0;
}
