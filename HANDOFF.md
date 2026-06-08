# WFE Project Handoff — v1.0

This document contains the complete context and current state for **WFE v1.0**. Read this before writing any code.

## 1. Project Status

**WFE** is a high-performance, GPU-accelerated Numerical Weather Prediction engine in C++20/CUDA. It has reached **v1.0** — the core dynamics engine is functional across 1D, 2D, and 3D.

**Completed Phases:**
- **Phase 1 (Done):** 1D Shallow Water Equations on GPU — validated against analytical dam-break.
- **Phase 2 (Done):** 2D Non-Hydrostatic Compressible Euler — density current (Robert 1993) + mountain wave (Schär 2002).
- **Phase 3 (Done, basic):** 3D dynamics — warm bubble test case (Bryan & Fritsch 2002).

**Remaining Phases:**
- **Phase 4:** Operational pipeline — GFS/ICON-EU GRIB2 ingest, Zarr output, web viewer.
- **Phase 5:** Multi-GPU domain decomposition, ensemble forecasting.

## 2. Resolved Issues

### ✅ Vertical Velocity Blowup — FIXED
The exponential w-blowup caused by well-balanced boundary error has been resolved. The fix bypasses WENO5/HLLC entirely at wall faces (f=0, f=nz) and uses direct hydrostatic pressure extrapolation: `p'_wall = p'_cell ± ½Δz·ρ'·g`.

### ✅ Acoustic Sub-Cycling — IMPLEMENTED
Full Forward-Backward acoustic integration with N_SPLIT=10 sub-steps per RK3 stage is implemented and working in both 2D and 3D solvers.

### ✅ Rayleigh Sponge — IMPLEMENTED
Sin²-profile Rayleigh damping absorbs upward-propagating gravity waves at the domain top.

### ✅ Smagorinsky Turbulence — IMPLEMENTED
Dynamic Smagorinsky closure (Cs=0.18, Prt=1/3) provides subgrid-scale diffusion for both momentum and potential temperature.

### ✅ Mountain Wave + Terrain — IMPLEMENTED
Schär (2002) test case with bell-shaped terrain and constant-N base state. Terrain slope flux correction in `k_z_fluxes`.

## 3. Build & Run

### Windows (CMake + Visual Studio + CUDA)
```powershell
cmake -B build
cmake --build build --config Release
.\build\Release\wfe2d.exe --nx 256 --nz 64 --tend 900 --cfl 0.4
```

### Linux (Makefile)
```bash
make clean all
./build/wfe2d --nx 256 --nz 64 --tend 900 --cfl 0.4
```

## 4. Next Steps for Future Sessions

| Priority | Task | Details |
|---|---|---|
| 1 | GFS GRIB2 reader | Real initial and lateral boundary conditions |
| 2 | NetCDF/Zarr output | Replace CSV for operational-scale I/O |
| 3 | Thompson microphysics | 8-class cloud/precip CUDA kernel |
| 4 | YSU PBL | Planetary boundary layer parameterization |
| 5 | Web viewer | GitHub Pages auto-updating forecast page |
| 6 | Multi-GPU | NCCL-based domain decomposition |

## 5. Key Files

| File | Purpose |
|---|---|
| `src/solver/cuda/euler2d_gpu.cu` (1150 lines) | ⭐ 2D GPU solver — all kernels |
| `src/solver/cuda/euler3d_gpu.cu` (1000 lines) | 3D GPU solver — same scheme + y |
| `cases/density_current.hpp` | Robert (1993) cold bubble IC |
| `cases/mountain_wave.hpp` | Schär (2002) mountain wave IC + terrain |
| `ARCHITECTURE.md` | Full technical reference |
