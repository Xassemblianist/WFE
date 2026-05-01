<h1 align="center">Atmosfer</h1>

<p align="center">
  <i>Modern numerical weather prediction, written from scratch in C++20 + CUDA.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-design%20phase-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/language-C%2B%2B20%20%2B%20CUDA-76B900?style=flat-square" />
  <img src="https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square" />
</p>

---

> **Status:** This repository is in the design phase &mdash; no production code yet. The README and roadmap document the vision; implementation begins with Phase 1 (1D shallow water on GPU). Progress is tracked openly.

## What this is

Atmosfer is a clean-room implementation of a research-grade non-hydrostatic atmospheric model, designed for modern GPUs from the ground up.

The dominant weather model in research today, **WRF** (Weather Research and Forecasting), is roughly 1.5 million lines of Fortran 90 written starting in the late 1990s. Its data layouts, MPI communication patterns, and memory model predate CUDA. Porting WRF to GPUs has been an active research effort for over a decade and remains incomplete.

**Atmosfer takes the opposite approach:** start with the equations, target the hardware (Hopper / Blackwell-class GPUs), and let the code shape itself accordingly.

## Three pillars

### 1. Dynamical core
- Fully compressible non-hydrostatic equations
- Terrain-following (sigma-pressure hybrid) vertical coordinate
- Finite-volume spatial discretization, third-order WENO advection
- Split-explicit acoustic time stepping (Klemp, Skamarock &amp; Dudhia, 2007)
- Single-precision compute path with mixed-precision option for tensor cores

### 2. Pluggable physics
Each parameterization is an independent CUDA kernel with a stable interface, so any one can be swapped without touching the dynamics:
- **Microphysics:** Thompson 8-class
- **Planetary boundary layer:** YSU (Hong, 2006)
- **Land surface:** Noah-MP
- **Radiation:** RRTMG (longwave + shortwave)
- **Cumulus:** off by default at convection-permitting resolution

### 3. Operational I/O
- Reads GFS or ICON-EU GRIB2 initial &amp; boundary conditions
- Writes Zarr-format output for streaming to a web viewer
- Drives a public forecast page hosted at `xassemblianist.github.io/atmosfer`

## Demo target

An operational forecast pipeline for the Eastern Mediterranean / Antalya basin:
- **Domain:** ~500&times;500 km, centered on Antalya
- **Resolution:** 1 km horizontal, 60 vertical levels
- **Range:** 48 hours
- **Cadence:** 00 UTC and 12 UTC, twice daily
- **Output:** browser-renderable forecast page, auto-published to GitHub Pages

## Roadmap

| Phase | Scope | Primary deliverable |
|---|---|---|
| **1** | 1D shallow-water equations on a single GPU, idealized initial conditions | Validation against analytical dam-break solution |
| **2** | 2D non-hydrostatic compressible atmosphere, idealized cases | Density-current and 2D mountain-wave reproduction (Straka 1993, Schär 2002) |
| **3** | 3D dynamical core, real terrain, basic physics (microphysics + PBL) | Reproduce a documented historical convective event |
| **4** | Operational pipeline, GFS / ICON-EU ingest, Zarr output, web viewer | Live Antalya forecast running twice daily |
| **5** | Multi-GPU domain decomposition, ensemble forecasting | 16-member ensemble at convection-permitting scale |

Each phase produces something testable and publishable on its own. No phase is hidden behind another.

## Why this is worth doing

- **There is a real gap.** Modern fully-GPU NWP is an open problem. MPAS, FV3, and IFS are all Fortran-first; their GPU paths are partial and bolted on.
- **It composes with [XasmAI](https://github.com/Xassemblianist/XasmAI).** Phase 5+ opens the door to ML-augmented forecasting (e.g., learned subgrid closures, FourCastNet-style emulators) trained with my own engine.
- **It has a real-world target.** A working Antalya forecast page is a tangible outcome &mdash; not a paper, not a benchmark, a thing the public can see.

## References

The core algorithmic references this work draws on:

- Skamarock, W.C. &amp; Klemp, J.B. (2008). *A time-split nonhydrostatic atmospheric model for weather research and forecasting applications.* J. Comput. Phys.
- Wicker, L.J. &amp; Skamarock, W.C. (2002). *Time-splitting methods for elastic models using forward time schemes.* Mon. Wea. Rev.
- Klemp, J.B., Skamarock, W.C. &amp; Dudhia, J. (2007). *Conservative split-explicit time integration methods for the compressible nonhydrostatic equations.* Mon. Wea. Rev.

## License

MIT &mdash; see [LICENSE](LICENSE).
