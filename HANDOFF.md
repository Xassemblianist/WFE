# CPPWRF Project Handoff & Phase 2 Blueprint

This document contains the complete context, current technical state, and the final required fixes for **CPPWRF (Phase 2)**. Please read this entirely before writing any code.

## 1. Project Overarching Goals (CPPWRF)
**CPPWRF** is being developed as a high-performance, GPU-accelerated Numerical Weather Prediction (NWP) engine written in C++ and CUDA. The ultimate objective is to surpass the legacy CPU-based WRF (Weather Research and Forecasting) model by utilizing modern GPU hardware architectures (e.g., RTX 5070 Ti) from the ground up, moving away from legacy Fortran code.

**Project Roadmap:**
- **Phase 1 (Completed):** 1D Shallow Water Equations (SWE) on GPU (validated successfully).
- **Phase 2 (CURRENT):** 2D Non-Hydrostatic Compressible Euler equations (Atmospheric Engine).
- **Future Phases:** 3D dynamics, moisture physics (microphysics), Coriolis terms, and terrain-following coordinates.

## 2. Phase 2 Current Implementation Details
We are currently building the 2D non-hydrostatic atmospheric dynamics core (`src/solver/cuda/euler2d_gpu.cu`).
- **Governing Equations:** Compressible Euler equations in conservative variables ($\rho, \rho u, \rho w, \rho \theta$).
- **Time Integration:** Wicker-Skamarock Split-Explicit SSP-RK3.
- **Reconstruction:** WENO5 (5th-order weighted essentially non-oscillatory scheme) for spatial accuracy.
- **Numerical Flux:** **HLLC Riemann Solver**. (We recently upgraded from the Rusanov flux to HLLC to successfully couple pressure and velocity, eliminating explicit Euler checkerboard decoupling).
- **Base State Formulation:** The model solves for the full variables but analytically subtracts the hydrostatic background state ($\rho_b, p_b, \theta_b$) during flux reconstruction to strictly preserve well-balanced properties.
- **Test Cases:** 
  1. Robert (1993) Density Current (currently being validated).
  2. Schär et al. (2002) Mountain Wave.

## 3. The Current Blocker: Exponential Vertical Blowup
The solver is currently experiencing a severe exponential blowup strictly in the vertical velocity ($w$), reaching 14 m/s in just 3 seconds of simulation time even when initialized with the exact unperturbed base state (`dtheta_cold = 0.0`). 

**Root Cause Analysis:**
The instability is a **well-balanced boundary error**. At the top and bottom rigid walls ($f=0$ and $f=nz$), the hydrostatic balance $\frac{\partial p'}{\partial z} = -\rho' g$ MUST be exactly maintained. However, the ghost cells for $\rho$ and $\theta$ were populated using a zero-gradient boundary condition. WENO5 sees this zero-gradient state and constructs a polynomial with a zero derivative at the boundary. 

As a result, the vertical pressure gradient at the wall becomes exactly zero ($\frac{\partial p'}{\partial z} = 0$), leaving the gravitational term $-\rho' g$ completely unbalanced. This factor-of-2 discrete imbalance creates an artificial massive forcing at the wall that pumps energy into acoustic waves, causing an exponential acoustic blowup.

## 4. The Required Fix
To fix this, we must BYPASS the WENO reconstruction at the boundaries $f=0$ and $f=nz$. Instead, we must explicitly enforce an exact hydrostatic pressure extrapolation from the adjacent interior cell center directly to the wall face.

**Task 1: Apply this code fix to `k_z_fluxes` in `src/solver/cuda/euler2d_gpu.cu`:**

```cpp
    // In k_z_fluxes, replace the boundary handling at f==0 and f==nz with the following exact extrapolation:
    
    if (f == 0) {
        const int fidx = f * stride + ix_;
        const int icell = halo * stride + ix_; // bottom interior cell center
        const Real p_cell = pressure(rhoTh[icell]);
        const Real p_prime = p_cell - p_b[halo];
        const Real r_prime = rho[icell] - rho_b[halo];
        
        // Hydrostatic extrapolation to z=0 (distance is dz/2)
        const Real p_prime_wall = p_prime + 0.5 * dz * r_prime * atm::g;
        
        Fz_rho [fidx] = 0.0;
        Fz_rhou[fidx] = 0.0;
        Fz_rhow[fidx] = p_prime_wall;
        Fz_rhoTh[fidx] = 0.0;
        return;
    }
    if (f == nz) {
        const int fidx = f * stride + ix_;
        const int icell = (halo + nz - 1) * stride + ix_; // top interior cell center
        const Real p_cell = pressure(rhoTh[icell]);
        const Real p_prime = p_cell - p_b[halo + nz - 1];
        const Real r_prime = rho[icell] - rho_b[halo + nz - 1];
        
        // Hydrostatic extrapolation to z=H (distance is dz/2)
        const Real p_prime_wall = p_prime - 0.5 * dz * r_prime * atm::g;
        
        Fz_rho [fidx] = 0.0;
        Fz_rhou[fidx] = 0.0;
        Fz_rhow[fidx] = p_prime_wall;
        Fz_rhoTh[fidx] = 0.0;
        return;
    }
```

*Note: You must ensure that the `k_z_fluxes` CUDA kernel accepts `Real dz` as its last argument, and that the wrapper function `launch_z_fluxes` passes `g_.dz` into it.*

## 5. Next Steps for the Session
1. **Fix & Verify:** Apply the hydrostatic boundary extrapolation fix described above.
2. **Equilibrium Test:** Compile and run `./build/cppwrf2d --tend 3 --cfl 0.4` with `dtheta_cold = 0.0` (in `cases/density_current.hpp`). The velocities MUST remain exactly $0.00$ at all times, confirming exact hydrostatic equilibrium.
3. **Density Current Test:** Change `dtheta_cold` back to `-15.0` and run the simulation. The solver should now be completely stable. Add a simple Python script to plot the generated CSV snapshots to verify the physical structure of the density current.
4. **Mountain Wave:** Once validated, move on to the Schär et al. (2002) Mountain Wave test case.
5. **Damping/Sponge Layer:** Add Rayleigh damping near the top boundary to absorb upward-propagating gravity waves.
