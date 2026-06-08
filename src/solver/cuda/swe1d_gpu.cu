/*
 * WFE — Phase 1
 * GPU solver: 1D SWE, WENO3 + HLLC + SSP-RK3
 * Targets: sm_75 (RTX 2060) ... sm_120 (RTX 5070 Ti / Blackwell)
 *
 * dt is computed asynchronously every DT_UPDATE_FREQ steps via pinned memory
 * + cudaMemcpyAsync, so the GPU pipeline runs without a blocking sync per step.
 */

#include "swe1d_gpu.hpp"
#include <cmath>
#include <stdexcept>
#include <string>
#include <cstring>

// ─── Error helpers ────────────────────────────────────────────────────────────

#define CUDA_CHECK(call) do {                                               \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess)                                                  \
        throw std::runtime_error(std::string("CUDA error: ")               \
            + cudaGetErrorString(_e) + " at " __FILE__ ":"                 \
            + std::to_string(__LINE__));                                    \
} while(0)

// ─── Device helpers ───────────────────────────────────────────────────────────

__device__ __forceinline__ Real dev_u(const State& q) {
    return (q.h > 1.0e-10) ? q.hu / q.h : 0.0;
}
__device__ __forceinline__ Real dev_a(const State& q) {
    return (q.h > 1.0e-10) ? sqrt(9.81 * q.h) : 0.0;
}
__device__ __forceinline__ Flux physical_flux_dev(const State& q) {
    const Real u = dev_u(q);
    return { q.hu, u * q.hu + 0.5 * 9.81 * q.h * q.h };
}

// ─── WENO3 helpers ────────────────────────────────────────────────────────────

__device__ __forceinline__ Real weno3L(Real qm1, Real q0, Real qp1) {
    const Real b0 = (q0 - qm1) * (q0 - qm1);
    const Real b1 = (qp1 - q0) * (qp1 - q0);
    const Real a0 = (1.0/3.0) / ((b0 + 1.0e-6) * (b0 + 1.0e-6));
    const Real a1 = (2.0/3.0) / ((b1 + 1.0e-6) * (b1 + 1.0e-6));
    const Real w  = 1.0 / (a0 + a1);
    return (a0 * (-0.5*qm1 + 1.5*q0) + a1 * (0.5*q0 + 0.5*qp1)) * w;
}

// q0=q_i, qp1=q_{i+1}, qp2=q_{i+2} where cell i is LEFT of the interface.
// Stencil 0 {i+1,i+2}: qs0=3/2*qp1-1/2*qp2, d=2/3.  Stencil 1 {i,i+1}: qs1=avg, d=1/3.
__device__ __forceinline__ Real weno3R(Real q0, Real qp1, Real qp2) {
    const Real b0 = (qp2 - qp1) * (qp2 - qp1);  // stencil {i+1, i+2}
    const Real b1 = (qp1 - q0)  * (qp1 - q0);   // stencil {i,   i+1}
    const Real a0 = (2.0/3.0) / ((b0 + 1.0e-6) * (b0 + 1.0e-6));
    const Real a1 = (1.0/3.0) / ((b1 + 1.0e-6) * (b1 + 1.0e-6));
    const Real w  = 1.0 / (a0 + a1);
    return (a0 * (1.5*qp1 - 0.5*qp2) + a1 * (0.5*q0 + 0.5*qp1)) * w;
}

// ─── Kernel: transmissive halos ───────────────────────────────────────────────

__global__ void kernel_fill_halos(State* q, int HALO, int nx) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < HALO) {
        q[HALO - 1 - k]  = q[HALO];
        q[HALO + nx + k] = q[HALO + nx - 1];
    }
}

// ─── Kernel: WENO3 reconstruction ─────────────────────────────────────────────

__global__ void kernel_weno3(const State* __restrict__ q,
                              State* __restrict__ qL,
                              State* __restrict__ qR,
                              int HALO, int nx)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f > nx) return;

    const int il = HALO + f - 1;
    State left;
    left.h  = weno3L(q[il-1].h,  q[il].h,  q[il+1].h);
    left.hu = weno3L(q[il-1].hu, q[il].hu, q[il+1].hu);
    if (left.h < 1.0e-10) { left.h = 1.0e-10; left.hu = 0.0; }
    qL[f] = left;

    // Right state: same center cell il, stencil {f-1, f, f+1} = {il, il+1, il+2}.
    State right;
    right.h  = weno3R(q[il].h,  q[il+1].h,  q[il+2].h);
    right.hu = weno3R(q[il].hu, q[il+1].hu, q[il+2].hu);
    if (right.h < 1.0e-10) { right.h = 1.0e-10; right.hu = 0.0; }
    qR[f] = right;
}

// ─── Kernel: HLLC flux ────────────────────────────────────────────────────────

__global__ void kernel_hllc(const State* __restrict__ qL,
                             const State* __restrict__ qR,
                             Flux* __restrict__ F,
                             int nfaces)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nfaces) return;

    const State L = qL[f], R = qR[f];
    const Real uL = dev_u(L), uR = dev_u(R);
    const Real aL = dev_a(L), aR = dev_a(R);

    const Real sL = fmin(uL - aL, uR - aR);
    const Real sR = fmax(uL + aL, uR + aR);

    if (sL >= 0.0) { F[f] = physical_flux_dev(L); return; }
    if (sR <= 0.0) { F[f] = physical_flux_dev(R); return; }

    const Real denom = L.h * (sL - uL) - R.h * (sR - uR);
    const Real sStar = (denom != 0.0)
        ? (L.hu*(sL-uL) - R.hu*(sR-uR) + 9.81*(0.5*R.h*R.h - 0.5*L.h*L.h)) / denom
        : 0.0;

    Flux FL = physical_flux_dev(L);
    Flux FR = physical_flux_dev(R);
    Flux Fs;

    if (sStar >= 0.0) {
        const Real fac = L.h * (sL - uL) / (sL - sStar);
        Fs.mass     = FL.mass     + sL * (fac        - L.h);
        Fs.momentum = FL.momentum + sL * (fac*sStar  - L.hu);
    } else {
        const Real fac = R.h * (sR - uR) / (sR - sStar);
        Fs.mass     = FR.mass     + sR * (fac        - R.h);
        Fs.momentum = FR.momentum + sR * (fac*sStar  - R.hu);
    }
    F[f] = Fs;
}

// ─── Kernel: SSP-RK3 stage update ────────────────────────────────────────────
// dst[i] = coeff_old * q_old[i] + coeff_stage * (q_stage[i] + dt * L[i])

__global__ void kernel_rk_update(State* __restrict__ dst,
                                 const State* __restrict__ q_old,
                                 const State* __restrict__ q_stage,
                                 const Flux* __restrict__ F,
                                 Real dt, Real inv_dx,
                                 Real coeff_old, Real coeff_stage,
                                 int HALO, int nx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;
    const int gi = HALO + i;

    const Real dh  = -(F[i+1].mass     - F[i].mass)     * inv_dx;
    const Real dhu = -(F[i+1].momentum - F[i].momentum) * inv_dx;

    Real h_new  = coeff_old * q_old[gi].h  + coeff_stage * (q_stage[gi].h  + dt * dh);
    Real hu_new = coeff_old * q_old[gi].hu + coeff_stage * (q_stage[gi].hu + dt * dhu);
    if (h_new < 1.0e-10) { h_new = 1.0e-10; hu_new = 0.0; }
    dst[gi] = {h_new, hu_new};
}

// ─── Kernel: wave speed reduction for CFL ────────────────────────────────────

__global__ void kernel_wave_speed(const State* __restrict__ q,
                                  Real* __restrict__ smax_out,
                                  int HALO, int nx)
{
    extern __shared__ Real smem[];
    const int i  = blockIdx.x * blockDim.x + threadIdx.x;
    const int ti = threadIdx.x;

    Real s = 0.0;
    if (i < nx) s = fabs(dev_u(q[HALO + i])) + dev_a(q[HALO + i]);
    smem[ti] = s;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (ti < stride) smem[ti] = fmax(smem[ti], smem[ti + stride]);
        __syncthreads();
    }
    if (ti == 0)
        atomicMax((unsigned long long*)smax_out,
                  __double_as_longlong(smem[0]));
}

// ─── SWE1D_GPU implementation ─────────────────────────────────────────────────

SWE1D_GPU::SWE1D_GPU(int nx, Real dx, Real cfl)
    : nx_(nx), dx_(dx), cfl_(cfl), t_(0.0), step_n_(0)
{
    const int N = nx + 2 * wfe::HALO;

    CUDA_CHECK(cudaMalloc(&d_q_,   N * sizeof(State)));
    CUDA_CHECK(cudaMalloc(&d_q1_,  N * sizeof(State)));
    CUDA_CHECK(cudaMalloc(&d_q2_,  N * sizeof(State)));
    CUDA_CHECK(cudaMalloc(&d_qL_,  (nx + 1) * sizeof(State)));
    CUDA_CHECK(cudaMalloc(&d_qR_,  (nx + 1) * sizeof(State)));
    CUDA_CHECK(cudaMalloc(&d_F_,   (nx + 1) * sizeof(Flux)));
    CUDA_CHECK(cudaMalloc(&d_smax_, sizeof(Real)));

    // Pinned host buffer for async dt copy
    CUDA_CHECK(cudaMallocHost(&h_smax_, sizeof(Real)));
    *h_smax_ = 0.0;

    CUDA_CHECK(cudaEventCreate(&ev_smax_ready_));
}

SWE1D_GPU::~SWE1D_GPU() {
    cudaFree(d_q_);
    cudaFree(d_q1_);
    cudaFree(d_q2_);
    cudaFree(d_qL_);
    cudaFree(d_qR_);
    cudaFree(d_F_);
    cudaFree(d_smax_);
    cudaFreeHost(h_smax_);
    cudaEventDestroy(ev_smax_ready_);
}

void SWE1D_GPU::set_state(const std::vector<State>& ic) {
    const int N = nx_ + 2 * wfe::HALO;
    std::vector<State> buf(N);
    for (int i = 0; i < nx_; ++i) buf[wfe::HALO + i] = ic[i];
    for (int k = 0; k < wfe::HALO; ++k) {
        buf[wfe::HALO - 1 - k]   = ic[0];
        buf[wfe::HALO + nx_ + k] = ic[nx_ - 1];
    }
    CUDA_CHECK(cudaMemcpy(d_q_, buf.data(), N * sizeof(State), cudaMemcpyHostToDevice));
    t_ = 0.0; step_n_ = 0; smax_pending_ = false;

    // Bootstrap dt with a blocking compute
    CUDA_CHECK(cudaMemset(d_smax_, 0, sizeof(Real)));
    int threads = 256, blocks = (nx_ + threads - 1) / threads;
    kernel_wave_speed<<<blocks, threads, threads*sizeof(Real)>>>(
        d_q_, d_smax_, wfe::HALO, nx_);
    CUDA_CHECK(cudaDeviceSynchronize());
    Real smax;
    CUDA_CHECK(cudaMemcpy(&smax, d_smax_, sizeof(Real), cudaMemcpyDeviceToHost));
    dt_ = (smax > 0.0) ? cfl_ * dx_ / smax : cfl_ * dx_;
}

void SWE1D_GPU::async_compute_dt() {
    CUDA_CHECK(cudaMemset(d_smax_, 0, sizeof(Real)));
    int threads = 256, blocks = (nx_ + threads - 1) / threads;
    kernel_wave_speed<<<blocks, threads, threads*sizeof(Real)>>>(
        d_q_, d_smax_, wfe::HALO, nx_);
    // Async copy so we don't block; record event to know when it's done
    CUDA_CHECK(cudaMemcpyAsync(h_smax_, d_smax_, sizeof(Real),
                               cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_smax_ready_));
    smax_pending_ = true;
}

void SWE1D_GPU::collect_dt() {
    if (!smax_pending_) return;
    // Wait only for the async copy to finish (not for all GPU work)
    CUDA_CHECK(cudaEventSynchronize(ev_smax_ready_));
    Real smax = *h_smax_;
    if (smax > 0.0) dt_ = cfl_ * dx_ / smax;
    smax_pending_ = false;
}

void SWE1D_GPU::launch_fill_halos(State* d_q) {
    constexpr int threads = 32;
    kernel_fill_halos<<<1, threads>>>(d_q, wfe::HALO, nx_);
}

void SWE1D_GPU::launch_reconstruct(const State* d_q) {
    const int nfaces = nx_ + 1;
    const int threads = 256;
    const int blocks  = (nfaces + threads - 1) / threads;
    kernel_weno3<<<blocks, threads>>>(d_q, d_qL_, d_qR_, wfe::HALO, nx_);
}

void SWE1D_GPU::launch_fluxes() {
    const int nfaces  = nx_ + 1;
    const int threads = 256;
    const int blocks  = (nfaces + threads - 1) / threads;
    kernel_hllc<<<blocks, threads>>>(d_qL_, d_qR_, d_F_, nfaces);
}

void SWE1D_GPU::launch_update(State* d_dst, const State* d_q_old,
                               const State* d_q_stage,
                               Real dt, Real coeff_old, Real coeff_stage)
{
    const int threads = 256;
    const int blocks  = (nx_ + threads - 1) / threads;
    kernel_rk_update<<<blocks, threads>>>(
        d_dst, d_q_old, d_q_stage, d_F_,
        dt, 1.0 / dx_, coeff_old, coeff_stage,
        wfe::HALO, nx_);
}

void SWE1D_GPU::step() {
    // Collect dt from the previous async request (if any).
    // For step_n_ > 0 and step_n_ % DT_UPDATE_FREQ == 0, a request was
    // already kicked off at the end of the previous step.
    if (step_n_ > 0 && (step_n_ % DT_UPDATE_FREQ) == 0)
        collect_dt();

    const Real dt = dt_;

    // ── Stage 1: q1 = q + dt * L(q) ──────────────────────────────────────────
    launch_reconstruct(d_q_);
    launch_fluxes();
    launch_update(d_q1_, d_q_, d_q_, dt, 0.0, 1.0);
    launch_fill_halos(d_q1_);

    // ── Stage 2: q2 = 3/4 q + 1/4 (q1 + dt*L(q1)) ───────────────────────────
    launch_reconstruct(d_q1_);
    launch_fluxes();
    launch_update(d_q2_, d_q_, d_q1_, dt, 0.75, 0.25);
    launch_fill_halos(d_q2_);

    // ── Stage 3: q = 1/3 q + 2/3 (q2 + dt*L(q2)) ────────────────────────────
    launch_reconstruct(d_q2_);
    launch_fluxes();
    launch_update(d_q_, d_q_, d_q2_, dt, 1.0/3.0, 2.0/3.0);
    launch_fill_halos(d_q_);

    t_ += dt;
    step_n_++;

    // Kick off the next async dt update (result collected at start of next step).
    if ((step_n_ % DT_UPDATE_FREQ) == 0)
        async_compute_dt();
}

std::vector<State> SWE1D_GPU::get_state() const {
    // Flush any pending work before reading back
    CUDA_CHECK(cudaDeviceSynchronize());
    const int N = nx_ + 2 * wfe::HALO;
    std::vector<State> buf(N);
    CUDA_CHECK(cudaMemcpy(buf.data(), d_q_, N * sizeof(State), cudaMemcpyDeviceToHost));
    return std::vector<State>(buf.begin() + wfe::HALO,
                              buf.begin() + wfe::HALO + nx_);
}
