/*
 * WFE — 3D non-hydrostatic compressible Euler (GPU)
 * Scheme : WENO5 reconstruction · HLLC flux · split-explicit SSP-RK3
 * Layout : SoA [nz+2h][ny+2h][nx+2h], x is fastest dimension (coalesced)
 * Physics: perturbation form (p-pb, ρ-ρb) for well-balanced hydrostatics
 * 5 variables: ρ, ρu, ρv, ρw, ρθ   (θ = potential temperature)
 */
#include "euler3d_gpu.cuh"
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cstring>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){     \
    printf("CUDA %s:%d: %s\n",__FILE__,__LINE__,               \
           cudaGetErrorString(_e)); exit(1);} } while(0)

// ─── WENO5 (Jiang-Shu 1996) ───────────────────────────────────────────────────
__device__ __forceinline__ Real weno5L(Real m2,Real m1,Real c0,Real p1,Real p2)
{
    const Real b0=(13./12.)*(m2-2*m1+c0)*(m2-2*m1+c0)+.25*(m2-4*m1+3*c0)*(m2-4*m1+3*c0);
    const Real b1=(13./12.)*(m1-2*c0+p1)*(m1-2*c0+p1)+.25*(m1-p1)*(m1-p1);
    const Real b2=(13./12.)*(c0-2*p1+p2)*(c0-2*p1+p2)+.25*(3*c0-4*p1+p2)*(3*c0-4*p1+p2);
    constexpr Real eps=1e-6;
    const Real a0=0.1/((b0+eps)*(b0+eps));
    const Real a1=0.6/((b1+eps)*(b1+eps));
    const Real a2=0.3/((b2+eps)*(b2+eps));
    const Real w=1./(a0+a1+a2);
    return ((a0*(2*m2-7*m1+11*c0)+a1*(-m1+5*c0+2*p1)+a2*(2*c0+5*p1-p2))/6.)*w;
}
__device__ __forceinline__ Real weno5R(Real m2,Real m1,Real c0,Real p1,Real p2)
{
    const Real b0=(13./12.)*(m2-2*m1+c0)*(m2-2*m1+c0)+.25*(m2-4*m1+3*c0)*(m2-4*m1+3*c0);
    const Real b1=(13./12.)*(m1-2*c0+p1)*(m1-2*c0+p1)+.25*(m1-p1)*(m1-p1);
    const Real b2=(13./12.)*(c0-2*p1+p2)*(c0-2*p1+p2)+.25*(3*c0-4*p1+p2)*(3*c0-4*p1+p2);
    constexpr Real eps=1e-6;
    const Real a0=0.3/((b0+eps)*(b0+eps));
    const Real a1=0.6/((b1+eps)*(b1+eps));
    const Real a2=0.1/((b2+eps)*(b2+eps));
    const Real w=1./(a0+a1+a2);
    return ((a0*(-m2+5*m1+2*c0)+a1*(2*m1+5*c0-p1)+a2*(11*c0-7*p1+2*p2))/6.)*w;
}

// ─── EOS ──────────────────────────────────────────────────────────────────────
__device__ __forceinline__ Real pressure3d(Real rhoTh) {
    return atm::p0 * pow(max(atm::Rd * rhoTh / atm::p0, 1e-10), atm::gamma);
}

// ─── HLLC: x-direction (normal = u) ──────────────────────────────────────────
__device__ __forceinline__ void hllc_x(
    Real& Fr, Real& Fru, Real& Frv, Real& Frw, Real& FrT,
    Real rL,Real ruL,Real rvL,Real rwL,Real rTL,Real pL,
    Real rR,Real ruR,Real rvR,Real rwR,Real rTR,Real pR,Real pb)
{
    const Real uL=ruL/max(rL,1e-10), uR=ruR/max(rR,1e-10);
    const Real vL=rvL/max(rL,1e-10), vR=rvR/max(rR,1e-10);
    const Real wL=rwL/max(rL,1e-10), wR=rwR/max(rR,1e-10);
    const Real aL=sqrt(atm::gamma*pL/max(rL,1e-10));
    const Real aR=sqrt(atm::gamma*pR/max(rR,1e-10));
    const Real SL=min(uL-aL,uR-aR), SR=max(uL+aL,uR+aR);
    // physical fluxes
    const Real fLr=ruL,  fRr=ruR;
    const Real fLu=ruL*uL+(pL-pb), fRu=ruR*uR+(pR-pb);
    const Real fLv=ruL*vL, fRv=ruR*vR;
    const Real fLw=ruL*wL, fRw=ruR*wR;
    const Real fLT=rTL*uL, fRT=rTR*uR;
    if(SL>=0){Fr=fLr;Fru=fLu;Frv=fLv;Frw=fLw;FrT=fLT;return;}
    if(SR<=0){Fr=fRr;Fru=fRu;Frv=fRv;Frw=fRw;FrT=fRT;return;}
    const Real Ss=(pR-pL+ruL*(SL-uL)-ruR*(SR-uR))/(rL*(SL-uL)-rR*(SR-uR)-1e-15);
    if(Ss>=0){
        const Real c=1./(SL-Ss-1e-15), rs=rL*(SL-uL)*c;
        Fr =fLr +SL*(rs       -rL);
        Fru=fLu +SL*(rs*Ss    -ruL);
        Frv=fLv +SL*(rs*vL    -rvL);
        Frw=fLw +SL*(rs*wL    -rwL);
        FrT=fLT +SL*(rs*rTL/rL-rTL);
    } else {
        const Real c=1./(SR-Ss+1e-15), rs=rR*(SR-uR)*c;
        Fr =fRr +SR*(rs       -rR);
        Fru=fRu +SR*(rs*Ss    -ruR);
        Frv=fRv +SR*(rs*vR    -rvR);
        Frw=fRw +SR*(rs*wR    -rwR);
        FrT=fRT +SR*(rs*rTR/rR-rTR);
    }
}

// ─── HLLC: y-direction (normal = v) ──────────────────────────────────────────
__device__ __forceinline__ void hllc_y(
    Real& Fr, Real& Fru, Real& Frv, Real& Frw, Real& FrT,
    Real rL,Real ruL,Real rvL,Real rwL,Real rTL,Real pL,
    Real rR,Real ruR,Real rvR,Real rwR,Real rTR,Real pR,Real pb)
{
    const Real uL=ruL/max(rL,1e-10), uR=ruR/max(rR,1e-10);
    const Real vL=rvL/max(rL,1e-10), vR=rvR/max(rR,1e-10);
    const Real wL=rwL/max(rL,1e-10), wR=rwR/max(rR,1e-10);
    const Real aL=sqrt(atm::gamma*pL/max(rL,1e-10));
    const Real aR=sqrt(atm::gamma*pR/max(rR,1e-10));
    const Real SL=min(vL-aL,vR-aR), SR=max(vL+aL,vR+aR);
    const Real fLr=rvL,  fRr=rvR;
    const Real fLu=rvL*uL, fRu=rvR*uR;
    const Real fLv=rvL*vL+(pL-pb), fRv=rvR*vR+(pR-pb);
    const Real fLw=rvL*wL, fRw=rvR*wR;
    const Real fLT=rTL*vL, fRT=rTR*vR;
    if(SL>=0){Fr=fLr;Fru=fLu;Frv=fLv;Frw=fLw;FrT=fLT;return;}
    if(SR<=0){Fr=fRr;Fru=fRu;Frv=fRv;Frw=fRw;FrT=fRT;return;}
    const Real Ss=(pR-pL+rvL*(SL-vL)-rvR*(SR-vR))/(rL*(SL-vL)-rR*(SR-vR)-1e-15);
    if(Ss>=0){
        const Real c=1./(SL-Ss-1e-15), rs=rL*(SL-vL)*c;
        Fr =fLr +SL*(rs       -rL);
        Fru=fLu +SL*(rs*uL    -ruL);
        Frv=fLv +SL*(rs*Ss    -rvL);
        Frw=fLw +SL*(rs*wL    -rwL);
        FrT=fLT +SL*(rs*rTL/rL-rTL);
    } else {
        const Real c=1./(SR-Ss+1e-15), rs=rR*(SR-vR)*c;
        Fr =fRr +SR*(rs       -rR);
        Fru=fRu +SR*(rs*uR    -ruR);
        Frv=fRv +SR*(rs*Ss    -rvR);
        Frw=fRw +SR*(rs*wR    -rwR);
        FrT=fRT +SR*(rs*rTR/rR-rTR);
    }
}

// ─── HLLC: z-direction (normal = w) ──────────────────────────────────────────
__device__ __forceinline__ void hllc_z3d(
    Real& Fr, Real& Fru, Real& Frv, Real& Frw, Real& FrT,
    Real rL,Real ruL,Real rvL,Real rwL,Real rTL,Real pL,
    Real rR,Real ruR,Real rvR,Real rwR,Real rTR,Real pR,Real pb)
{
    const Real uL=ruL/max(rL,1e-10), uR=ruR/max(rR,1e-10);
    const Real vL=rvL/max(rL,1e-10), vR=rvR/max(rR,1e-10);
    const Real wL=rwL/max(rL,1e-10), wR=rwR/max(rR,1e-10);
    const Real aL=sqrt(atm::gamma*pL/max(rL,1e-10));
    const Real aR=sqrt(atm::gamma*pR/max(rR,1e-10));
    const Real SL=min(wL-aL,wR-aR), SR=max(wL+aL,wR+aR);
    const Real fLr=rwL,  fRr=rwR;
    const Real fLu=rwL*uL, fRu=rwR*uR;
    const Real fLv=rwL*vL, fRv=rwR*vR;
    const Real fLw=rwL*wL+(pL-pb), fRw=rwR*wR+(pR-pb);
    const Real fLT=rTL*wL, fRT=rTR*wR;
    if(SL>=0){Fr=fLr;Fru=fLu;Frv=fLv;Frw=fLw;FrT=fLT;return;}
    if(SR<=0){Fr=fRr;Fru=fRu;Frv=fRv;Frw=fRw;FrT=fRT;return;}
    const Real Ss=(pR-pL+rwL*(SL-wL)-rwR*(SR-wR))/(rL*(SL-wL)-rR*(SR-wR)-1e-15);
    if(Ss>=0){
        const Real c=1./(SL-Ss-1e-15), rs=rL*(SL-wL)*c;
        Fr =fLr +SL*(rs       -rL);
        Fru=fLu +SL*(rs*uL    -ruL);
        Frv=fLv +SL*(rs*vL    -rvL);
        Frw=fLw +SL*(rs*Ss    -rwL);
        FrT=fLT +SL*(rs*rTL/rL-rTL);
    } else {
        const Real c=1./(SR-Ss+1e-15), rs=rR*(SR-wR)*c;
        Fr =fRr +SR*(rs       -rR);
        Fru=fRu +SR*(rs*uR    -ruR);
        Frv=fRv +SR*(rs*vR    -rvR);
        Frw=fRw +SR*(rs*Ss    -rwR);
        FrT=fRT +SR*(rs*rTR/rR-rTR);
    }
}

// ─── Halo fill: X direction (transmissive) ───────────────────────────────────
__global__ void k_fill_x_halos(Real* r,Real* ru,Real* rv,Real* rw,Real* rT,
                                int sy,int sz,int nx,int ny,int nz,int h)
{
    const int k  = blockIdx.x*blockDim.x+threadIdx.x; // halo index 0..h-1
    const int iy = blockIdx.y*blockDim.y+threadIdx.y;
    const int iz = blockIdx.z*blockDim.z+threadIdx.z;
    if(k>=h||iy>=ny+2*h||iz>=nz+2*h) return;
    const int iL=h*1+iy*sy+iz*sz;       // leftmost interior x
    const int iR=(h+nx-1)*1+iy*sy+iz*sz; // rightmost interior x
#define XFILL(arr) arr[(h-1-k)+iy*sy+iz*sz]=arr[iL]; \
                   arr[(h+nx+k)+iy*sy+iz*sz]=arr[iR];
    XFILL(r) XFILL(ru) XFILL(rv) XFILL(rw) XFILL(rT)
#undef XFILL
}

// ─── Halo fill: Y direction (periodic) ───────────────────────────────────────
__global__ void k_fill_y_halos(Real* r,Real* ru,Real* rv,Real* rw,Real* rT,
                                int sy,int sz,int nx,int ny,int nz,int h)
{
    const int ix = blockIdx.x*blockDim.x+threadIdx.x;
    const int k  = blockIdx.y*blockDim.y+threadIdx.y; // halo index 0..h-1
    const int iz = blockIdx.z*blockDim.z+threadIdx.z;
    if(ix>=nx+2*h||k>=h||iz>=nz+2*h) return;
    const int iBot=ix+(h  )*sy+iz*sz;         // bottom interior y
    const int iTop=ix+(h+ny-1)*sy+iz*sz;      // top interior y
#define YFILL(arr) arr[ix+(h-1-k)*sy+iz*sz]=arr[iTop+(ny-1-k)*sy-h*sy]; \
                   arr[ix+(h+ny+k)*sy+iz*sz]=arr[iBot+k*sy];
    YFILL(r) YFILL(ru) YFILL(rv) YFILL(rw) YFILL(rT)
#undef YFILL
}

// ─── Halo fill: Z direction (well-balanced wall) ─────────────────────────────
__global__ void k_fill_z_halos(Real* r,Real* ru,Real* rv,Real* rw,Real* rT,
                                const Real* rho_b,const Real* rhoTh_b,
                                int sy,int sz,int nx,int ny,int nz,int h)
{
    const int ix = blockIdx.x*blockDim.x+threadIdx.x;
    const int iy = blockIdx.y*blockDim.y+threadIdx.y;
    const int k  = blockIdx.z*blockDim.z+threadIdx.z; // halo index 0..h-1
    if(ix>=nx+2*h||iy>=ny+2*h||k>=h) return;
    const int gi0=ix+iy*sy+(h+k)*sz;       // bottom interior cell at layer k
    const int gi1=ix+iy*sy+(h+nz-1-k)*sz; // top interior cell at layer k
    // ghost indices
    const int gb=ix+iy*sy+(h-1-k)*sz;
    const int gt=ix+iy*sy+(h+nz+k)*sz;
    // bottom: well-balanced ρ/ρθ, reflective w
    r [gb] = r [gi0] - rho_b[h+k]    + rho_b[h-1-k];
    ru[gb] = ru[gi0];
    rv[gb] = rv[gi0];
    rw[gb] = -rw[gi0];
    rT[gb] = rT[gi0] - rhoTh_b[h+k] + rhoTh_b[h-1-k];
    // top: same
    r [gt] = r [gi1] - rho_b[h+nz-1-k]    + rho_b[h+nz+k];
    ru[gt] = ru[gi1];
    rv[gt] = rv[gi1];
    rw[gt] = -rw[gi1];
    rT[gt] = rT[gi1] - rhoTh_b[h+nz-1-k] + rhoTh_b[h+nz+k];
}

// ─── X-direction fluxes ───────────────────────────────────────────────────────
// Thread (fx, iy, iz): interface fx in [0,nx], iy in [0,ny-1], iz in [0,nz-1]
// Flux index: iz*(ny*(nx+1)) + iy*(nx+1) + fx
__global__ void k_x_fluxes_3d(
    const Real* rho,const Real* rhou,const Real* rhov,const Real* rhow,const Real* rhoTh,
    Real* Fx0,Real* Fx1,Real* Fx2,Real* Fx3,Real* Fx4,
    const Real* p_b,const Real* rho_b,const Real* rhoTh_b,
    int sy,int sz,int nx,int ny,int nz,int h)
{
    const int fx = blockIdx.x*blockDim.x+threadIdx.x;
    const int iy = blockIdx.y*blockDim.y+threadIdx.y;
    const int iz = blockIdx.z*blockDim.z+threadIdx.z;
    if(fx>nx||iy>=ny||iz>=nz) return;

    const int iz_=h+iz, iy_=h+iy;
    const int il=h+fx-1; // left cell x-index in full array (il-2..il+3 stencil)

    Real r[6],ru[6],rv[6],rw[6],rT[6];
    for(int k=-2;k<=3;++k){
        const int gi=iz_*sz+iy_*sy+(il+k);
        r [k+2]=rho  [gi]-rho_b[iz_];
        ru[k+2]=rhou [gi];
        rv[k+2]=rhov [gi];
        rw[k+2]=rhow [gi];
        rT[k+2]=rhoTh[gi]-rhoTh_b[iz_];
    }

    const Real rL =weno5L(r[0],r[1],r[2],r[3],r[4])+rho_b[iz_];
    const Real rR =weno5R(r[1],r[2],r[3],r[4],r[5])+rho_b[iz_];
    const Real rTL=weno5L(rT[0],rT[1],rT[2],rT[3],rT[4])+rhoTh_b[iz_];
    const Real rTR=weno5R(rT[1],rT[2],rT[3],rT[4],rT[5])+rhoTh_b[iz_];
    const Real ruL=weno5L(ru[0],ru[1],ru[2],ru[3],ru[4]);
    const Real ruR=weno5R(ru[1],ru[2],ru[3],ru[4],ru[5]);
    const Real rvL=weno5L(rv[0],rv[1],rv[2],rv[3],rv[4]);
    const Real rvR=weno5R(rv[1],rv[2],rv[3],rv[4],rv[5]);
    const Real rwL=weno5L(rw[0],rw[1],rw[2],rw[3],rw[4]);
    const Real rwR=weno5R(rw[1],rw[2],rw[3],rw[4],rw[5]);

    const Real pL=pressure3d(rTL), pR=pressure3d(rTR), pb=p_b[iz_];
    const int fi=iz*(ny*(nx+1))+iy*(nx+1)+fx;
    hllc_x(Fx0[fi],Fx1[fi],Fx2[fi],Fx3[fi],Fx4[fi],
           rL,ruL,rvL,rwL,rTL,pL, rR,ruR,rvR,rwR,rTR,pR,pb);
}

// ─── Y-direction fluxes ───────────────────────────────────────────────────────
// Thread (ix, fy, iz): interface fy in [0,ny], ix in [0,nx-1], iz in [0,nz-1]
// Flux index: iz*((ny+1)*nx) + fy*nx + ix
__global__ void k_y_fluxes_3d(
    const Real* rho,const Real* rhou,const Real* rhov,const Real* rhow,const Real* rhoTh,
    Real* Fy0,Real* Fy1,Real* Fy2,Real* Fy3,Real* Fy4,
    const Real* p_b,const Real* rho_b,const Real* rhoTh_b,
    int sy,int sz,int nx,int ny,int nz,int h)
{
    const int ix = blockIdx.x*blockDim.x+threadIdx.x;
    const int fy = blockIdx.y*blockDim.y+threadIdx.y;
    const int iz = blockIdx.z*blockDim.z+threadIdx.z;
    if(ix>=nx||fy>ny||iz>=nz) return;

    const int iz_=h+iz, ix_=h+ix;
    const int il=h+fy-1; // left (south) y-index in full array

    Real r[6],ru[6],rv[6],rw[6],rT[6];
    for(int k=-2;k<=3;++k){
        const int gi=iz_*sz+(il+k)*sy+ix_;
        r [k+2]=rho  [gi]-rho_b[iz_];
        ru[k+2]=rhou [gi];
        rv[k+2]=rhov [gi];
        rw[k+2]=rhow [gi];
        rT[k+2]=rhoTh[gi]-rhoTh_b[iz_];
    }

    const Real rL =weno5L(r[0],r[1],r[2],r[3],r[4])+rho_b[iz_];
    const Real rR =weno5R(r[1],r[2],r[3],r[4],r[5])+rho_b[iz_];
    const Real rTL=weno5L(rT[0],rT[1],rT[2],rT[3],rT[4])+rhoTh_b[iz_];
    const Real rTR=weno5R(rT[1],rT[2],rT[3],rT[4],rT[5])+rhoTh_b[iz_];
    const Real ruL=weno5L(ru[0],ru[1],ru[2],ru[3],ru[4]);
    const Real ruR=weno5R(ru[1],ru[2],ru[3],ru[4],ru[5]);
    const Real rvL=weno5L(rv[0],rv[1],rv[2],rv[3],rv[4]);
    const Real rvR=weno5R(rv[1],rv[2],rv[3],rv[4],rv[5]);
    const Real rwL=weno5L(rw[0],rw[1],rw[2],rw[3],rw[4]);
    const Real rwR=weno5R(rw[1],rw[2],rw[3],rw[4],rw[5]);

    const Real pL=pressure3d(rTL), pR=pressure3d(rTR), pb=p_b[iz_];
    const int fi=iz*((ny+1)*nx)+fy*nx+ix;
    hllc_y(Fy0[fi],Fy1[fi],Fy2[fi],Fy3[fi],Fy4[fi],
           rL,ruL,rvL,rwL,rTL,pL, rR,ruR,rvR,rwR,rTR,pR,pb);
}

// ─── Z-direction fluxes ───────────────────────────────────────────────────────
// Thread (ix, iy, fz): interface fz in [0,nz], ix in [0,nx-1], iy in [0,ny-1]
// Flux index: fz*(ny*nx) + iy*nx + ix
__global__ void k_z_fluxes_3d(
    const Real* rho,const Real* rhou,const Real* rhov,const Real* rhow,const Real* rhoTh,
    Real* Fz0,Real* Fz1,Real* Fz2,Real* Fz3,Real* Fz4,
    const Real* p_b,const Real* rho_b,const Real* rhoTh_b,const Real* pi_b,
    int sy,int sz,int nx,int ny,int nz,int h,Real dz)
{
    const int ix = blockIdx.x*blockDim.x+threadIdx.x;
    const int iy = blockIdx.y*blockDim.y+threadIdx.y;
    const int fz = blockIdx.z*blockDim.z+threadIdx.z;
    if(ix>=nx||iy>=ny||fz>nz) return;

    const int ix_=h+ix, iy_=h+iy;
    const int fi=fz*(ny*nx)+iy*nx+ix;

    // Wall BCs: pressure extrapolation at bottom (fz==0) and top (fz==nz)
    if(fz==0){
        const int ic=h*sz+iy_*sy+ix_;
        const Real pp=pressure3d(rhoTh[ic])-p_b[h];
        const Real rp=rho[ic]-rho_b[h];
        Fz0[fi]=0; Fz1[fi]=0; Fz2[fi]=0;
        Fz3[fi]=pp+0.5*dz*rp*atm::g;
        Fz4[fi]=0;
        return;
    }
    if(fz==nz){
        const int ic=(h+nz-1)*sz+iy_*sy+ix_;
        const Real pp=pressure3d(rhoTh[ic])-p_b[h+nz-1];
        const Real rp=rho[ic]-rho_b[h+nz-1];
        Fz0[fi]=0; Fz1[fi]=0; Fz2[fi]=0;
        Fz3[fi]=pp-0.5*dz*rp*atm::g;
        Fz4[fi]=0;
        return;
    }

    const int il=h+fz-1; // bottom z-index
    Real r[6],ru[6],rv[6],rw[6],rT[6];
    for(int k=-2;k<=3;++k){
        const int gi=(il+k)*sz+iy_*sy+ix_;
        r [k+2]=rho  [gi]-rho_b[il+k];
        ru[k+2]=rhou [gi];
        rv[k+2]=rhov [gi];
        rw[k+2]=rhow [gi];
        rT[k+2]=rhoTh[gi]-rhoTh_b[il+k];
    }

    const Real rL_p=weno5L(r[0],r[1],r[2],r[3],r[4]);
    const Real rR_p=weno5R(r[1],r[2],r[3],r[4],r[5]);
    const Real rTL_p=weno5L(rT[0],rT[1],rT[2],rT[3],rT[4]);
    const Real rTR_p=weno5R(rT[1],rT[2],rT[3],rT[4],rT[5]);
    const Real ruL=weno5L(ru[0],ru[1],ru[2],ru[3],ru[4]);
    const Real ruR=weno5R(ru[1],ru[2],ru[3],ru[4],ru[5]);
    const Real rvL=weno5L(rv[0],rv[1],rv[2],rv[3],rv[4]);
    const Real rvR=weno5R(rv[1],rv[2],rv[3],rv[4],rv[5]);
    const Real rwL=weno5L(rw[0],rw[1],rw[2],rw[3],rw[4]);
    const Real rwR=weno5R(rw[1],rw[2],rw[3],rw[4],rw[5]);

    // Hydrostatic base state at interface: exact pi_b interpolation
    const Real pi_f    =0.5*(pi_b[il]+pi_b[il+1]);
    const Real th_bar  =atm::p0/(atm::Rd*rho_b[il])*pow(pi_b[il],atm::Cv/atm::Rd);
    const Real rho_b_f =(atm::p0/(atm::Rd*th_bar))*pow(pi_f,atm::Cv/atm::Rd);
    const Real rT_b_f  =rho_b_f*th_bar;
    const Real pb       =atm::p0*pow(pi_f,atm::Cp/atm::Rd);

    const Real rL=rL_p+rho_b_f, rR=rR_p+rho_b_f;
    const Real rTL=rTL_p+rT_b_f, rTR=rTR_p+rT_b_f;
    const Real pL=pressure3d(rTL), pR=pressure3d(rTR);

    hllc_z3d(Fz0[fi],Fz1[fi],Fz2[fi],Fz3[fi],Fz4[fi],
             rL,ruL,rvL,rwL,rTL,pL, rR,ruR,rvR,rwR,rTR,pR,pb);
}

// ─── Slow tendencies: T = -(dFx/dx + dFy/dy + dFz/dz) ───────────────────────
__global__ void k_slow_tendencies_3d(
    Real* T0,Real* T1,Real* T2,Real* T3,Real* T4,
    const Real* Fx0,const Real* Fx1,const Real* Fx2,const Real* Fx3,const Real* Fx4,
    const Real* Fy0,const Real* Fy1,const Real* Fy2,const Real* Fy3,const Real* Fy4,
    const Real* Fz0,const Real* Fz1,const Real* Fz2,const Real* Fz3,const Real* Fz4,
    Real idx,Real idy,Real idz,
    int sy,int sz,int nx,int ny,int nz,int h)
{
    const int ix=blockIdx.x*blockDim.x+threadIdx.x;
    const int iy=blockIdx.y*blockDim.y+threadIdx.y;
    const int iz=blockIdx.z*blockDim.z+threadIdx.z;
    if(ix>=nx||iy>=ny||iz>=nz) return;

    const int gi=(iz+h)*sz+(iy+h)*sy+(ix+h);
    // Fx: [nz][ny][nx+1] → left face of cell ix is at index ix
    const int fxi=iz*(ny*(nx+1))+iy*(nx+1)+ix;
    // Fy: [nz][ny+1][nx] → bottom face of cell iy is at index iy
    const int fyi=iz*((ny+1)*nx)+iy*nx+ix;
    // Fz: [nz+1][ny][nx] → bottom face of cell iz is at index iz
    const int fzi=iz*(ny*nx)+iy*nx+ix;

#define TEND(Ti,Fxi,Fyi,Fzi) Ti[gi]=-(Fxi[fxi+1]-Fxi[fxi])*idx \
                                     -(Fyi[fyi+nx]-Fyi[fyi])*idy \
                                     -(Fzi[fzi+ny*nx]-Fzi[fzi])*idz;
    TEND(T0,Fx0,Fy0,Fz0)
    TEND(T1,Fx1,Fy1,Fz1)
    TEND(T2,Fx2,Fy2,Fz2)
    TEND(T3,Fx3,Fy3,Fz3)
    TEND(T4,Fx4,Fy4,Fz4)
#undef TEND
}

// ─── Acoustic sub-step: momentum update ──────────────────────────────────────
// Forward: ru,rv,rw += dtt*(T_slow - ∇p' + buoyancy)
__global__ void k_acoustic_mom_3d(
    Real* ru,Real* rv,Real* rw,
    const Real* rho,const Real* rhoTh,
    const Real* Tru,const Real* Trv,const Real* Trw,
    const Real* p_b,const Real* rho_b,
    Real dtt,Real idx,Real idy,Real idz,
    int sy,int sz,int nx,int ny,int nz,int h)
{
    const int ix=blockIdx.x*blockDim.x+threadIdx.x;
    const int iy=blockIdx.y*blockDim.y+threadIdx.y;
    const int iz=blockIdx.z*blockDim.z+threadIdx.z;
    if(ix>=nx||iy>=ny||iz>=nz) return;
    const int gi=(iz+h)*sz+(iy+h)*sy+(ix+h);

    const Real pp=pressure3d(rhoTh[gi])-p_b[iz+h];
    // Neighbors for central difference gradient
    const Real ppxp=pressure3d(rhoTh[gi+1    ])-p_b[iz+h];
    const Real ppxm=pressure3d(rhoTh[gi-1    ])-p_b[iz+h];
    const Real ppyp=pressure3d(rhoTh[gi+sy   ])-p_b[iz+h];
    const Real ppym=pressure3d(rhoTh[gi-sy   ])-p_b[iz+h];
    const Real ppzp=pressure3d(rhoTh[gi+sz   ])-p_b[iz+h+1];
    const Real ppzm=pressure3d(rhoTh[gi-sz   ])-p_b[iz+h-1];
    (void)pp;

    const Real dpdx=0.5*(ppxp-ppxm)*idx;
    const Real dpdy=0.5*(ppyp-ppym)*idy;
    const Real dpdz=0.5*(ppzp-ppzm)*idz;
    const Real buoy=-(rho[gi]-rho_b[iz+h])*atm::g;

    ru[gi]+=dtt*(Tru[gi]-dpdx);
    rv[gi]+=dtt*(Trv[gi]-dpdy);
    rw[gi]+=dtt*(Trw[gi]-dpdz+buoy);
}

// ─── Acoustic sub-step: mass + energy update ─────────────────────────────────
// Backward: ρ, ρθ updated using NEW momentum (divergence)
__global__ void k_acoustic_mass_3d(
    Real* rho,Real* rhoTh,
    const Real* r0,const Real* rT0,
    const Real* ru,const Real* rv,const Real* rw,
    const Real* Tr,const Real* TrT,
    Real dtt,Real idx,Real idy,Real idz,
    int sy,int sz,int nx,int ny,int nz,int h)
{
    const int ix=blockIdx.x*blockDim.x+threadIdx.x;
    const int iy=blockIdx.y*blockDim.y+threadIdx.y;
    const int iz=blockIdx.z*blockDim.z+threadIdx.z;
    if(ix>=nx||iy>=ny||iz>=nz) return;
    const int gi=(iz+h)*sz+(iy+h)*sy+(ix+h);

    const Real div=0.5*(ru[gi+1 ]-ru[gi-1 ])*idx
                  +0.5*(rv[gi+sy]-rv[gi-sy])*idy
                  +0.5*(rw[gi+sz]-rw[gi-sz])*idz;

    rho  [gi]+=dtt*(Tr [gi]-div);
    const Real theta=rT0[gi]/max(r0[gi],1e-10);
    rhoTh[gi]+=dtt*(TrT[gi]-theta*div);
    if(rho[gi]<1e-10) rho[gi]=1e-10;
}

// ─── RK3 stage combine ────────────────────────────────────────────────────────
__global__ void k_rk_combine_3d(
    Real* d0,Real* d1,Real* d2,Real* d3,Real* d4,
    const Real* o0,const Real* o1,const Real* o2,const Real* o3,const Real* o4,
    const Real* n0,const Real* n1,const Real* n2,const Real* n3,const Real* n4,
    Real c0,Real c1,int total_sz)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=total_sz) return;
    d0[i]=c0*o0[i]+c1*n0[i]; d1[i]=c0*o1[i]+c1*n1[i];
    d2[i]=c0*o2[i]+c1*n2[i]; d3[i]=c0*o3[i]+c1*n3[i];
    d4[i]=c0*o4[i]+c1*n4[i];
}

// ─── Rayleigh sponge ─────────────────────────────────────────────────────────
__global__ void k_rayleigh_3d(
    Real* rho,Real* rhou,Real* rhov,Real* rhow,Real* rhoTh,
    const Real* rho_b,const Real* rhoTh_b,
    Real u_bar,Real z_bot,Real z_top,Real alpha_max,Real dt,Real dz,
    int sy,int sz,int nx,int ny,int nz,int h)
{
    const int ix=blockIdx.x*blockDim.x+threadIdx.x;
    const int iy=blockIdx.y*blockDim.y+threadIdx.y;
    const int iz=blockIdx.z*blockDim.z+threadIdx.z;
    if(ix>=nx||iy>=ny||iz>=nz) return;
    const Real z=(iz+0.5)*dz;
    if(z<=z_bot) return;
    const Real zn=(z-z_bot)/(z_top-z_bot);
    const Real sv=sin(0.5*M_PI*zn);
    const Real alpha=alpha_max*sv*sv;
    const Real damp=1./(1.+alpha*dt);
    const int gi=(iz+h)*sz+(iy+h)*sy+(ix+h);
    const Real rb=rho_b[iz+h], rTb=rhoTh_b[iz+h];
    const Real rub=rb*u_bar;
    rho  [gi]=rb +(rho  [gi]-rb )*damp;
    rhou [gi]=rub+(rhou [gi]-rub)*damp;
    rhov [gi]=    rhov  [gi]     *damp;
    rhow [gi]=    rhow  [gi]     *damp;
    rhoTh[gi]=rTb+(rhoTh[gi]-rTb)*damp;
}

// ─── Wave speed reduction ─────────────────────────────────────────────────────
__global__ void k_wave_speed_3d(
    const Real* rho,const Real* rhou,const Real* rhov,
    const Real* rhow,const Real* rhoTh,
    Real* smax_out,int sy,int sz,int nx,int ny,int nz,int h)
{
    extern __shared__ Real sdata[];
    const int ix=blockIdx.x*blockDim.x+threadIdx.x;
    const int iy=blockIdx.y*blockDim.y+threadIdx.y;
    const int iz=blockIdx.z*blockDim.z+threadIdx.z;
    Real s=0;
    if(ix<nx&&iy<ny&&iz<nz){
        const int gi=(iz+h)*sz+(iy+h)*sy+(ix+h);
        const Real r=max(rho[gi],1e-10);
        const Real a=sqrt(atm::gamma*pressure3d(rhoTh[gi])/r);
        const Real u=abs(rhou[gi])/r, v=abs(rhov[gi])/r, w=abs(rhow[gi])/r;
        s=max(u+a, max(v+a, w+a));
    }
    const int tid=threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.x*blockDim.y;
    sdata[tid]=s;
    __syncthreads();
    const int bsz=blockDim.x*blockDim.y*blockDim.z;
    for(int off=bsz/2;off>0;off>>=1){
        if(tid<off) sdata[tid]=max(sdata[tid],sdata[tid+off]);
        __syncthreads();
    }
    if(tid==0) atomicMax((int*)smax_out,(int)__float_as_int((float)sdata[0]));
}

// ─── Kernel: Smagorinsky + explicit diffusion (3D) ───────────────────────────
// Two modes: fixed K (K_smag_coef≤0) or Smagorinsky (K_smag_coef>0)
// 3D deformation: |S|² = 2(S11²+S22²+S33²+2S12²+2S13²+2S23²)
// Diffusion: T_ρφ += K * ∇²(ρφ),  φ ∈ {u,v,w,θ}
__global__ void k_diffusion_3d(
    Real* T1, Real* T2, Real* T3, Real* T4,
    const Real* rho, const Real* rhou, const Real* rhov,
    const Real* rhow, const Real* rhoTh,
    Real K_m_fixed, Real K_theta_fixed,
    Real K_smag_coef, Real Prt_inv,
    Real inv_dx, Real inv_dy, Real inv_dz,
    int sy, int sz, int nx, int ny, int nz, int h)
{
    const int ix = blockIdx.x*blockDim.x + threadIdx.x;
    const int iy = blockIdx.y*blockDim.y + threadIdx.y;
    const int iz = blockIdx.z*blockDim.z + threadIdx.z;
    if (ix >= nx || iy >= ny || iz >= nz) return;

    const int gi = (iz+h)*sz + (iy+h)*sy + (ix+h);

    Real K_m, K_theta;
    if (K_smag_coef > 0.0) {
        const Real u_e = rhou[gi+1 ] / max(rho[gi+1 ], Real(1e-10));
        const Real u_w = rhou[gi-1 ] / max(rho[gi-1 ], Real(1e-10));
        const Real u_n = rhou[gi+sy] / max(rho[gi+sy], Real(1e-10));
        const Real u_s = rhou[gi-sy] / max(rho[gi-sy], Real(1e-10));
        const Real u_t = rhou[gi+sz] / max(rho[gi+sz], Real(1e-10));
        const Real u_b = rhou[gi-sz] / max(rho[gi-sz], Real(1e-10));
        const Real v_e = rhov[gi+1 ] / max(rho[gi+1 ], Real(1e-10));
        const Real v_w = rhov[gi-1 ] / max(rho[gi-1 ], Real(1e-10));
        const Real v_n = rhov[gi+sy] / max(rho[gi+sy], Real(1e-10));
        const Real v_s = rhov[gi-sy] / max(rho[gi-sy], Real(1e-10));
        const Real v_t = rhov[gi+sz] / max(rho[gi+sz], Real(1e-10));
        const Real v_b = rhov[gi-sz] / max(rho[gi-sz], Real(1e-10));
        const Real w_e = rhow[gi+1 ] / max(rho[gi+1 ], Real(1e-10));
        const Real w_w = rhow[gi-1 ] / max(rho[gi-1 ], Real(1e-10));
        const Real w_n = rhow[gi+sy] / max(rho[gi+sy], Real(1e-10));
        const Real w_s = rhow[gi-sy] / max(rho[gi-sy], Real(1e-10));
        const Real w_t = rhow[gi+sz] / max(rho[gi+sz], Real(1e-10));
        const Real w_b = rhow[gi-sz] / max(rho[gi-sz], Real(1e-10));

        const Real S11 = Real(0.5)*(u_e-u_w)*inv_dx;
        const Real S22 = Real(0.5)*(v_n-v_s)*inv_dy;
        const Real S33 = Real(0.5)*(w_t-w_b)*inv_dz;
        const Real S12 = Real(0.25)*((u_n-u_s)*inv_dy + (v_e-v_w)*inv_dx);
        const Real S13 = Real(0.25)*((u_t-u_b)*inv_dz + (w_e-w_w)*inv_dx);
        const Real S23 = Real(0.25)*((v_t-v_b)*inv_dz + (w_n-w_s)*inv_dy);
        const Real Smag = sqrt(Real(2)*(S11*S11 + S22*S22 + S33*S33
                                      + Real(2)*(S12*S12 + S13*S13 + S23*S23)));
        K_m     = K_smag_coef * Smag;
        K_theta = K_m * Prt_inv;
    } else {
        K_m     = K_m_fixed;
        K_theta = K_theta_fixed;
    }

    const Real idx2 = inv_dx*inv_dx, idy2 = inv_dy*inv_dy, idz2 = inv_dz*inv_dz;
    auto lapl = [&](const Real* q) -> Real {
        return (q[gi-1 ] - Real(2)*q[gi] + q[gi+1 ]) * idx2
             + (q[gi-sy] - Real(2)*q[gi] + q[gi+sy]) * idy2
             + (q[gi-sz] - Real(2)*q[gi] + q[gi+sz]) * idz2;
    };
    T1[gi] += K_m     * lapl(rhou);
    T2[gi] += K_m     * lapl(rhov);
    T3[gi] += K_m     * lapl(rhow);
    T4[gi] += K_theta * lapl(rhoTh);
}

// ─── Kernel: Conservation diagnostics (3D) ───────────────────────────────────
// Per-block partial sums: mass, KE (u²+v²+w²), PE, rhoTheta
// d_out[4*nblocks]: [mass|KE|PE|rhoTh] interleaved by block
__global__ void k_diagnostics_3d(
    const Real* __restrict__ rho,
    const Real* __restrict__ rhou,
    const Real* __restrict__ rhov,
    const Real* __restrict__ rhow,
    const Real* __restrict__ rhoTh,
    Real* __restrict__ d_out,
    Real dV, Real dz,
    int sy, int sz, int nx, int ny, int nz, int h)
{
    extern __shared__ Real sdata[];
    const int ix  = blockIdx.x*blockDim.x + threadIdx.x;
    const int iy  = blockIdx.y*blockDim.y + threadIdx.y;
    const int iz  = blockIdx.z*blockDim.z + threadIdx.z;
    const int tid = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    const int bsz = blockDim.x*blockDim.y*blockDim.z;

    Real lm = 0, lke = 0, lpe = 0, lrT = 0;
    if (ix < nx && iy < ny && iz < nz) {
        const int gi = (iz+h)*sz + (iy+h)*sy + (ix+h);
        const Real r = rho[gi];
        const Real u = (r > Real(1e-10)) ? rhou[gi]/r : Real(0);
        const Real v = (r > Real(1e-10)) ? rhov[gi]/r : Real(0);
        const Real w = (r > Real(1e-10)) ? rhow[gi]/r : Real(0);
        const Real z = (iz + Real(0.5)) * dz;
        lm  = r;
        lke = Real(0.5)*r*(u*u + v*v + w*w);
        lpe = r * atm::g * z;
        lrT = rhoTh[gi];
    }

    sdata[tid]       = lm;
    sdata[tid+bsz]   = lke;
    sdata[tid+2*bsz] = lpe;
    sdata[tid+3*bsz] = lrT;
    __syncthreads();

    for (int s = bsz/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid]       += sdata[tid+s];
            sdata[tid+bsz]   += sdata[tid+bsz+s];
            sdata[tid+2*bsz] += sdata[tid+2*bsz+s];
            sdata[tid+3*bsz] += sdata[tid+3*bsz+s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const int bid = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y;
        d_out[4*bid+0] = sdata[0]       * dV;
        d_out[4*bid+1] = sdata[bsz]     * dV;
        d_out[4*bid+2] = sdata[2*bsz]   * dV;
        d_out[4*bid+3] = sdata[3*bsz]   * dV;
    }
}

// ─── Memory management ───────────────────────────────────────────────────────
void Euler3D_GPU::alloc_arrays()
{
    const size_t N  = g_.size()*sizeof(Real);
    const size_t Nx = (size_t)(g_.nx+1)*g_.ny*g_.nz*sizeof(Real);
    const size_t Ny = (size_t)g_.nx*(g_.ny+1)*g_.nz*sizeof(Real);
    const size_t Nz = (size_t)g_.nx*g_.ny*(g_.nz+1)*sizeof(Real);
    auto al=[](Real*& p,size_t sz){CK(cudaMalloc(&p,sz));CK(cudaMemset(p,0,sz));};
    for(int i=0;i<5;++i){
        al(d_q_[i],N); al(d_q1_[i],N); al(d_q2_[i],N); al(d_T_[i],N);
        al(d_Fx_[i],Nx); al(d_Fy_[i],Ny); al(d_Fz_[i],Nz);
    }
    const size_t Nb=(g_.nz+2*g_.halo)*sizeof(Real);
    al(d_rho_b_,Nb); al(d_p_b_,Nb); al(d_pi_b_,Nb); al(d_rhoTh_b_,Nb);
    CK(cudaMalloc(&d_smax_,sizeof(Real)));
    CK(cudaHostAlloc(&h_smax_,sizeof(Real),cudaHostAllocDefault));
    CK(cudaEventCreate(&ev_smax_ready_));
}

void Euler3D_GPU::free_arrays()
{
    auto fr=[](Real* p){if(p)cudaFree(p);};
    for(int i=0;i<5;++i){
        fr(d_q_[i]);fr(d_q1_[i]);fr(d_q2_[i]);fr(d_T_[i]);
        fr(d_Fx_[i]);fr(d_Fy_[i]);fr(d_Fz_[i]);
    }
    fr(d_rho_b_);fr(d_p_b_);fr(d_pi_b_);fr(d_rhoTh_b_);
    fr(d_smax_);if(h_smax_)cudaFreeHost(h_smax_);
    fr(d_diag_);
    cudaEventDestroy(ev_smax_ready_);
}

// ─── Constructor / Destructor ─────────────────────────────────────────────────
Euler3D_GPU::Euler3D_GPU(int nx,int ny,int nz,Real dx,Real dy,Real dz,Real cfl)
    : cfl_(cfl), t_(0.0)
{
    g_.nx=nx; g_.ny=ny; g_.nz=nz;
    g_.dx=dx; g_.dy=dy; g_.dz=dz;
    g_.halo=atm::HALO3;
    alloc_arrays();
}
Euler3D_GPU::~Euler3D_GPU(){ free_arrays(); }

// ─── set_sponge_layer ─────────────────────────────────────────────────────────
void Euler3D_GPU::set_sponge_layer(Real z_bot,Real z_top,Real tau,Real u_bar){
    sponge_z_bot_=z_bot; sponge_z_top_=z_top;
    sponge_alpha_=(tau>0)?1./tau:0.;
    sponge_u_bar_=u_bar;
}

// ─── set_diffusion / set_smagorinsky ──────────────────────────────────────────
void Euler3D_GPU::set_diffusion(Real K_m, Real K_theta) {
    K_m_        = K_m;
    K_theta_    = (K_theta < 0.0) ? K_m : K_theta;
    K_smag_coef_= 0.0;
}

void Euler3D_GPU::set_smagorinsky(Real Cs, Real Prt) {
    const Real Delta = std::cbrt(g_.dx * g_.dy * g_.dz);
    K_smag_coef_ = (Cs * Delta) * (Cs * Delta);
    Prt_         = Prt;
    K_m_         = 0.0;
    K_theta_     = 0.0;
}

// ─── get_diagnostics ──────────────────────────────────────────────────────────
std::array<Real,4> Euler3D_GPU::get_diagnostics() const
{
    const int nx=g_.nx, ny=g_.ny, nz=g_.nz, h=g_.halo;
    const int sy=g_.stride_y(), sz=g_.stride_z();
    dim3 thr(8,8,4);
    dim3 blk((nx+7)/8,(ny+7)/8,(nz+3)/4);
    const int nblocks = (int)(blk.x * blk.y * blk.z);
    const int bsz     = thr.x * thr.y * thr.z;

    if (!d_diag_) CK(cudaMalloc(&d_diag_, 4*nblocks*sizeof(Real)));
    CK(cudaMemset(d_diag_, 0, 4*nblocks*sizeof(Real)));

    const int smem = 4 * bsz * (int)sizeof(Real);
    const Real dV  = g_.dx * g_.dy * g_.dz;
    k_diagnostics_3d<<<blk,thr,smem>>>(
        d_q_[0],d_q_[1],d_q_[2],d_q_[3],d_q_[4],
        d_diag_, dV, g_.dz, sy, sz, nx, ny, nz, h);

    std::vector<Real> h_buf(4*nblocks, Real(0));
    CK(cudaMemcpy(h_buf.data(), d_diag_, 4*nblocks*sizeof(Real), cudaMemcpyDeviceToHost));

    std::array<Real,4> res = {Real(0),Real(0),Real(0),Real(0)};
    for (int b = 0; b < nblocks; ++b) {
        res[0] += h_buf[4*b+0];
        res[1] += h_buf[4*b+1];
        res[2] += h_buf[4*b+2];
        res[3] += h_buf[4*b+3];
    }
    return res;
}

// ─── State upload ─────────────────────────────────────────────────────────────
void Euler3D_GPU::set_state(const std::vector<Real>& rho,const std::vector<Real>& rhou,
                             const std::vector<Real>& rhov,const std::vector<Real>& rhow,
                             const std::vector<Real>& rhoTh)
{
    const int h=g_.halo, nx=g_.nx, ny=g_.ny, nz=g_.nz;
    const int sy=g_.stride_y(), sz=g_.stride_z();
    const size_t fsz=g_.size()*sizeof(Real);
    // Zero full arrays (sets halos to zero)
    for(int v=0;v<5;++v) CK(cudaMemset(d_q_[v],0,fsz));
    // Upload interior cells
    const Real* src[5]={rho.data(),rhou.data(),rhov.data(),rhow.data(),rhoTh.data()};
    std::vector<Real> full(g_.size(),0);
    for(int v=0;v<5;++v){
        std::fill(full.begin(),full.end(),0);
        for(int iz=0;iz<nz;++iz)
            for(int iy=0;iy<ny;++iy)
                for(int ix=0;ix<nx;++ix)
                    full[(iz+h)*sz+(iy+h)*sy+(ix+h)]=src[v][iz*ny*nx+iy*nx+ix];
        CK(cudaMemcpy(d_q_[v],full.data(),fsz,cudaMemcpyHostToDevice));
    }
}

void Euler3D_GPU::set_base_state(const std::vector<Real>& rho_b,
                                  const std::vector<Real>& pi_b)
{
    const int nz_full=g_.nz+2*g_.halo;
    std::vector<Real> p_b(nz_full), rT_b(nz_full);
    for(int iz=0;iz<nz_full;++iz){
        p_b [iz]=atm::p0*pow(pi_b[iz],atm::Cp/atm::Rd);
        rT_b[iz]=p_b[iz]/(atm::Rd*pi_b[iz]);
    }
    CK(cudaMemcpy(d_rho_b_,  rho_b.data(), nz_full*sizeof(Real), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_pi_b_,   pi_b.data(),  nz_full*sizeof(Real), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_p_b_,    p_b.data(),   nz_full*sizeof(Real), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_rhoTh_b_,rT_b.data(),  nz_full*sizeof(Real), cudaMemcpyHostToDevice));
}

// ─── State download ───────────────────────────────────────────────────────────
std::vector<Real> Euler3D_GPU::download_interior(const Real* d_ptr) const
{
    const int h=g_.halo, nx=g_.nx, ny=g_.ny, nz=g_.nz;
    const int sy=g_.stride_y(), sz=g_.stride_z();
    std::vector<Real> full(g_.size());
    CK(cudaMemcpy(full.data(),d_ptr,g_.size()*sizeof(Real),cudaMemcpyDeviceToHost));
    std::vector<Real> out(nx*ny*nz);
    for(int iz=0;iz<nz;++iz)
        for(int iy=0;iy<ny;++iy)
            for(int ix=0;ix<nx;++ix)
                out[iz*ny*nx+iy*nx+ix]=full[(iz+h)*sz+(iy+h)*sy+(ix+h)];
    return out;
}
std::vector<Real> Euler3D_GPU::get_rho()   const { return download_interior(d_q_[0]); }
std::vector<Real> Euler3D_GPU::get_rhou()  const { return download_interior(d_q_[1]); }
std::vector<Real> Euler3D_GPU::get_rhov()  const { return download_interior(d_q_[2]); }
std::vector<Real> Euler3D_GPU::get_rhow()  const { return download_interior(d_q_[3]); }
std::vector<Real> Euler3D_GPU::get_rhoTh() const { return download_interior(d_q_[4]); }

// ─── Launch helpers ───────────────────────────────────────────────────────────
void Euler3D_GPU::launch_fill_halos(Real** q)
{
    const int h=g_.halo, nx=g_.nx, ny=g_.ny, nz=g_.nz;
    const int sy=g_.stride_y(), sz=g_.stride_z();
    // X halos: loop over iy and iz, fill h ghost cells each side
    {
        dim3 thr(h,8,8);
        dim3 blk(1,(ny+2*h+7)/8,(nz+2*h+7)/8);
        k_fill_x_halos<<<blk,thr>>>(q[0],q[1],q[2],q[3],q[4],sy,sz,nx,ny,nz,h);
    }
    // Y halos: periodic
    {
        dim3 thr(8,h,8);
        dim3 blk((nx+2*h+7)/8,1,(nz+2*h+7)/8);
        k_fill_y_halos<<<blk,thr>>>(q[0],q[1],q[2],q[3],q[4],sy,sz,nx,ny,nz,h);
    }
    // Z halos: well-balanced wall
    {
        dim3 thr(8,8,h);
        dim3 blk((nx+2*h+7)/8,(ny+2*h+7)/8,1);
        k_fill_z_halos<<<blk,thr>>>(q[0],q[1],q[2],q[3],q[4],
                                    d_rho_b_,d_rhoTh_b_,sy,sz,nx,ny,nz,h);
    }
}

void Euler3D_GPU::launch_slow_tendencies(Real** q)
{
    launch_fill_halos(q);
    const int h=g_.halo, nx=g_.nx, ny=g_.ny, nz=g_.nz;
    const int sy=g_.stride_y(), sz=g_.stride_z();
    // X fluxes: threads (fx, iy, iz)
    {
        dim3 thr(32,4,4);
        dim3 blk((nx+1+31)/32,(ny+3)/4,(nz+3)/4);
        k_x_fluxes_3d<<<blk,thr>>>(q[0],q[1],q[2],q[3],q[4],
            d_Fx_[0],d_Fx_[1],d_Fx_[2],d_Fx_[3],d_Fx_[4],
            d_p_b_,d_rho_b_,d_rhoTh_b_,sy,sz,nx,ny,nz,h);
    }
    // Y fluxes: threads (ix, fy, iz)
    {
        dim3 thr(32,4,4);
        dim3 blk((nx+31)/32,(ny+1+3)/4,(nz+3)/4);
        k_y_fluxes_3d<<<blk,thr>>>(q[0],q[1],q[2],q[3],q[4],
            d_Fy_[0],d_Fy_[1],d_Fy_[2],d_Fy_[3],d_Fy_[4],
            d_p_b_,d_rho_b_,d_rhoTh_b_,sy,sz,nx,ny,nz,h);
    }
    // Z fluxes: threads (ix, iy, fz)
    {
        dim3 thr(32,4,4);
        dim3 blk((nx+31)/32,(ny+3)/4,(nz+1+3)/4);
        k_z_fluxes_3d<<<blk,thr>>>(q[0],q[1],q[2],q[3],q[4],
            d_Fz_[0],d_Fz_[1],d_Fz_[2],d_Fz_[3],d_Fz_[4],
            d_p_b_,d_rho_b_,d_rhoTh_b_,d_pi_b_,sy,sz,nx,ny,nz,h,g_.dz);
    }
    // Tendency accumulation
    {
        dim3 thr(8,8,4);
        dim3 blk((nx+7)/8,(ny+7)/8,(nz+3)/4);
        k_slow_tendencies_3d<<<blk,thr>>>(
            d_T_[0],d_T_[1],d_T_[2],d_T_[3],d_T_[4],
            d_Fx_[0],d_Fx_[1],d_Fx_[2],d_Fx_[3],d_Fx_[4],
            d_Fy_[0],d_Fy_[1],d_Fy_[2],d_Fy_[3],d_Fy_[4],
            d_Fz_[0],d_Fz_[1],d_Fz_[2],d_Fz_[3],d_Fz_[4],
            1./g_.dx,1./g_.dy,1./g_.dz,sy,sz,nx,ny,nz,h);
    }
    // Smagorinsky / fixed-K diffusion (adds to existing tendencies)
    if (K_m_ > 0.0 || K_smag_coef_ > 0.0) {
        dim3 thr(8,8,4);
        dim3 blk((nx+7)/8,(ny+7)/8,(nz+3)/4);
        k_diffusion_3d<<<blk,thr>>>(
            d_T_[1],d_T_[2],d_T_[3],d_T_[4],
            q[0],q[1],q[2],q[3],q[4],
            K_m_,K_theta_,K_smag_coef_,Real(1)/Prt_,
            Real(1)/g_.dx,Real(1)/g_.dy,Real(1)/g_.dz,
            sy,sz,nx,ny,nz,h);
    }
}

void Euler3D_GPU::launch_acoustic_step(Real** q,Real** q0,Real dtt)
{
    launch_fill_halos(q);
    const int h=g_.halo, nx=g_.nx, ny=g_.ny, nz=g_.nz;
    const int sy=g_.stride_y(), sz=g_.stride_z();
    dim3 thr(8,8,4); dim3 blk((nx+7)/8,(ny+7)/8,(nz+3)/4);
    k_acoustic_mom_3d<<<blk,thr>>>(q[1],q[2],q[3], q[0],q[4],
        d_T_[1],d_T_[2],d_T_[3],
        d_p_b_,d_rho_b_,dtt,1./g_.dx,1./g_.dy,1./g_.dz,
        sy,sz,nx,ny,nz,h);
    k_acoustic_mass_3d<<<blk,thr>>>(q[0],q[4], q0[0],q0[4], q[1],q[2],q[3],
        d_T_[0],d_T_[4],dtt,1./g_.dx,1./g_.dy,1./g_.dz,
        sy,sz,nx,ny,nz,h);
}

void Euler3D_GPU::launch_rk_combine(Real** dst,Real** qo,Real** qn,Real co,Real cn)
{
    dim3 thr(256); dim3 blk((g_.size()+255)/256);
    k_rk_combine_3d<<<blk,thr>>>(dst[0],dst[1],dst[2],dst[3],dst[4],
        qo[0],qo[1],qo[2],qo[3],qo[4],
        qn[0],qn[1],qn[2],qn[3],qn[4],co,cn,(int)g_.size());
}

void Euler3D_GPU::launch_rayleigh_damping(Real** q,Real dt)
{
    if(sponge_z_bot_<0||sponge_alpha_==0) return;
    const int h=g_.halo, nx=g_.nx, ny=g_.ny, nz=g_.nz;
    const int sy=g_.stride_y(), sz=g_.stride_z();
    dim3 thr(8,8,4); dim3 blk((nx+7)/8,(ny+7)/8,(nz+3)/4);
    k_rayleigh_3d<<<blk,thr>>>(q[0],q[1],q[2],q[3],q[4],
        d_rho_b_,d_rhoTh_b_,sponge_u_bar_,sponge_z_bot_,sponge_z_top_,
        sponge_alpha_,dt,g_.dz,sy,sz,nx,ny,nz,h);
}

// ─── Asynchronous CFL timestep ────────────────────────────────────────────────
void Euler3D_GPU::async_compute_dt(Real** q)
{
    const int h=g_.halo, nx=g_.nx, ny=g_.ny, nz=g_.nz;
    const int sy=g_.stride_y(), sz=g_.stride_z();
    CK(cudaMemset(d_smax_,0,sizeof(Real)));
    dim3 thr(8,8,4);
    dim3 blk((nx+7)/8,(ny+7)/8,(nz+3)/4);
    const int smem=thr.x*thr.y*thr.z*sizeof(Real);
    k_wave_speed_3d<<<blk,thr,smem>>>(q[0],q[1],q[2],q[3],q[4],
        d_smax_,sy,sz,nx,ny,nz,h);
    CK(cudaMemcpyAsync(h_smax_,d_smax_,sizeof(Real),cudaMemcpyDeviceToHost));
    CK(cudaEventRecord(ev_smax_ready_));
    smax_pending_=true;
}

void Euler3D_GPU::collect_dt()
{
    if(!smax_pending_) return;
    CK(cudaEventSynchronize(ev_smax_ready_));
    smax_pending_=false;
    const float smax=*reinterpret_cast<float*>(h_smax_);
    if(smax>0)
        dt_=Real(cfl_)*min(g_.dx, min(g_.dy,g_.dz))/Real(smax);
}

// ─── Main time step: SSP-RK3 + split-explicit acoustic sub-cycling ────────────
void Euler3D_GPU::step()
{
    if(step_n_%DT_UPDATE_FREQ==0){
        collect_dt();
        async_compute_dt(d_q_);
        if(dt_==0.0) collect_dt();
    }
    const Real dt=dt_;
    const Real dtt=dt/N_SPLIT;
    const size_t fsz=g_.size()*sizeof(Real);

    // Stage 1: q1 = q + dt*L(q)  [slow tendencies computed once, acoustic sub-cycled]
    launch_slow_tendencies(d_q_);
    for(int v=0;v<5;++v) CK(cudaMemcpy(d_q1_[v],d_q_[v],fsz,cudaMemcpyDeviceToDevice));
    for(int m=0;m<N_SPLIT;++m) launch_acoustic_step(d_q1_,d_q_,dtt);
    launch_rayleigh_damping(d_q1_,dt);

    // Stage 2: q2 = 3/4*q + 1/4*(q1 + dt*L(q1))
    launch_slow_tendencies(d_q1_);
    for(int v=0;v<5;++v) CK(cudaMemcpy(d_q2_[v],d_q1_[v],fsz,cudaMemcpyDeviceToDevice));
    for(int m=0;m<N_SPLIT;++m) launch_acoustic_step(d_q2_,d_q1_,dtt);
    launch_rk_combine(d_q2_,d_q_,d_q2_,0.75,0.25);
    launch_rayleigh_damping(d_q2_,dt);

    // Stage 3: q^(n+1) = 1/3*q^n + 2/3*(q2 + dt*L(q2))
    launch_slow_tendencies(d_q2_);
    // Save q^(2) into d_q1_ as fixed reference; d_q_ (= q^n) stays untouched
    for(int v=0;v<5;++v) CK(cudaMemcpy(d_q1_[v],d_q2_[v],fsz,cudaMemcpyDeviceToDevice));
    for(int m=0;m<N_SPLIT;++m) launch_acoustic_step(d_q2_,d_q1_,dtt);
    launch_rk_combine(d_q_,d_q_,d_q2_,1./3.,2./3.);
    launch_rayleigh_damping(d_q_,dt);

    t_+=dt; ++step_n_;
}
