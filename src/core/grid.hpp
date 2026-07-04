#pragma once

#include <cstddef>

#include "core/precision.hpp"

#if defined(__CUDACC__)
#define WFE_HD __host__ __device__
#else
#define WFE_HD
#endif

namespace wfe {

// Arakawa C-grid, i-en-hizli bellek yerlesimi, her yonde ng ghost hucre.
// Tum alanlar (u, v, w, skalarlar) ayni tampon boyutunu paylasir; z'de +1
// seviye w icin ayrilmistir. Gecerli araliklar:
//   skalarlar (thp, pip): i in [0,nx), j in [0,ny), k in [0,nz)
//   u: i in [0,nx) (hucre i'nin sol yuzu), periyodiklikle u(nx)=u(0) ghost'ta
//   v: j in [0,ny) benzer sekilde
//   w: k in [0,nz] (hucre alt/ust yuzleri), w(0)=w(nz)=0 (rijit sinir)
struct GDims {
  int nx, ny, nz;   // ic hucre sayilari
  int ng;           // ghost genisligi (5. mertebe stencil icin 3)
  int NX, NY, NZ;   // ghost dahil tampon boyutlari
  real dx, dy, dz;  // grid araliklari [m]

  WFE_HD size_t idx(int i, int j, int k) const {
    return ((size_t)(k + ng) * NY + (j + ng)) * NX + (i + ng);
  }
  size_t npts() const { return (size_t)NX * NY * NZ; }
};

inline GDims make_grid(int nx, int ny, int nz, int ng, real dx, real dy, real dz) {
  return GDims{nx, ny, nz, ng, nx + 2 * ng, ny + 2 * ng, nz + 1 + 2 * ng, dx, dy, dz};
}

// Kolon cozuculerin (akustik, PBL, mikrofizik) yerel dizi siniri: nz+1 <= bu.
inline constexpr int MAX_COLUMN_LEVELS = 320;

} // namespace wfe
