#include "core/metric.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#include "core/cuda_check.hpp"

namespace wfe {

void Metric::build(const GDims& g, const Config& cfg) {
  release();
  n1_ = (size_t)g.NZ;
  n2_ = (size_t)g.NX * g.NY;

  // --- dikey zeta seviyeleri (gerilmis veya duzgun) ---
  h_zeta_w.assign(n1_, 0);
  h_zeta_c.assign(n1_, 0);
  std::vector<real> dzc(n1_, 0), dzw(n1_, 0);

  std::string stretch = cfg.get_str("stretch", "none");
  std::vector<double> zw(g.nz + 1, 0.0);
  if (stretch == "geometric") {
    double dz0 = cfg.get_real("dz0", g.dz);
    double ratio = cfg.get_real("dz_ratio", (real)1.05);
    double dzmax = cfg.get_real("dz_max", g.dz * 4);
    double d = dz0;
    for (int k = 1; k <= g.nz; ++k) {
      zw[k] = zw[k - 1] + d;
      d = std::min(d * ratio, dzmax);
    }
  } else {
    for (int k = 0; k <= g.nz; ++k) zw[k] = (double)k * g.dz;
  }
  zt_ = (real)zw[g.nz];

  // ghost'lar: sabit uzatma (dz sinirdaki degerle)
  auto zw_at = [&](int k) -> double {
    if (k < 0) return zw[0] + (double)k * (zw[1] - zw[0]);
    if (k > g.nz) return zw[g.nz] + (double)(k - g.nz) * (zw[g.nz] - zw[g.nz - 1]);
    return zw[k];
  };
  for (int kk = 0; kk < (int)n1_; ++kk) {
    int k = kk - g.ng;
    h_zeta_w[kk] = (real)zw_at(k);
    h_zeta_c[kk] = (real)(0.5 * (zw_at(k) + zw_at(k + 1)));
  }
  for (int kk = 0; kk < (int)n1_ - 1; ++kk)
    dzc[kk] = h_zeta_w[kk + 1] - h_zeta_w[kk];
  dzc[n1_ - 1] = dzc[n1_ - 2];
  for (int kk = 1; kk < (int)n1_; ++kk)
    dzw[kk] = h_zeta_c[kk] - h_zeta_c[kk - 1];
  dzw[0] = dzw[1];

  // --- arazi (padded 2D, ghost'lar dahil analitik) ---
  h_h.assign(n2_, 0);
  h_jac.assign(n2_, 0);
  std::vector<real> hx(n2_, 0), hy(n2_, 0), hxu(n2_, 0), hyv(n2_, 0);

  std::string terrain = cfg.get_str("terrain", "none");
  real h0 = cfg.get_real("mtn_h0", 250);
  real xa = cfg.get_real("mtn_a", 5000);
  real lam = cfg.get_real("mtn_lambda", 4000);
  real xc = cfg.get_real("mtn_xc", g.nx * g.dx / 2);
  real yc = cfg.get_real("mtn_yc", g.ny * g.dy / 2);
  real ya = cfg.get_real("mtn_ya", xa);
  const real pi = (real)3.14159265358979323846;

  auto h_of = [&](double x, double y) -> double {
    if (terrain == "schaer") {
      double xr = x - xc;
      double env = std::exp(-(xr / xa) * (xr / xa));
      double c = std::cos(pi * xr / lam);
      return h0 * env * c * c;
    }
    if (terrain == "gauss") {
      double xr = (x - xc) / xa, yr = (y - yc) / ya;
      return h0 * std::exp(-xr * xr - yr * yr);
    }
    return 0.0;
  };

  for (int j = -g.ng; j < g.ny + g.ng; ++j)
    for (int i = -g.ng; i < g.nx + g.ng; ++i) {
      double x = ((double)i + 0.5) * g.dx;
      double y = ((double)j + 0.5) * g.dy;
      size_t c = idx2(g, i, j);
      h_h[c] = (real)h_of(x, y);
      h_jac[c] = (zt_ - h_h[c]) / zt_;
      hx[c] = (real)((h_of(x + g.dx, y) - h_of(x - g.dx, y)) / (2.0 * g.dx));
      hy[c] = (real)((h_of(x, y + g.dy) - h_of(x, y - g.dy)) / (2.0 * g.dy));
      hxu[c] = (real)((h_of(x, y) - h_of(x - g.dx, y)) / g.dx);   // u noktasi (i-1/2)
      hyv[c] = (real)((h_of(x, y) - h_of(x, y - g.dy)) / g.dy);   // v noktasi (j-1/2)
    }

  // tek tampon: 4 dikey [n1] + 6 yatay [n2]
  size_t total = 4 * n1_ + 6 * n2_;
  WFE_CUDA_CHECK(cudaMalloc(&d_all_, total * sizeof(real)));
  size_t off = 0;
  auto up = [&](const std::vector<real>& v) {
    WFE_CUDA_CHECK(cudaMemcpy(d_all_ + off, v.data(), v.size() * sizeof(real),
                              cudaMemcpyHostToDevice));
    off += v.size();
  };
  up(h_zeta_w);
  up(h_zeta_c);
  up(dzc);
  up(dzw);
  up(h_h);
  up(hx);
  up(hy);
  up(hxu);
  up(hyv);
  up(h_jac);
}

void Metric::release() {
  if (d_all_) {
    cudaFree(d_all_);
    d_all_ = nullptr;
  }
}

DevMetric Metric::dev() const {
  DevMetric m{};
  size_t off = 0;
  m.zeta_w = d_all_ + off; off += n1_;
  m.zeta_c = d_all_ + off; off += n1_;
  m.dzeta_c = d_all_ + off; off += n1_;
  m.dzeta_w = d_all_ + off; off += n1_;
  m.h = d_all_ + off; off += n2_;
  m.hx = d_all_ + off; off += n2_;
  m.hy = d_all_ + off; off += n2_;
  m.hx_u = d_all_ + off; off += n2_;
  m.hy_v = d_all_ + off; off += n2_;
  m.jac = d_all_ + off; off += n2_;
  m.zt = zt_;
  return m;
}

} // namespace wfe
