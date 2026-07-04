#include "core/base_state.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#include "core/constants.hpp"
#include "core/cuda_check.hpp"

namespace wfe {

void BaseState::build(const GDims& g, const Config& cfg, const Metric& metric) {
  release();

  std::string prof = cfg.get_str("profile", "isentropic");
  real theta0 = cfg.get_real("theta0", 300);
  real bvN = cfg.get_real("bv_N", (real)0.01);
  real windu = cfg.get_real("wind_u", 0);
  bool constN = (prof == "constant_N");
  if (!constN && prof != "isentropic") {
    std::fprintf(stderr, "bilinmeyen profile: %s\n", prof.c_str());
    std::exit(1);
  }

  auto theta_of_z = [&](double z) -> double {
    if (constN) return theta0 * std::exp((double)bvN * bvN * z / phys::grav);
    return theta0;
  };
  auto dtheta_dz = [&](double z) -> double {
    if (constN) return theta_of_z(z) * (double)bvN * bvN / phys::grav;
    return 0;
  };

  // Hidrostatik pi(z) icin ince tablo (1 m adim, cift hassasiyet), sonra
  // grid noktalarina dogrusal interpolasyon.
  double zmin = -(double)(g.ng + 1) * g.dz * 4 - 100.0;
  double zmax = (double)metric.zt() + (double)(g.ng + 1) * g.dz * 4 + 100.0;
  const double dzt = 1.0;
  int ntab = (int)((zmax - zmin) / dzt) + 2;
  int i0 = (int)(-zmin / dzt);  // z=0 indeksi
  std::vector<double> pitab(ntab);
  pitab[i0] = 1.0;
  for (int it = i0 + 1; it < ntab; ++it) {
    double zm = (it - 1 - i0 + 0.5) * dzt;
    pitab[it] = pitab[it - 1] - phys::grav * dzt / ((double)phys::cp * theta_of_z(zm));
  }
  for (int it = i0 - 1; it >= 0; --it) {
    double zm = (it - i0 + 0.5) * dzt;
    pitab[it] = pitab[it + 1] + phys::grav * dzt / ((double)phys::cp * theta_of_z(zm));
  }
  auto pi_at = [&](double z) -> double {
    double s = (z - zmin) / dzt;
    int it = (int)s;
    if (it < 0) it = 0;
    if (it > ntab - 2) it = ntab - 2;
    double f = s - it;
    return pitab[it] * (1 - f) + pitab[it + 1] * f;
  };
  auto rho_of = [&](double pi, double th) -> double {
    return (double)phys::p00 * std::pow(pi, (double)phys::cv / phys::Rd) /
           ((double)phys::Rd * th);
  };

  size_t n = g.npts();
  std::vector<real> b_thb(n), b_pib(n), b_rhob(n), b_dth(n), b_thbw(n), b_rhobw(n);
  for (int k = -g.ng; k < g.nz + 1 + g.ng; ++k)
    for (int j = -g.ng; j < g.ny + g.ng; ++j)
      for (int i = -g.ng; i < g.nx + g.ng; ++i) {
        size_t c = g.idx(i, j, k);
        double zc = metric.z_at(g, i, j, metric.h_zeta_c[k + g.ng]);
        double zw = metric.z_at(g, i, j, metric.h_zeta_w[k + g.ng]);
        double pic = pi_at(zc), piw = pi_at(zw);
        b_thb[c] = (real)theta_of_z(zc);
        b_pib[c] = (real)pic;
        b_rhob[c] = (real)rho_of(pic, theta_of_z(zc));
        b_dth[c] = (real)dtheta_dz(zc);
        b_thbw[c] = (real)theta_of_z(zw);
        b_rhobw[c] = (real)rho_of(piw, theta_of_z(zw));
      }

  thb_.alloc(n);    thb_.upload(b_thb.data());
  pib_.alloc(n);    pib_.upload(b_pib.data());
  rhob_.alloc(n);   rhob_.upload(b_rhob.data());
  dthbdz_.alloc(n); dthbdz_.upload(b_dth.data());
  thbw_.alloc(n);   thbw_.upload(b_thbw.data());
  rhobw_.alloc(n);  rhobw_.upload(b_rhobw.data());

  h_ub.assign((size_t)g.NZ, windu);
  WFE_CUDA_CHECK(cudaMalloc(&d_ub_, g.NZ * sizeof(real)));
  WFE_CUDA_CHECK(cudaMemcpy(d_ub_, h_ub.data(), g.NZ * sizeof(real),
                            cudaMemcpyHostToDevice));
}

void BaseState::release() {
  thb_.release();
  pib_.release();
  rhob_.release();
  dthbdz_.release();
  thbw_.release();
  rhobw_.release();
  if (d_ub_) {
    cudaFree(d_ub_);
    d_ub_ = nullptr;
  }
}

DevProf BaseState::dev() const {
  return DevProf{thb_.d, pib_.d, rhob_.d, dthbdz_.d, thbw_.d, rhobw_.d, d_ub_};
}

} // namespace wfe
