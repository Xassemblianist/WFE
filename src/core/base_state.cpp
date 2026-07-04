#include "core/base_state.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

#include "core/constants.hpp"
#include "core/cuda_check.hpp"

namespace wfe {

namespace {
// Tetens doygunluk karisim orani (p [Pa], T [K]).
double qsat_of(double p, double T) {
  double es = 610.78 * std::exp(17.269 * (T - 273.16) / (T - 35.86));
  return 0.622 * es / (p - es);
}
} // namespace

void BaseState::build(const GDims& g, const Config& cfg, const Metric& metric) {
  release();

  std::string prof = cfg.get_str("profile", "isentropic");
  real theta0 = cfg.get_real("theta0", 300);
  real bvN = cfg.get_real("bv_N", (real)0.01);
  bool constN = (prof == "constant_N");
  bool wk82 = (prof == "wk82");
  has_moisture_ = wk82;
  if (!constN && !wk82 && prof != "isentropic") {
    std::fprintf(stderr, "bilinmeyen profile: %s\n", prof.c_str());
    std::exit(1);
  }

  // WK82 sounding parametreleri
  const double z_tr = 12000.0, th_tr = 343.0, T_tr = 213.0, qv_max = 0.014;

  auto theta_of_z = [&](double z) -> double {
    if (wk82) {
      if (z <= z_tr) return 300.0 + (th_tr - 300.0) * std::pow(z / z_tr, 1.25);
      return th_tr * std::exp((double)phys::grav * (z - z_tr) / ((double)phys::cp * T_tr));
    }
    if (constN) return theta0 * std::exp((double)bvN * bvN * z / phys::grav);
    return theta0;
  };
  auto relhum_of_z = [&](double z) -> double {
    if (z <= z_tr) return 1.0 - 0.75 * std::pow(z / z_tr, 1.25);
    return 0.25;
  };

  // Ince z tablosu (1 m): once kuru theta, sonra nemliyse qv-thv-pi iterasyonu.
  double zmin = -(double)(g.ng + 1) * g.dz * 4 - 100.0;
  double zmax = (double)metric.zt() + (double)(g.ng + 1) * g.dz * 4 + 100.0;
  const double dzt = 1.0;
  int ntab = (int)((zmax - zmin) / dzt) + 2;
  int i0 = (int)(-zmin / dzt);
  std::vector<double> pitab(ntab), qvtab(ntab, 0.0), thvtab(ntab);

  for (int it = 0; it < ntab; ++it)
    thvtab[it] = theta_of_z(zmin + it * dzt);  // ilk tahmin: kuru

  auto integrate_pi = [&]() {
    pitab[i0] = 1.0;
    for (int it = i0 + 1; it < ntab; ++it) {
      double thv = 0.5 * (thvtab[it - 1] + thvtab[it]);
      pitab[it] = pitab[it - 1] - phys::grav * dzt / ((double)phys::cp * thv);
    }
    for (int it = i0 - 1; it >= 0; --it) {
      double thv = 0.5 * (thvtab[it] + thvtab[it + 1]);
      pitab[it] = pitab[it + 1] + phys::grav * dzt / ((double)phys::cp * thv);
    }
  };
  integrate_pi();

  if (wk82) {
    for (int iter = 0; iter < 5; ++iter) {
      for (int it = 0; it < ntab; ++it) {
        double z = zmin + it * dzt;
        double th = theta_of_z(z);
        double pi = pitab[it];
        double T = th * pi;
        double p = (double)phys::p00 * std::pow(pi, (double)phys::cp / phys::Rd);
        double qv = std::min(relhum_of_z(z) * qsat_of(p, T), qv_max);
        if (z < 0) qv = qvtab[it > 0 ? it : 0];  // yuzey alti ghost: sabit uzat
        qvtab[it] = qv;
        thvtab[it] = th * (1.0 + (double)phys::eps61 * qv);
      }
      integrate_pi();
    }
  }

  auto interp = [&](const std::vector<double>& tab, double z) -> double {
    double s = (z - zmin) / dzt;
    int it = (int)s;
    if (it < 0) it = 0;
    if (it > ntab - 2) it = ntab - 2;
    double f = s - it;
    return tab[it] * (1 - f) + tab[it + 1] * f;
  };
  auto rho_of = [&](double pi, double thv) -> double {
    return (double)phys::p00 * std::pow(pi, (double)phys::cv / phys::Rd) /
           ((double)phys::Rd * thv);
  };

  size_t n = g.npts();
  std::vector<real> b_thb(n), b_pib(n), b_rhob(n), b_dth(n), b_thbw(n), b_rhobw(n),
      b_thvb(n), b_thvbw(n);
  h_qvb_.assign(n, 0);
  for (int k = -g.ng; k < g.nz + 1 + g.ng; ++k)
    for (int j = -g.ng; j < g.ny + g.ng; ++j)
      for (int i = -g.ng; i < g.nx + g.ng; ++i) {
        size_t c = g.idx(i, j, k);
        double zc = metric.z_at(g, i, j, metric.h_zeta_c[k + g.ng]);
        double zw = metric.z_at(g, i, j, metric.h_zeta_w[k + g.ng]);
        double thc = theta_of_z(zc), thw = theta_of_z(zw);
        double qvc = wk82 ? interp(qvtab, zc) : 0.0;
        double qvw = wk82 ? interp(qvtab, zw) : 0.0;
        double thvc = thc * (1.0 + (double)phys::eps61 * qvc);
        double thvw = thw * (1.0 + (double)phys::eps61 * qvw);
        double pic = interp(pitab, zc), piw = interp(pitab, zw);
        b_thb[c] = (real)thc;
        b_pib[c] = (real)pic;
        b_rhob[c] = (real)rho_of(pic, thvc);
        b_dth[c] = (real)((theta_of_z(zc + 1.0) - theta_of_z(zc - 1.0)) / 2.0);
        b_thbw[c] = (real)thw;
        b_rhobw[c] = (real)rho_of(piw, thvw);
        b_thvb[c] = (real)thvc;
        b_thvbw[c] = (real)thvw;
        h_qvb_[c] = (real)qvc;
      }

  thb_.alloc(n);    thb_.upload(b_thb.data());
  pib_.alloc(n);    pib_.upload(b_pib.data());
  rhob_.alloc(n);   rhob_.upload(b_rhob.data());
  dthbdz_.alloc(n); dthbdz_.upload(b_dth.data());
  thbw_.alloc(n);   thbw_.upload(b_thbw.data());
  rhobw_.alloc(n);  rhobw_.upload(b_rhobw.data());
  thvb_.alloc(n);   thvb_.upload(b_thvb.data());
  thvbw_.alloc(n);  thvbw_.upload(b_thvbw.data());
  qvb_.alloc(n);    qvb_.upload(h_qvb_.data());
  if (!wk82) h_qvb_.clear();

  // Taban ruzgari: sabit veya tanh kesme profili (WK82)
  std::string wprof = cfg.get_str("wind_profile", "const");
  real windu = cfg.get_real("wind_u", 0);
  real us = cfg.get_real("wind_us", 25);
  real zs = cfg.get_real("wind_zs", 3000);
  real uoff = cfg.get_real("wind_offset", 0);
  h_ub.assign((size_t)g.NZ, windu);
  if (wprof == "tanh") {
    for (int kk = 0; kk < g.NZ; ++kk) {
      double z = metric.h_zeta_c[kk];  // duz zeminli firtina testlerinde zeta ~ z
      if (z < 0) z = 0;
      h_ub[kk] = (real)(us * std::tanh(z / zs) + uoff);
    }
  }
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
  thvb_.release();
  thvbw_.release();
  qvb_.release();
  if (d_ub_) {
    cudaFree(d_ub_);
    d_ub_ = nullptr;
  }
}

DevProf BaseState::dev() const {
  return DevProf{thb_.d,  pib_.d,   rhob_.d, dthbdz_.d, thbw_.d,
                 rhobw_.d, thvb_.d, thvbw_.d, qvb_.d,    d_ub_};
}

} // namespace wfe
