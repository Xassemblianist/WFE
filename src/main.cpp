#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "core/base_state.hpp"
#include "core/config.hpp"
#include "core/cuda_check.hpp"
#include "core/grid.hpp"
#include "core/metric.hpp"
#include "dynamics/boundary.hpp"
#include "dynamics/integrator.hpp"
#include "dynamics/kernels.hpp"
#include "io/input.hpp"
#include "io/writer.hpp"
#include "physics/surface.hpp"

namespace wfe {
namespace {

// cos^2 profilli theta' kabarcigi (sicak: WK98; soguk: Straka yogunluk akintisi).
// bubble_dtheta = 0 ise atlanir. Yukseklik fiziksel z ile olculur.
void init_bubble(const GDims& g, const Config& cfg, const Metric& metric, State& s) {
  real dth = cfg.get_real("bubble_dtheta", (real)0);
  if (dth == (real)0) return;
  real xc = cfg.get_real("bubble_xc", g.nx * g.dx / 2);
  real yc = cfg.get_real("bubble_yc", g.ny * g.dy / 2);
  real zc = cfg.get_real("bubble_zc", (real)2000);
  real xr = cfg.get_real("bubble_xr", (real)2000);
  real yr = cfg.get_real("bubble_yr", (real)2000);
  real zr = cfg.get_real("bubble_zr", (real)2000);

  std::vector<real> h(g.npts(), 0);
  const real pi = (real)3.14159265358979323846;
  for (int k = 0; k < g.nz; ++k)
    for (int j = 0; j < g.ny; ++j)
      for (int i = 0; i < g.nx; ++i) {
        real x = ((real)i + (real)0.5) * g.dx;
        real y = ((real)j + (real)0.5) * g.dy;
        real z = metric.z_at(g, i, j, metric.h_zeta_c[k + g.ng]);
        real rx = (x - xc) / xr, ry = (y - yc) / yr, rz = (z - zc) / zr;
        real r = std::sqrt(rx * rx + ry * ry + rz * rz);
        if (r < (real)1) {
          real c = std::cos(pi * r / 2);
          h[g.idx(i, j, k)] = dth * c * c;
        }
      }
  s.thp.upload(h.data());
}

// Gercek veri baslangici: GFS'ten interpolasyonlanmis tam alanlar
// pertubasyonlara donusturulur (u,v merkez->yuz, th->th', pi->pi').
void init_from_input(const GDims& g, const BaseState& base, const InputData& in,
                     State& s) {
  auto src = [&](const std::vector<real>& a, int i, int j, int k) {
    return a[((size_t)k * g.ny + j) * g.nx + i];
  };
  std::vector<real> h(g.npts(), 0);
  for (int k = 0; k < g.nz; ++k)  // u yuzleri
    for (int j = 0; j < g.ny; ++j)
      for (int i = 0; i <= g.nx; ++i) {
        int il = i > 0 ? i - 1 : 0, ir = i < g.nx ? i : g.nx - 1;
        h[g.idx(i, j, k)] = (real)0.5 * (src(in.u, il, j, k) + src(in.u, ir, j, k));
      }
  s.u.upload(h.data());
  std::fill(h.begin(), h.end(), (real)0);
  for (int k = 0; k < g.nz; ++k)  // v yuzleri
    for (int j = 0; j < g.ny; ++j)
      for (int i = 0; i < g.nx; ++i) {
        int jl = j > 0 ? j - 1 : 0;
        h[g.idx(i, j, k)] = (real)0.5 * (src(in.v, i, jl, k) + src(in.v, i, j, k));
      }
  s.v.upload(h.data());
  std::fill(h.begin(), h.end(), (real)0);
  for (int k = 0; k < g.nz; ++k)
    for (int j = 0; j < g.ny; ++j)
      for (int i = 0; i < g.nx; ++i) {
        size_t c = g.idx(i, j, k);
        h[c] = src(in.th, i, j, k) - base.h_thb3[c];
      }
  s.thp.upload(h.data());
  std::fill(h.begin(), h.end(), (real)0);
  for (int k = 0; k < g.nz; ++k)
    for (int j = 0; j < g.ny; ++j)
      for (int i = 0; i < g.nx; ++i) {
        size_t c = g.idx(i, j, k);
        h[c] = src(in.pi, i, j, k) - base.h_pib3[c];
      }
  s.pip.upload(h.data());
  std::fill(h.begin(), h.end(), (real)0);
  for (int k = 0; k < g.nz; ++k)
    for (int j = 0; j < g.ny; ++j)
      for (int i = 0; i < g.nx; ++i) h[g.idx(i, j, k)] = src(in.qv, i, j, k);
  s.qv.upload(h.data());
}

// Baslangic nemi: qv = taban profili q̄v.
void init_moisture(const GDims& g, const BaseState& base, State& s) {
  if (!base.has_moisture()) return;
  std::vector<real> h(g.npts(), 0);
  for (int k = 0; k < g.nz; ++k)
    for (int j = 0; j < g.ny; ++j)
      for (int i = 0; i < g.nx; ++i) h[g.idx(i, j, k)] = base.qvb_at(g, i, j, k);
  s.qv.upload(h.data());
}

// Baslangic ruzgari = taban ruzgar profili (u yuzleri 0..nx dahil).
void init_wind(const GDims& g, const BaseState& base, State& s) {
  std::vector<real> h(g.npts(), 0);
  for (int k = 0; k < g.nz; ++k) {
    real ub = base.h_ub[k + g.ng];
    if (ub == (real)0) continue;
    for (int j = 0; j < g.ny; ++j)
      for (int i = 0; i <= g.nx; ++i) h[g.idx(i, j, k)] = ub;
  }
  s.u.upload(h.data());
}

struct Diag {
  real wmax, thmin, thmax, pipmax, qcmax, qrmax;
  bool finite;
};

Diag diagnose(const GDims& g, const State& s, std::vector<real>& buf, bool moist) {
  Diag d{0, 0, 0, 0, 0, 0, true};
  s.w.download(buf.data());
  for (int k = 0; k <= g.nz; ++k)
    for (int j = 0; j < g.ny; ++j)
      for (int i = 0; i < g.nx; ++i) {
        real v = buf[g.idx(i, j, k)];
        if (!std::isfinite(v)) d.finite = false;
        if (std::fabs(v) > d.wmax) d.wmax = std::fabs(v);
      }
  s.thp.download(buf.data());
  for (int k = 0; k < g.nz; ++k)
    for (int j = 0; j < g.ny; ++j)
      for (int i = 0; i < g.nx; ++i) {
        real v = buf[g.idx(i, j, k)];
        if (!std::isfinite(v)) d.finite = false;
        if (v < d.thmin) d.thmin = v;
        if (v > d.thmax) d.thmax = v;
      }
  s.pip.download(buf.data());
  for (int k = 0; k < g.nz; ++k)
    for (int j = 0; j < g.ny; ++j)
      for (int i = 0; i < g.nx; ++i) {
        real v = std::fabs(buf[g.idx(i, j, k)]);
        if (!std::isfinite(v)) d.finite = false;
        if (v > d.pipmax) d.pipmax = v;
      }
  if (moist) {
    s.qc.download(buf.data());
    for (size_t c = 0; c < buf.size(); ++c)
      if (buf[c] > d.qcmax) d.qcmax = buf[c];
    s.qr.download(buf.data());
    for (size_t c = 0; c < buf.size(); ++c) {
      if (!std::isfinite(buf[c])) d.finite = false;
      if (buf[c] > d.qrmax) d.qrmax = buf[c];
    }
  }
  return d;
}

} // namespace
} // namespace wfe

int main(int argc, char** argv) {
  using namespace wfe;

  std::string cfg_path = argc > 1 ? argv[1] : "cases/warm_bubble.ini";
  Config cfg;
  if (!cfg.load(cfg_path)) {
    std::fprintf(stderr, "config okunamadi: %s\n", cfg_path.c_str());
    return 1;
  }

  int nx = cfg.get_int("nx", 100);
  int ny = cfg.get_int("ny", 100);
  int nz = cfg.get_int("nz", 50);
  real dx = cfg.get_real("dx", 200);
  real dy = cfg.get_real("dy", 200);
  real dz = cfg.get_real("dz", 200);
  real dt = cfg.get_real("dt", (real)0.25);
  real t_end = cfg.get_real("t_end", 1000);
  real out_every = cfg.get_real("out_interval", 100);
  real diag_every = cfg.get_real("diag_interval", 10);
  std::string out_dir = cfg.get_str("out_dir", "out/warm_bubble");

  GDims g = make_grid(nx, ny, nz, 3, dx, dy, dz);

  int dev = 0;
  cudaDeviceProp prop{};
  WFE_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::printf("WFE | GPU: %s | grid %dx%dx%d (dx=%gm) | dt=%gs | t_end=%gs\n",
              prop.name, nx, ny, nz, (double)dx, (double)dt, (double)t_end);
  std::printf("bellek: ~%.1f MB (16 alan x %zu nokta, %s)\n",
              16.0 * g.npts() * sizeof(real) / 1e6, g.npts(),
              sizeof(real) == 4 ? "FP32" : "FP64");

  DynParams dp;
  dp.diff_K = cfg.get_real("diff_K", 0);
  dp.coriolis_f = cfg.get_real("coriolis_f", 0);
  dp.rayleigh_zd = cfg.get_real("rayleigh_zd", -1);
  dp.rayleigh_alpha = cfg.get_real("rayleigh_alpha", 0);
  dp.acoustic_ns = cfg.get_int("acoustic_ns", 6);
  dp.acoustic_beta = cfg.get_real("acoustic_beta", (real)0.2);
  dp.acoustic_smdiv = cfg.get_real("acoustic_smdiv", (real)0.1);
  dp.bc_x_open = cfg.get_str("bc_x", "periodic") == "open";
  dp.bc_y_open = cfg.get_str("bc_y", "periodic") == "open";
  dp.cstar = cfg.get_real("cstar", 30);
  dp.w_damping = cfg.get_str("w_damping", "off") == "on";
  dp.w_abort = cfg.get_real("w_abort", 150);

  // akustik CFL kontrolu: en hizli ses dalgasi yatay alt-adimi sinirlar
  {
    real cs = 350;  // tipik maksimum ses hizi [m/s]
    real dtau = dt / std::max(1, dp.acoustic_ns);
    real lim = (real)0.5 * std::min(dx, dy) / cs;
    if (dtau > lim)
      std::fprintf(stderr,
                   "UYARI: akustik alt-adim %.2fs > guvenli sinir %.2fs "
                   "(dt<=%.1fs veya acoustic_ns>=%d onerilir)\n",
                   (double)dtau, (double)lim, (double)(lim * dp.acoustic_ns),
                   (int)std::ceil(dt / lim));
  }

  bool file_mode = cfg.get_str("profile", "isentropic") == "file";
  InputData input;
  if (file_mode) {
    if (!input.load(g, cfg.get_str("input_dir", ""))) return 1;
    std::printf("girdi: %s (baslangic %s, %d sinir dosyasi @ %.0fs)\n",
                cfg.get_str("input_dir", "").c_str(), input.start.c_str(),
                input.n_bdy, (double)input.bdy_interval);
    dp.coriolis_use_ub = false;  // gercek veri: Coriolis tam ruzgara etkir
  }

  Metric metric;
  metric.build(g, cfg, file_mode ? &input.h : nullptr,
               file_mode ? &input.fcor : nullptr);

  BaseState base;
  ProfileTables tables{&input.prof_z, &input.prof_th, &input.prof_qv, &input.prof_u};
  base.build(g, cfg, metric, file_mode ? &tables : nullptr);
  dp.moisture = base.has_moisture();

  Integrator integ;
  integ.init(g, base.dev(), metric.dev(), dp);
  BdyManager bdy;
  SfcPBL phys;
  if (file_mode) {
    init_from_input(g, base, input, integ.state());
    bdy.init(g, dp, &input, base.h_thb3, base.h_pib3, cfg.get_int("bdy_width", 8),
             cfg.get_real("bdy_tau", 600));
    integ.set_boundary(&bdy);
    if (cfg.get_str("physics", "none") == "simple") {
      // start = YYYYMMDDHH: gun-of-year + UTC saat
      const std::string& st = input.start;
      int mo = std::atoi(st.substr(4, 2).c_str());
      int dy = std::atoi(st.substr(6, 2).c_str());
      int hh = std::atoi(st.substr(8, 2).c_str());
      static const int cum[12] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
      int doy = cum[mo - 1] + dy;
      phys.init(g, input, (real)hh, doy);
      integ.set_physics(&phys);
      std::printf("fizik: yuzey katmani + PBL + toprak + radyasyon (doy=%d, %02d UTC)\n",
                  doy, hh);
    }
  } else {
    init_bubble(g, cfg, metric, integ.state());
    init_wind(g, base, integ.state());
    init_moisture(g, base, integ.state());
  }

  Writer writer;
  bool phys_on = cfg.get_str("physics", "none") == "simple" && file_mode;
  writer.init(g, out_dir, dt, dp.moisture, phys_on);
  {  // hucre merkezi fiziksel yukseklikleri (gorselleştirme icin)
    std::vector<float> zc((size_t)g.nx * g.ny * g.nz);
    size_t o = 0;
    for (int k = 0; k < g.nz; ++k)
      for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
          zc[o++] = (float)metric.z_at(g, i, j, metric.h_zeta_c[k + g.ng]);
    writer.write_static("zc", zc);
  }
  writer.write(integ.state(), 0, 0);

  // provenans: surum, git hash, hassasiyet, config ekosu
  {
    std::string p = out_dir + "/run_info.txt";
    FILE* f = std::fopen(p.c_str(), "w");
    if (f) {
      std::fprintf(f, "wfe_version = 0.5.0\ngit = %s\nprecision = %s\ngpu = %s\n\n",
#ifdef WFE_GIT_HASH
                   WFE_GIT_HASH,
#else
                   "unknown",
#endif
                   sizeof(real) == 4 ? "fp32" : "fp64", prop.name);
      for (const auto& [k, v] : cfg.raw()) std::fprintf(f, "%s = %s\n", k.c_str(), v.c_str());
      std::fclose(f);
    }
  }
  for (const auto& k : cfg.unused())
    std::fprintf(stderr, "UYARI: config anahtari hic okunmadi (yazim hatasi?): %s\n",
                 k.c_str());

  std::vector<real> diag_buf(g.npts());
  int nsteps = (int)std::ceil(t_end / dt);
  int diag_steps = std::max(1, (int)std::round(diag_every / dt));
  int out_steps = std::max(1, (int)std::round(out_every / dt));

  auto t0 = std::chrono::steady_clock::now();
  for (int step = 1; step <= nsteps; ++step) {
    integ.step(dt, (step - 1) * dt);
    real t = step * dt;

    if (step % 10 == 0) {  // ucuz patlama/NaN bekcisi (GPU reduce)
      float wm = field_absmax(integ.state().w);
      if (!std::isfinite(wm) || wm > (float)dp.w_abort) {
        std::fprintf(stderr,
                     "HATA: |w|max=%.1f m/s esigi asti (adim %d, t=%.0fs) — "
                     "acil cikti yazilip durduruluyor\n",
                     wm, step, (double)t);
        writer.write(integ.state(), step, t);
        return 3;
      }
    }
    if (step % diag_steps == 0 || step == nsteps) {
      WFE_CUDA_CHECK(cudaDeviceSynchronize());
      Diag d = diagnose(g, integ.state(), diag_buf, dp.moisture);
      if (dp.moisture)
        std::printf("adim %6d  t=%7.1fs  |w|max=%7.3f  th'=[%+.2f,%+.2f]K  "
                    "qc=%.2fg/kg qr=%.2fg/kg\n",
                    step, (double)t, (double)d.wmax, (double)d.thmin, (double)d.thmax,
                    (double)d.qcmax * 1000, (double)d.qrmax * 1000);
      else
        std::printf("adim %6d  t=%7.1fs  |w|max=%7.3f  th'=[%+.3f,%+.3f]K  |pi'|max=%.2e\n",
                    step, (double)t, (double)d.wmax, (double)d.thmin, (double)d.thmax,
                    (double)d.pipmax);
      if (!d.finite) {
        std::fprintf(stderr, "HATA: NaN/Inf tespit edildi, simulasyon durduruldu.\n");
        return 2;
      }
    }
    if (step % out_steps == 0 || step == nsteps) {
      writer.write(integ.state(), step, t);
      if (dp.moisture) writer.write_field2d(integ.rain(), "rain", step);
      if (phys_on) writer.write_field2d(phys.tsk(), "tsk", step);
    }
  }
  WFE_CUDA_CHECK(cudaDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();
  double wall = std::chrono::duration<double>(t1 - t0).count();
  std::printf("bitti: %d adim, %.1f s duvar zamani, %.1f adim/s, gercek-zaman orani %.0fx\n",
              nsteps, wall, nsteps / wall, (double)t_end / wall);
  integ.print_profile();

  base.release();
  return 0;
}
