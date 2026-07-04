#include "io/input.hpp"

#include <cstdio>

#include "core/config.hpp"

namespace wfe {

namespace {
bool read_f32(FILE* f, std::vector<real>& dst, size_t n) {
  dst.resize(n);
  std::vector<float> tmp(n);
  if (std::fread(tmp.data(), sizeof(float), n, f) != n) return false;
  for (size_t i = 0; i < n; ++i) dst[i] = (real)tmp[i];
  return true;
}
} // namespace

bool InputData::load(const GDims& g, const std::string& dir) {
  dir_ = dir;
  Config meta;
  if (!meta.load(dir + "/wfe_input.ini")) {
    std::fprintf(stderr, "girdi meta okunamadi: %s/wfe_input.ini\n", dir.c_str());
    return false;
  }
  if (meta.get_int("nx", -1) != g.nx || meta.get_int("ny", -1) != g.ny ||
      meta.get_int("nz", -1) != g.nz) {
    std::fprintf(stderr, "girdi grid boyutlari case ile uyusmuyor\n");
    return false;
  }
  int npf = meta.get_int("np_prof", 0);
  bdy_interval = meta.get_real("bdy_interval", 10800);
  n_bdy = meta.get_int("n_bdy", 0);
  start = meta.get_str("start", "");

  FILE* f = std::fopen((dir + "/wfe_init.bin").c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "wfe_init.bin acilamadi\n");
    return false;
  }
  size_t n2 = (size_t)g.nx * g.ny;
  size_t n3 = n2 * g.nz;
  bool ok = read_f32(f, prof_z, npf) && read_f32(f, prof_th, npf) &&
            read_f32(f, prof_qv, npf) && read_f32(f, prof_u, npf) &&
            read_f32(f, h, n2) && read_f32(f, fcor, n2) && read_f32(f, u, n3) &&
            read_f32(f, v, n3) && read_f32(f, th, n3) && read_f32(f, pi, n3) &&
            read_f32(f, qv, n3);
  std::fclose(f);
  if (!ok) std::fprintf(stderr, "wfe_init.bin eksik/bozuk\n");
  return ok;
}

bool InputData::load_bdy(const GDims& g, int idx, std::vector<real> fields[5]) const {
  char path[512];
  std::snprintf(path, sizeof(path), "%s/wfe_bdy_%03d.bin", dir_.c_str(),
                (int)(idx * bdy_interval / 3600));
  FILE* f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "sinir dosyasi acilamadi: %s\n", path);
    return false;
  }
  size_t n3 = (size_t)g.nx * g.ny * g.nz;
  bool ok = true;
  for (int v = 0; v < 5 && ok; ++v) ok = read_f32(f, fields[v], n3);
  std::fclose(f);
  if (!ok) std::fprintf(stderr, "sinir dosyasi eksik/bozuk: %s\n", path);
  return ok;
}

} // namespace wfe
