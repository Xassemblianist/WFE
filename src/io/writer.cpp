#include "io/writer.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <filesystem>

#include "core/cuda_check.hpp"

namespace wfe {

void Writer::init(const GDims& g, const std::string& out_dir, real dt, bool moisture,
                  bool has_tsk) {
  g_ = g;
  dir_ = out_dir;
  moist_ = moisture;
  std::filesystem::create_directories(dir_);
  full_.resize(g.npts());
  out_.resize((size_t)g.nx * g.ny * (g.nz + 1));

  std::string meta = dir_ + "/meta.json";
  FILE* f = std::fopen(meta.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "meta.json yazilamadi: %s\n", meta.c_str());
    std::exit(1);
  }
  std::fprintf(f,
               "{\n"
               "  \"nx\": %d, \"ny\": %d, \"nz\": %d,\n"
               "  \"dx\": %g, \"dy\": %g, \"dz\": %g,\n"
               "  \"dt\": %g,\n"
               "  \"vars\": {\"u\": %d, \"v\": %d, \"w\": %d, \"thp\": %d, \"pip\": %d",
               g.nx, g.ny, g.nz, (double)g.dx, (double)g.dy, (double)g.dz,
               (double)dt, g.nz, g.nz, g.nz + 1, g.nz, g.nz);
  if (moist_)
    std::fprintf(f, ", \"qv\": %d, \"qc\": %d, \"qr\": %d, \"rain\": 1", g.nz, g.nz,
                 g.nz);
  if (has_tsk) std::fprintf(f, ", \"tsk\": 1");
  std::fprintf(f, "}\n}\n");
  std::fclose(f);
}

void Writer::write(const State& s, int step, real t) {
  write_field(s.u.d, "u", step, g_.nz);
  write_field(s.v.d, "v", step, g_.nz);
  write_field(s.w.d, "w", step, g_.nz + 1);
  write_field(s.thp.d, "thp", step, g_.nz);
  write_field(s.pip.d, "pip", step, g_.nz);
  if (moist_) {
    write_field(s.qv.d, "qv", step, g_.nz);
    write_field(s.qc.d, "qc", step, g_.nz);
    write_field(s.qr.d, "qr", step, g_.nz);
  }
  std::printf("  cikti yazildi: step %d (t = %.1f s)\n", step, (double)t);
}

void Writer::write_field2d(const Field3D& f2, const char* name, int step) {
  std::vector<real> h2(f2.n);
  f2.download(h2.data());
  std::vector<float> o((size_t)g_.nx * g_.ny);
  size_t idx = 0;
  for (int j = 0; j < g_.ny; ++j)
    for (int i = 0; i < g_.nx; ++i)
      o[idx++] = (float)h2[(size_t)(j + g_.ng) * g_.NX + (i + g_.ng)];
  char path[512];
  std::snprintf(path, sizeof(path), "%s/%s_%06d.bin", dir_.c_str(), name, step);
  FILE* f = std::fopen(path, "wb");
  if (!f) {
    std::fprintf(stderr, "cikti dosyasi acilamadi: %s\n", path);
    std::exit(1);
  }
  std::fwrite(o.data(), sizeof(float), o.size(), f);
  std::fclose(f);
}

void Writer::write_static(const char* name, const std::vector<float>& data) {
  char path[512];
  std::snprintf(path, sizeof(path), "%s/%s.bin", dir_.c_str(), name);
  FILE* f = std::fopen(path, "wb");
  if (!f) {
    std::fprintf(stderr, "cikti dosyasi acilamadi: %s\n", path);
    std::exit(1);
  }
  std::fwrite(data.data(), sizeof(float), data.size(), f);
  std::fclose(f);
}

void Writer::write_field(const real* dev, const char* name, int step, int nzlev) {
  WFE_CUDA_CHECK(cudaMemcpy(full_.data(), dev, full_.size() * sizeof(real),
                            cudaMemcpyDeviceToHost));
  size_t o = 0;
  for (int k = 0; k < nzlev; ++k)
    for (int j = 0; j < g_.ny; ++j)
      for (int i = 0; i < g_.nx; ++i)
        out_[o++] = (float)full_[g_.idx(i, j, k)];

  char path[512];
  std::snprintf(path, sizeof(path), "%s/%s_%06d.bin", dir_.c_str(), name, step);
  FILE* f = std::fopen(path, "wb");
  if (!f) {
    std::fprintf(stderr, "cikti dosyasi acilamadi: %s\n", path);
    std::exit(1);
  }
  std::fwrite(out_.data(), sizeof(float), o, f);
  std::fclose(f);
}

} // namespace wfe
