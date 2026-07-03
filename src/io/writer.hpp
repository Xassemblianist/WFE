#pragma once

#include <string>
#include <vector>

#include "core/grid.hpp"
#include "dynamics/state.hpp"

namespace wfe {

// Anlik goruntuleri ham float32 binary olarak yazar (ghost'suz ic bolge,
// i-en-hizli). Boyutlar ve degisken listesi out_dir/meta.json icinde.
class Writer {
 public:
  void init(const GDims& g, const std::string& out_dir, real dt);
  void write(const State& s, int step, real t);

 private:
  void write_field(const real* dev, const char* name, int step, int nzlev);

  GDims g_{};
  std::string dir_;
  std::vector<real> full_;
  std::vector<float> out_;
};

} // namespace wfe
