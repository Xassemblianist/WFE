#pragma once

#include <string>
#include <vector>

#include "core/field3d.hpp"
#include "core/grid.hpp"
#include "dynamics/state.hpp"

namespace wfe {

// Anlik goruntuleri ham float32 binary olarak yazar (ghost'suz ic bolge,
// i-en-hizli). Boyutlar ve degisken listesi out_dir/meta.json icinde.
class Writer {
 public:
  void init(const GDims& g, const std::string& out_dir, real dt, bool moisture,
            bool has_tsk);
  void write(const State& s, int step, real t);
  // Genel 2D alan yazimi (yagis birikimi, yuzey sicakligi, ...): padded
  // device tampondan ghost'suz float32.
  void write_field2d(const Field3D& f, const char* name, int step);
  // Statik alan (or. hucre merkezi yukseklikleri "zc"): ghost'suz, host verisi.
  void write_static(const char* name, const std::vector<float>& data);

 private:
  void write_field(const real* dev, const char* name, int step, int nzlev);

  GDims g_{};
  std::string dir_;
  bool moist_ = false;
  std::vector<real> full_;
  std::vector<float> out_;
};

} // namespace wfe
