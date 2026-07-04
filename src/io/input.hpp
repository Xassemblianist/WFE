#pragma once

#include <string>
#include <vector>

#include "core/grid.hpp"
#include "core/precision.hpp"

namespace wfe {

// prep_gfs.py'nin urettigi gercek veri girdisi (wfe_input.ini + wfe_init.bin
// + wfe_bdy_FFF.bin). Tum 3B alanlar interior (nz*ny*nx, i-en-hizli), 2B
// alanlar (ny*nx), float32.
class InputData {
 public:
  bool load(const GDims& g, const std::string& dir);

  // Sinir dosyasi idx'i (0..n_bdy-1) 5 alani okur: u, v, th, pi, qv (interior).
  bool load_bdy(const GDims& g, int idx, std::vector<real> fields[5]) const;

  std::vector<real> prof_z, prof_th, prof_qv, prof_u;  // taban tablolari
  std::vector<real> h, fcor;                            // [ny*nx]
  std::vector<real> tsk, land, lat, lon;                // [ny*nx] yuzey alanlari (v2)
  std::vector<real> u, v, th, pi, qv;                   // [nz*ny*nx], t=0
  real bdy_interval = 10800;
  int n_bdy = 0;
  std::string start;  // YYYYMMDDHH

 private:
  std::string dir_;
};

} // namespace wfe
