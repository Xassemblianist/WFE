#pragma once

#include <vector>

#include "core/field3d.hpp"
#include "core/grid.hpp"
#include "dynamics/params.hpp"
#include "io/input.hpp"

namespace wfe {

// Davies sinir relaksasyonu: acik kenarlarda genisligi bdy_width hucre olan
// bolgede cozum, zamana dogrusal interpole edilmis GFS hedef alanlarina
// (u, v, theta', pi', qv) nudge edilir. Katsayi kenardan iceri cos^2 rampasi
// ile azalir; en dis hucrede 1/bdy_tau.
class BdyManager {
 public:
  void init(const GDims& g, const DynParams& dp, const InputData* input,
            const std::vector<real>& thb3, const std::vector<real>& pib3,
            int bdy_width, real bdy_tau, real nudge_tau);
  void update(real t);        // gerekirse sonraki sinir dosyasina gec
  real tfrac(real t) const;
  bool active() const { return input_ != nullptr; }

  Field3D lo_[5], hi_[5];  // hedefler: u, v, thp, pip, qv (padded tampon)
  Field3D wgt_;            // 2D relaksasyon katsayisi [1/s]
  real t_lo_ = 0, t_hi_ = 0;

 private:
  void load_into(int idx, Field3D* dst);

  GDims g_{};
  const InputData* input_ = nullptr;
  const std::vector<real>* thb3_ = nullptr;
  const std::vector<real>* pib3_ = nullptr;
  int idx_hi_ = 1;
  std::vector<real> hbuf_;
};

} // namespace wfe
