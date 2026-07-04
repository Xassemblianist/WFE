#pragma once

#include <cmath>

#include "core/grid.hpp"  // WFE_HD

namespace wfe::thermo {

// Tetens doygunluk karisim orani (p [Pa], T [K]). Host (double, taban durumu)
// ve device (real, mikrofizik/yuzey) tarafindan paylasilir.
template <typename T>
WFE_HD inline T qsat_tetens(T p, T temp) {
  T es = (T)610.78 * exp((T)17.269 * (temp - (T)273.16) / (temp - (T)35.86));
  T den = p - es;
  if (den < (T)1) den = (T)1;
  return (T)0.622 * es / den;
}

} // namespace wfe::thermo
