#pragma once

#include "core/precision.hpp"

namespace wfe::phys {

inline constexpr real grav = (real)9.81;     // yercekimi ivmesi [m s-2]
inline constexpr real cp   = (real)1004.5;   // kuru hava sabit basinc isi kapasitesi [J kg-1 K-1]
inline constexpr real Rd   = (real)287.04;   // kuru hava gaz sabiti [J kg-1 K-1]
inline constexpr real cv   = cp - Rd;        // sabit hacim isi kapasitesi
inline constexpr real p00  = (real)1.0e5;    // referans basinc [Pa]

} // namespace wfe::phys
