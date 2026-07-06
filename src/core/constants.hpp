#pragma once

#include "core/precision.hpp"

namespace wfe::phys {

inline constexpr real grav = (real)9.81;     // yercekimi ivmesi [m s-2]
inline constexpr real cp   = (real)1004.5;   // kuru hava sabit basinc isi kapasitesi [J kg-1 K-1]
inline constexpr real Rd   = (real)287.04;   // kuru hava gaz sabiti [J kg-1 K-1]
inline constexpr real cv   = cp - Rd;        // sabit hacim isi kapasitesi
inline constexpr real p00  = (real)1.0e5;    // referans basinc [Pa]
inline constexpr real Lv   = (real)2.5e6;    // buharlasma gizli isisi [J kg-1]
inline constexpr real Lf   = (real)3.34e5;   // erime gizli isisi [J kg-1]
inline constexpr real Ls   = Lv + Lf;        // sublimasyon gizli isisi [J kg-1]
inline constexpr real Rv   = (real)461.5;    // su buhari gaz sabiti [J kg-1 K-1]
inline constexpr real eps61 = (real)0.61;    // sanal sicaklik katsayisi (Rv/Rd - 1)
inline constexpr real Tfrz = (real)273.15;   // donma noktasi [K]

} // namespace wfe::phys
