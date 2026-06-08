<h1 align="center">WFE — Weather Forecast Engine</h1>

<p align="center">
  <i>GPU-first sayısal hava tahmini motoru — C++20 ve CUDA ile sıfırdan.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/sürüm-v1.0-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/dil-C%2B%2B20%20%2B%20CUDA-76B900?style=flat-square" />
  <img src="https://img.shields.io/badge/platform-Windows%20%2B%20Linux-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/lisans-MIT-lightgrey?style=flat-square" />
</p>

---

## Bu nedir

WFE, modern NVIDIA GPU'lar için sıfırdan tasarlanmış, araştırma-sınıfı bir **non-hidrostatik atmosfer modeli**dir. ~1.5 milyon satırlık Fortran tabanlı WRF'e modern bir alternatif sunar.

**Temel fark:** WRF GPU'ya sonradan uyarlanırken, WFE doğrudan GPU-first olarak tasarlandı — veri yapıları, bellek düzeni ve çekirdek mimarisi CUDA için optimize edilmiştir.

## Özellikler

### Dinamik Çekirdek
- Tam sıkıştırılabilir non-hidrostatik Euler denklemleri (2D ve 3D)
- WENO5 (5. derece) uzaysal rekonstrüksiyon (Jiang-Shu 1996)
- HLLC Riemann çözücü (basınç-hız bağlaşımı)
- Split-explicit akustik alt-döngü (Forward-Backward, N_SPLIT=10)
- SSP-RK3 zaman entegrasyonu (Wicker-Skamarock 2002)
- Well-balanced pertürbasyon formülasyonu (ρ', p', θ')
- CFL-adaptif zaman adımı (asenkron GPU redüksiyon)

### Fizik Parametrizasyonları
- **Türbülans:** Smagorinsky kapatma şeması (Cs=0.18, Prt=1/3)
- **Sönümleme:** Rayleigh sponge katmanı (üst sınır yerçekimi dalgası absorpsiyonu)
- **Arazi:** Terrain-following eğim flux düzeltmesi

### Doğrulama Test Vakaları
| Test Vakası | Boyut | Durum |
|---|---|---|
| Dam Break (analitik çözüm) | 1D | ✅ Doğrulandı |
| Robert (1993) Density Current | 2D | ✅ Çalışıyor |
| Schär (2002) Mountain Wave | 2D | ✅ Çalışıyor |
| Bryan & Fritsch (2002) Warm Bubble | 3D | ✅ Çalışıyor |

### Teknik Altyapı
- SoA (Structure of Arrays) bellek düzeni — coalesced GPU erişimi
- Çapraz platform desteği (Windows MSVC + Linux GCC)
- CMake ve Makefile yapılandırma sistemi
- Korunumluluk tanılama (kütle, KE, PE — GPU paralel redüksiyon)

## Yol Haritası

| Faz | Kapsam | Durum |
|---|---|---|
| **1** | 1D Shallow Water — GPU üzerinde dam-break | ✅ Tamamlandı |
| **2** | 2D Non-Hidrostatik Euler — density current, dağ dalgası | ✅ Tamamlandı |
| **3** | 3D Dinamik çekirdek — sıcak kabarcık (warm bubble) | ✅ Tamamlandı (temel) |
| **4** | Operasyonel pipeline — GFS/ICON-EU ingest, Zarr çıktı, web viewer | 🔲 Planlandı |
| **5** | Çoklu GPU, ensemble forecasting | 🔲 Planlandı |

## Derleme

### Gereksinimler
- C++20 destekli derleyici (MSVC 19.30+, GCC 11+, Clang 14+)
- NVIDIA CUDA Toolkit 12.0+ (`nvcc`)
- CMake 3.20+

### Windows (CMake + Visual Studio)
```powershell
cmake -B build
cmake --build build --config Release
```

### Linux (Makefile)
```bash
make clean all
```

### Linux (CMake)
```bash
cmake -B build && cmake --build build
```

## Çalıştırma

### 1D Dam Break
```bash
./build/Release/wfe --gpu --nx 1000 --tend 0.2
```

### 2D Density Current (Robert 1993)
```bash
./build/Release/wfe2d --nx 256 --nz 64 --tend 900 --cfl 0.4
```

### 2D Mountain Wave (Schär 2002)
```bash
./build/Release/wfe_mw --nx 500 --nz 105 --tend 10000 --cfl 0.4
```

### 3D Warm Bubble (Bryan & Fritsch 2002)
```bash
./build/Release/wfe3d --nx 100 --ny 100 --nz 50 --tend 600 --cfl 0.4
```

## Proje Yapısı

```
WFE/
├── CMakeLists.txt              — CMake yapılandırması (4 hedef: wfe, wfe2d, wfe_mw, wfe3d)
├── Makefile                    — Linux Makefile (g++ + nvcc)
├── ARCHITECTURE.md             — Detaylı mimari referans dokümanı
│
├── src/
│   ├── types.hpp               — 1D tipler: Real=double, State, Flux
│   ├── types2d.hpp             — 2D tipler: atmosferik sabitler, EOS, Grid2D
│   ├── types3d.hpp             — 3D tipler: Grid3D, BaseState3D
│   ├── main.cpp                — 1D dam-break giriş noktası (CPU/GPU)
│   ├── main2d.cpp              — 2D density current giriş noktası
│   ├── main_mw.cpp             — 2D dağ dalgası giriş noktası
│   ├── main3d.cpp              — 3D sıcak kabarcık giriş noktası
│   │
│   ├── solver/
│   │   ├── swe1d.hpp/cpp       — CPU 1D SWE çözücü
│   │   └── cuda/
│   │       ├── swe1d_gpu.hpp/cu    — GPU 1D SWE çözücü
│   │       ├── euler2d_gpu.cuh/cu  — ⭐ GPU 2D Euler çözücü (~1150 satır)
│   │       └── euler3d_gpu.cuh/cu  — GPU 3D Euler çözücü (~1000 satır)
│   │
│   └── io/
│       ├── output.hpp/cpp      — CSV yazıcı (1D)
│
├── cases/
│   ├── dam_break.hpp           — 1D dam-break başlangıç koşulları
│   ├── density_current.hpp     — Robert (1993) 2D soğuk kabarcık
│   └── mountain_wave.hpp       — Schär (2002) dağ dalgası
│
└── scripts/
    └── plot_density_current.py — θ' kontur çizici (matplotlib)
```

## Referanslar

- Skamarock, W.C. & Klemp, J.B. (2008). *A time-split nonhydrostatic atmospheric model.* J. Comput. Phys.
- Wicker, L.J. & Skamarock, W.C. (2002). *Time-splitting methods for elastic models.* Mon. Wea. Rev.
- Klemp, J.B., Skamarock, W.C. & Dudhia, J. (2007). *Conservative split-explicit time integration.* Mon. Wea. Rev.
- Jiang, G.-S. & Shu, C.-W. (1996). *Efficient implementation of weighted ENO schemes.* J. Comput. Phys.
- Robert, A. (1993). *Bubble convection experiments with a semi-implicit formulation of the Euler equations.* J. Atmos. Sci.
- Schär, C. et al. (2002). *A new terrain-following vertical coordinate formulation.* Mon. Wea. Rev.
- Toro, E.F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics.* Springer.

## Lisans

MIT — bkz. [LICENSE](LICENSE).
