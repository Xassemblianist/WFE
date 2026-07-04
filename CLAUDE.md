# WFE — Weather Forecast Engine

Sıfırdan yazılan C++/CUDA bölgesel sayısal hava tahmin modeli (WRF muadili hedef).
Bilimsel tasarım: docs/EQUATIONS.md, yol haritası: docs/ROADMAP.md, kod mimarisi: docs/ARCHITECTURE.md.

## Build (Windows, VS Build Tools 2022 + CUDA 13.2)

CMake/Ninja sistem PATH'inde yok; Build Tools içindekiler kullanılır. PowerShell'den:

```
cmd /s /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -no_logo && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build'
```

Çift hassasiyet için: `-DWFE_DOUBLE=ON` (varsayılan FP32; RTX 2060'ta FP64 çok yavaştır, FP32 kalmalı).

## Çalıştırma

```
build\wfe.exe cases\warm_bubble.ini
```

Çıktılar config'deki `out_dir` altına ham float32 binary + `meta.json` olarak yazılır.
Görselleştirme: `python tools\plot_slice.py <out_dir> --var thp --step <N>` (numpy+matplotlib gerekir).

Operasyonel gerçek tahmin (tek komut): `python tools\run_forecast.py cases\turkey.ini --hours 24`
(gereken pip paketleri: numpy matplotlib eccodes cartopy netCDF4 xarray; Python
`%LOCALAPPDATA%\Programs\Python\Python312\python.exe`). Doğrulama: `tools\verify.py`.

## Test

Her anlamlı değişiklikten sonra: `python tools\run_tests.py` (6 doğrulama
vakası, sayısal kapılar, ~90 s; çıkış kodu 0 = PASS). CUDA hata ayıklama:
`WFE_SYNC=1` ile koş.

## Kod kuralları

- `real` tipi (`src/core/precision.hpp`) her yerde kullanılır; kernel içinde çıplak `double` sabiti yazma (FP32'de gizli dönüşüm maliyeti).
- Tüm alanlar tek boyutlandırma şemasını paylaşır: `GDims::idx(i,j,k)`, ghost genişliği `ng=3`, i-en-hızlı (coalesced). Staggered değişkenler aynı tampon boyutunu kullanır, sadece geçerli aralıkları farklıdır (bkz. docs/ARCHITECTURE.md).
- Her `cudaMalloc/Memcpy` `WFE_CUDA_CHECK` ile sarılır; kernel launch'lardan sonra `wfe::check_kernel()`.
- Fiziksel sabitler `src/core/constants.hpp` içinde (`phys::grav`, `phys::cp`, ...); kernel içine sihirli sayı gömme.
- Yeni prognostik değişken eklerken: `State`'e alan, BC kerneli, tendency kerneli, `update_state` ve `Writer` listesi birlikte güncellenir.
