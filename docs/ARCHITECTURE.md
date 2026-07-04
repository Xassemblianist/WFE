# Kod mimarisi

## Temel ilkeler

- **Tek bellek şeması:** Tüm alanlar (staggered dahil) aynı tampon boyutunu ve
  `GDims::idx(i,j,k)` indekslemesini paylaşır: i-en-hızlı (coalesced erişim),
  her yönde `ng=3` ghost, z'de w için +1 seviye. Staggered değişkenler sadece
  geçerli aralıklarıyla ayrışır (bkz. src/core/grid.hpp üstündeki yorum).
  Bu, kernel'lerde offset aritmetiğini tekilleştirir; bellek israfı ihmal edilir.
- **Hassasiyet politikası:** `real = float` (WFE_DOUBLE ile double). Turing'de FP64
  1/32 hızında; operasyonel NWP'de FP32 standarttır (ECMWF IFS). Kernel'lerde çıplak
  double sabiti yazılmaz.
- **Taban durumu ayrımı:** Hidrostatik dengedeki ρ̄, θ̄, π̄ profilleri 1B dizi olarak
  kernel'lere gider (`DevProf`); prognostik alanlar sapmadır. Bu FP32'de dinamik
  aralığı korur (tam basınç yerine ~10⁻⁴'lük π' taşınır).
- **Ghost doldurma sırası:** dikey → x periyodik (tüm j,k) → y periyodik (x ghost'ları
  dahil tüm i). Köşe tutarlılığı bu sıraya bağlıdır; değiştirme.

## Veri akışı (bir zaman adımı, split-explicit)

```
Integrator::step(dt)                          [src/dynamics/integrator.cpp]
  3 RK3 aşaması (m=0,1,2; süreler dt/3, dt/2, dt) için:
    est = (m==0 ? s_n : s_stage)              yavaş egilimlerin kaynağı
    apply_bcs(est)
    compute_mass_fluxes(est) -> mfx,mfy,mfz   ρ̄J ağırlıklı, kontravariant dikey
    compute_divergence -> div
    compute_tendencies(est, mf, div) -> tend  YAVAŞ: adveksiyon+difüzyon+Coriolis+Rayleigh
    s_work = kopya(s_n)                       hızlı sistem zaman-n'den başlar
    ns_m = {1, ns/2, ns} kez acoustic_substep(s_work, tend):
      k_acou_uv    u,v yatay explicit (π* diverjans sönümlemeli, arazi çapraz terimi)
      radyasyon/ghost BC'leri (u,v)
      k_acou_wpi   kolon başına thread: w-π' tridiagonal (Thomas), θ', yüzey w'si
      π' ghost'ları
    s_stage <-> s_work
  s_n <-> s_stage
```

Metrik terimler `DevMetric` (src/core/metric.hpp) ile taşınır; taban durumu
arazi-takip eden gridde 3B'dir (`DevProf`, src/core/base_state.hpp). Tendency
kernel'leri şimdilik "naif" (shared memory yok); kolon çözücü yerel dizilerle
(nz ≤ 319). Optimizasyon Faz 6'nın işi; önce doğruluk ve kapsam.

## Yeni prognostik değişken ekleme kontrol listesi

1. `State`'e Field3D alanı (src/dynamics/state.hpp) — alloc/copy_from/swap'a da ekle
2. `DevState`'e pointer (kernels.cu)
3. Tendency kernel'i + `compute_tendencies`'e launch
4. `update_state`'e çağrı
5. BC: periyodik listeye ekle + uygun dikey BC kerneli
6. `Writer::write`'a satır + meta.json vars listesi
7. Görselleştirme: tools/plot_slice.py otomatik çalışır (meta.json'dan okur)

## Sağlamlık altyapısı

- **Test süiti:** `python tools/run_tests.py` — 6 doğrulama vakası, sayısal
  kapılarla (~90 s). Her anlamlı değişiklikten sonra koşulmalı.
- **Çalışma-zamanı korumaları:** başlangıçta akustik CFL kontrolü (dt önerisiyle
  uyarı); her 10 adımda GPU max|w| bekçisi (NaN/patlamada acil çıktı + kod 3);
  opsiyonel `w_damping = on` (dikey Courant > 1'de WRF-tarzı yerel sönüm —
  gerçek veri koşularında açık).
- **Config doğrulama:** hiç okunmamış anahtarlar uyarı üretir (yazım hatası
  yakalar; prep'e ait `proj_*` hariç).
- **Provenans:** her koşu `out_dir/run_info.txt` yazar (sürüm, git hash,
  hassasiyet, GPU, tam config ekosu).
- **Hata ayıklama:** `WFE_SYNC=1` ortam değişkeni her kernel'den sonra
  senkronize eder — asenkron CUDA hataları tam yerinde yakalanır.
- **FP64:** `-DWFE_DOUBLE=ON` derlenir ve warm bubble FP32 ile 3 ondalık
  aynı sonucu verir (hassasiyet politikası doğrulaması).
- Ortak yardımcılar: `core/thermo.hpp` (qsat), `MAX_COLUMN_LEVELS` (grid.hpp).

## Bilinen bilinçli borçlar

- Difüzyon Laplasyeni arazi çapraz terimlerini ihmal eder (düz gridde tam;
  arazili gerçek durumlarda fizik kapanımı Faz 4'te bunu değiştirecek).
- Rayleigh katmanı ζ~z varsayar (katman düz tepeye yakın olduğundan iyi yaklaşım).
- Kolon çözücü yerel dizileri local memory'ye taşar; Faz 6'da shared memory /
  register blocking ile optimize edilecek.
- Açık sınırda skalar ghost'lar sıfır-gradyan (Faz 3'te GFS'ten zamana bağlı
  sınır beslemesi + relaxation zone gelince yenilenecek).
