# WFE — Weather Forecast Engine

Sıfırdan C++20/CUDA ile yazılan **bölgesel sayısal hava tahmin modeli**. Uzun vadeli hedef:
WRF'nin yerini alan, GPU-yerlisi, NOAA GFS verisiyle gerçek tarih-saatli operasyonel tahmin
üreten tam bir sistem.

## Mevcut durum (Faz 4 v1 tamamlandı — FİZİKLİ GERÇEK TAHMİN)

- **Fizik v1**: Louis yüzey katmanı, yerel-K PBL (implicit), levha toprak,
  basit radyasyon (günlük döngü) → alt seviye rüzgâr becerisi belirgin düzeldi


- **Gerçek veri tahmini**: `tools/prep_gfs.py` GFS'i NOMADS'tan indirir, Lambert
  gridine interpole eder; model GFS başlangıç + 3 saatlik Davies sınır beslemesiyle
  koşar. Türkiye 24h @ 12 km: **52 saniye** (1671× gerçek-zaman). Jet seviyesi
  rüzgârda ve sıcaklıkta persistansı yener (fizik paketleri henüz yok → Faz 4).
- **Nemli dinamik + Kessler mikrofiziği**: WK82 süperhücre testinde fırtına
  bölünmesi (w_max ~45 m/s, 457× gerçek-zaman)

```
# operasyonel tahmin: tek komut (en guncel GFS dongusunu kendisi bulur)
python tools\run_forecast.py cases\turkey.ini --hours 24
# -> out/turkey/map_*.png (kiyi cizgili urunler), wfe_out.nc, dogrulama raporu
```


- 3B, tam sıkıştırılabilir, **non-hidrostatik dinamik çekirdek** (Klemp–Wilhelmson denklem seti)
- Arakawa C-grid, 5. mertebe upwind adveksiyon, WS2002 RK3 + **split-explicit akustik
  alt-adımlama** (dikey implicit tridiagonal çözücü) → zaman adımı adveksiyon-CFL'li
- **Gal-Chen arazi-takip eden koordinat** (gerilebilir dikey grid), Coriolis, Rayleigh
  üst katmanı, açık/radyasyon yanal sınırlar, sabit-N stratifiye taban durumu
- Tamamı GPU'da (CUDA), FP32; RTX 2060'ta Schär testi gerçek zamandan ~690× hızlı
- Doğrulama süiti: sıcak kabarcık, Straka yoğunluk akıntısı, Schär dağ dalgası,
  arazide durağanlık, Galilean değişmezlik (sonuçlar: docs/EQUATIONS.md)

Ayrıntılar: [docs/EQUATIONS.md](docs/EQUATIONS.md) (denklem seti ve ayrıklaştırma),
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (kod mimarisi),
[docs/ROADMAP.md](docs/ROADMAP.md) (GFS ingest'e giden fazlar).

## Hızlı başlangıç

Build (Windows, VS Build Tools 2022 + CUDA ≥ 12): bkz. [CLAUDE.md](CLAUDE.md)

```
build\wfe.exe cases\warm_bubble.ini
python tools\plot_slice.py out\warm_bubble --var thp --step 4000
```

## Dizin yapısı

```
src/core/      hassasiyet, sabitler, grid, GPU alan tamponu, config, taban durumu
src/dynamics/  prognostik durum, CUDA kernel'leri, RK3 entegratör
src/physics/   (Faz 2+) mikrofizik, radyasyon, PBL, yüzey
src/io/        binary çıktı yazıcı (ileride: GRIB2 okuyucu, NetCDF)
cases/         test senaryosu config'leri
tools/         görselleştirme / doğrulama script'leri
docs/          bilimsel ve mimari dokümantasyon
```
