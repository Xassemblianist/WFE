# WFE — Weather Forecast Engine

Sıfırdan C++20/CUDA ile yazılan **bölgesel sayısal hava tahmin modeli**. Uzun vadeli hedef:
WRF'nin yerini alan, GPU-yerlisi, NOAA GFS verisiyle gerçek tarih-saatli operasyonel tahmin
üreten tam bir sistem.

## Mevcut durum (Faz 0 tamamlandı)

- 3B, tam sıkıştırılabilir, **non-hidrostatik dinamik çekirdek** (Klemp–Wilhelmson denklem seti)
- Arakawa C-grid, 5. mertebe upwind adveksiyon, Wicker–Skamarock RK3 zaman entegrasyonu
- Tamamı GPU'da (CUDA), FP32; RTX 2060'ta 100×100×50 grid gerçek zamandan ~107× hızlı
- Doğrulama: klasik yükselen sıcak kabarcık testi → literatürle uyumlu mantar termali

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
