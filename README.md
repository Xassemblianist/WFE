# WFE — Weather Forecast Engine

Sıfırdan C++20/CUDA ile yazılan, **GPU-yerlisi bölgesel sayısal hava tahmin modeli**.
NOAA GFS verisiyle gerçek tarih-saatli operasyonel tahmin üretir; bağımsız istasyon
gözlemleriyle (METAR) doğrulanmış olarak birçok alanda persistansı yener. WRF muadili hedef.

RTX 2060'ta Türkiye 24h @ 12 km ≈ **1500× gerçek-zaman** (60 saniye).

```
# operasyonel tahmin: tek komut (en guncel GFS dongusunu kendisi bulur)
python tools\run_forecast.py cases\turkey.ini --hours 24
# -> haritalar (out/turkey/map_*.png) + NetCDF + GFS/METAR dogrulama raporu
```

## Yetenekler

**Dinamik çekirdek** — 3B tam sıkıştırılabilir non-hidrostatik (Klemp–Wilhelmson);
Arakawa C-grid, 5. mertebe upwind adveksiyon, WS2002 RK3 + split-explicit akustik
alt-adımlama (dikey implicit); Gal-Chen arazi-takip eden koordinat, harita faktörleri
(Lambert), Coriolis, Rayleigh üst katmanı, açık/radyasyon yanal sınırlar. Çekirdek
**yön-tarafsızlığı kanıtlı** (u–v ayna simetrisi 2×10⁻⁶).

**Fizik** — nonlocal (Troen-Mahrt) PBL + karşı-gradyan + PBL yüksekliği teşhisi;
Louis yüzey katmanı; çok katmanlı (Noah-benzeri) toprak; iki-akı broadband radyasyon;
Kessler warm-rain + karışık-faz **buz mikrofiziği**; toprak nemli evaporasyon;
pozitif-tanımlı nem adveksiyonu (Skamarock — negatif su/sahte yağış yok).

**Gerçek veri + operasyonel** — GFS GRIB2 ingest (ecCodes), Lambert projeksiyon +
rüzgâr rotasyonu, Davies sınır relaksasyonu, **iç bölge analiz-nudging** (GFS-güdümlü
LAM); 2m sıcaklık / 10m rüzgâr / PBL yüksekliği ürünleri; cartopy kıyı çizgili haritalar;
NetCDF çıktı; **topluluk (ensemble)** tahmini (belirsizlik yayılımı).

**Doğrulama** — GFS analizine (`verify.py`), gerçek istasyonlara (`verify_metar.py`,
yükseklik-düzeltmeli) ve çok-döngülü istatistiğe (`verify_multi.py`) karşı;
7 vakalı otomatik regresyon süiti (`run_tests.py`).

## Doğrulanmış sonuçlar (2026-07, GFS + METAR)

| Test | Sonuç |
|---|---|
| Gerçek tahmin 24h (GFS f024, iç nudging) | u **+%30**, qv **+%1** persistansı yener; çok-döngü robust |
| Gerçek tahmin 48h (GFS f048) | u **+%53**, θ +%14, v +%9, qv +%33 — stabil, uzun menzil değer katar |
| İstasyon gözlemi (METAR, 337 ist.) | 10m rüzgâr yanlılık ~0, RMSE 2.9 m/s; 2m T yanlılık −2.5°C (yüks.düz.) |
| WK82 süperhücre | fırtına bölünmesi, w_max ~45 m/s, çift yağış şeridi |
| Straka / Schär / arazide durağanlık | cephe 14.5 km / dağ dalgası deseni / |w|=0 makine kesinliği |
| Topluluk (6 üye) | yayılım deseni fiziksel: konvektif/dağlık belirsizlik yüksek |

Tam tablo ve referanslar: [docs/EQUATIONS.md](docs/EQUATIONS.md).

## Hızlı başlangıç

Build (Windows, VS Build Tools 2022 + CUDA ≥ 12): bkz. [CLAUDE.md](CLAUDE.md).

```
build\wfe.exe cases\warm_bubble.ini                 # idealize test
python tools\run_tests.py                           # 7 vakali regresyon suiti
python tools\run_forecast.py cases\turkey.ini --hours 24   # operasyonel gercek tahmin
python tools\run_ensemble.py cases\turkey.ini --members 6  # topluluk (belirsizlik)
```

Ayrıntılar: [docs/EQUATIONS.md](docs/EQUATIONS.md) (denklem seti, ayrıklaştırma, doğrulama),
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (kod mimarisi, sağlamlık altyapısı),
[docs/ROADMAP.md](docs/ROADMAP.md) (fazlar ve durum).

## Dizin yapısı

```
src/core/      hassasiyet, sabitler, grid, GPU alan tamponu, config, metrik, taban durumu, termo
src/dynamics/  prognostik durum, CUDA kernel'leri, RK3+akustik entegratör, sınır yöneticisi
src/physics/   Kessler+buz mikrofiziği, yüzey katmanı + PBL + toprak + radyasyon
src/io/        GFS girdi okuyucu, binary çıktı yazıcı
cases/         idealize testler + turkey.ini (operasyonel)
tools/         prep (GFS→Lambert), haritalar, NetCDF, doğrulama (GFS/METAR/çok-döngü), topluluk, test
docs/          bilimsel ve mimari dokümantasyon
```

## Sağlamlık

Her koşu provenans yazar (`run_info.txt`: sürüm, git hash, hassasiyet, config).
Çalışma-zamanı korumaları: akustik CFL uyarısı, GPU patlama/NaN bekçisi, WRF-tarzı
w-sönümleme. Test süiti sayısal kapıların yanı sıra çekirdek simetri + nem pozitifliği
+ arazide durağanlık garantilerini denetler. FP32 (varsayılan) ve FP64 derlenir.
