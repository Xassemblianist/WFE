<h1 align="center">WFE — Weather Forecast Engine</h1>

<p align="center">
  <i>Modern sayısal hava tahmini — sıfırdan, C++20 ve CUDA ile.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/durum-tasar%C4%B1m%20a%C5%9Fmas%C4%B1-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/dil-C%2B%2B20%20%2B%20CUDA-76B900?style=flat-square" />
  <img src="https://img.shields.io/badge/lisans-MIT-lightgrey?style=flat-square" />
</p>

---

> **Durum:** Bu repo şu an tasarım aşamasındadır — üretim kodu yok. README ve yol haritası viziyon dokümanıdır; uygulama Faz 1 (tek GPU üzerinde 1D shallow water denklemleri) ile başlar. İlerleme açıkça takip edilir.

## Bu nedir

WFE, modern GPU'lar için sıfırdan tasarlanmış, araştırma-sınıfı bir non-hidrostatik atmosfer modelidir.

Bugün araştırmada baskın olan hava modeli **WRF** (Weather Research and Forecasting), 1990'ların sonunda Fortran 90 ile yazılmış yaklaşık 1.5 milyon satırlık bir kod tabanıdır. Veri yapıları, MPI iletişim desenleri ve bellek modeli CUDA öncesinden gelir. WRF'i GPU'ya taşıma 10 yılı aşkın süredir aktif bir araştırma çabası ve hâlâ tamamlanmadı.

**WFE tersine bir yaklaşım benimser:** denklemlerden başla, donanımı (Hopper / Blackwell sınıfı GPU'lar) hedefle, kod kendi kendine şekillensin.

## Üç sütun

### 1. Dinamik çekirdek
- Tam sıkıştırılabilir non-hidrostatik denklemler
- Terrain-following (sigma-basınç hibrit) dikey koordinat
- Sonlu hacim uzaysal ayrıklaştırma, üçüncü dereceden WENO advection
- Split-explicit akustik zaman adımı (Klemp, Skamarock ve Dudhia, 2007)
- Tek hassasiyetli compute path; tensor core'lar için karışık hassasiyet seçeneği

### 2. Eklenebilir fizik
Her parametrizasyon, kararlı bir arayüze sahip bağımsız bir CUDA kernel'idir; herhangi biri dinamiklere dokunmadan değiştirilebilir:
- **Mikrofizik:** Thompson 8-sınıf
- **Sınır tabakası (PBL):** YSU (Hong, 2006)
- **Yüzey:** Noah-MP
- **Radyasyon:** RRTMG (uzun ve kısa dalga)
- **Konvektif şema:** convection-permitting çözünürlükte varsayılan kapalı

### 3. Operasyonel G/Ç
- GFS veya ICON-EU GRIB2 başlangıç ve sınır koşullarını okur
- Web viewer'a streaming için Zarr çıktısı yazar
- `xassemblianist.github.io/wfe` adresinde canlı tahmin sayfası besler

## Demo hedefi

Doğu Akdeniz / Antalya havzası için operasyonel tahmin pipeline'ı:
- **Bölge:** ~500&times;500 km, Antalya merkezli
- **Çözünürlük:** 1 km yatay, 60 dikey seviye
- **Aralık:** 48 saat
- **Sıklık:** 00 UTC ve 12 UTC, günde iki kez
- **Çıktı:** tarayıcıda render edilebilir tahmin sayfası, otomatik GitHub Pages'e yayınlanır

## Yol haritası

| Faz | Kapsam | Birincil çıktı |
|---|---|---|
| **1** | Tek GPU üzerinde 1D shallow-water, idealize başlangıç koşulları | Analitik dam-break çözümü ile doğrulama |
| **2** | 2D non-hidrostatik sıkıştırılabilir atmosfer, idealize vakalar | Density-current ve 2D dağ dalgası reprodüksiyonu (Straka 1993, Schär 2002) |
| **3** | 3D dinamik çekirdek, gerçek topografya, temel fizik (mikrofizik + PBL) | Belgelenmiş tarihsel konvektif olayın yeniden üretilmesi |
| **4** | Operasyonel pipeline, GFS / ICON-EU ingest, Zarr çıktı, web viewer | Günde iki kez çalışan canlı Antalya tahmini |
| **5** | Çoklu GPU domain decomposition, ensemble forecasting | Convection-permitting ölçekte 16 üyeli ensemble |

Her faz tek başına test edilebilir ve yayımlanabilir.

## Neden değerli

- **Gerçek bir boşluk var.** Modern tam-GPU NWP açık bir problem. MPAS, FV3 ve IFS Fortran-first; GPU yolları kısmi ve sonradan eklenmiş.
- **[XasmAI](https://github.com/Xassemblianist/XasmAI) ile birleşir.** Faz 5+ , kendi motorumla eğittiğim ML-augmented forecasting (öğrenilmiş subgrid closure'lar, FourCastNet-tarzı emülatörler) için kapı açar.
- **Gerçek dünya hedefi var.** Çalışan bir Antalya tahmin sayfası elle tutulur bir çıktı — paper değil, benchmark değil, halkın görebileceği bir şey.

## Referanslar

Bu çalışmanın dayandığı temel algoritmik referanslar:

- Skamarock, W.C. &amp; Klemp, J.B. (2008). *A time-split nonhydrostatic atmospheric model for weather research and forecasting applications.* J. Comput. Phys.
- Wicker, L.J. &amp; Skamarock, W.C. (2002). *Time-splitting methods for elastic models using forward time schemes.* Mon. Wea. Rev.
- Klemp, J.B., Skamarock, W.C. &amp; Dudhia, J. (2007). *Conservative split-explicit time integration methods for the compressible nonhydrostatic equations.* Mon. Wea. Rev.

## Lisans

MIT &mdash; bkz. [LICENSE](LICENSE).
