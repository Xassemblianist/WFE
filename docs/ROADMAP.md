# Yol haritası

Hedef: WRF muadili, GPU-yerlisi bölgesel tahmin sistemi — GFS'ten başlangıç/sınır
koşulu alıp Türkiye ve çevresi için gerçek tarih-saatli tahmin üretmek.

## Faz 0 — Çekirdek altyapı + kuru dinamik çekirdek ✅ (2026-07-03)

- [x] CMake/CUDA build, FP32/FP64 anahtarı, GPU alan/grid/config altyapısı
- [x] KW78 sıkıştırılabilir non-hidrostatik denklemler, C-grid, WS2002 RK3,
      5. mertebe upwind adveksiyon (tamamen explicit)
- [x] Sıcak kabarcık doğrulaması + binary çıktı + Python görselleştirme

## Faz 1 — Gerçek çekirdek yetenekleri ✅ (2026-07-04)

- [x] Split-explicit akustik alt-adımlama (yatay explicit, dikey implicit) → dt 6×
- [x] Arazi-takip eden dikey koordinat (Gal-Chen) + gerilmiş dikey grid (geometric)
- [x] Açık/radyasyon yanal sınır koşulları + üst Rayleigh sönümleme katmanı
- [x] Coriolis (f-plane) + genel taban durumu (isentropic / constant_N; sounding Faz 3'te)
- [x] Doğrulama: Straka yoğunluk akıntısı, Schär dağ dalgası, arazide durağanlık,
      Galilean değişmezlik, split-explicit/explicit regresyonu (docs/EQUATIONS.md tablosu)

## Faz 2 — Nem ve mikrofizik ✅ (2026-07-04, buz mikrofiziği hariç)

- [x] Nem değişkenleri (qv, qc, qr), nemli kaldırma g(θ'/θ̄+0.61qv'−qc−qr), θ̄v
      ile tutarlı hidrostatik taban ve PGF
- [x] Kessler warm-rain mikrofiziği (doygunluk ayarı, otokonv., akresyon,
      buharlaşma, sedimentasyon, yüzey yağış birikimi)
- [x] WK82 sounding (nem + tanh rüzgâr kesmesi)
- [x] Doğrulama: WK82 süperhücre — fırtına bölünmesi (ayna-simetrik sağ/sol
      hareketli), w_max 40-48 m/s, yağış şeritleri; kuru regresyonlar birebir
- [ ] WSM6 sınıfı buz mikrofiziği (kar/graupel; kış yağışları için — Faz 4 ile)
- [ ] Pozitif-tanımlı nem adveksiyonu (5. mertebe şemanın alt-aşımlarını keser)

## Faz 3 — Gerçek veri: WPS muadili ✅ (2026-07-04, harita faktörleri hariç)

- [x] Lambert konformal projeksiyon (ileri/ters + rüzgâr rotasyonu, prep'te)
- [x] GFS GRIB2 okuma (ecCodes/pip) + NOMADS filter otomatik indirme (tools/prep_gfs.py)
- [x] Yatay bilinear + dikey interpolasyon, θ̄v-tutarlı taban, GFS topografyası,
      f(lat) 2D Coriolis, tam-θv PGF
- [x] Davies sınır relaksasyon bölgesi (cos² rampa, 3 saatlik GFS beslemesi)
- [x] İlk gerçek tahmin: Türkiye 24h @ 12 km, 52 s duvar zamanı (1671× gerçek-zaman);
      doğrulama GFS f024'e karşı: θ persistansı +%14, jet seviyesi u +%20 yener;
      alt seviyeler PBL fiziksiz kaybeder (Faz 4 gerekçesi)
- [ ] Harita faktörleri mx,my (şu an m=1 yaklaşımı; Türkiye alanında ~%1-2 hata)
- [ ] Statik arazi tipi/albedo alanları (Faz 4 yüzey fiziğiyle birlikte)

## Faz 4 — Fizik parametrizasyonları

- [ ] Radyasyon (kısa/uzun dalga, RRTMG sınıfı basitleştirilmiş başlangıç)
- [ ] Yüzey katmanı + toprak modeli (Noah sınıfı basitleştirilmiş)
- [ ] PBL şeması (YSU sınıfı) + alt-grid türbülans (Smagorinsky/TKE)

## Faz 5 — Operasyonel sistem

- [ ] Uçtan uca pipeline: indir → ön işle → koş → görselleştir (haritalı ürünler)
- [ ] NetCDF çıktı, tahmin doğrulama metrikleri (gözlemlerle karşılaştırma)
- [ ] Nesting (iç içe alan) — opsiyonel

## Faz 6 — Performans

- [ ] Kernel optimizasyonu (shared memory, kernel füzyonu, occupancy)
- [ ] Asenkron I/O, CUDA stream örtüşmesi
- [ ] Çoklu GPU (halo exchange) — donanım el verirse
