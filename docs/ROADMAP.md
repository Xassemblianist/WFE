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

## Faz 2 — Nem ve mikrofizik

- [ ] Nem değişkenleri (qv, qc, qr) + yoğunluk sıcaklığı θ_ρ ile kaldırma
- [ ] Kessler warm-rain mikrofiziği → sonra WSM6 sınıfı buz mikrofiziği
- [ ] Doğrulama: Bryan-Fritsch nemli benchmark, süperhücre simülasyonu

## Faz 3 — Gerçek veri: WPS muadili

- [ ] Lambert konformal projeksiyon + harita faktörleri
- [ ] GFS GRIB2 okuyucu (ecCodes) + otomatik indirme (NOMADS)
- [ ] Yatay/dikey interpolasyon, dengeleme, statik alanlar (topografya, arazi tipi)
- [ ] Zamana bağlı sınır koşulu beslemesi (boundary relaxation zone)

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
