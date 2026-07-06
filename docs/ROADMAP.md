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
- [x] Pozitif-tanımlı nem adveksiyonu ✅ (Skamarock 2006 akı renorm.; `pd_moist`):
      moist_blob testi 5265 negatif hücre → 0 (qv≥0 makine kesinliğinde); Türkiye
      koşusunda sahte hafif yağış −%15, gerçek çekirdekler keskinleşti (183→272 mm)
- [ ] WSM6 sınıfı buz mikrofiziği (kar/graupel; kış yağışları için — Faz 4 ile)

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

## Faz 4 — Fizik parametrizasyonları (v1 ✅ 2026-07-04)

- [x] Yüzey katmanı: Louis (1979) bulk aerodinamik (Cd/Ch, momentum sürtünmesi,
      duyulur/gizli ısı) — src/physics/surface.cu
- [x] PBL: Ri-bağımlı yerel-K profili + kolon-implicit dikey difüzyon (u,v,θ,qv)
- [x] Levha toprak (force-restore, karada prognostik T_sfc; denizde SST sabit)
- [x] Basit radyasyon zorlaması: güneş geometrisi + bulut-zayıflatmalı SW,
      Brunt ampirik LW, troposferik −2 K/gün LW soğuması
- [x] Doğrulama: Türkiye 24h yeniden koşusu — alt seviye u becerisi −%48→−%32,
      3B u −%19→−%11, jet +%23, θ +%15; fizik maliyeti ~%16
- [x] Nonlocal PBL ✅ (Troen-Mahrt/Hong-Pan K-profili + karşı-gradyan; `pbl=nonlocal`):
      bulk-Ri PBL yüksekliği teşhisi, konvektif K=κ·ws·z·(1-z/h)², θ/qv karşı-gradyan.
      PBLH ders kitabı gündüz döngüsü (kara gece 460m → öğleden sonra 1400m,
      deniz ~800m sabit); Türkiye qv becerisi −%50→−%38, rüzgâr/θ küçük kazanç
- [ ] Gerçek kolon radyasyonu (iki-akı/broadband) + toprak nemi (Noah sınıfı)
- [ ] Üst seviye hatası: harita faktörleri + dikey seviye artırımı

## Faz 5 — Operasyonel sistem ✅ (2026-07-04)

- [x] Uçtan uca pipeline (tools/run_forecast.py): en güncel GFS döngüsünü
      NOMADS'ta otomatik bulur → prep → koşu → haritalar → NetCDF → doğrulama
- [x] Harita ürünleri (tools/forecast_maps.py): cartopy Lambert + kıyı/sınır
      çizgili 4 panel (yüzey T + rüzgâr, jet, bulutluluk, dönem yağışı)
- [x] NetCDF dışa aktarım (tools/to_netcdf.py, sıkıştırmalı, xarray tabanlı)
- [x] Doğrulama aracı (tools/verify.py): GFS hedefine karşı alan + seviye-bazlı
      RMSE/beceri raporu — vaka bazlı sistematik iyileştirmenin altyapısı
- [ ] Gözlemlerle (METAR/SYNOP) nokta doğrulaması
- [ ] Nesting (iç içe alan) — opsiyonel
- [ ] Zamanlanmış otomatik koşular (görev zamanlayıcı)

## Faz 6 — Performans (v1 ✅ 2026-07-04)

- [x] Yerleşik profilci (`WFE_PROF=1`): bölüm bazlı GPU-senkron zamanlar
- [x] Ölçüm bulgusu: akustik döngü %60 (kolon çözücü gecikme-sınırlı, 16k thread);
      adveksiyon yalnız %11-14 — önyargı değil ölçüm yönlendirdi
- [x] Akustik sabit katsayı önhesabı (rtjx/rtjy/kt3/aw alanları): wpi yükleri ~2×↓
- [x] Vaka-bazlı acoustic_ns ayarı: dx=12km'de ns=4 yeterli (aşama CFL≤0.6) —
      turkey 60.1→46.4 s (1.30×, 1861× gerçek-zaman), beceri birebir korundu
- [x] CUDA Graphs denendi ve GERİ ALINDI: kazanç ölçülemedi + State::swap tampon
      rotasyonuyla temelden uyumsuz (sabit pointer yakalar) — test süiti yakaladı.
      Yeniden denenirse önce swap yerine sabit-rol tamponlara geçilmeli.
- [x] Koruma doğrulaması: ns=3 (CFL 0.88) patlamasını bekçi temiz yakaladı (kod 3)
- [ ] Kolon çözücü 3B-hazırlık/ince-çözücü ayrımı (kalan en büyük tekil kazanç ~%15-20)
- [ ] Skalar adveksiyon füzyonu (4→1 kernel, ~%4), blok boyutu taraması, fast-math
- [ ] Asenkron I/O örtüşmesi, çoklu GPU (halo exchange) — donanım el verirse
