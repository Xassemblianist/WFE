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
- [x] Harita faktörleri ✅ (izotropik Lambert m; akustik PGF + süreklilik):
      m(lat) standart paralellerde tam 1, Türkiye'de 0.998-1.003 (etki ~%0.3,
      alan küçük); idealize m=1 bit-özdeş (schaer_rest tam sıfır). Advection
      m-ölçekleme + ∂m/∂x higher-order (aynı %2 mertebe) olarak bırakıldı
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
- [x] Gözlemlerle METAR nokta doğrulaması ✅ (tools/verify_metar.py): aviationweather
      API'sinden gerçek istasyon gözlemi, model 2m T / 10m rüzgâr en-yakın eşleme.
      İlk sonuç (07-06 12Z, 337 istasyon): 10m rüzgâr bias +0.3 RMSE 2.9 m/s (iyi);
      2m T bias −5.7°C (öğle soğuk yanlılığı → levha ısıl ataleti; gelecek ayar)
- [x] İç bölge analiz-nudging ✅ (`nudge_tau`, GFS-güdümlü LAM standardı): Davies
      kenar relaksasyonuna zayıf iç taban (1/nudge_tau) eklenir; büyük ölçekler
      GFS'e bağlı, küçük ölçekler serbest. Türkiye 6h nudging: u −%21→**+%17**,
      qv −%72→**+%1** (persistansı yener), θ RMSE 2.9→2.36; METAR 2m T yanlılığı
      −4.2→**−2.5°C**, 10m rüzgâr yanlılığı ~0. Tüm skill metriklerinde büyük kazanç.
      Güç ayarı çok-vaka + bağımsız METAR ile: 3h > 6h (θ −%1→+%7, u +%17→+%30;
      METAR 2m RMSE 4.47→4.15) → tautolojik değil, gerçek kazanç. nudge_tau=3h varsayılan
- [x] (negatif sonuç) Dikey çözünürlük nz=40→56 test edildi: GFS skill'i
      iyileştirmedi (rüzgâr/θ marjinal kötü, %45 yavaş) → üst-seviye hatası
      çözünürlük-kaynaklı değil; geri alındı
- [x] Topluluk (ensemble) tahmini ✅ (tools/run_ensemble.py): IC θ' korelasyonlu
      pertürbasyon, N üye, ortalama + yayılım. Yayılım deseni fiziksel (konvektif/
      dağlık belirsizlik yüksek, deniz düşük). θ'-only az-dağılımlı; tam ensemble gelecek
- [x] Uzatılmış menzil 48h ✅: stabil, GFS f048'de u +%53/θ +%14 (persistansı yener)
- [~] Cumulus parametrizasyonu — DEĞERLENDİRİLDİ/ERTELENDİ: 12km gri-bölgede konveksiyon
      zaten kısmen çözülüyor; cumulus çift-sayım riski + faydası yağış gözlemi olmadan
      doğrulanamaz → iyi-ayarlı sistemi bozma riski değmez
- [~] Spektral/PBL-üstü seçici nudging — DEĞERLENDİRİLDİ/ERTELENDİ: uniform nudging
      2m'yi BL'yi GFS'e çekerek iyileştiriyor; BL'yi serbest bırakmak kazancı kaybettirir
- [ ] Nesting (iç içe alan) — opsiyonel
- [ ] Zamanlanmış otomatik koşular (görev zamanlayıcı)
- [~] 2m sıcaklık soğuk yanlılığı — kısmen ele alındı (2026-07-06): SW su buharı
      soğurma katsayısı düzeltildi (0.02→0.008, ~3x fazlaydı), LW opaklık artırıldı,
      levha ısı kapasitesi düşürüldü; verify_metar'a yükseklik (lapse) düzeltmesi
      eklendi. Etkin yanlılık −5.7 (ham) → −4.2°C (yüks.düz.). Kalan ~4°C spin-down/
      denge sorunu → Noah-sınıfı toprak modeli + çok-vakalı kalibrasyon (gelecek)
- [x] Çok katmanlı toprak modeli ✅ (Noah-benzeri, 4 katman 0.1/0.3/0.6/1.0m,
      implicit 1B ısı difüzyonu; tek levha yerine): doğru ısı iletimi + ısıl
      bellek + fiziksel toprak profili. 24h headline etkisi marjinal (2m yanlılığı
      atmosferik-baskın olduğundan); regresyon yok, WRF-Noah muadili altyapı
- [ ] Toprak nemi difüzyonu + bitki örtüsü (tam Noah) — çok-günlük tahminler için

## Faz 7 — Yüksek çözünürlüklü yerel model ✅ (2026-07-06)

- [x] Yüksek çöz. gerçek arazi (tools/get_terrain.py): AWS Terrain Tiles
      (terrarium, anahtarsız açık veri, ~30-90m) indirme + RGB→yükseklik decode
      + Web Mercator mozaik; prep alan-ortalamalı model gridine interpole eder
- [x] cases/antalya.ini: 2.5 km, Antalya körfezi + Toros (arazi 0→2588m çözülür;
      GFS 28km bunu ~1000m'ye ezerdi), akustik CFL'e uygun dt=9s, sınır-sürüklü
      (nudge_tau=0 → iç bölge terrain detayını serbest geliştirir)
- [x] Doğrulama: dik Toros arazisinde STABİL (|w|~12-17 m/s, patlama yok);
      **orografik detay** üretir (kıyı-dağ sıcaklık kontrastı 6-30°C, arazi-kilitli
      orografik yağış, arazi-kanalize rüzgâr) — 12km'nin göremediği; METAR (Antalya
      istasyonları): 2m T RMSE 2.3°C (kaba koşudan daha iyi — arazi doğru çözülür)
- [x] 1 km konveksiyon-çözücü (cases/antalya1km.ini, zoom-11 arazi 0→2755m):
      en dik Toros'ta STABİL (|w| sınırlı, sıkı akustik CFL dt=4s); model 12km→1km
      tüm operasyonel ölçek aralığında çalışır (WRF en-ince-nest rejimi)
- [x] Operasyonel entegrasyon: run_forecast.py yüksek çöz. için get_terrain'i
      otomatik çağırır (tek komut: arazi→prep→koşu→ürünler)
- [ ] İki-yönlü nesting (12km ↔ 2.5km çift yönlü besleme) — tek-yönlü hazır
- [ ] Sub-km çözünürlük (LES yaklaşımı) — daha güçlü GPU ile

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
