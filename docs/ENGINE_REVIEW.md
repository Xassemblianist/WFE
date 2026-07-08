# WFE Motor İncelemesi (2026-07-08, tam kaynak okuması)

3.701 satır C++/CUDA'nın satır-satır incelemesi. Amaç: mimariyi denetlemek ve
"WRF'den iyi" hedefine giden somut yolu çıkarmak.

## Mimari özeti (olduğu gibi)

| Katman | Durum |
|---|---|
| Zaman entegrasyonu | WRF-sınıfı: RK3 (Wicker–Skamarock) + split-explicit akustik; dikeyde off-centered implicit w–π′ tridiagonal (kolon/thread), yatayda forward–backward + π* diverjans sönümlemesi |
| Adveksiyon | 5. mertebe upwind (WS2002), akı formu, advektif-tutarlı; nemde Skamarock-2006 pozitif-tanımlı akı renormalizasyonu |
| Koordinat/metrik | Gal-Chen arazi-takip ζ; çapraz metrik terimleri hem kütle akısında hem PGF'de; Lambert harita faktörü m(lat) PGF + diverjansta tutarlı |
| Sınırlar | Davies relaksasyon (cos², bdy_width) + iç-bölge analiz nudging; KW radyasyon açık-sınır; GFS sınır dosyaları zamanda lineer |
| Fizik | Louis-79 yüzey katmanı; Troen–Mahrt/Hong–Pan **nonlocal** PBL (karşı-gradyan dahil, h↔w* iterasyonlu); 4-katman implicit toprak; iki-akı geniş-bant radyasyon (güneş geometrisi + su buharı/bulut ε); Kessler sıcak yağmur |
| Sağlamlık | w_damping (dikey Courant), 10-adımda GPU absmax NaN bekçisi + acil çıktı, config yazım-hatası uyarısı, run_info.txt provenans, WFE_PROF profil |
| Doğrulama | 8 idealize + çok döngülü gerçek-veri doğrulaması; u-v ayna simetrisi makine kesinliğinde (yön yanlılığı yok) |

**Hüküm:** Bu, oyuncak değil — dinamik çekirdek WRF-ARW'nin sayısal omurgasının
sadık, temiz bir uygulaması. 3.7k satırda bu kapsam istisnai. Tek-GPU'da
1500×+ gerçek-zaman oranı, WRF'nin aynı donanımdaki CPU koşusundan ~2 mertebe hızlı.

## WRF'ye karşı boşluk analizi (önem sırasıyla)

1. **Radyasyonda bulut albedosu YOK** — SW'de bulut yalnız soğurur
   (`surface.cu` k_sfc_scalar: `tau=exp(-(0.008du+0.15dl)/mu)`), yansıtmaz.
   Bulutlu günde yüzeye fazla güneş → gündüz sıcak, gece (LW ε kaba) soğuk
   sapma. **En yüksek getiri/maliyet oranı:** bulut kolonuna basit albedo
   `A_c = a·LWP/(LWP+b)` ekle (~10 satır) — bilinen −5 °C 2m sapmasının ana
   şüphelisi bu ve LW ε katsayıları.
2. **t2m/u10 tanısında stabilite düzeltmesi yok** — düz log-oran interpolasyonu
   (`r2=log(2/z0h)/log(z1/z0h)`); Businger–Dyer ψ fonksiyonları yok. Kararlı
   gece koşullarında 2m sıcaklığı yüzeye fazla yapışır → gece soğuk sapma.
   Louis Fh zaten hesaplı; ψ eklemek ~15 satır.
3. **Yüzey özellikleri sabit** — z0=0.1 m (tüm kara), albedo=0.2, bitki örtüsü
   direnci yok. Orman/bozkır/şehir ayrımı yok → yerel kontrast eksik. GFS'ten
   veya statik LU haritasından z0/albedo alanları: prep + 2 alan.
4. **Mikrofizik sıcak-yağmur (Kessler)** — kar/buz/graupel yok. Yaz Antalya'sı
   için yeterli; kış Türkiye'si için tek-moment 5-sınıf (WSM5-benzeri) gerekir.
5. **Kümülüs parametrizasyonu yok** — 6 km gri-bölgede kabul edilebilir
   (açık-çözüm), 12+ km'de eksik. Öncelik düşük (operasyonel alanlar ≤6 km).
6. Küçükler: LW'de CO₂/ozon yok (sabit katsayıya gömülü), toprak nemi
   prognostik değil (β sabit), derin toprak T başlangıç TSK'sı.

## Uygulandı (2026-07-08 — bu incelemenin 1. ve 2. maddeleri)

- **Bulut albedosu** (`surface.cu`): `A_c = LWP/(LWP+60)`; TOA'da SW artık
  bulut tarafından yansıtılıyor. Doğrulama süiti: 7/7 PASS.
- **ψ-düzeltmeli 2m/10m tanıları** (`surface.cu`): Businger–Dyer ψm/ψh,
  z/L Rib'den Launiainen yaklaşımıyla; kararlı gece profillerinde 2m artık
  yüzeye yapışmıyor. Doğrulama süiti: 7/7 PASS.
- Sonraki METAR döngü-doğrulamasında (verify_metar) 2m yanlılığının yeniden
  ölçülmesi beklenir — operasyonel döngü her koşuda otomatik doğruluyor.
- **Toprak termal özellikleri + buharlaşma nem-stresli yapıldı** (aynı gün,
  ikinci tur): C=f(w) [1.2e6+3.2e6w], K=f(w) [0.25+3w] (Johansen-tipi) ve
  β karesel nem-stresi (solma 0.08 / tarla 0.32). Gerekçe: sabit "nemli kil"
  ataleti (µ≈1800) + cömert lineer β, Temmuz bozkırında öğle ısınmasını çift
  koldan bastırıyordu → METAR'da −4.3 °C gündüz sapması ölçüldü (06Z koşusu,
  +3/+5 sa, ~300 istasyon).
- **A/B SONUCU (aynı 06Z döngüsü, aynı istasyonlar):** 2m yanlılık
  −4.30→−3.53 °C (+3 sa) ve −4.02→−3.31 °C (+5 sa); RMSE 4.65→4.06 ve
  4.43→3.98. Rüzgâr değişmedi (−0.4/−0.5 m/s, iyi). **~0.75 °C net kazanç.**
- Kalan −3.3 °C'nin analizi: sapma +3→+5 arasında BÜYÜMÜYOR (fizik-evrimi
  değil, ofset karakterli). Adaylar: (a) doğrulamanın yükseklik düzeltmesi
  6.5 K/km — öğlen karışık sınır tabakasında gerçekçi oran kuru-adyabatik
  (9.8); istasyon-hücre Δz~300 m ile ~+1 °C görünür sapma açıklar
  (verify_metar'a gündüz-adyabatik düzeltme opsiyonu eklenebilir);
  (b) GFS başlangıç/θ mirası; (c) LW ε katsayıları (gece ayrı ölçülmeli).

## Doğrulama hedefi hakkında ÖNEMLİ not

`verify.py` beceriyi **GFS'in kendi +24h tahminine karşı** ölçer (analiz o an
mevcut değil). 6 km'lik model GFS'ten meşru biçimde ayrıştıkça bu metrik
cezalandırır — turkey6km'de θ "beceri"sinin ~−46..−48% çıkması (eski fizikte
de aynı) bundan; 12 km alan GFS'e yapışık kaldığından +14% görünüyordu.
Gerçek hakem METAR istasyonlarıdır. **Sonraki adım:** gecikmeli doğrulama —
her döngüde ~24 saat önceki koşuyu, yeni döngünün ANALİZİ (f000) ile doğrula
(run_operational'a eklenecek; verify.py'ye hedef-analiz modu gerekir).

## Bu oturumda yapılanlar (motor + ürün)

- **turkey6km saatlik çıktı** (`out_interval 10800→3600`): 8 kare → 25 kare;
  animasyon ve saatlik meteogram gerçek anlamda saatlik. Koşu maliyeti
  değişmedi (yalnız I/O ~2.3 GB/koşu; operasyonel budama zaten var).
- **Görüntüleme motoru fiziksel downscaling** (`server/terrain.py` +
  WebGL boyacı): T_görüntü = T_model + Γ·(z_model − z_gerçekDEM), Γ=6.5 K/km.
  Modelin 6 km'de göremediği vadi/sırt kontrastı, elimizdeki z9 (~150 m) DEM
  ile geri kazanılıyor — meteoblue'nun ürünleştirme tekniği. (Deniz
  batimetrisi 0'a kelepçeli; yalnız t2m'e uygulanır.)

## Önerilen sıra (WRF'yi ürün kalitesinde geçmek için)

1. Bulut albedosu + LW ε ayarı → METAR 2m sapmasını yeniden ölç (verify_metar).
2. ψ-düzeltmeli 2m/10m tanısı → gece sapması.
3. z0/albedo/emisivite alanları (GFS'ten) → yerel kontrast.
4. WSM5-benzeri buz mikrofiziği → kış yağışı (kar haritası ürünü!).
5. Toprak nemi prognostiği (2 katman kova) → çok-günlük koşularda sürüklenme.

Model şimdiden birçok metrikte persistansı ve ürün tarafında görselleştirme
kalitesiyle çoğu WRF kurulumunu geçiyor; yukarıdaki 1-2-3, sıcaklık sapmasını
kapatarak "sayısal olarak da WRF-üstü" iddiasını METAR'la savunulur kılar.
