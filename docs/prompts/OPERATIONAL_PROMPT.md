# GÖREV: WFE Yerel Operasyonel Sistem (1-2 hafta kesintisiz) + API genişletme

> Projeyi hiç bilmeyen bir Opus 4.8 oturumunu soğuktan başarıya taşımak için yazıldı.
> Depo: github.com/Xassemblianist/WFE. İhtiyacın olan kritik teknik bilgi burada
> damıtıldı — yeniden keşfetme, kullan. Kalite çıtası yüksek: sistem 1-2 hafta
> insan müdahalesi olmadan güvenilir çalışmalı.

## Bağlam

WFE: sıfırdan C++/CUDA bölgesel hava tahmin modeli. Windows 11, **RTX 2060 (6 GB,
sm_75)**, CUDA 13.2, VS Build Tools 2022. Model olgun ve doğrulanmış (GFS + METAR'a
karşı persistansı yener). Mevcut operasyonel pipeline: `tools/run_forecast.py`
(en güncel GFS döngüsünü bulur → prep → koşu → haritalar → NetCDF → doğrulama).

Sahibi şimdilik **yerel PC'de** iki modeli **1-2 hafta sürekli** koşturmak istiyor:
1. **Türkiye — İYİ ÇÖZÜNÜRLÜK** (6 km öneri, tüm ülke, gerçek yüksek çöz. arazi).
   Şu an `cases/turkey.ini` 12 km; bunu 6 km'ye çıkar.
2. **Antalya — 2.5 km yüksek çözünürlük** (`cases/antalya.ini` hazır).

## Derleme + çalıştırma (KRİTİK)

```
# Derleme (PowerShell). CMake/Ninja sistem PATH'inde YOK; Build Tools içindekiler:
cmd /s /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64 -no_logo && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build'

# Python: %LOCALAPPDATA%\Programs\Python\Python312\python.exe ; bağımlılıklar requirements.txt
# Test süiti (her model değişikliğinden SONRA 7/7 kalmalı, ~90 s):
python tools\run_tests.py
```

## Bildiğim kritik teknik detaylar (bunları kullan)

**Akustik CFL (çözünürlük → zaman adımı):** Split-explicit çekirdekte RK3 aşama-0
akustik alt-adımı `dtau0 = dt/3` ve stabilite için `dt/3 < 0.5·dx/c` (c≈350 m/s).
Yani **dt < 1.5·dx/350**. Tablo (`acoustic_ns=4` hepsi):

| dx | dt (kullan) |
|---|---|
| 12 km | 30 |
| 6 km | 15 |
| 4 km | 12 |
| 2.5 km | 9 |
| 1 km | 4 |

Model başlangıçta CFL uyarısı basar; aşarsan uyarır. `w_damping=on` operasyonel güvenlik.

**Yüksek çözünürlüklü arazi (6 km ve altı için ŞART):** GFS orografisi 28 km'de
pürüzsüz — 6 km grid için işe yaramaz. `tools/get_terrain.py cases/<case>.ini --zoom Z`
AWS Terrain Tiles'tan (anahtarsız, ~30-90 m) gerçek arazi indirir; case'de
`terrain_source = tiles`. Zoom: **6 km→9, 2.5 km→10, 1 km→11**. (Pillow gerekir.)
prep bunu alan-ortalamalı model gridine koyar. Dik arazide model STABİL (Antalya
2.5 km'de 2588 m Toros'ta patlama yok — doğrulandı).

**Pipeline (bir koşu):** `get_terrain` (bir kez, önbelleğe alınır) → `prep_gfs.py
--date --cycle --hours` → `wfe.exe case.ini t_end=<saat*3600>` → `forecast_maps.py`
→ `to_netcdf.py` → `verify.py`/`verify_metar.py`. `run_forecast.py` bunların hepsini
yapar + en güncel GFS döngüsünü otomatik bulur + terrain_source=tiles ise get_terrain'i
çağırır. `--hours` için t_end'i otomatik geçirir.

**Ayarlar (doğrulanmış en iyi):** `nudge_tau=10800` (3 saat iç analiz-nudging —
GFS-güdümlü LAM'de büyük ölçekleri GFS'e bağlar; çok-vaka + METAR ile 6h'tan iyi
çıktı), `pbl=nonlocal`, `pd_moist=on`, `w_damping=on`, `physics=simple`,
`rayleigh_zd=17000-18000`, `bc_x=open bc_y=open`.

**Runtime (RTX 2060, 24h tahmin):** 12 km ~60 s; 2.5 km ~210 s; 6 km tahmini ~7-10 dk;
1 km çok ağır. Prep (GFS indirme) ~2-4 dk. → 6 km Türkiye + 2.5 km Antalya bir döngüde
~15-20 dk. Günde 4 döngü = ~1-1.5 saat GPU/gün — sürdürülebilir.

**GFS zamanlaması:** döngüler 00/06/12/18Z; `f024` döngüden ~4-4.5 saat sonra hazır.
`run_forecast.py:latest_cycle()` uygunluğu kontrol eder. prep 3-denemeli indirir.
NOMADS bazen 502/aksama verir — retry + bir sonraki döngüye geç.

**Robustluk (mevcut):** başlangıç CFL uyarısı, her 10 adımda GPU patlama/NaN bekçisi
(aşarsa acil çıktı + çıkış kodu 3), config yazım-hatası uyarısı, provenans
(`out/<r>/run_info.txt`). Çıkış kodu 0 = başarı.

**Ürünler:** `out/<bölge>/<değişken>_<adım:06d>.bin` (u,v,w,thp,pip,qv,qc,qr 3B;
t2m,u10,rain,pblh,tsk 2B), `map_<adım>.png`, `meta.json`, `wfe_out.nc`, `zc.bin`.
Alan lat/lon: `<input_dir>/wfe_init.bin` (bkz. tools/verify_metar.py ayrıştırma).

## Yapılacaklar

**1. `cases/turkey6km.ini`** — `turkey.ini`'yi baz al; `dx=dy=6000`, alanı tüm
Türkiye+çevresini kapsayacak şekilde ayarla (ör. `nx=320 ny=200`, merkez
lat0=39 lon0=35, standart paraleller 35/43), `dt=15 acoustic_ns=4`,
`terrain_source=tiles`, `stretch=geometric dz0=200 dz_ratio=1.08 dz_max=700 nz≈44`,
`nudge_tau=10800`, fizik + pd_moist + w_damping açık. Sonra:
`get_terrain cases/turkey6km.ini --zoom 9` (tüm-Türkiye, çok karo — sabırlı ol),
prep + kısa koşu ile **stabiliteyi doğrula** (patlama yok), `run_tests.py` 7/7 kalmalı.
Not: 6 km çok ağır gelirse 8 km (dt=20) düş; sahibi "iyi çözünürlük" istiyor, 6 km hedef.

**2. `tools/run_operational.py`** — yerel, gözetimsiz operasyonel döngü:
```
python tools/run_operational.py --days 14 --regions turkey6km antalya
```
Mantık (sonsuz döngü, `--days` sonra dur):
- En güncel uygun GFS döngüsünü bul (`run_forecast.latest_cycle` mantığı).
- İşlenmemiş yeni döngü varsa: her bölge için `run_forecast.py` çağır (prep + model +
  haritalar + netcdf + doğrulama). Ürünleri sakla + "latest" işaretçisini güncelle.
- HER ŞEYİ bir log dosyasına yaz (zaman, döngü, bölge, süre, çıkış kodu, doğrulama özeti).
- ~20-30 dk uyu, tekrarla.
- **Robust:** her bölge koşusu try/except içinde; bir bölge/döngü çökerse LOGLA ve
  DEVAM ET — döngü asla ölmemeli. Ağ/GFS aksamasında döngüyü atla, sonraki denesin.
- **Disk yönetimi:** `out/` büyür; N günden (ör. 3) eski zaman-damgalı koşuları sil.
- Windows'ta cron yok — bu Python döngüsü açık bırakılarak çalışır (veya Görev
  Zamanlayıcı ile). Kararlı, tekrar-başlatılabilir olsun (kaldığı yerden devam).

**3. API genişletme (`server/`) — frontend için** (bkz. `docs/prompts/FRONTEND_PROMPT.md`):
- `GET /overlay/{region}/{field}/{step}.png`: **tek alan**, eksenи/çerçevesi olmayan,
  şeffaf arka planlı, alan-bazlı renk eşlemeli PNG (matplotlib, `bbox_inches='tight'`,
  `axis('off')`, `transparent=True`). `field` ∈ {t2m, wind, precip, cloud, mslp}.
- `GET /products/{region}` manifestine **`bounds:[west,south,east,north]`** (wfe_init.bin
  lat/lon'dan) ve **`fields:[...]`** ekle.
- **CORS** aktifleştir (`CORSMiddleware`, geliştirmede `allow_origins=["*"]`).
- `latest` manifesti: her bölge için güncel koşuya işaret eden bir uç
  (`GET /products/{region}` zaten en güncel out/'u okuyor — koru).

**4. Doğrulama:**
- `run_tests.py` 7/7 (model/case değişikliğinden sonra).
- turkey6km: dik arazide stabil + orografik detay üretir; `verify.py` persistansı yener;
  `verify_metar.py` makul (2m T RMSE ~2-3°C, 10m rüzgâr yanlılık ~0).
- Operasyonel döngü: en az 2 döngü uçtan uca insan müdahalesi olmadan tamamlanır; log temiz.

## Not

- Sahibi commit'lerde AI/Claude görünmesini İSTEMİYOR — `Co-Authored-By` EKLEME,
  `git config user` `Xassemblianist <omerkaan20102003@gmail.com>` olarak kalsın.
- İleride GCP (spot GPU) hedefi var (`deploy/` hazır) ama ŞU AN yerel RTX 2060.
- Test süiti + `docs/EQUATIONS.md` doğrulama tablosu + `docs/ROADMAP.md` durum için
  bak. Her anlamlı değişiklikten sonra `run_tests.py`.
