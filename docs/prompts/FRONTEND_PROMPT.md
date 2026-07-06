# GÖREV: WFE Hava Durumu Web Sitesi (Frontend)

> Bu prompt, projeyi hiç bilmeyen bir Opus 4.8 oturumunu soğuktan başarıya taşımak
> için yazıldı. Repo kökünde `README.md`, `docs/EQUATIONS.md`, `server/app.py` var —
> ama ihtiyacın olan her şey burada. Depo: github.com/Xassemblianist/WFE

## Bağlam: WFE nedir

WFE (Weather Forecast Engine), sıfırdan C++/CUDA ile yazılmış, GPU'da çalışan bölgesel
sayısal hava tahmin modeli (WRF muadili). Sahibinin RTX 2060'lı yerel PC'sinde çalışıyor
ve **iki bölge** için gerçek GFS verisiyle operasyonel tahmin üretiyor:

- **Türkiye** (~6 km, tüm ülke) — genel tahmin
- **Antalya** (2.5 km, gerçek yüksek çözünürlüklü arazi) — Toros + kıyı detayı

Model her GFS döngüsünde (günde 4×) koşuyor; ürünler `out/<bölge>/` altında birikiyor.
Model çıktıları GFS'e ve gerçek istasyon gözlemlerine (METAR) karşı doğrulanmış; birçok
alanda persistansı yeniyor (24h u +%30, 48h u +%53). **Senin görevin:** bu tahminleri
gösteren, modern, profesyonel, güvenilir bir hava durumu web sitesi yapmak.

**Kalite çıtası:** windy.com / ventusky / meteoblue seviyesinde temiz, akıcı, hızlı.
Sahibi "frontendde Opus'un eline kimse su dökemez" dedi — o çıtayı karşıla.

## Mevcut API (tüketeceğin backend)

FastAPI, `server/app.py`. Yerelde çalıştırma:
```
python -m uvicorn app:app --app-dir server --port 8000
```
Base URL: `http://localhost:8000`. Otomatik dokümantasyon: `/docs`.

Mevcut endpoint'ler ve yanıt şekilleri:

| Endpoint | Yanıt |
|---|---|
| `GET /health` | `{status, model, regions:[...]}` |
| `GET /regions` | `[{id, title, desc, default_hours}]` |
| `GET /products/{region}` | `{region, title, available, init (ISO UTC), dx_m, nx, ny, steps:[{step, fhour}], maps:[dosya adları]}` |
| `GET /products/{region}/map/{name}` | PNG (4-panelli matplotlib kompozit) |
| `GET /point/{region}?lat=&lon=` | `{region, lat, lon, grid:{i,j,elev_m,grid_lat,grid_lon}, init, series:[{valid (ISO), fhour, t2m_C, wind10_ms, precip_mm}]}` |
| `POST /run/{region}?hours=` | `{job_id}` (arka planda koşu tetikler; GPU'lu makinede) |
| `GET /run/{job_id}` | `{job_id, state}` |

`region` ∈ {`turkey`, `antalya`, `antalya1km`}. `init` koşunun başlangıç zamanı;
`series` saatlik nokta tahmini (2m sıcaklık °C, 10m rüzgâr m/s, saatlik yağış mm).

## İstenen site özellikleri

1. **İnteraktif harita** (maplibre-gl önerilir): model alanı üzerinde tahmin alanı
   overlay'i (sıcaklık / rüzgâr / yağış / bulut). Haritada tıkla → o noktanın tahmini
   yan panelde. Türkiye ↔ Antalya bölge geçişi (Antalya'ya zoom).
2. **Zaman kaydırıcı**: forecast saatleri arasında gez + oynat (animasyon).
3. **Katman seçici**: sıcaklık / rüzgâr / yağış / bulutluluk.
4. **Nokta tahmini paneli (meteogram)**: seçili nokta veya aranan şehir için
   sıcaklık + rüzgâr + yağış grafikleri (saatlik). Şehir arama (Türk şehirleri).
5. **Ana sayfa / özet**: öne çıkan şehirler için mevcut durum kartları.
6. **Responsive** (mobil + masaüstü), **koyu/açık tema**, **Türkçe** arayüz.
7. Model hakkında kısa bir "hakkında" sayfası (WFE nedir, doğrulama sonuçları —
   docs/EQUATIONS.md tablosundan alınabilir; güven verir).

## API SÖZLEŞMESİ — backend ile koordine et (ÖNEMLİ)

Mevcut API interaktif harita overlay'i için yeterli DEĞİL: yalnızca 4-panelli
matplotlib PNG servis ediyor (bu bir "galeri" görünümü için iyi, ama harita üstüne
bindirilecek tek-alan georeferanslı raster değil). Gerçek windy-tarzı harita için
backend'den (bu işi paralel yürüten diğer Opus oturumu / API prompt'u) şunları iste
ve sözleşmeyi netleştir:

- `GET /overlay/{region}/{field}/{step}.png` — **tek alan**, şeffaf arka planlı,
  renk-eşlemeli PNG (harita overlay'i). `field` ∈ {t2m, wind, precip, cloud, mslp}.
- `GET /products/{region}` manifestine **`bounds: [west,south,east,north]`** (lat/lon)
  ve **`fields: [...]`** ekle — overlay'i haritaya georeferanslamak için.
- (İsteğe bağlı) `GET /field/{region}/{field}/{step}.json` — ham ızgara değerleri
  (istemci-tarafı renklendirme/kontur için).
- **CORS**: ayrı bir dev sunucudan (ör. Vite :5173) API'ye erişmek için FastAPI'de
  CORS aktif edilmeli (`fastapi.middleware.cors.CORSMiddleware`, `allow_origins=["*"]`
  geliştirmede). Backend prompt'una bunu ekletmen gerekebilir.

Not: Alan sınırları (bounds) `out/<bölge>` yanındaki prep verisinden (`wfe_init.bin`
içindeki lat/lon) türetilir; backend bunu manifeste ekleyebilir. Antalya alanı Lambert
projeksiyonlu olduğundan hafif eğik; maplibre'de düz lat/lon overlay yeterli yaklaşım.

## Teknik yönlendirme

- **Stack**: React + Vite veya SvelteKit; harita için **maplibre-gl**; grafikler için
  recharts / chart.js / uPlot. TypeScript tercih edilir.
- Şehir listesi: statik `cities.json` (Türk il/ilçe merkezleri + lat/lon).
- Renk eşlemeleri meteoroloji standardına yakın olsun (sıcaklık: mavi→kırmızı;
  yağış: beyaz→mavi→mor; rüzgâr: sarı→kırmızı).
- Performans: overlay PNG'leri önbelleğe al; zaman kaydırıcıda ön-yükleme.
- Erişilebilirlik + hızlı ilk boyama; gereksiz ağır bağımlılıktan kaçın.

## Teslimatlar

- `web-app/` dizini (repo kökünde), çalışan `npm run dev`, kısa README (kurulum + API
  base URL yapılandırması).
- Yerel API'ye (`localhost:8000`) karşı test edilmiş; en az Türkiye + Antalya bölgeleri
  ve bir şehir için nokta tahmini çalışıyor.
- Mevcut minimal `web/` (index.html + app.js) referans/başlangıç noktası olabilir —
  onu profesyonel bir uygulamaya taşı.

## Çalıştırıp test etme

```
# API (ayrı terminal)
python -m uvicorn app:app --app-dir server --port 8000
# örnek veriler out/turkey ve out/antalya altında mevcut (önceki koşulardan)
curl http://localhost:8000/regions
curl "http://localhost:8000/point/antalya?lat=36.9&lon=30.7"
```

Sahibi Antalya'da yaşıyor; Antalya deneyimi özellikle iyi olsun. Türkçe, temiz, hızlı,
güven veren bir ürün hedefle.
