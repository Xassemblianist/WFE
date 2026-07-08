# WFE — Hava Durumu Web Arayüzü

WFE tahmin modelinin (bkz. depo kökü) interaktif web arayüzü. React + Vite +
TypeScript + MapLibre GL + Recharts.

Öne çıkanlar (windy-tarzı istemci render motoru):

- **İstemci-tarafı alan render'ı** — model ham verisi (16-bit paketli veri-PNG)
  tarayıcıda çözülür, renk LUT'u + süperörnekleme + Bayer dithering ile boyanır.
- **Zamanda akıcı interpolasyon** — oynatma sırasında kareler arasında kesirli
  harmanlama (windy'deki pürüzsüz animasyon hissi).
- **Rüzgâr partikül animasyonu** — modelin gerçek alt-seviye u/v rüzgârıyla
  taşınan binlerce akış çizgisi (rüzgâr katmanında).
- **İmleç altında değer okuma** — haritada gezdirin, o noktadaki değeri görün
  (ters-bilineer quad projeksiyonu ile).
- Nokta meteogramı, şehir arama, bölge geçişi (Türkiye 6 km ↔ Antalya 2.5/1 km),
  koyu/açık tema, responsive, Türkçe.

## Hızlı başlangıç

```bash
# 1) API (depo kökünden, ayrı terminal)
python -m uvicorn app:app --app-dir server --port 8000

# 2) Web arayüzü (bu dizinde)
npm install
npm run dev            # http://localhost:5173
```

## API tabanını yapılandırma

Varsayılan `http://localhost:8000`. Farklı sunucu için `.env` oluşturun
(`.env.example`'ı kopyalayın): `VITE_API_BASE=http://sunucu:8000`

Kullanılan uç noktalar:

| Uç nokta | Kullanım |
|---|---|
| `GET /regions`, `GET /products/{bölge}` | bölgeler + manifest (adımlar, bounds/**corners**, fields, init) |
| `GET /data/{bölge}/{alan}/{adım}.png` | ham alan verisi — 16-bit R/G paketli (istemci render) |
| `GET /uv/{bölge}/{adım}.png` | u/v rüzgâr bileşenleri — 8-bit (partiküller) |
| `GET /colormap` | renk skalaları + değer aralıkları (LUT + legend) |
| `GET /point/{bölge}?lat=&lon=[&run=]` | nokta tahmini zaman serisi (meteogram) |
| `GET /runs/{bölge}` | gezinebilir koşular: güncel + arşiv (geçmiş tahminler) |
| `GET /terrain/{bölge}/{model\|hires}.png` | yükseklik alanları (t2m arazi-detaylandırma) |
| `GET /overlay/...` | sunucu-tarafı renkli PNG (eski arayüz / yedek) |

`?run=YYYYMMDDHH` parametresi `products`/`data`/`point` uçlarında geçmiş
(arşivlenmiş) koşuyu seçer — arşivde yalnız yüzey alanları (t2m/rüzgâr/yağış)
bulunur. Arşiv, `tools/run_operational.py` döngüsünce her koşuda beslenir.

> Veri-PNG kodlama aralıkları `server/overlay.py DATA_RANGES` ile
> `src/lib/ranges.ts` arasında birebir eşleşmelidir.

## Komutlar

```bash
npm run dev        # geliştirme (HMR)
npm run build      # tsc + üretim derlemesi (dist/)
npm run preview    # üretim önizleme
```

## Yapı

```
src/
  api.ts               API istemcisi + tipler
  config.ts            API tabanı (VITE_API_BASE)
  lib/
    ranges.ts          veri-PNG aralıkları (backend ile aynı)
    gridLoader.ts      veri-PNG → Float32Array (img→canvas çözümü)
    lut.ts             renk skalası → 1024'lük LUT
    painter.ts         alan boyacısı (zaman harmanı + süperörnekleme + dither)
    quad.ts            dört-köşe georeferans (ileri/ters bilineer)
    particles.ts       rüzgâr partikül katmanı
  components/          MapView (motor entegrasyonu) · LayerRail · TimeBar ·
                       SearchBar · PointPanel · Meteogram · icons
  pages/               Ana Sayfa · Harita · Hakkında
  data/cities.json     Türkiye şehir/ilçe merkezleri
```
