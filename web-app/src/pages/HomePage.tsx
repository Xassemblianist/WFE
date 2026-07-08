import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type Manifest, type SeriesPoint } from '../api'
import { round1 } from '../lib/format'
import { IcWind, IcRain, IcLayers, IcPin, IcThermo, IcArrowRight, IcGithub } from '../components/icons'
import { WSun, WPartly, WCloud, WRain, WWindy } from '../components/icons'

interface Featured {
  name: string
  province: string
  lat: number
  lon: number
}

const FEATURED: Featured[] = [
  { name: 'Antalya', province: 'Antalya', lat: 36.897, lon: 30.713 },
  { name: 'Alanya', province: 'Antalya', lat: 36.544, lon: 31.999 },
  { name: 'İstanbul', province: 'İstanbul', lat: 41.008, lon: 28.978 },
  { name: 'Ankara', province: 'Ankara', lat: 39.933, lon: 32.859 },
  { name: 'İzmir', province: 'İzmir', lat: 38.423, lon: 27.143 },
  { name: 'Bursa', province: 'Bursa', lat: 40.188, lon: 29.061 },
]

interface CityNow extends Featured {
  temp: number | null
  wind: number | null
  precip: number
  glyph: 'sun' | 'partly' | 'cloud' | 'rain' | 'windy'
  spark: number[]
}

const GLYPHS = {
  sun: WSun,
  partly: WPartly,
  cloud: WCloud,
  rain: WRain,
  windy: WWindy,
}

function glyphFor(temp: number | null, precip: number, wind: number | null): CityNow['glyph'] {
  if (precip >= 0.5) return 'rain'
  if ((wind ?? 0) >= 12) return 'windy'
  if ((temp ?? 0) >= 27) return 'sun'
  if ((temp ?? 0) >= 15) return 'partly'
  return 'cloud'
}

function inBounds(m: Manifest | undefined, lat: number, lon: number): boolean {
  if (!m?.bounds) return false
  const [w, s, e, n] = m.bounds
  return lon >= w && lon <= e && lat >= s && lat <= n
}

function Sparkline({ values, color }: { values: number[]; color: string }) {
  if (values.length < 2) return null
  const w = 150
  const h = 30
  const mn = Math.min(...values)
  const mx = Math.max(...values)
  const span = mx - mn || 1
  const pts = values
    .map((v, i) => `${((i / (values.length - 1)) * w).toFixed(1)},${(h - 3 - ((v - mn) / span) * (h - 6)).toFixed(1)}`)
    .join(' ')
  return (
    <svg className="spark" width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" opacity="0.85" />
    </svg>
  )
}

export default function HomePage() {
  const nav = useNavigate()
  const [cards, setCards] = useState<CityNow[] | null>(null)
  const [apiDown, setApiDown] = useState(false)

  useEffect(() => {
    ;(async () => {
      try {
        const regs = await api.regions()
        const mans: Record<string, Manifest> = {}
        await Promise.all(
          regs.map(async (r) => {
            try {
              mans[r.id] = await api.manifest(r.id)
            } catch {
              /* yoksay */
            }
          })
        )
        const now = Date.now()
        const results = await Promise.all(
          FEATURED.map(async (c): Promise<CityNow> => {
            const cand = Object.values(mans)
              .filter((m) => m.available && inBounds(m, c.lat, c.lon))
              .sort((a, b) => (a.dx_m || 0) - (b.dx_m || 0))
            const region = cand[0]?.region
            let temp: number | null = null
            let wind: number | null = null
            let precip = 0
            let spark: number[] = []
            if (region) {
              try {
                const d = await api.point(region, c.lat, c.lon)
                const rows: SeriesPoint[] = d.series?.filter((s) => s.t2m_C !== null) ?? []
                if (rows.length) {
                  let best = rows[0]
                  let bd = Infinity
                  for (const r of rows) {
                    const dd = Math.abs(new Date(r.valid).getTime() - now)
                    if (dd < bd) {
                      bd = dd
                      best = r
                    }
                  }
                  temp = best.t2m_C
                  wind = best.wind10_ms
                  precip = best.precip_mm
                  spark = rows.map((r) => r.t2m_C!) as number[]
                }
              } catch {
                /* yoksay */
              }
            }
            return { ...c, temp, wind, precip, glyph: glyphFor(temp, precip, wind), spark }
          })
        )
        setCards(results)
      } catch {
        setApiDown(true)
      }
    })()
  }, [])

  const openCity = (c: Featured) =>
    nav(`/harita?lat=${c.lat}&lon=${c.lon}&name=${encodeURIComponent(c.name)}`)

  return (
    <div className="page">
      <div className="container">
        <section className="hero">
          <div className="kicker">
            <span className="pulse" />
            Operasyonel · günde 4 döngü
          </div>
          <h1>
            Türkiye için <span className="grad">GPU-yerlisi</span> hava tahmini
          </h1>
          <p>
            WFE, sıfırdan C++/CUDA ile yazılmış bölgesel sayısal hava tahmin modelidir. Gerçek GFS
            verisiyle beslenir, tek bir RTX GPU'da koşar; Türkiye ve Antalya için gerçek arazili,
            yüksek çözünürlüklü tahmin üretir.
          </p>
          <div className="cta">
            <button className="btn primary" onClick={() => nav('/harita')}>
              Haritayı Aç <IcArrowRight size={16} />
            </button>
            <button className="btn" onClick={() => nav('/hakkinda')}>
              Model Hakkında
            </button>
          </div>
          <div className="hero-stats">
            <div className="hstat">
              <b>6 km</b>
              <span>Türkiye çözünürlüğü</span>
            </div>
            <div className="hstat">
              <b>2.5 km</b>
              <span>Antalya, gerçek arazi</span>
            </div>
            <div className="hstat">
              <b>+%53</b>
              <span>rüzgâr becerisi @48 sa*</span>
            </div>
            <div className="hstat">
              <b>~0</b>
              <span>rüzgâr yanlılığı (METAR)</span>
            </div>
          </div>
        </section>

        <h2 className="section-title">Öne çıkan şehirler</h2>
        {apiDown && (
          <p className="muted">
            API'ye ulaşılamadı — <code>localhost:8000</code> çalışıyor mu?
          </p>
        )}
        <div className="city-grid">
          {(cards ?? FEATURED.map((f) => ({ ...f, temp: null, wind: null, precip: 0, glyph: 'partly' as const, spark: [] }))).map(
            (c, i) => {
              const G = GLYPHS[c.glyph]
              return (
                <button key={i} className="city-card" onClick={() => openCity(c)}>
                  <div className="top">
                    <div>
                      <div className="name">{c.name}</div>
                      <div className="prov">{c.province}</div>
                    </div>
                    <G size={32} />
                  </div>
                  <div className="temp">{cards ? `${round1(c.temp)}°` : '—'}</div>
                  <Sparkline values={c.spark} color="var(--ac)" />
                  <div className="meta">
                    <span>
                      <IcWind size={13} />
                      {cards ? `${round1(c.wind)} m/s` : '—'}
                    </span>
                    <span>
                      <IcRain size={13} />
                      {cards ? `${round1(c.precip)} mm` : '—'}
                    </span>
                  </div>
                </button>
              )
            }
          )}
        </div>

        <h2 className="section-title">Neler sunuyor</h2>
        <div className="feature-grid">
          <div className="feature">
            <div className="ico">
              <IcLayers size={20} />
            </div>
            <h3>Canlı tahmin katmanları</h3>
            <p>
              Sıcaklık, rüzgâr, yağış, bulut ve basınç — model ham çıktısından tarayıcıda
              renklendirilir, zaman içinde akıcı biçimde interpolasyonlanır.
            </p>
          </div>
          <div className="feature">
            <div className="ico">
              <IcWind size={20} />
            </div>
            <h3>Rüzgâr akış animasyonu</h3>
            <p>
              Rüzgâr katmanında binlerce partikül gerçek model rüzgârıyla taşınır — akışı, konverjansı
              ve dağ etkilerini bir bakışta görün.
            </p>
          </div>
          <div className="feature">
            <div className="ico">
              <IcPin size={20} />
            </div>
            <h3>Nokta tahmini</h3>
            <p>
              Haritaya tıklayın veya şehir arayın; saatlik sıcaklık, rüzgâr ve yağış meteogramı anında
              açılır. İmleci gezdirin, değeri okuyun.
            </p>
          </div>
          <div className="feature">
            <div className="ico">
              <IcThermo size={20} />
            </div>
            <h3>Doğrulanmış model</h3>
            <p>
              GFS analizi ve METAR istasyon gözlemlerine karşı çok-döngülü doğrulama; rüzgârda
              kalıcılık tahminini her döngüde yener.
            </p>
          </div>
        </div>

        <footer className="site-footer">
          <span>WFE — sıfırdan yazılmış C++/CUDA bölgesel NWP modeli. *GFS f048'e karşı u-rüzgâr RMSE iyileşmesi.</span>
          <a href="https://github.com/Xassemblianist/WFE" target="_blank" rel="noreferrer">
            <IcGithub size={15} /> GitHub
          </a>
        </footer>
      </div>
    </div>
  )
}
