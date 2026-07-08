import type { PointForecast, SeriesPoint } from '../api'
import Meteogram from './Meteogram'
import { round1, windLabel, fmtValidLong } from '../lib/format'
import { IcClose } from './icons'
import { WSun, WPartly, WCloud, WRain, WWindy } from './icons'

interface Props {
  open: boolean
  loading: boolean
  data: PointForecast | null
  error: string | null
  title: string
  focusISO: string | null
  onClose: () => void
}

function nearestRow(data: PointForecast, focusISO: string | null): SeriesPoint | null {
  const rows = data.series.filter((s) => s.t2m_C !== null)
  if (!rows.length) return null
  if (!focusISO) return rows[0]
  const ft = new Date(focusISO).getTime()
  let best = rows[0]
  let bd = Infinity
  for (const r of rows) {
    const d = Math.abs(new Date(r.valid).getTime() - ft)
    if (d < bd) {
      bd = d
      best = r
    }
  }
  return best
}

function condition(row: SeriesPoint): { label: string; glyph: JSX.Element } {
  const t = row.t2m_C ?? 0
  const w = row.wind10_ms ?? 0
  const p = row.precip_mm
  if (p >= 1) return { label: 'Yağışlı', glyph: <WRain size={54} /> }
  if (p >= 0.2) return { label: 'Hafif yağış', glyph: <WRain size={54} /> }
  if (w >= 12) return { label: 'Rüzgârlı', glyph: <WWindy size={54} /> }
  if (t >= 28) return { label: 'Açık, sıcak', glyph: <WSun size={54} /> }
  if (t >= 16) return { label: 'Az bulutlu', glyph: <WPartly size={54} /> }
  return { label: 'Bulutlu', glyph: <WCloud size={54} /> }
}

export default function PointPanel({ open, loading, data, error, title, focusISO, onClose }: Props) {
  const focus = data ? nearestRow(data, focusISO) : null
  const cond = focus ? condition(focus) : null
  return (
    <aside className={`point-panel ${open ? 'open' : ''}`} aria-hidden={!open}>
      <div className="point-head">
        <div style={{ flex: 1 }}>
          <h3 className="point-title">{title}</h3>
          {data && (
            <div className="point-sub">
              {data.grid.grid_lat.toFixed(3)}°, {data.grid.grid_lon.toFixed(3)}° · yükseklik{' '}
              {Math.round(data.grid.elev_m)} m
            </div>
          )}
        </div>
        <button className="icon-btn" onClick={onClose} aria-label="Kapat" style={{ width: 34, height: 34 }}>
          <IcClose size={15} />
        </button>
      </div>

      <div className="point-body">
        {loading && (
          <div style={{ display: 'grid', placeItems: 'center', padding: '48px 0' }}>
            <div className="spinner" />
          </div>
        )}
        {!loading && error && <p className="muted">{error}</p>}
        {!loading && data && focus && cond && (
          <>
            <div className="point-hero">
              {cond.glyph}
              <div>
                <div className="big">{round1(focus.t2m_C)}°</div>
                <div className="cond">{cond.label}</div>
              </div>
            </div>
            {focusISO && <div className="point-time">{fmtValidLong(focusISO)}</div>}
            <div className="stat-row">
              <div className="stat">
                <div className="v">
                  {round1(focus.wind10_ms)}
                  <small>m/s</small>
                </div>
                <div className="k">Rüzgâr · {windLabel(focus.wind10_ms)}</div>
              </div>
              <div className="stat">
                <div className="v">
                  {round1(focus.precip_mm)}
                  <small>mm</small>
                </div>
                <div className="k">Yağış (3 sa)</div>
              </div>
              <div className="stat">
                <div className="v">
                  {Math.round(data.grid.elev_m)}
                  <small>m</small>
                </div>
                <div className="k">Rakım</div>
              </div>
            </div>
            <Meteogram series={data.series} />
          </>
        )}
        {!loading && data && !focus && <p className="muted">Bu nokta için tahmin serisi yok.</p>}
      </div>
    </aside>
  )
}
