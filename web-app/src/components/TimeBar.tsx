import { useMemo } from 'react'
import type { StepInfo, ColormapMeta } from '../api'
import { gradientFromMeta, legendTicks } from '../lib/fields'
import { fmtValidLong } from '../lib/format'
import { IcPlay, IcPause } from './icons'

export const SPEEDS = [
  { label: '0.5×', sps: 0.55 },
  { label: '1×', sps: 1.1 },
  { label: '2×', sps: 2.2 },
]

interface Props {
  steps: StepInfo[]
  initISO: string | null
  timePos: number // kesirli adım index'i
  playing: boolean
  speedIdx: number
  meta: ColormapMeta | null
  onScrub: (v: number) => void
  onTogglePlay: () => void
  onSpeed: (i: number) => void
}

function validAt(initISO: string, steps: StepInfo[], pos: number): Date {
  const i0 = Math.min(Math.floor(pos), steps.length - 1)
  const i1 = Math.min(i0 + 1, steps.length - 1)
  const f = pos - i0
  const fh = steps[i0].fhour + (steps[i1].fhour - steps[i0].fhour) * f
  return new Date(new Date(initISO).getTime() + fh * 3600 * 1000)
}

const DAY_NAMES = ['Paz', 'Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt']
const MONTHS = ['Oca', 'Şub', 'Mar', 'Nis', 'May', 'Haz', 'Tem', 'Ağu', 'Eyl', 'Eki', 'Kas', 'Ara']

export default function TimeBar({
  steps,
  initISO,
  timePos,
  playing,
  speedIdx,
  meta,
  onScrub,
  onTogglePlay,
  onSpeed,
}: Props) {
  const max = Math.max(steps.length - 1, 0)
  const pos = Math.min(timePos, max)

  // Gün sınırı işaretleri (yerel gece yarısına en yakın adımlar arası konum)
  const dayMarks = useMemo(() => {
    if (!initISO || steps.length < 2) return []
    const out: { pct: number; label: string }[] = []
    const t0 = validAt(initISO, steps, 0).getTime()
    const t1 = validAt(initISO, steps, max).getTime()
    const d = new Date(t0)
    d.setHours(0, 0, 0, 0)
    d.setDate(d.getDate() + 1)
    while (d.getTime() < t1) {
      const pct = ((d.getTime() - t0) / (t1 - t0)) * 100
      out.push({ pct, label: `${DAY_NAMES[d.getDay()]} ${d.getDate()} ${MONTHS[d.getMonth()]}` })
      d.setDate(d.getDate() + 1)
    }
    return out
  }, [initISO, steps, max])

  if (!steps.length) return null
  const cur = initISO ? validAt(initISO, steps, pos) : null
  const i0 = Math.min(Math.floor(pos), max)
  const fh = steps[i0].fhour + (steps[Math.min(i0 + 1, max)].fhour - steps[i0].fhour) * (pos - i0)
  const pct = max > 0 ? (pos / max) * 100 : 0
  const ticks = meta ? legendTicks(meta, 6) : []
  const fmtTick = (v: number) =>
    Math.abs(v) >= 100 ? Math.round(v).toString() : Math.abs(v) >= 10 ? Math.round(v).toString() : (Math.round(v * 10) / 10).toString()

  return (
    <div className="timebar glass">
      {meta && (
        <div className="legend-strip">
          <div className="legend-grad" style={{ background: gradientFromMeta(meta) }} />
          <div className="legend-ticks">
            {ticks.map((t, i) => (
              <span key={i}>{fmtTick(t)}</span>
            ))}
          </div>
          <div className="legend-unit">{meta.unit}</div>
        </div>
      )}
      <div className="timebar-main">
        <button className="play-btn" onClick={onTogglePlay} aria-label={playing ? 'Duraklat' : 'Oynat'}>
          {playing ? <IcPause size={17} /> : <IcPlay size={17} />}
        </button>
        <div className="time-readout">
          <div className="valid">{cur ? fmtValidLong(cur.toISOString()) : `t+${fh.toFixed(0)} s`}</div>
          <div className="fhour">Tahmin +{Math.round(fh)} saat</div>
        </div>
        <div className="time-track">
          <div className="time-progress" style={{ width: `calc(${pct}% )` }} />
          <input
            type="range"
            min={0}
            max={max}
            step={0.01}
            value={pos}
            onChange={(e) => onScrub(Number(e.target.value))}
            aria-label="Tahmin zamanı"
          />
          <div className="time-days">
            {dayMarks.map((m, i) => (
              <span key={i}>
                <span className="tick" style={{ left: `${m.pct}%` }} />
                <span className="day" style={{ left: `${m.pct}%` }}>
                  {m.label}
                </span>
              </span>
            ))}
          </div>
        </div>
        <div className="speed-seg">
          {SPEEDS.map((s, i) => (
            <button key={s.label} className={i === speedIdx ? 'active' : ''} onClick={() => onSpeed(i)}>
              {s.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
