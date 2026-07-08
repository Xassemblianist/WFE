import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts'
import type { SeriesPoint } from '../api'
import { fmtValidLong, fmtHourOnly } from '../lib/format'

interface Row {
  t: number
  label: string
  temp: number | null
  wind: number | null
  precip: number
}

function buildRows(series: SeriesPoint[]): Row[] {
  return series
    .filter((s) => s.t2m_C !== null)
    .map((s) => ({
      t: new Date(s.valid).getTime(),
      label: s.valid,
      temp: s.t2m_C,
      wind: s.wind10_ms,
      precip: s.precip_mm,
    }))
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function TipBox({ active, payload, unit }: any) {
  if (!active || !payload?.length) return null
  const p = payload[0].payload as Row
  return (
    <div
      style={{
        background: 'var(--glass-strong)',
        backdropFilter: 'blur(12px)',
        border: '1px solid var(--line)',
        borderRadius: 10,
        padding: '7px 11px',
        fontSize: 12,
        boxShadow: 'var(--shadow-sm)',
      }}
    >
      <div style={{ color: 'var(--tx3)', marginBottom: 3, fontWeight: 560 }}>{fmtValidLong(p.label)}</div>
      <b style={{ fontSize: 14 }}>
        {payload[0].value}
        {unit}
      </b>
    </div>
  )
}

const axis = { fill: 'var(--tx3)', fontSize: 10, fontWeight: 600 }

export default function Meteogram({ series }: { series: SeriesPoint[] }) {
  const rows = buildRows(series)
  if (rows.length < 2) return <p className="muted">Bu nokta için yeterli veri yok.</p>
  const ticks = rows.filter((r) => new Date(r.t).getHours() % 6 === 0).map((r) => r.t)
  const xProps = {
    dataKey: 't',
    type: 'number' as const,
    scale: 'time' as const,
    domain: ['dataMin', 'dataMax'] as [string, string],
    ticks,
    tickFormatter: (t: number) => fmtHourOnly(new Date(t).toISOString()),
    tick: axis,
    tickLine: false,
    axisLine: false,
  }
  const grid = <CartesianGrid stroke="var(--line)" vertical={false} />

  return (
    <div>
      <div className="chart-block">
        <h4>
          <span className="dot" style={{ background: '#fb923c' }} /> Sıcaklık (°C)
        </h4>
        <ResponsiveContainer width="100%" height={132}>
          <AreaChart data={rows} margin={{ top: 8, right: 4, left: -22, bottom: 0 }}>
            <defs>
              <linearGradient id="tg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#fb923c" stopOpacity={0.5} />
                <stop offset="100%" stopColor="#fb923c" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            {grid}
            <XAxis {...xProps} />
            <YAxis tick={axis} tickLine={false} axisLine={false} width={36} domain={['auto', 'auto']} />
            <Tooltip content={(p) => <TipBox {...p} unit="°C" />} cursor={{ stroke: 'var(--line2)' }} />
            <Area type="monotone" dataKey="temp" stroke="#fb923c" strokeWidth={2.2} fill="url(#tg)" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-block">
        <h4>
          <span className="dot" style={{ background: '#60a5fa' }} /> Yağış (mm / 3 sa)
        </h4>
        <ResponsiveContainer width="100%" height={104}>
          <BarChart data={rows} margin={{ top: 8, right: 4, left: -22, bottom: 0 }}>
            {grid}
            <XAxis {...xProps} />
            <YAxis tick={axis} tickLine={false} axisLine={false} width={36} allowDecimals={false} domain={[0, 'auto']} />
            <Tooltip content={(p) => <TipBox {...p} unit=" mm" />} cursor={{ fill: 'var(--glass-soft)' }} />
            <Bar dataKey="precip" fill="#60a5fa" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-block">
        <h4>
          <span className="dot" style={{ background: '#34d399' }} /> Rüzgâr (m/s)
        </h4>
        <ResponsiveContainer width="100%" height={104}>
          <AreaChart data={rows} margin={{ top: 8, right: 4, left: -22, bottom: 0 }}>
            <defs>
              <linearGradient id="wg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#34d399" stopOpacity={0.4} />
                <stop offset="100%" stopColor="#34d399" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            {grid}
            <XAxis {...xProps} />
            <YAxis tick={axis} tickLine={false} axisLine={false} width={36} domain={[0, 'auto']} />
            <Tooltip content={(p) => <TipBox {...p} unit=" m/s" />} cursor={{ stroke: 'var(--line2)' }} />
            <Area type="monotone" dataKey="wind" stroke="#34d399" strokeWidth={2.2} fill="url(#wg)" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
