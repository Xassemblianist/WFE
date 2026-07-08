import type { FieldKey, ColormapMeta } from '../api'

export interface FieldDef {
  key: FieldKey
  label: string
  short: string
  unit: string
  /** arayüz vurgu rengi (katman rayı, grafikler) */
  accent: string
  /** hover değeri biçimi */
  fmt: (v: number) => string
}

export const FIELD_DEFS: Record<FieldKey, FieldDef> = {
  t2m: {
    key: 't2m',
    label: 'Sıcaklık',
    short: '2 m sıcaklık',
    unit: '°C',
    accent: '#fb923c',
    fmt: (v) => `${v.toFixed(1)} °C`,
  },
  wind: {
    key: 'wind',
    label: 'Rüzgâr',
    short: '10 m rüzgâr',
    unit: 'm/s',
    accent: '#34d399',
    fmt: (v) => `${v.toFixed(1)} m/s`,
  },
  precip: {
    key: 'precip',
    label: 'Yağış',
    short: '3 saatlik yağış',
    unit: 'mm',
    accent: '#60a5fa',
    fmt: (v) => `${v.toFixed(1)} mm`,
  },
  cloud: {
    key: 'cloud',
    label: 'Bulut',
    short: 'Bulutluluk',
    unit: 'g/kg',
    accent: '#a5b4cb',
    fmt: (v) => `${v.toFixed(2)} g/kg`,
  },
  mslp: {
    key: 'mslp',
    label: 'Basınç',
    short: 'Deniz sv. basıncı',
    unit: 'hPa',
    accent: '#f472b6',
    fmt: (v) => `${Math.round(v)} hPa`,
  },
}

export const FIELD_ORDER: FieldKey[] = ['t2m', 'wind', 'precip', 'cloud', 'mslp']

/** Colormap stop'larından yatay CSS gradyanı (değere göre konumlanmış). */
export function gradientFromMeta(m: ColormapMeta): string {
  const vs = m.stops.map((s) => s.value)
  const lo = vs[0]
  const hi = vs[vs.length - 1]
  const span = hi - lo || 1
  const parts = m.stops.map((s) => {
    const pct = ((s.value - lo) / span) * 100
    const [r, g, b] = s.rgba
    return `rgb(${r},${g},${b}) ${pct.toFixed(1)}%`
  })
  return `linear-gradient(90deg, ${parts.join(', ')})`
}

/** Legend için düzgün etiket değerleri. */
export function legendTicks(m: ColormapMeta, count = 6): number[] {
  const vs = m.stops.map((s) => s.value)
  const lo = vs[0]
  const hi = vs[vs.length - 1]
  const out: number[] = []
  for (let i = 0; i < count; i++) out.push(lo + ((hi - lo) * i) / (count - 1))
  return out
}
