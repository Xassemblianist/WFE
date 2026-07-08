// Renk skalası → 1024-girişli arama tablosu (LUT).
import type { ColormapMeta } from '../api'

export interface LUT {
  rgba: Uint8ClampedArray // 1024 * 4
  lo: number
  hi: number
}

export function buildLUT(meta: ColormapMeta): LUT {
  const stops = meta.stops
  const lo = stops[0].value
  const hi = stops[stops.length - 1].value
  const n = 1024
  const rgba = new Uint8ClampedArray(n * 4)
  let si = 0
  for (let i = 0; i < n; i++) {
    const v = lo + ((hi - lo) * i) / (n - 1)
    while (si < stops.length - 2 && v > stops[si + 1].value) si++
    const a = stops[si]
    const b = stops[si + 1]
    const f = b.value === a.value ? 0 : Math.min(1, Math.max(0, (v - a.value) / (b.value - a.value)))
    rgba[i * 4] = a.rgba[0] + (b.rgba[0] - a.rgba[0]) * f
    rgba[i * 4 + 1] = a.rgba[1] + (b.rgba[1] - a.rgba[1]) * f
    rgba[i * 4 + 2] = a.rgba[2] + (b.rgba[2] - a.rgba[2]) * f
    rgba[i * 4 + 3] = (a.rgba[3] + (b.rgba[3] - a.rgba[3]) * f) * 255
  }
  return { rgba, lo, hi }
}
