// Veri-PNG kodlama aralıkları — server/overlay.py DATA_RANGES ile BİREBİR.
// Skaler alanlar 16-bit R/G paketli; rüzgâr bileşenleri 8-bit (R=u, G=v).
import type { FieldKey } from '../api'

export const DATA_RANGES: Record<FieldKey, [number, number]> = {
  t2m: [-45, 50],
  wind: [0, 45],
  precip: [0, 80],
  cloud: [0, 8],
  mslp: [960, 1050],
}

export const UV_RANGE = 45 // m/s, ± her iki bileşen

// server/terrain.py ELEV_RANGE ile birebir
export const ELEV_RANGE: [number, number] = [-500, 5500]

// t2m arazi-detaylandırma lapse oranı [K/m]
export const LAPSE_RATE = 0.0065
