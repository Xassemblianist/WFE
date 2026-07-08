// WFE Tahmin API istemcisi + tip tanımları.
import { API_BASE, apiUrl } from './config'

export type FieldKey = 't2m' | 'wind' | 'precip' | 'cloud' | 'mslp'

export interface RegionInfo {
  id: string
  title: string
  desc: string
  default_hours: number
}

export interface StepInfo {
  step: number
  fhour: number
}

export interface Manifest {
  region: string
  title: string
  available: boolean
  init: string | null
  dx_m: number
  nx: number
  ny: number
  steps: StepInfo[]
  maps: string[]
  run?: string | null // arşivlenmiş koşu (YYYYMMDDHH) — null/undefined = güncel
  fields?: FieldKey[]
  corners?: [number, number][] // [TL, TR, BR, BL] her biri [lon, lat]
  bounds?: [number, number, number, number] // [w, s, e, n]
  center?: [number, number] // [lon, lat]
}

export interface RunInfo {
  cycle: string // YYYYMMDDHH
  init: string
  current: boolean
}

export interface SeriesPoint {
  valid: string
  fhour: number
  t2m_C: number | null
  wind10_ms: number | null
  precip_mm: number
}

export interface PointForecast {
  region: string
  lat: number
  lon: number
  grid: {
    i: number
    j: number
    elev_m: number
    grid_lat: number
    grid_lon: number
  }
  init: string
  series: SeriesPoint[]
  error?: string
}

export interface ColorStop {
  value: number
  rgba: [number, number, number, number]
}

export interface ColormapMeta {
  field: FieldKey
  label: string
  unit: string
  opacity: number
  stops: ColorStop[]
}

async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(apiUrl(path))
  if (!r.ok) throw new Error(`${r.status} ${path}`)
  return r.json() as Promise<T>
}

export const api = {
  base: API_BASE,
  health: () => getJSON<{ status: string; model: string; regions: string[] }>('/health'),
  regions: () => getJSON<RegionInfo[]>('/regions'),
  manifest: (region: string, run?: string | null) =>
    getJSON<Manifest>(`/products/${region}${run ? `?run=${run}` : ''}`),
  runs: (region: string) => getJSON<RunInfo[]>(`/runs/${region}`),
  colormaps: () => getJSON<ColormapMeta[]>('/colormap'),
  point: (region: string, lat: number, lon: number, run?: string | null) =>
    getJSON<PointForecast>(
      `/point/${region}?lat=${lat.toFixed(4)}&lon=${lon.toFixed(4)}${run ? `&run=${run}` : ''}`
    ),
  overlayUrl: (region: string, field: FieldKey, step: number, version?: string | null) =>
    apiUrl(`/overlay/${region}/${field}/${step}.png`) +
    (version ? `?v=${encodeURIComponent(version)}` : ''),
}
