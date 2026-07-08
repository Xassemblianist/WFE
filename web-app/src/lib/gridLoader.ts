// Veri-PNG yükleyici: <img crossOrigin> → canvas → Float32Array.
// fetch() yerine <img> kullanılır (her ortamda çalışır, HTTP önbelleği bedava).
// Görüntü satır-0 = kuzey; ızgara dizileri de bu sırada tutulur.
import { apiUrl } from '../config'
import { DATA_RANGES, UV_RANGE, ELEV_RANGE } from './ranges'
import type { FieldKey } from '../api'

export interface Grid {
  data: Float32Array
  w: number
  h: number
  /** kaynak görüntü — WebGL boyacı dokuları doğrudan bundan yükler */
  img: HTMLImageElement
}

export interface UVGrid {
  u: Float32Array
  v: Float32Array
  w: number
  h: number
}

const scalarCache = new Map<string, Promise<Grid | null>>()
const uvCache = new Map<string, Promise<UVGrid | null>>()

function loadImage(url: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => resolve(null)
    img.src = url
  })
}

function readPixels(img: HTMLImageElement): Uint8ClampedArray {
  const cv = document.createElement('canvas')
  cv.width = img.naturalWidth
  cv.height = img.naturalHeight
  const ctx = cv.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(img, 0, 0)
  return ctx.getImageData(0, 0, cv.width, cv.height).data
}

export function loadScalar(
  region: string,
  field: FieldKey,
  step: number,
  version: string | null,
  run?: string | null
): Promise<Grid | null> {
  const key = `${region}/${field}/${step}/${version ?? ''}/${run ?? ''}`
  let p = scalarCache.get(key)
  if (!p) {
    const params = new URLSearchParams()
    if (version) params.set('v', version)
    if (run) params.set('run', run)
    const q = params.toString()
    const url = apiUrl(`/data/${region}/${field}/${step}.png`) + (q ? `?${q}` : '')
    p = loadImage(url).then((img) => {
      if (!img) return null
      return decode16(img, DATA_RANGES[field])
    })
    scalarCache.set(key, p)
  }
  return p
}

function decode16(img: HTMLImageElement, [lo, hi]: [number, number]): Grid {
  const px = readPixels(img)
  const w = img.naturalWidth
  const h = img.naturalHeight
  const span = (hi - lo) / 65535
  const out = new Float32Array(w * h)
  for (let i = 0, j = 0; i < out.length; i++, j += 4) {
    out[i] = lo + (px[j] * 256 + px[j + 1]) * span
  }
  return { data: out, w, h, img }
}

const elevCache = new Map<string, Promise<Grid | null>>()

/** Yükseklik alanı (model / hires) — t2m arazi-detaylandırma için. */
export function loadElev(region: string, kind: 'model' | 'hires'): Promise<Grid | null> {
  const key = `${region}/${kind}`
  let p = elevCache.get(key)
  if (!p) {
    p = loadImage(apiUrl(`/terrain/${region}/${kind}.png`)).then((img) =>
      img ? decode16(img, ELEV_RANGE) : null
    )
    elevCache.set(key, p)
  }
  return p
}

export function loadUV(region: string, step: number, version: string | null): Promise<UVGrid | null> {
  const key = `${region}/${step}/${version ?? ''}`
  let p = uvCache.get(key)
  if (!p) {
    const url =
      apiUrl(`/uv/${region}/${step}.png`) + (version ? `?v=${encodeURIComponent(version)}` : '')
    p = loadImage(url).then((img) => {
      if (!img) return null
      const px = readPixels(img)
      const w = img.naturalWidth
      const h = img.naturalHeight
      const u = new Float32Array(w * h)
      const v = new Float32Array(w * h)
      const k = (2 * UV_RANGE) / 255
      for (let i = 0, j = 0; i < u.length; i++, j += 4) {
        u[i] = -UV_RANGE + px[j] * k
        v[i] = -UV_RANGE + px[j + 1] * k
      }
      return { u, v, w, h }
    })
    uvCache.set(key, p)
  }
  return p
}

/** (gx, gy) kesirli ızgara koordinatında bilineer örnekleme (kenar-kelepçeli). */
export function sampleGrid(g: Float32Array, w: number, h: number, gx: number, gy: number): number {
  const x = Math.min(Math.max(gx, 0), w - 1.001)
  const y = Math.min(Math.max(gy, 0), h - 1.001)
  const x0 = x | 0
  const y0 = y | 0
  const fx = x - x0
  const fy = y - y0
  const i = y0 * w + x0
  const a = g[i]
  const b = g[i + 1]
  const c = g[i + w]
  const d = g[i + w + 1]
  return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy
}
