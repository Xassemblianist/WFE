// Skaler alan boyacısı: iki zaman adımını harmanlar (temporal interpolasyon),
// süperörnekleme + Bayer dithering ile kendi canvas'ına çizer. Canvas maplibre
// `canvas` kaynağı olarak haritaya bindirilir.
import type { Grid } from './gridLoader'
import type { LUT } from './lut'

// 4×4 Bayer matrisi — bantlaşmayı kırar (LUT indeksine ±~2 giriş)
const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]

export class FieldPainter {
  readonly canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private img: ImageData | null = null
  private blend: Float32Array | null = null
  private gw = 0
  private gh = 0
  private ss = 2

  constructor() {
    this.canvas = document.createElement('canvas')
    this.ctx = this.canvas.getContext('2d')!
  }

  /** Izgara boyutuna göre yapılandır. Boyut değiştiyse true döner (kaynak yeniden kurulmalı). */
  configure(gw: number, gh: number): boolean {
    // Hedef ~700px genişlik: küçük ızgaralar daha çok süperörneklenir
    const ss = Math.min(5, Math.max(2, Math.round(700 / gw)))
    const W = gw * ss
    const H = gh * ss
    if (gw === this.gw && gh === this.gh && this.canvas.width === W) return false
    this.gw = gw
    this.gh = gh
    this.ss = ss
    this.canvas.width = W
    this.canvas.height = H
    this.img = this.ctx.createImageData(W, H)
    this.blend = new Float32Array(gw * gh)
    return true
  }

  /**
   * g0/g1 arasında frac ile harmanla ve boya. g1 yoksa yalnız g0.
   * Dönen değer: örneklenebilir harman ızgarası (hover okuması için).
   */
  paint(g0: Grid, g1: Grid | null, frac: number, lut: LUT): Float32Array {
    this.configure(g0.w, g0.h)
    const { gw, gh, ss } = this
    const blend = this.blend!
    const d0 = g0.data
    if (g1 && frac > 0.001) {
      const d1 = g1.data
      const a = frac
      for (let i = 0; i < blend.length; i++) blend[i] = d0[i] + (d1[i] - d0[i]) * a
    } else {
      blend.set(d0)
    }

    const img = this.img!
    const px = img.data
    const { rgba, lo, hi } = lut
    const scale = 1023 / (hi - lo)
    const W = gw * ss
    const H = gh * ss
    const inv = 1 / ss
    let p = 0
    for (let y = 0; y < H; y++) {
      const gy = Math.min(y * inv, gh - 1.001)
      const y0 = gy | 0
      const fy = gy - y0
      const row0 = y0 * gw
      const row1 = row0 + gw
      const by = (y & 3) * 4
      for (let x = 0; x < W; x++) {
        const gx = Math.min(x * inv, gw - 1.001)
        const x0 = gx | 0
        const fx = gx - x0
        const a = blend[row0 + x0]
        const b = blend[row0 + x0 + 1]
        const c = blend[row1 + x0]
        const d = blend[row1 + x0 + 1]
        const v = a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy
        let idx = (v - lo) * scale + (BAYER[by + (x & 3)] - 7.5) * 0.25
        if (idx < 0) idx = 0
        else if (idx > 1023) idx = 1023
        const q = (idx | 0) * 4
        px[p] = rgba[q]
        px[p + 1] = rgba[q + 1]
        px[p + 2] = rgba[q + 2]
        px[p + 3] = rgba[q + 3]
        p += 4
      }
    }
    this.ctx.putImageData(img, 0, 0)
    return blend
  }
}
