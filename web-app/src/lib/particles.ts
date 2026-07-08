// Rüzgâr partikül animasyonu (windy-tarzı akış çizgileri).
// Partiküller quad-uzayında (s,t ∈ [0,1]²) yaşar; her karede model rüzgârıyla
// taşınır ve ekrana harita projeksiyonuyla çizilir — pan/zoom otomatik izlenir.
import type { Map as MLMap } from 'maplibre-gl'
import { quadPoint, type Quad } from './quad'
import { sampleGrid, type UVGrid } from './gridLoader'

interface P {
  s: number
  t: number
  age: number
  px: number // önceki ekran x (CSS px), NaN = henüz yok
  py: number
}

// Fiziksel hız → animasyon: u [m/s] / SPEED_K = quad-oranı/kare.
// Domain genişliğinden bağımsız benzer ekran hızı verir (bkz. dt oranlaması).
const SPEED_K = 16000

export class WindParticles {
  private map: MLMap
  private canvas: HTMLCanvasElement
  private ctx: CanvasRenderingContext2D
  private parts: P[] = []
  private uv: UVGrid | null = null
  private quad: Quad | null = null
  private aspect = 1 // metersX / metersY
  private raf = 0
  private running = false
  private skipDraw = false
  private color = 'rgba(235,245,255,0.85)'
  private onMove = () => {
    this.clear()
    this.skipDraw = true
  }
  private onResize = () => this.resize()

  constructor(map: MLMap) {
    this.map = map
    this.canvas = document.createElement('canvas')
    Object.assign(this.canvas.style, {
      position: 'absolute',
      inset: '0',
      pointerEvents: 'none',
      zIndex: '3',
    } as CSSStyleDeclaration)
    map.getContainer().appendChild(this.canvas)
    this.ctx = this.canvas.getContext('2d')!
    this.resize()
    map.on('move', this.onMove)
    map.on('resize', this.onResize)
  }

  setTheme(dark: boolean) {
    this.color = dark ? 'rgba(235,245,255,0.85)' : 'rgba(20,35,60,0.72)'
  }

  setWind(uv: UVGrid | null, quad: Quad | null, metersX: number, metersY: number) {
    this.uv = uv
    this.quad = quad
    this.aspect = metersY > 0 ? metersX / metersY : 1
    if (uv && this.parts.length === 0) this.seed()
  }

  private seed() {
    const n = Math.min(2800, Math.round((this.canvas.clientWidth * this.canvas.clientHeight) / 420))
    this.parts = Array.from({ length: Math.max(600, n) }, () => this.spawn())
  }

  private spawn(): P {
    return { s: Math.random(), t: Math.random(), age: 60 + Math.random() * 160, px: NaN, py: NaN }
  }

  private resize() {
    const c = this.map.getContainer()
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    this.canvas.width = c.clientWidth * dpr
    this.canvas.height = c.clientHeight * dpr
    this.canvas.style.width = c.clientWidth + 'px'
    this.canvas.style.height = c.clientHeight + 'px'
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    this.clear()
  }

  clear() {
    this.ctx.save()
    this.ctx.setTransform(1, 0, 0, 1, 0, 0)
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    this.ctx.restore()
    for (const p of this.parts) {
      p.px = NaN
      p.py = NaN
    }
  }

  start() {
    if (this.running) return
    this.running = true
    const tick = () => {
      if (!this.running) return
      this.frame()
      this.raf = requestAnimationFrame(tick)
    }
    this.raf = requestAnimationFrame(tick)
  }

  stop() {
    this.running = false
    cancelAnimationFrame(this.raf)
    this.clear()
  }

  destroy() {
    this.stop()
    this.map.off('move', this.onMove)
    this.map.off('resize', this.onResize)
    this.canvas.remove()
  }

  private frame() {
    const { uv, quad, ctx } = this
    if (!uv || !quad) return
    const w = this.canvas.clientWidth
    const h = this.canvas.clientHeight

    // iz soldurma
    ctx.globalCompositeOperation = 'destination-in'
    ctx.fillStyle = 'rgba(0,0,0,0.94)'
    ctx.fillRect(0, 0, w, h)
    ctx.globalCompositeOperation = 'source-over'
    ctx.strokeStyle = this.color
    ctx.lineWidth = 1.15
    ctx.lineCap = 'round'
    ctx.beginPath()

    const draw = !this.skipDraw
    this.skipDraw = false

    for (const p of this.parts) {
      const gu = sampleGrid(uv.u, uv.w, uv.h, p.s * (uv.w - 1), p.t * (uv.h - 1))
      const gv = sampleGrid(uv.v, uv.w, uv.h, p.s * (uv.w - 1), p.t * (uv.h - 1))
      p.s += gu / SPEED_K
      p.t -= (gv * this.aspect) / SPEED_K // t ekseni güneye artar
      p.age -= 1
      if (p.s < 0 || p.s > 1 || p.t < 0 || p.t > 1 || p.age <= 0) {
        Object.assign(p, this.spawn())
        continue
      }
      const [lng, lat] = quadPoint(quad, p.s, p.t)
      const pt = this.map.project([lng, lat])
      if (draw && !Number.isNaN(p.px) && Math.abs(pt.x - p.px) < 40 && Math.abs(pt.y - p.py) < 40) {
        ctx.moveTo(p.px, p.py)
        ctx.lineTo(pt.x, pt.y)
      }
      p.px = pt.x
      p.py = pt.y
    }
    ctx.stroke()
  }
}
