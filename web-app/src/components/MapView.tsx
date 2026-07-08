import { useEffect, useMemo, useRef } from 'react'
import maplibregl from 'maplibre-gl'
import { BASEMAP } from '../lib/basemap'
import { loadScalar, loadUV, loadElev, sampleGrid, type Grid } from '../lib/gridLoader'
import { buildLUT } from '../lib/lut'
import { FieldPainter } from '../lib/painter'
import { GLPainter } from '../lib/glpainter'
import { WindParticles } from '../lib/particles'
import { quadInverse, type Quad } from '../lib/quad'
import { FIELD_DEFS } from '../lib/fields'
import { ELEV_RANGE, DATA_RANGES, LAPSE_RATE } from '../lib/ranges'
import type { Manifest, ColormapMeta, FieldKey } from '../api'

type Coords4 = [[number, number], [number, number], [number, number], [number, number]]

interface Props {
  theme: 'dark' | 'light'
  manifest: Manifest | null
  region: string
  field: FieldKey
  run: string | null // arşivlenmiş koşu (geçmiş tahmin); null = güncel
  timePos: number // kesirli adım index'i
  meta: ColormapMeta | null
  onClick: (lng: number, lat: number) => void
  marker: { lng: number; lat: number } | null
  flyTarget: { lng: number; lat: number; zoom?: number; nonce: number } | null
}

const SRC = 'wfe-field'
const LYR = 'wfe-field-layer'
const OSRC = 'wfe-outline'
const OLYR = 'wfe-outline-layer'

function firstSymbolId(map: maplibregl.Map): string | undefined {
  for (const l of map.getStyle().layers || []) if (l.type === 'symbol') return l.id
  return undefined
}

function outlineGeo(c: Quad): GeoJSON.Feature {
  return {
    type: 'Feature',
    properties: {},
    geometry: { type: 'Polygon', coordinates: [[...c, c[0]]] },
  }
}

interface HoverInfo {
  g0: Grid
  g1: Grid | null
  frac: number
  field: FieldKey
  elev: { m: Grid; h: Grid } | null
}

export default function MapView({
  theme,
  manifest,
  region,
  field,
  run,
  timePos,
  meta,
  onClick,
  marker,
  flyTarget,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chipRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const markerRef = useRef<maplibregl.Marker | null>(null)
  const glRef = useRef<GLPainter | null>(null)
  const cpuRef = useRef<FieldPainter | null>(null)
  const particlesRef = useRef<WindParticles | null>(null)
  const gridsRef = useRef(new Map<number, Grid>())
  const elevRef = useRef<{ m: Grid; h: Grid } | null>(null)
  const genRef = useRef(0)
  const srcDimsRef = useRef('')
  const hoverRef = useRef<HoverInfo | null>(null)
  const pauseTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const rafRef = useRef(0)
  const lastRegion = useRef<string | null>(null)

  if (!glRef.current) glRef.current = new GLPainter()
  if (!glRef.current.ok && !cpuRef.current) cpuRef.current = new FieldPainter()
  const paintCanvas = glRef.current.ok ? glRef.current.canvas : cpuRef.current!.canvas

  const lut = useMemo(() => (meta ? buildLUT(meta) : null), [meta])
  const runKey = run ?? 'cur'

  const stateRef = useRef({ manifest, region, field, timePos, lut, theme, meta, runKey })
  stateRef.current = { manifest, region, field, timePos, lut, theme, meta, runKey }

  /** Mevcut kesirli zamana göre alanı boya ve haritaya bindir. */
  const renderRef = useRef<() => void>(() => {})
  renderRef.current = () => {
    const map = mapRef.current
    const { manifest: man, timePos: pos, lut: l, field: f, region: reg, meta: mt } = stateRef.current
    if (!map || !man?.corners || !l || !mt || !man.steps.length) return
    if (!map.isStyleLoaded()) {
      map.once('idle', () => scheduleRender())
      return
    }
    const steps = man.steps
    const max = steps.length - 1
    const i0 = Math.min(Math.floor(pos), max)
    const i1 = Math.min(i0 + 1, max)
    const frac = Math.min(Math.max(pos - i0, 0), 1)
    let g0 = gridsRef.current.get(steps[i0].step) ?? null
    let g1 = i1 !== i0 ? gridsRef.current.get(steps[i1].step) ?? null : null
    if (!g0) {
      for (let d = 1; d <= max && !g0; d++) {
        g0 =
          gridsRef.current.get(steps[Math.max(0, i0 - d)]?.step) ??
          gridsRef.current.get(steps[Math.min(max, i0 + d)]?.step) ??
          null
      }
      g1 = null
    }
    if (!g0) return

    const elev = f === 't2m' ? elevRef.current : null
    hoverRef.current = { g0, g1, frac: g1 ? frac : 0, field: f, elev }

    const gl = glRef.current!
    if (gl.ok) {
      gl.setLUT(l, mt.field)
      const rk = stateRef.current.runKey
      const tex0 = gl.texture(`f/${reg}/${rk}/${f}/${steps[i0].step}`, g0.img)
      const tex1 = g1 ? gl.texture(`f/${reg}/${rk}/${f}/${steps[i1].step}`, g1.img) : null
      // çıkış çözünürlüğü: bölge başına sabit (K× model; arazi hires ile aynı)
      const outW = Math.min(g0.w * 6, 2048)
      const outH = Math.round((g0.h * outW) / g0.w)
      gl.paint({
        tex0,
        tex1,
        frac,
        outW,
        outH,
        gridW: g0.w,
        gridH: g0.h,
        dataRange: DATA_RANGES[f],
        lutRange: [l.lo, l.hi],
        enhance: elev
          ? {
              zh: gl.texture(`terr/${reg}/hires`, elev.h.img),
              zm: gl.texture(`terr/${reg}/model`, elev.m.img),
              elevRange: ELEV_RANGE,
              lapse: LAPSE_RATE,
            }
          : null,
      })
    } else {
      cpuRef.current!.paint(g0, g1, g1 ? frac : 0, l)
    }

    // maplibre canvas kaynağı
    const corners = man.corners as Quad
    const dims = `${paintCanvas.width}x${paintCanvas.height}`
    if (srcDimsRef.current !== dims && map.getLayer(LYR)) {
      map.removeLayer(LYR)
      map.removeSource(SRC)
    }
    let src = map.getSource(SRC) as maplibregl.CanvasSource | undefined
    if (!src) {
      map.addSource(SRC, {
        type: 'canvas',
        canvas: paintCanvas,
        coordinates: corners as unknown as Coords4,
        animate: false,
      })
      map.addLayer(
        {
          id: LYR,
          type: 'raster',
          source: SRC,
          paint: { 'raster-opacity': mt.opacity ?? 0.9, 'raster-fade-duration': 0, 'raster-resampling': 'linear' },
        },
        firstSymbolId(map)
      )
      src = map.getSource(SRC) as maplibregl.CanvasSource
      srcDimsRef.current = dims
    } else {
      src.setCoordinates(corners as unknown as Coords4)
      map.setPaintProperty(LYR, 'raster-opacity', mt.opacity ?? 0.9)
    }
    try {
      src.play()
      if (pauseTimer.current) clearTimeout(pauseTimer.current)
      pauseTimer.current = setTimeout(() => {
        try {
          ;(map.getSource(SRC) as maplibregl.CanvasSource | undefined)?.pause()
        } catch {
          /* yoksay */
        }
      }, 90)
    } catch {
      /* yoksay */
    }

    // alan sınırı
    const osrc = map.getSource(OSRC) as maplibregl.GeoJSONSource | undefined
    if (osrc) osrc.setData(outlineGeo(corners))
    else {
      map.addSource(OSRC, { type: 'geojson', data: outlineGeo(corners) })
      map.addLayer({
        id: OLYR,
        type: 'line',
        source: OSRC,
        paint: { 'line-color': '#7dd3fc', 'line-width': 1.1, 'line-opacity': 0.3, 'line-dasharray': [3, 2.5] },
      })
    }
  }

  const scheduleRender = () => {
    cancelAnimationFrame(rafRef.current)
    // gizli sekmede rAF askıya alınır — ilk boyamanın hiç olmaması yerine
    // zamanlayıcıya düş (sekme öne gelince kare hazır olur)
    if (document.hidden) {
      setTimeout(() => renderRef.current(), 16)
      return
    }
    rafRef.current = requestAnimationFrame(() => renderRef.current())
  }

  // ---- harita (bir kez) ----
  useEffect(() => {
    if (!containerRef.current) return
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: BASEMAP[theme],
      center: [35, 39],
      zoom: 4.6,
      attributionControl: false,
      dragRotate: false,
    })
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right')
    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-right')
    map.touchZoomRotate.disableRotation()
    map.getCanvas().style.cursor = 'crosshair'
    map.on('click', (e) => onClick(e.lngLat.lng, e.lngLat.lat))
    map.on('load', () => {
      srcDimsRef.current = ''
      scheduleRender()
    })
    map.on('styledata', () => {
      if (!map.getSource(SRC)) srcDimsRef.current = ''
      scheduleRender()
    })

    // hover değer okuması (arazi-düzeltmeli — boyanan pikselle tutarlı)
    const chip = chipRef.current
    map.on('mousemove', (e) => {
      const { manifest: man } = stateRef.current
      const hv = hoverRef.current
      if (!chip || !man?.corners || !hv) return
      const st = quadInverse(man.corners as Quad, e.lngLat.lng, e.lngLat.lat)
      if (!st) {
        chip.style.opacity = '0'
        return
      }
      const [s, t] = st
      const samp = (g: Grid) => sampleGrid(g.data, g.w, g.h, s * (g.w - 1), t * (g.h - 1))
      let v = samp(hv.g0)
      if (hv.g1 && hv.frac > 0) v = v + (samp(hv.g1) - v) * hv.frac
      if (hv.elev) v += LAPSE_RATE * (samp(hv.elev.m) - samp(hv.elev.h))
      chip.textContent = FIELD_DEFS[hv.field].fmt(v)
      chip.style.opacity = '1'
      chip.style.left = `${e.point.x}px`
      chip.style.top = `${e.point.y}px`
    })
    map.getCanvas().addEventListener('mouseleave', () => {
      if (chip) chip.style.opacity = '0'
    })

    particlesRef.current = new WindParticles(map)
    particlesRef.current.setTheme(theme === 'dark')
    mapRef.current = map
    return () => {
      particlesRef.current?.destroy()
      particlesRef.current = null
      map.remove()
      mapRef.current = null
      glRef.current?.destroy()
      glRef.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ---- bölge/alan/koşu değişimi: kareleri yükle ----
  useEffect(() => {
    if (!manifest?.steps.length || !manifest.available) return
    const gen = ++genRef.current
    gridsRef.current = new Map()
    hoverRef.current = null
    const version = manifest.init
    manifest.steps.forEach((s, i) => {
      loadScalar(region, field, s.step, version, run).then((g) => {
        if (gen !== genRef.current || !g) return
        gridsRef.current.set(s.step, g)
        const cur = stateRef.current.timePos
        if (Math.abs(i - cur) < 1.5 || gridsRef.current.size === manifest.steps.length) scheduleRender()
      })
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [region, field, run, manifest])

  // ---- arazi (bölge başına bir kez) ----
  useEffect(() => {
    elevRef.current = null
    let cancelled = false
    Promise.all([loadElev(region, 'model'), loadElev(region, 'hires')]).then(([m, h]) => {
      if (cancelled) return
      elevRef.current = m && h ? { m, h } : null
      scheduleRender()
    })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [region])

  // ---- zaman/lut değişimi ----
  useEffect(() => {
    scheduleRender()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timePos, lut])

  // ---- rüzgâr partikülleri ----
  const uvStep = manifest?.steps.length
    ? manifest.steps[Math.min(Math.round(timePos), manifest.steps.length - 1)].step
    : null
  useEffect(() => {
    const parts = particlesRef.current
    if (!parts) return
    parts.setTheme(theme === 'dark')
    // arşiv koşularında u/v bileşenleri yok — partiküller yalnız güncel koşuda
    if (field !== 'wind' || run || !manifest?.corners || uvStep === null) {
      parts.stop()
      return
    }
    let cancelled = false
    loadUV(region, uvStep, manifest.init).then((uv) => {
      if (cancelled || !uv || !manifest.corners) return
      const metersX = manifest.nx * manifest.dx_m
      const metersY = manifest.ny * manifest.dx_m
      parts.setWind(uv, manifest.corners as Quad, metersX, metersY)
      parts.start()
    })
    return () => {
      cancelled = true
    }
  }, [field, region, run, uvStep, manifest, theme])

  // ---- tema ----
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    srcDimsRef.current = ''
    map.setStyle(BASEMAP[theme])
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme])

  // ---- bölgeye uç ----
  useEffect(() => {
    const map = mapRef.current
    if (!map || !manifest?.bounds) return
    if (lastRegion.current === region) return
    lastRegion.current = region
    const [w, s, e, n] = manifest.bounds
    map.fitBounds(
      [
        [w, s],
        [e, n],
      ],
      { padding: { top: 70, bottom: 130, left: 70, right: 80 }, duration: 900 }
    )
  }, [region, manifest])

  // ---- işaretçi ----
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    if (!marker) {
      markerRef.current?.remove()
      markerRef.current = null
      return
    }
    if (!markerRef.current) {
      const el = document.createElement('div')
      el.innerHTML =
        '<svg width="28" height="38" viewBox="0 0 30 40"><path d="M15 0C6.7 0 0 6.7 0 15c0 10 15 25 15 25s15-15 15-25C30 6.7 23.3 0 15 0z" fill="#0f8bd0" stroke="#fff" stroke-width="2"/><circle cx="15" cy="15" r="5" fill="#fff"/></svg>'
      markerRef.current = new maplibregl.Marker({ element: el, anchor: 'bottom' })
    }
    markerRef.current.setLngLat([marker.lng, marker.lat]).addTo(map)
  }, [marker])

  // ---- şehir aramasında uç ----
  useEffect(() => {
    const map = mapRef.current
    if (!map || !flyTarget) return
    map.easeTo({
      center: [flyTarget.lng, flyTarget.lat],
      zoom: flyTarget.zoom ?? Math.max(map.getZoom(), 7.5),
      duration: 900,
    })
  }, [flyTarget])

  return (
    <div className="map-wrap" ref={containerRef}>
      <div ref={chipRef} className="hover-chip" style={{ opacity: 0 }} />
    </div>
  )
}
