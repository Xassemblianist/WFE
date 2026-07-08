import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import MapView from '../components/MapView'
import LayerRail from '../components/LayerRail'
import TimeBar, { SPEEDS } from '../components/TimeBar'
import PointPanel from '../components/PointPanel'
import SearchBar, { type City } from '../components/SearchBar'
import { useTheme } from '../theme'
import {
  api,
  type Manifest,
  type ColormapMeta,
  type FieldKey,
  type PointForecast,
  type RegionInfo,
  type RunInfo,
} from '../api'

const RUN_MONTHS = ['Oca', 'Şub', 'Mar', 'Nis', 'May', 'Haz', 'Tem', 'Ağu', 'Eyl', 'Eki', 'Kas', 'Ara']

/** "2026070806" → "8 Tem 06Z" */
function runLabel(cycle: string): string {
  const mo = parseInt(cycle.slice(4, 6), 10) - 1
  const d = parseInt(cycle.slice(6, 8), 10)
  return `${d} ${RUN_MONTHS[mo]} ${cycle.slice(8)}Z`
}

function inBounds(m: Manifest | undefined, lat: number, lon: number): boolean {
  if (!m?.bounds) return false
  const [w, s, e, n] = m.bounds
  return lon >= w && lon <= e && lat >= s && lat <= n
}

/** "Antalya (2.5 km)" -> "Antalya · 2.5km" */
function regionLabel(title: string): string {
  const m = title.match(/^(.*?)\s*\(([\d.]+)\s*km\)/)
  return m ? `${m[1]} · ${m[2]}km` : title
}

export default function MapPage() {
  const { theme } = useTheme()
  const [params, setParams] = useSearchParams()

  const [regions, setRegions] = useState<RegionInfo[]>([])
  const [manifests, setManifests] = useState<Record<string, Manifest>>({})
  const [region, setRegion] = useState<string>(params.get('region') || 'turkey')
  const [colormaps, setColormaps] = useState<Record<string, ColormapMeta>>({})
  const [field, setField] = useState<FieldKey>((params.get('field') as FieldKey) || 't2m')
  const [runs, setRuns] = useState<RunInfo[]>([])
  const [run, setRun] = useState<string | null>(null) // null = güncel koşu
  const [runManifest, setRunManifest] = useState<Manifest | null>(null)
  const [timePos, setTimePos] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speedIdx, setSpeedIdx] = useState(1)

  const [marker, setMarker] = useState<{ lng: number; lat: number } | null>(null)
  const [flyTarget, setFlyTarget] = useState<{ lng: number; lat: number; zoom?: number; nonce: number } | null>(null)
  const [panelOpen, setPanelOpen] = useState(false)
  const [point, setPoint] = useState<PointForecast | null>(null)
  const [pointLoading, setPointLoading] = useState(false)
  const [pointError, setPointError] = useState<string | null>(null)
  const [pointTitle, setPointTitle] = useState('Seçili nokta')
  const nonceRef = useRef(0)

  const manifest = run ? runManifest : manifests[region] || null
  const steps = useMemo(() => manifest?.steps ?? [], [manifest])
  const fields = (manifest?.fields ?? ['t2m', 'wind', 'precip', 'cloud', 'mslp']) as FieldKey[]
  const activeCmap = colormaps[field] || null

  // ---- koşu listesi (bölge başına) + arşiv manifesti ----
  useEffect(() => {
    setRun(null)
    setRunManifest(null)
    api.runs(region).then(setRuns).catch(() => setRuns([]))
  }, [region])

  useEffect(() => {
    if (!run) {
      setRunManifest(null)
      return
    }
    let cancelled = false
    api.manifest(region, run).then((m) => {
      if (!cancelled) {
        setRunManifest(m)
        setTimePos(0)
        setPlaying(false)
      }
    })
    return () => {
      cancelled = true
    }
  }, [region, run])

  // arşiv koşusunda mevcut olmayan katmandan güncel katmana düş
  useEffect(() => {
    if (manifest?.fields && !manifest.fields.includes(field)) setField('t2m')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [manifest])

  // ---- ilk yükleme ----
  useEffect(() => {
    ;(async () => {
      try {
        const [regs, cmaps] = await Promise.all([api.regions(), api.colormaps()])
        setRegions(regs)
        const cm: Record<string, ColormapMeta> = {}
        cmaps.forEach((c) => (cm[c.field] = c))
        setColormaps(cm)
        const entries = await Promise.all(
          regs.map(async (r) => {
            try {
              return [r.id, await api.manifest(r.id)] as const
            } catch {
              return [r.id, null] as const
            }
          })
        )
        const mm: Record<string, Manifest> = {}
        entries.forEach(([id, m]) => {
          if (m) mm[id] = m
        })
        setManifests(mm)
        setRegion((cur) => (mm[cur]?.available ? cur : Object.keys(mm).find((k) => mm[k].available) || cur))
      } catch (e) {
        console.error('API yüklenemedi', e)
      }
    })()
  }, [])

  // ---- derin bağlantı: ?lat=&lon=&name= ----
  useEffect(() => {
    const lat = parseFloat(params.get('lat') || '')
    const lon = parseFloat(params.get('lon') || '')
    if (!isNaN(lat) && !isNaN(lon) && Object.keys(manifests).length) {
      selectPoint(lat, lon, params.get('name') || undefined)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [manifests])

  // ---- URL senkronu ----
  useEffect(() => {
    const p = new URLSearchParams(params)
    p.set('region', region)
    p.set('field', field)
    setParams(p, { replace: true })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [region, field])

  // ---- akıcı oynatma (kesirli zaman, ~30fps) ----
  useEffect(() => {
    if (!playing || steps.length < 2) return
    let raf = 0
    let last = performance.now()
    let acc = 0
    const sps = SPEEDS[speedIdx].sps
    const tick = (now: number) => {
      acc += now - last
      last = now
      if (acc >= 33) {
        const dt = acc / 1000
        acc = 0
        setTimePos((p) => {
          const next = p + dt * sps
          return next >= steps.length - 1 ? 0 : next
        })
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, speedIdx, steps.length])

  // adım sayısı değişince kelepçele
  useEffect(() => {
    setTimePos((p) => Math.min(p, Math.max(steps.length - 1, 0)))
  }, [steps.length])

  const focusISO = useMemo(() => {
    if (!steps.length || !manifest?.init) return null
    const max = steps.length - 1
    const i0 = Math.min(Math.floor(timePos), max)
    const i1 = Math.min(i0 + 1, max)
    const fh = steps[i0].fhour + (steps[i1].fhour - steps[i0].fhour) * (timePos - i0)
    return new Date(new Date(manifest.init).getTime() + fh * 3600 * 1000).toISOString()
  }, [steps, timePos, manifest])

  // ---- nokta seçimi ----
  const selectPoint = useCallback(
    async (lat: number, lon: number, name?: string) => {
      const ids = Object.keys(manifests)
      let target = region
      const containing = ids.filter((r) => manifests[r].available && inBounds(manifests[r], lat, lon))
      if (containing.length) {
        target = containing.sort((a, b) => (manifests[a].dx_m || 0) - (manifests[b].dx_m || 0))[0]
      }
      // arşiv koşusu görüntülenirken bölge terfisi yapma — aynı koşudan oku
      if (run) target = region
      if (target !== region) setRegion(target)
      setMarker({ lng: lon, lat })
      setPanelOpen(true)
      setPointTitle(name || 'Seçili nokta')
      setPointLoading(true)
      setPointError(null)
      setPoint(null)
      try {
        const d = await api.point(target, lat, lon, run)
        if (d.error) {
          setPointError('Bu nokta model alanının dışında.')
        } else {
          setPoint(d)
          if (!name) setPointTitle(`${d.grid.grid_lat.toFixed(2)}°, ${d.grid.grid_lon.toFixed(2)}°`)
        }
      } catch {
        setPointError('Tahmin alınamadı — API çalışıyor mu?')
      } finally {
        setPointLoading(false)
      }
    },
    [manifests, region, run]
  )

  const onCity = (c: City) => {
    setFlyTarget({ lng: c.lon, lat: c.lat, zoom: 8, nonce: ++nonceRef.current })
    selectPoint(c.lat, c.lon, c.name)
  }

  const availableRegions = regions.filter((r) => manifests[r.id]?.available)

  return (
    <div className="map-page">
      <MapView
        theme={theme}
        manifest={manifest}
        region={region}
        field={field}
        run={run}
        timePos={timePos}
        meta={activeCmap}
        onClick={(lng, lat) => selectPoint(lat, lng)}
        marker={marker}
        flyTarget={flyTarget}
      />

      {!manifest && (
        <div className="center-msg">
          <div className="spinner" />
        </div>
      )}
      {manifest && !manifest.available && <div className="center-msg">Bu bölge için henüz koşu yok.</div>}

      <div className="map-topleft">
        <SearchBar onSelect={onCity} />
        {availableRegions.length > 1 && (
          <div className="region-seg glass">
            {availableRegions.map((r) => (
              <button
                key={r.id}
                className={region === r.id ? 'active' : ''}
                onClick={() => setRegion(r.id)}
                title={r.title}
              >
                {regionLabel(r.title)}
              </button>
            ))}
          </div>
        )}
        {runs.length > 1 && (
          <div className="run-select glass">
            <span className="run-label">Koşu</span>
            <select
              value={run ?? ''}
              onChange={(e) => setRun(e.target.value || null)}
              aria-label="Model koşusu seç"
            >
              {runs.map((r) => (
                <option key={r.cycle} value={r.current ? '' : r.cycle}>
                  {runLabel(r.cycle)}
                  {r.current ? ' · güncel' : ''}
                </option>
              ))}
            </select>
            {run && <span className="run-badge">geçmiş</span>}
          </div>
        )}
      </div>

      <LayerRail fields={fields} value={field} onChange={setField} />

      <TimeBar
        steps={steps}
        initISO={manifest?.init ?? null}
        timePos={timePos}
        playing={playing}
        speedIdx={speedIdx}
        meta={activeCmap}
        onScrub={(v) => {
          setTimePos(v)
          setPlaying(false)
        }}
        onTogglePlay={() => setPlaying((p) => !p)}
        onSpeed={setSpeedIdx}
      />

      <PointPanel
        open={panelOpen}
        loading={pointLoading}
        data={point}
        error={pointError}
        title={pointTitle}
        focusISO={focusISO}
        onClose={() => setPanelOpen(false)}
      />
    </div>
  )
}
