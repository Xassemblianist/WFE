import { useMemo, useState } from 'react'
import citiesData from '../data/cities.json'
import { IcSearch } from './icons'

export interface City {
  name: string
  province: string
  lat: number
  lon: number
}

const CITIES = citiesData as City[]

/** Türkçe duyarsız normalleştirme. */
function norm(s: string): string {
  return s
    .toLocaleLowerCase('tr-TR')
    .replaceAll('ı', 'i')
    .replaceAll('ş', 's')
    .replaceAll('ğ', 'g')
    .replaceAll('ü', 'u')
    .replaceAll('ö', 'o')
    .replaceAll('ç', 'c')
    .trim()
}

interface Props {
  onSelect: (c: City) => void
}

export default function SearchBar({ onSelect }: Props) {
  const [q, setQ] = useState('')
  const [open, setOpen] = useState(false)
  const [hl, setHl] = useState(0)

  const results = useMemo(() => {
    const nq = norm(q)
    if (!nq) return []
    const starts: City[] = []
    const contains: City[] = []
    for (const c of CITIES) {
      const nn = norm(c.name)
      if (nn.startsWith(nq)) starts.push(c)
      else if (nn.includes(nq) || norm(c.province).startsWith(nq)) contains.push(c)
    }
    return [...starts, ...contains].slice(0, 8)
  }, [q])

  function pick(c: City) {
    setQ('')
    setOpen(false)
    onSelect(c)
  }

  return (
    <div className="search-pill glass">
      <IcSearch size={17} />
      <input
        value={q}
        placeholder="Şehir ara…"
        aria-label="Şehir ara"
        onChange={(e) => {
          setQ(e.target.value)
          setOpen(true)
          setHl(0)
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 160)}
        onKeyDown={(e) => {
          if (!results.length) return
          if (e.key === 'ArrowDown') {
            e.preventDefault()
            setHl((h) => Math.min(h + 1, results.length - 1))
          } else if (e.key === 'ArrowUp') {
            e.preventDefault()
            setHl((h) => Math.max(h - 1, 0))
          } else if (e.key === 'Enter') {
            e.preventDefault()
            pick(results[hl])
          } else if (e.key === 'Escape') {
            setOpen(false)
          }
        }}
      />
      {open && results.length > 0 && (
        <div className="suggestions">
          {results.map((c, i) => (
            <button
              key={`${c.name}-${c.province}-${i}`}
              className={i === hl ? 'hl' : ''}
              onMouseDown={(e) => {
                // input blur'u tıklamayı yutmasın
                e.preventDefault()
                pick(c)
              }}
              onMouseEnter={() => setHl(i)}
            >
              <span>{c.name}</span>
              <span className="prov">{c.province}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
