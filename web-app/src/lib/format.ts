// Zaman ve sayı biçimlendirme yardımcıları (Türkçe, yerel saat).

const DAYS = ['Paz', 'Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt']
const MONTHS = ['Oca', 'Şub', 'Mar', 'Nis', 'May', 'Haz', 'Tem', 'Ağu', 'Eyl', 'Eki', 'Kas', 'Ara']

/** ISO zaman -> "Sal 14:00" (yerel saat) */
export function fmtValidShort(iso: string): string {
  const d = new Date(iso)
  return `${DAYS[d.getDay()]} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

/** ISO -> "6 Tem, Salı 14:00" */
export function fmtValidLong(iso: string): string {
  const d = new Date(iso)
  return `${d.getDate()} ${MONTHS[d.getMonth()]}, ${fullDay(d.getDay())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

/** ISO -> "6 Tem 15:00" (kısa, gün+saat) */
export function fmtDayHour(iso: string): string {
  const d = new Date(iso)
  return `${pad(d.getDate())} ${MONTHS[d.getMonth()]} ${pad(d.getHours())}:00`
}

export function fmtHourOnly(iso: string): string {
  const d = new Date(iso)
  return `${pad(d.getHours())}:00`
}

export function fmtInit(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return `${d.getDate()} ${MONTHS[d.getMonth()]} ${pad(d.getHours())}:00Z başlangıç`
}

function fullDay(i: number): string {
  return ['Pazar', 'Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi'][i]
}
function pad(n: number): string {
  return n < 10 ? `0${n}` : `${n}`
}

export function round1(n: number | null | undefined): string {
  if (n === null || n === undefined) return '—'
  return (Math.round(n * 10) / 10).toString()
}

/** Rüzgâr yön/şiddet -> kısa tanım (yalnızca şiddet var). */
export function windLabel(ms: number | null): string {
  if (ms === null) return '—'
  if (ms < 0.5) return 'sakin'
  if (ms < 3.3) return 'hafif'
  if (ms < 7.9) return 'orta'
  if (ms < 13.8) return 'kuvvetli'
  if (ms < 20.7) return 'fırtınamsı'
  return 'fırtına'
}
