// El yazımı SVG ikon seti — tutarlı 24px stroke stili.
import type { SVGProps } from 'react'

type IP = SVGProps<SVGSVGElement> & { size?: number }

function Base({ size = 18, children, ...rest }: IP) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...rest}
    >
      {children}
    </svg>
  )
}

export const IcSearch = (p: IP) => (
  <Base {...p}>
    <circle cx="11" cy="11" r="7" />
    <path d="m20 20-3.2-3.2" />
  </Base>
)

export const IcPlay = (p: IP) => (
  <Base {...p} fill="currentColor" stroke="none">
    <path d="M8 5.8v12.4a.9.9 0 0 0 1.37.77l10-6.2a.9.9 0 0 0 0-1.54l-10-6.2A.9.9 0 0 0 8 5.8z" />
  </Base>
)

export const IcPause = (p: IP) => (
  <Base {...p} fill="currentColor" stroke="none">
    <rect x="6.5" y="5" width="3.6" height="14" rx="1.2" />
    <rect x="14" y="5" width="3.6" height="14" rx="1.2" />
  </Base>
)

export const IcClose = (p: IP) => (
  <Base {...p} strokeWidth={2.1}>
    <path d="M6 6l12 12M18 6L6 18" />
  </Base>
)

export const IcSun = (p: IP) => (
  <Base {...p}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 3v1.8M12 19.2V21M3 12h1.8M19.2 12H21M5.6 5.6l1.3 1.3M17.1 17.1l1.3 1.3M18.4 5.6l-1.3 1.3M6.9 17.1l-1.3 1.3" />
  </Base>
)

export const IcMoon = (p: IP) => (
  <Base {...p} fill="currentColor" stroke="none">
    <path d="M20.3 13.6A8.3 8.3 0 0 1 10.4 3.7a.6.6 0 0 0-.8-.7 9 9 0 1 0 11.4 11.4.6.6 0 0 0-.7-.8z" />
  </Base>
)

export const IcThermo = (p: IP) => (
  <Base {...p}>
    <path d="M10 13.6V5a2 2 0 1 1 4 0v8.6a4.4 4.4 0 1 1-4 0z" />
    <circle cx="12" cy="17.5" r="1.6" fill="currentColor" stroke="none" />
    <path d="M12 15.8V9" strokeWidth={1.4} />
  </Base>
)

export const IcWind = (p: IP) => (
  <Base {...p}>
    <path d="M3 8.5h10.5a2.5 2.5 0 1 0-2.4-3.2" />
    <path d="M3 13h15.5a2.8 2.8 0 1 1-2.6 3.8" />
    <path d="M3 17.4h7" />
  </Base>
)

export const IcRain = (p: IP) => (
  <Base {...p}>
    <path d="M7 15a4.5 4.5 0 0 1 1.1-8.86A5.5 5.5 0 0 1 18.6 7.5 3.8 3.8 0 0 1 18 15z" />
    <path d="M8.5 18.2 7.7 20.4M12.5 18.2l-.8 2.2M16.5 18.2l-.8 2.2" />
  </Base>
)

export const IcCloud = (p: IP) => (
  <Base {...p}>
    <path d="M6.5 18.5a4.5 4.5 0 0 1 .6-8.96A5.8 5.8 0 0 1 18.4 10a4.2 4.2 0 0 1-.9 8.5z" />
  </Base>
)

export const IcGauge = (p: IP) => (
  <Base {...p}>
    <path d="M4.5 17.5a8.5 8.5 0 1 1 15 0" />
    <path d="m12 14 3.8-4.6" />
    <circle cx="12" cy="14.5" r="1.4" fill="currentColor" stroke="none" />
  </Base>
)

export const IcPin = (p: IP) => (
  <Base {...p}>
    <path d="M12 21s-6.5-5.7-6.5-10.3a6.5 6.5 0 0 1 13 0C18.5 15.3 12 21 12 21z" />
    <circle cx="12" cy="10.5" r="2.3" />
  </Base>
)

export const IcLayers = (p: IP) => (
  <Base {...p}>
    <path d="m12 3 8.5 4.7L12 12.4 3.5 7.7 12 3z" />
    <path d="m4.5 12.2 7.5 4.2 7.5-4.2M4.5 16.4l7.5 4.2 7.5-4.2" />
  </Base>
)

export const IcArrowRight = (p: IP) => (
  <Base {...p}>
    <path d="M4 12h15M13.5 6.5 19 12l-5.5 5.5" />
  </Base>
)

export const IcGithub = (p: IP) => (
  <Base {...p} fill="currentColor" stroke="none">
    <path d="M12 2a10 10 0 0 0-3.16 19.5c.5.09.68-.22.68-.48v-1.7c-2.78.6-3.37-1.34-3.37-1.34-.45-1.16-1.11-1.47-1.11-1.47-.9-.62.07-.6.07-.6 1 .07 1.53 1.03 1.53 1.03.9 1.52 2.34 1.08 2.91.83.09-.65.35-1.09.63-1.34-2.22-.25-4.56-1.11-4.56-4.94 0-1.1.39-1.99 1.03-2.69-.1-.25-.45-1.27.1-2.64 0 0 .84-.27 2.75 1.02a9.58 9.58 0 0 1 5 0c1.91-1.3 2.75-1.02 2.75-1.02.55 1.37.2 2.39.1 2.64.64.7 1.03 1.6 1.03 2.69 0 3.84-2.34 4.68-4.57 4.93.36.31.68.92.68 1.85V21c0 .27.18.58.69.48A10 10 0 0 0 12 2z" />
  </Base>
)

/* ---- Hava durumu glifleri (çok renkli, dolgulu) ---- */
type WP = { size?: number }

export const WSun = ({ size = 28 }: WP) => (
  <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
    <circle cx="16" cy="16" r="6.5" fill="#fbbf24" />
    <g stroke="#fbbf24" strokeWidth="2.2" strokeLinecap="round">
      <path d="M16 3.5v3M16 25.5v3M3.5 16h3M25.5 16h3M7.2 7.2l2.1 2.1M22.7 22.7l2.1 2.1M24.8 7.2l-2.1 2.1M9.3 22.7l-2.1 2.1" />
    </g>
  </svg>
)

export const WPartly = ({ size = 28 }: WP) => (
  <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
    <circle cx="12" cy="11" r="5.5" fill="#fbbf24" />
    <path
      d="M11 25a5 5 0 0 1 1-9.9A6.2 6.2 0 0 1 24 16.6a4.6 4.6 0 0 1-1 9.4z"
      fill="#cbd5e1"
    />
  </svg>
)

export const WCloud = ({ size = 28 }: WP) => (
  <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
    <path
      d="M9 24a5.5 5.5 0 0 1 1.2-10.86A6.8 6.8 0 0 1 23.4 14 5 5 0 0 1 22.5 24z"
      fill="#a8b6c8"
    />
  </svg>
)

export const WRain = ({ size = 28 }: WP) => (
  <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
    <path
      d="M9 20a5.5 5.5 0 0 1 1.2-10.86A6.8 6.8 0 0 1 23.4 10 5 5 0 0 1 22.5 20z"
      fill="#9db0c4"
    />
    <g stroke="#60a5fa" strokeWidth="2.2" strokeLinecap="round">
      <path d="M11 23.5 10 26.5M16 23.5l-1 3M21 23.5l-1 3" />
    </g>
  </svg>
)

export const WWindy = ({ size = 28 }: WP) => (
  <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden>
    <g stroke="#7dd3fc" strokeWidth="2.4" strokeLinecap="round" fill="none">
      <path d="M4 11h14.5a3.2 3.2 0 1 0-3-4.2" />
      <path d="M4 17h20a3.4 3.4 0 1 1-3.2 4.6" />
      <path d="M4 23h9" />
    </g>
  </svg>
)
