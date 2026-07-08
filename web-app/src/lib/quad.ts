// Dört-köşe (quad) georeferans matematiği.
// Köşe sırası [TL, TR, BR, BL] (satır 0 = kuzey), her köşe [lng, lat].
// s ∈ [0,1] batı→doğu, t ∈ [0,1] kuzey→güney — veri görüntüsüyle aynı yön.

export type Quad = [number, number][]

/** (s,t) → [lng,lat] bilineer. */
export function quadPoint(q: Quad, s: number, t: number): [number, number] {
  const [tl, tr, br, bl] = q
  const topLng = tl[0] + (tr[0] - tl[0]) * s
  const botLng = bl[0] + (br[0] - bl[0]) * s
  const topLat = tl[1] + (tr[1] - tl[1]) * s
  const botLat = bl[1] + (br[1] - bl[1]) * s
  return [topLng + (botLng - topLng) * t, topLat + (botLat - topLat) * t]
}

/**
 * [lng,lat] → (s,t) ters bilineer (Newton, 4 iterasyon).
 * Alan dışındaysa null (küçük toleransla).
 */
export function quadInverse(q: Quad, lng: number, lat: number): [number, number] | null {
  let s = 0.5
  let t = 0.5
  for (let i = 0; i < 5; i++) {
    const [x, y] = quadPoint(q, s, t)
    const rx = x - lng
    const ry = y - lat
    // Jacobian (analitik)
    const [tl, tr, br, bl] = q
    const dxds = (tr[0] - tl[0]) * (1 - t) + (br[0] - bl[0]) * t
    const dxdt = (bl[0] - tl[0]) * (1 - s) + (br[0] - tr[0]) * s
    const dyds = (tr[1] - tl[1]) * (1 - t) + (br[1] - bl[1]) * t
    const dydt = (bl[1] - tl[1]) * (1 - s) + (br[1] - tr[1]) * s
    const det = dxds * dydt - dxdt * dyds
    if (Math.abs(det) < 1e-12) return null
    s -= (rx * dydt - ry * dxdt) / det
    t -= (ry * dxds - rx * dyds) / det
  }
  const eps = 0.002
  if (s < -eps || s > 1 + eps || t < -eps || t > 1 + eps) return null
  return [Math.min(1, Math.max(0, s)), Math.min(1, Math.max(0, t))]
}
