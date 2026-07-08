// Ücretsiz, anahtarsız MapLibre vektör taban haritaları (CARTO).
// Koyu/açık temaya göre seçilir; etiketler overlay üstünde kalır.
export const BASEMAP: Record<'dark' | 'light', string> = {
  dark: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
  light: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
}
