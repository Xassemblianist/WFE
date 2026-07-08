import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { ThemeProvider } from './theme'
import '@fontsource-variable/inter'
import './index.css'
import 'maplibre-gl/dist/maplibre-gl.css'

// Not: StrictMode kasıtlı olarak kullanılmıyor — geliştirmede bileşenleri iki
// kez mount/unmount ederek MapLibre harita örneğini kurup hemen yıkıyor; bu da
// uçuş halindeki taban harita/overlay isteklerini iptal edip (ERR_ABORTED)
// konsolu kirletiyor. Tek-mount ile harita temiz kuruluyor.
createRoot(document.getElementById('root')!).render(
  <ThemeProvider>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </ThemeProvider>
)
